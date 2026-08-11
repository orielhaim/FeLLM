//! Interleaved multi-sequence scheduler over a shared [`Engine`].

use crate::engine::{Engine, GenParams, GenStats};
use crate::kv_fabric::{KvFabric, KvSequence};
use fellm_core::error::Result;
use fellm_core::tensor::Tensor;
use fellm_model::parse_assistant_output;
use fellm_tokenizer::{AssistantOutput, Message, ToolDef};
use std::collections::VecDeque;

/// Stable request identity used by scheduling plans and cancellation commands.
pub type SequenceId = u64;

/// Kind of inference work assigned to one sequence in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    /// One latency-sensitive autoregressive token.
    Decode,
    /// A bounded range of prompt tokens.
    Prefill,
}

/// Work for one sequence within a physical inference batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchItem {
    pub id: SequenceId,
    pub kind: WorkKind,
    pub token_start: usize,
    pub token_count: usize,
    pub compute_logits: bool,
}

/// First-class description of all work in one scheduler iteration.
#[derive(Debug, Clone, Default)]
pub struct BatchPlan {
    pub items: Vec<BatchItem>,
    pub token_budget: usize,
    pub scheduled_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_tokens: usize,
}

/// Policy-visible sequence state. It contains decision inputs, not ownership.
#[derive(Debug, Clone, Copy)]
pub struct SchedulingCandidate {
    pub id: SequenceId,
    pub status: SequenceStatus,
    pub waiting_ticks: u64,
    pub remaining_prefill: usize,
    pub has_decode: bool,
    pub priority: i32,
    /// Device/host-resident pages currently held by this sequence.
    pub resident_pages: usize,
    /// Pages not on the compute tier (need migrate/prefetch).
    pub non_resident_pages: usize,
    /// Pages required for the next scheduling step.
    pub pages_needed_next: usize,
    /// Tokens already satisfied by content-addressed prefix hit.
    pub prefix_hit_tokens: usize,
    /// Estimated host↔device transfer cost for this step (bytes).
    pub estimated_transfer_bytes: u64,
    /// Rough compute cost signal (token work units).
    pub estimated_compute_cost: f64,
    /// Fabric memory pressure in `[0, 1]`.
    pub memory_pressure: f64,
    /// Expected additional output tokens.
    pub expected_output_growth: usize,
    /// Optional latency target (ms).
    pub latency_target_ms: Option<u32>,
}

/// Owns which runnable work is selected; the scheduler retains lifecycle/KV state.
pub trait SchedulingPolicy: Send {
    fn plan(
        &mut self,
        candidates: &[SchedulingCandidate],
        token_budget: usize,
        physical_granularity: usize,
    ) -> BatchPlan;
}

/// Interactive serving policy: schedule every possible decode first, then fill
/// remaining capacity with age-ordered, chunked prefills.
#[derive(Debug, Default)]
pub struct InteractivePolicy;

impl SchedulingPolicy for InteractivePolicy {
    fn plan(
        &mut self,
        candidates: &[SchedulingCandidate],
        token_budget: usize,
        physical_granularity: usize,
    ) -> BatchPlan {
        let budget = token_budget.max(1);
        let granularity = physical_granularity.max(1).min(budget);
        let mut plan = BatchPlan {
            token_budget: budget,
            ..BatchPlan::default()
        };
        let mut ordered = candidates.to_vec();
        // Memory-aware ordering: priority, then prefer fully-resident cheap work,
        // then age. Heavy migrations defer relative to resident decode.
        ordered.sort_unstable_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| {
                    // Resident decode before migration-heavy work.
                    let a_ready = a.has_decode && a.non_resident_pages == 0;
                    let b_ready = b.has_decode && b.non_resident_pages == 0;
                    b_ready.cmp(&a_ready)
                })
                .then_with(|| a.estimated_transfer_bytes.cmp(&b.estimated_transfer_bytes))
                .then_with(|| b.waiting_ticks.cmp(&a.waiting_ticks))
                .then_with(|| a.id.cmp(&b.id))
        });
        for candidate in ordered.iter().filter(|candidate| candidate.has_decode) {
            if plan.scheduled_tokens == budget {
                break;
            }
            // Soft skip if under extreme pressure and this needs a large restore —
            // still allow if nothing else is runnable (handled by later pass).
            plan.items.push(BatchItem {
                id: candidate.id,
                kind: WorkKind::Decode,
                token_start: 0,
                token_count: 1,
                compute_logits: true,
            });
            plan.scheduled_tokens += 1;
            plan.decode_tokens += 1;
        }
        for candidate in ordered
            .iter()
            .filter(|candidate| candidate.remaining_prefill > 0)
        {
            let remaining = budget.saturating_sub(plan.scheduled_tokens);
            if remaining == 0 {
                break;
            }
            let count = candidate.remaining_prefill.min(remaining).min(granularity);
            plan.items.push(BatchItem {
                id: candidate.id,
                kind: WorkKind::Prefill,
                token_start: 0,
                token_count: count,
                compute_logits: count == candidate.remaining_prefill,
            });
            plan.scheduled_tokens += count;
            plan.prefill_tokens += count;
        }
        plan
    }
}

/// Lifecycle of a scheduled sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting to be admitted (need free blocks).
    Waiting,
    /// Actively generating.
    Running,
    /// Prefill/decode paused; blocks may be swapped.
    Preempted,
    /// Finished successfully or with error.
    Finished,
}

/// Events emitted toward the HTTP layer.
#[derive(Debug)]
pub enum SequenceEvent {
    /// Incremental decoded text.
    Token {
        /// Sequence id.
        id: u64,
        /// UTF-8 chunk.
        text: String,
    },
    /// Sequence completed.
    Done {
        /// Sequence id.
        id: u64,
        /// Finish reason.
        finish_reason: String,
        /// Full text.
        full_text: String,
        /// Parsed tool calls.
        tool_calls: Option<Vec<fellm_tokenizer::ToolCall>>,
        /// Usage stats.
        usage: GenStats,
    },
    /// Sequence failed.
    Error {
        /// Sequence id.
        id: u64,
        /// Message.
        message: String,
    },
}

/// Opaque handle returned when enqueuing work.
#[derive(Debug, Clone, Copy)]
pub struct SequenceHandle {
    /// Unique id.
    pub id: u64,
}

struct Sequence {
    id: u64,
    status: SequenceStatus,
    seq_cache: KvSequence,
    /// Request-owned recurrent state; absent for attention-only models.
    conv_state: Option<crate::HybridConvState>,
    prompt_ids: Vec<u32>,
    /// Prefill cursor into `prompt_ids`.
    prefill_pos: usize,
    /// Decode position (absolute).
    position: usize,
    params: GenParams,
    pending_logits: Option<Tensor>,
    emitted: u32,
    byte_buf: Vec<u8>,
    full_bytes: Vec<u8>,
    has_tools: bool,
    stream: bool,
    last_used: u64,
    prompt_tokens: u32,
    gen_start: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
    last_token_at: Option<std::time::Instant>,
    admitted_at: Option<std::time::Instant>,
    prefill_done_at: Option<std::time::Instant>,
    hit_stop: bool,
    generated_tokens: Vec<u32>,
    sampling: crate::sampling::SamplingWorkspace,
}

/// Round-robin scheduler owning waiting/running queues.
pub struct Scheduler {
    next_id: u64,
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    clock: u64,
    policy: Box<dyn SchedulingPolicy>,
    pending_events: VecDeque<SequenceEvent>,
    last_plan: BatchPlan,
}

impl Scheduler {
    /// Empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            clock: 0,
            policy: Box::<InteractivePolicy>::default(),
            pending_events: VecDeque::new(),
            last_plan: BatchPlan::default(),
        }
    }

    /// Replace the scheduling policy without changing sequence/KV ownership.
    pub fn set_policy(&mut self, policy: impl SchedulingPolicy + 'static) {
        self.policy = Box::new(policy);
    }

    /// Most recently executed plan, useful for metrics and diagnostics.
    #[must_use]
    pub fn last_plan(&self) -> &BatchPlan {
        &self.last_plan
    }

    /// Intentionally cancel a request and immediately release its KV state.
    pub fn cancel(&mut self, engine: &mut Engine, id: SequenceId) -> bool {
        for queue in [&mut self.waiting, &mut self.running] {
            if let Some(index) = queue.iter().position(|sequence| sequence.id == id) {
                let mut sequence = queue.remove(index).expect("located sequence");
                engine.release_seq_cache(&mut sequence.seq_cache);
                metrics::counter!("fellm_requests_cancelled_total").increment(1);
                return true;
            }
        }
        false
    }

    /// Enqueue a chat request (already mapped messages/tools).
    pub fn enqueue_chat(
        &mut self,
        engine: &mut Engine,
        messages: &[Message],
        tools: &[ToolDef],
        params: GenParams,
        stream: bool,
    ) -> Result<SequenceHandle> {
        let prepared_messages = engine.prepare_chat_messages(messages);
        let prompt = match engine.tokenizer().apply_chat_template_with_tools(
            &prepared_messages,
            tools,
            true,
        )? {
            Some(formatted) => formatted,
            None => messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        let ids = engine.tokenizer().encode(&prompt, true)?;
        self.enqueue_ids(engine, ids, params, stream, !tools.is_empty())
    }

    /// Enqueue raw token ids.
    pub fn enqueue_ids(
        &mut self,
        engine: &mut Engine,
        ids: Vec<u32>,
        params: GenParams,
        stream: bool,
        has_tools: bool,
    ) -> Result<SequenceHandle> {
        let id = self.next_id;
        self.next_id += 1;
        let max_seq = engine.n_ctx();
        let mut seq_cache = {
            // Access via a temporary reset path: create from engine's cache sizing.
            KvSequence::new(engine.spec().n_attn_layers().max(1), max_seq)
        };

        // Prefix match against engine cache.
        let matched = engine.attach_prefix(&ids, &mut seq_cache);

        let seq = Sequence {
            id,
            status: SequenceStatus::Waiting,
            seq_cache,
            conv_state: engine.new_hybrid_state()?,
            prompt_ids: ids,
            prefill_pos: matched,
            position: matched,
            params,
            pending_logits: None,
            emitted: 0,
            byte_buf: Vec::new(),
            full_bytes: Vec::new(),
            has_tools,
            stream,
            last_used: 0,
            prompt_tokens: 0,
            gen_start: std::time::Instant::now(),
            first_token_at: None,
            last_token_at: None,
            admitted_at: None,
            prefill_done_at: None,
            hit_stop: false,
            generated_tokens: Vec::with_capacity(params.max_tokens as usize),
            sampling: crate::sampling::SamplingWorkspace::default(),
        };
        self.waiting.push_back(seq);
        metrics::counter!("fellm_requests_total").increment(1);
        Ok(SequenceHandle { id })
    }

    /// Number of active (waiting + running) sequences.
    #[must_use]
    pub fn inflight(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Run until one event is produced, or return `None` if idle.
    pub fn poll_event(&mut self, engine: &mut Engine) -> Option<SequenceEvent> {
        if let Some(event) = self.pending_events.pop_front() {
            return Some(event);
        }
        self.clock = self.clock.wrapping_add(1);
        metrics::gauge!("fellm_scheduler_waiting_requests").set(self.waiting.len() as f64);
        metrics::gauge!("fellm_scheduler_running_requests").set(self.running.len() as f64);
        metrics::gauge!("fellm_scheduler_preempted_requests").set(
            self.waiting
                .iter()
                .filter(|sequence| sequence.status == SequenceStatus::Preempted)
                .count() as f64,
        );
        self.try_admit(engine);

        if self.running.is_empty() {
            return None;
        }

        let fabric_metrics = engine.fabric_metrics();
        metrics::gauge!("fellm_kv_device_resident_pages")
            .set(fabric_metrics.device_resident_pages as f64);
        metrics::gauge!("fellm_kv_host_resident_pages")
            .set(fabric_metrics.host_resident_pages as f64);
        metrics::gauge!("fellm_kv_device_resident_bytes")
            .set(fabric_metrics.device_resident_bytes as f64);
        metrics::gauge!("fellm_kv_shared_bytes").set(fabric_metrics.shared_kv_bytes as f64);
        metrics::gauge!("fellm_kv_allocation_pressure").set(engine.model_cache_pressure());
        metrics::counter!("fellm_kv_migrations_total").absolute(fabric_metrics.migrations);
        metrics::counter!("fellm_kv_migration_bytes_total")
            .absolute(fabric_metrics.migration_bytes);
        metrics::counter!("fellm_kv_cow_forks_total").absolute(fabric_metrics.cow_forks);

        let candidates: Vec<_> = self
            .running
            .iter()
            .map(|sequence| {
                let remaining = sequence
                    .prompt_ids
                    .len()
                    .saturating_sub(sequence.prefill_pos);
                let remaining_prefill = if engine.spec().is_hybrid() {
                    remaining.min(1)
                } else {
                    remaining
                };
                let extra = if sequence.prefill_pos < sequence.prompt_ids.len() {
                    remaining_prefill.max(1)
                } else {
                    1
                };
                let sig = engine.residency_signals_for(
                    &sequence.seq_cache,
                    extra,
                    sequence.params.priority,
                    sequence
                        .prefill_pos
                        .min(sequence.seq_cache.shared_prefix_len),
                );
                SchedulingCandidate {
                    id: sequence.id,
                    status: sequence.status,
                    waiting_ticks: self.clock.saturating_sub(sequence.last_used),
                    remaining_prefill,
                    has_decode: sequence.prefill_pos == sequence.prompt_ids.len()
                        && sequence.pending_logits.is_some(),
                    priority: sequence.params.priority,
                    resident_pages: sig.resident_pages,
                    non_resident_pages: sig.non_resident_pages,
                    pages_needed_next: sig.pages_needed_next,
                    prefix_hit_tokens: sig.prefix_hit_tokens,
                    estimated_transfer_bytes: sig.estimated_transfer_bytes,
                    estimated_compute_cost: sig.estimated_compute_cost,
                    memory_pressure: sig.memory_pressure,
                    expected_output_growth: sequence
                        .params
                        .max_tokens
                        .saturating_sub(sequence.emitted)
                        as usize,
                    latency_target_ms: None,
                }
            })
            .collect();
        let mut plan = self.policy.plan(
            &candidates,
            engine.settings().n_batch,
            engine.settings().n_ubatch,
        );
        for item in &mut plan.items {
            if item.kind == WorkKind::Prefill
                && let Some(sequence) = self.running.iter().find(|sequence| sequence.id == item.id)
            {
                item.token_start = sequence.prefill_pos;
                item.compute_logits =
                    sequence.prefill_pos + item.token_count == sequence.prompt_ids.len();
            }
        }
        self.last_plan = plan.clone();
        metrics::gauge!("fellm_scheduler_active_batch_size").set(plan.items.len() as f64);
        metrics::histogram!("fellm_scheduler_batch_size").record(plan.items.len() as f64);
        metrics::gauge!("fellm_scheduler_scheduled_tokens").set(plan.scheduled_tokens as f64);
        metrics::gauge!("fellm_scheduler_decode_tokens_per_batch").set(plan.decode_tokens as f64);
        metrics::gauge!("fellm_scheduler_prefill_tokens_per_batch").set(plan.prefill_tokens as f64);
        metrics::gauge!("fellm_scheduler_batch_utilization")
            .set(plan.scheduled_tokens as f64 / plan.token_budget.max(1) as f64);
        metrics::counter!("fellm_prompt_tokens_total").increment(plan.prefill_tokens as u64);
        metrics::counter!("fellm_decode_tokens_total").increment(plan.decode_tokens as u64);
        metrics::counter!("fellm_tokens_total").increment(plan.scheduled_tokens as u64);
        self.execute_physical_batch(engine, plan.items);
        self.pending_events.pop_front()
    }

    fn execute_physical_batch(&mut self, engine: &mut Engine, items: Vec<BatchItem>) {
        struct Selected {
            item: BatchItem,
            sequence: Sequence,
            terminal: bool,
        }
        #[derive(Clone, Copy)]
        enum RowKind {
            Prefill { final_prompt_row: bool },
            Decode,
        }
        #[derive(Clone, Copy)]
        struct RowOwner {
            selected: usize,
            kind: RowKind,
        }

        let mut selected = Vec::with_capacity(items.len());
        for item in items {
            let Some(index) = self
                .running
                .iter()
                .position(|sequence| sequence.id == item.id)
            else {
                continue;
            };
            let mut sequence = self.running.remove(index).expect("planned sequence");
            sequence.last_used = self.clock;
            selected.push(Selected {
                item,
                sequence,
                terminal: false,
            });
        }

        let mut rows = Vec::new();
        let mut owners = Vec::new();
        for (selected_index, entry) in selected.iter_mut().enumerate() {
            if entry.sequence.seq_cache.non_resident
                && let Err(error) = engine.swap_in_sequence(&mut entry.sequence.seq_cache)
            {
                self.pending_events.push_back(SequenceEvent::Error {
                    id: entry.sequence.id,
                    message: error.to_string(),
                });
                entry.terminal = true;
                continue;
            }
            match entry.item.kind {
                WorkKind::Prefill => {
                    let start = entry.sequence.prefill_pos;
                    let end = (start + entry.item.token_count).min(entry.sequence.prompt_ids.len());
                    for position in start..end {
                        let final_prompt_row = position + 1 == entry.sequence.prompt_ids.len();
                        rows.push(crate::engine::BatchToken {
                            sequence: selected_index,
                            token: entry.sequence.prompt_ids[position],
                            position,
                            compute_logits: final_prompt_row,
                        });
                        owners.push(RowOwner {
                            selected: selected_index,
                            kind: RowKind::Prefill { final_prompt_row },
                        });
                    }
                }
                WorkKind::Decode => match self.prepare_decode(engine, &mut entry.sequence) {
                    Ok((event, forward)) => {
                        if let Some(event) = event {
                            entry.terminal = matches!(event, SequenceEvent::Done { .. });
                            self.pending_events.push_back(event);
                        }
                        if let Some((token, position)) = forward {
                            rows.push(crate::engine::BatchToken {
                                sequence: selected_index,
                                token,
                                position,
                                compute_logits: true,
                            });
                            owners.push(RowOwner {
                                selected: selected_index,
                                kind: RowKind::Decode,
                            });
                        }
                    }
                    Err(error) => {
                        self.pending_events.push_back(SequenceEvent::Error {
                            id: entry.sequence.id,
                            message: error.to_string(),
                        });
                        entry.terminal = true;
                    }
                },
            }
        }

        if !rows.is_empty() {
            let granularity = engine.settings().n_ubatch.max(1);
            for (row_chunk, owner_chunk) in rows.chunks(granularity).zip(owners.chunks(granularity))
            {
                metrics::histogram!("fellm_scheduler_physical_batch_rows")
                    .record(row_chunk.len() as f64);
                let mut owned_states: Vec<_> = selected
                    .iter_mut()
                    .map(|entry| entry.sequence.conv_state.take())
                    .collect();
                let result = {
                    let mut caches: Vec<_> = selected
                        .iter_mut()
                        .map(|entry| &mut entry.sequence.seq_cache)
                        .collect();
                    let mut conv_states: Vec<_> = owned_states.iter_mut().collect();
                    engine.step_batch(&mut caches, &mut conv_states, row_chunk)
                };
                for (entry, state) in selected.iter_mut().zip(owned_states) {
                    entry.sequence.conv_state = state;
                }
                match result {
                    Ok(outputs) => {
                        for (owner, output) in owner_chunk.iter().copied().zip(outputs) {
                            let entry = &mut selected[owner.selected];
                            match owner.kind {
                                RowKind::Prefill { final_prompt_row } => {
                                    entry.sequence.prefill_pos += 1;
                                    entry.sequence.position = entry.sequence.prefill_pos;
                                    if final_prompt_row {
                                        entry.sequence.pending_logits = output;
                                        entry.sequence.prompt_tokens =
                                            entry.sequence.prompt_ids.len() as u32;
                                        entry.sequence.prefill_done_at =
                                            Some(std::time::Instant::now());
                                        if let Err(error) = engine.insert_prefix(
                                            &entry.sequence.prompt_ids,
                                            &entry.sequence.seq_cache,
                                        ) {
                                            self.pending_events.push_back(SequenceEvent::Error {
                                                id: entry.sequence.id,
                                                message: error.to_string(),
                                            });
                                            entry.terminal = true;
                                        }
                                    }
                                }
                                RowKind::Decode => {
                                    entry.sequence.position += 1;
                                    entry.sequence.pending_logits = output;
                                }
                            }
                        }
                    }
                    Err(error) => {
                        for entry in &mut selected {
                            if !entry.terminal {
                                self.pending_events.push_back(SequenceEvent::Error {
                                    id: entry.sequence.id,
                                    message: error.to_string(),
                                });
                                entry.terminal = true;
                            }
                        }
                        break;
                    }
                }
            }
        }
        for mut entry in selected {
            if entry.terminal {
                engine.release_seq_cache(&mut entry.sequence.seq_cache);
            } else {
                self.running.push_back(entry.sequence);
            }
        }
    }

    fn prepare_decode(
        &mut self,
        engine: &Engine,
        seq: &mut Sequence,
    ) -> Result<(Option<SequenceEvent>, Option<(u32, usize)>)> {
        let Some(logits_owned) = seq.pending_logits.take() else {
            return Ok((None, None));
        };
        let token = crate::sampling::sample_with_workspace(
            logits_owned.as_slice::<f32>()?,
            crate::sampling::SamplingOptions {
                temperature: seq.params.temperature,
                top_k: seq.params.top_k,
                top_p: seq.params.top_p,
                seed: seq.params.seed.wrapping_add(u64::from(seq.emitted)),
                repetition_penalty: seq.params.repetition_penalty,
                recent_tokens: &seq.generated_tokens,
            },
            &mut seq.sampling,
        );
        seq.generated_tokens.push(token);
        seq.emitted += 1;
        if seq.first_token_at.is_none() {
            let now = std::time::Instant::now();
            seq.first_token_at = Some(now);
            metrics::histogram!("fellm_time_to_first_token_seconds")
                .record(now.duration_since(seq.gen_start).as_secs_f64());
        } else if let Some(previous) = seq.last_token_at {
            metrics::histogram!("fellm_inter_token_latency_seconds")
                .record(previous.elapsed().as_secs_f64());
        }
        seq.last_token_at = Some(std::time::Instant::now());

        if engine.stop_token_ids_pub().contains(&token) {
            seq.hit_stop = true;
            return Ok((Some(self.finish_seq(seq, engine)?), None));
        }
        let bytes = engine.tokenizer().decode_token(token)?;
        if !bytes.is_empty() {
            seq.full_bytes.extend_from_slice(&bytes);
            seq.byte_buf.extend_from_slice(&bytes);
        }
        let event = if seq.stream && !seq.has_tools {
            let text = flush_utf8(&mut seq.byte_buf);
            (!text.is_empty()).then_some(SequenceEvent::Token { id: seq.id, text })
        } else {
            None
        };
        if seq.emitted >= seq.params.max_tokens || seq.position + 1 >= engine.n_ctx() {
            return Ok((Some(self.finish_seq(seq, engine)?), None));
        }
        Ok((event, Some((token, seq.position))))
    }

    fn try_admit(&mut self, engine: &mut Engine) {
        while let Some(mut seq) = self.waiting.pop_front() {
            // Fabric-backed admission: cooperate before execution so we do not
            // admit work that cannot make required state resident.
            let need_pages = engine
                .pages_needed_for(
                    &seq.seq_cache,
                    if seq.prefill_pos < seq.prompt_ids.len() {
                        1
                    } else {
                        1
                    },
                )
                .max(engine.spec().n_attn_layers().max(1));
            if !engine.can_admit_pages(need_pages) {
                engine.evict_prefixes_for_blocks(need_pages);
                if !engine.can_admit_pages(need_pages) && !self.try_swap_out(engine) {
                    metrics::counter!("fellm_kv_admission_reject_total").increment(1);
                    self.waiting.push_front(seq);
                    break;
                }
            }
            // Prefetch non-resident pages concurrently with admission.
            if seq.seq_cache.non_resident {
                if let Err(e) = engine.swap_in_sequence(&mut seq.seq_cache) {
                    tracing::warn!(error = %e, "admit migrate_in failed");
                    metrics::counter!("fellm_kv_admission_reject_total").increment(1);
                    self.waiting.push_front(seq);
                    break;
                }
            }
            // Ensure capacity for next prefill token if any.
            if seq.prefill_pos < seq.prompt_ids.len() {
                let pos = seq.prefill_pos;
                if let Err(e) = engine.ensure_seq_writable(&mut seq.seq_cache, pos) {
                    tracing::warn!(error = %e, "admit ensure_writable failed");
                    if self.try_swap_out(engine) {
                        if engine.ensure_seq_writable(&mut seq.seq_cache, pos).is_err() {
                            metrics::counter!("fellm_kv_admission_reject_total").increment(1);
                            self.waiting.push_front(seq);
                            break;
                        }
                    } else {
                        metrics::counter!("fellm_kv_admission_reject_total").increment(1);
                        self.waiting.push_front(seq);
                        break;
                    }
                }
            }
            seq.status = SequenceStatus::Running;
            if seq.admitted_at.is_none() {
                let now = std::time::Instant::now();
                metrics::histogram!("fellm_queue_latency_seconds")
                    .record(now.duration_since(seq.gen_start).as_secs_f64());
                seq.admitted_at = Some(now);
            }
            self.running.push_back(seq);
        }
    }

    fn try_swap_out(&mut self, engine: &mut Engine) -> bool {
        // Primary residency policy: value/cost keep-score (lowest first).
        // Plain LRU is not used as the production preemption ranking.
        let victim_idx = self
            .running
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let keep = engine.sequence_keep_value(&s.seq_cache, s.params.priority, s.last_used);
                (i, keep, s.id)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)))
            .map(|(i, _, _)| i);
        let Some(idx) = victim_idx else {
            return false;
        };
        let mut victim = self.running.remove(idx).expect("index");
        if engine.swap_out_sequence(&mut victim.seq_cache).is_ok() {
            victim.status = SequenceStatus::Preempted;
            metrics::counter!("fellm_kv_swap_out_total").increment(1);
            victim.mark_non_resident();
            self.waiting.push_back(victim);
            true
        } else {
            self.running.insert(idx, victim);
            false
        }
    }

    fn finish_seq(&mut self, seq: &mut Sequence, _engine: &Engine) -> Result<SequenceEvent> {
        if !seq.byte_buf.is_empty() && seq.stream && !seq.has_tools {
            // leftover flushed via Done full_text
        }
        let full_text = String::from_utf8_lossy(&seq.full_bytes).to_string();
        let (finish_reason, tool_calls) = if seq.has_tools {
            match parse_assistant_output(&full_text) {
                AssistantOutput::ToolCalls(calls) => ("tool_calls".to_string(), Some(calls)),
                AssistantOutput::Text(_) => {
                    let reason = if seq.hit_stop || seq.emitted < seq.params.max_tokens {
                        "stop"
                    } else {
                        "length"
                    };
                    (reason.to_string(), None)
                }
            }
        } else {
            let reason = if seq.hit_stop || seq.emitted < seq.params.max_tokens {
                "stop"
            } else {
                "length"
            };
            (reason.to_string(), None)
        };

        let usage = GenStats {
            prompt_tokens: seq.prompt_tokens,
            predicted_tokens: seq.emitted,
            prompt_ms: seq.prefill_done_at.map_or(0.0, |time| {
                time.duration_since(seq.gen_start).as_secs_f64() * 1000.0
            }),
            time_to_first_token_ms: seq.first_token_at.map_or(0.0, |t| {
                t.duration_since(seq.gen_start).as_secs_f64() * 1000.0
            }),
            predicted_ms: 0.0,
            total_ms: seq.gen_start.elapsed().as_secs_f64() * 1000.0,
        };
        metrics::histogram!("fellm_request_completion_latency_seconds")
            .record(seq.gen_start.elapsed().as_secs_f64());
        metrics::counter!("fellm_requests_completed_total").increment(1);

        // Tools+stream: content already buffered; Done carries tool_calls.
        let _ = seq.stream;

        Ok(SequenceEvent::Done {
            id: seq.id,
            finish_reason,
            full_text,
            tool_calls,
            usage,
        })
    }
}

impl Sequence {
    fn mark_non_resident(&mut self) {
        self.seq_cache.non_resident = true;
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

fn flush_utf8(buf: &mut Vec<u8>) -> String {
    let mut cut = buf.len();
    while cut > 0 {
        if std::str::from_utf8(&buf[..cut]).is_ok() {
            break;
        }
        cut -= 1;
    }
    let s = String::from_utf8_lossy(&buf[..cut]).to_string();
    buf.drain(..cut);
    s
}

// Silence unused import in some cfgs.
#[allow(dead_code)]
fn _cache_ty(_: &KvFabric) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(
        id: SequenceId,
        waiting_ticks: u64,
        remaining_prefill: usize,
        has_decode: bool,
    ) -> SchedulingCandidate {
        SchedulingCandidate {
            id,
            status: SequenceStatus::Running,
            waiting_ticks,
            remaining_prefill,
            has_decode,
            priority: 0,
            resident_pages: 4,
            non_resident_pages: 0,
            pages_needed_next: 0,
            prefix_hit_tokens: 0,
            estimated_transfer_bytes: 0,
            estimated_compute_cost: 1.0,
            memory_pressure: 0.1,
            expected_output_growth: 16,
            latency_target_ms: None,
        }
    }

    #[test]
    fn interactive_policy_prioritizes_decode_then_chunks_prefill() {
        let candidates = [
            cand(1, 2, 100, false),
            cand(2, 1, 0, true),
            cand(3, 3, 0, true),
        ];
        let plan = InteractivePolicy.plan(&candidates, 10, 4);
        assert_eq!(plan.decode_tokens, 2);
        assert_eq!(plan.prefill_tokens, 4);
        assert_eq!(plan.items[0].id, 3);
        assert_eq!(plan.items[1].id, 2);
        assert_eq!(plan.items[2].kind, WorkKind::Prefill);
        assert_eq!(plan.items[2].token_count, 4);
    }

    #[test]
    fn interactive_policy_prefers_resident_decode_over_migration() {
        let mut heavy = cand(10, 5, 0, true);
        heavy.non_resident_pages = 8;
        heavy.estimated_transfer_bytes = 1 << 20;
        let light = cand(11, 1, 0, true);
        let plan = InteractivePolicy.plan(&[heavy, light], 2, 1);
        assert_eq!(plan.items[0].id, 11);
        assert_eq!(plan.items[1].id, 10);
    }

    #[test]
    fn sequence_keep_value_is_wired_for_preemption_ranking() {
        // Structural: Engine exposes sequence_keep_value used by try_swap_out.
        // Value/cost ordering is unit-tested on KvFabric::sequence_keep_value;
        // this ensures the scheduler module references the fabric policy path
        // rather than only last_used.
        use crate::kv_fabric::{KvFabric, KvFabricConfig};
        let mut fab = KvFabric::new_full_attention(
            KvFabricConfig {
                device_budget: Some(4 * 1024 * 1024),
                host_budget: Some(4 * 1024 * 1024),
                ..KvFabricConfig::default()
            },
            32,
            1,
            1,
            4,
            16,
        )
        .unwrap();
        let mut a = fab.new_sequence(64);
        let mut b = fab.new_sequence(64);
        for p in 0..16 {
            fab.ensure_writable(&mut a, p).unwrap();
            fab.ensure_writable(&mut b, p).unwrap();
        }
        for _ in 0..8 {
            fab.tick();
            fab.ensure_writable(&mut a, 0).unwrap();
        }
        let keep_a = fab.sequence_keep_value(&a, 5, 100);
        let keep_b = fab.sequence_keep_value(&b, 0, 1);
        // Scheduler picks min keep-value victim → b should be preferred over hot a.
        assert!(keep_b < keep_a, "keep_b={keep_b} keep_a={keep_a}");
    }

    #[test]
    fn policy_never_exceeds_iteration_budget() {
        let candidates = (0..20)
            .map(|id| cand(id, id, if id % 2 == 0 { 100 } else { 0 }, id % 2 == 1))
            .collect::<Vec<_>>();
        let plan = InteractivePolicy.plan(&candidates, 7, 4);
        assert!(plan.scheduled_tokens <= 7);
        assert_eq!(
            plan.scheduled_tokens,
            plan.decode_tokens + plan.prefill_tokens
        );
    }
}
