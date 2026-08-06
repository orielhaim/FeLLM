//! Interleaved multi-sequence scheduler over a shared [`Engine`].

use crate::engine::{Engine, GenParams, GenStats};
use crate::paged::{CacheManager, SequenceCache};
use fellm_core::error::Result;
use fellm_core::tensor::Tensor;
use fellm_model::parse_assistant_output;
use fellm_tokenizer::{AssistantOutput, Message, ToolDef};
use std::collections::VecDeque;

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
    seq_cache: SequenceCache,
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
    hit_stop: bool,
}

/// Round-robin scheduler owning waiting/running queues.
pub struct Scheduler {
    next_id: u64,
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    clock: u64,
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
        }
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
        let prompt = match engine
            .tokenizer()
            .apply_chat_template_with_tools(messages, tools, true)?
        {
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
            SequenceCache::new(engine.spec().n_attn_layers().max(1), max_seq)
        };

        // Prefix match against engine cache.
        let matched = engine.attach_prefix(&ids, &mut seq_cache);

        let seq = Sequence {
            id,
            status: SequenceStatus::Waiting,
            seq_cache,
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
            hit_stop: false,
        };
        self.waiting.push_back(seq);
        Ok(SequenceHandle { id })
    }

    /// Number of active (waiting + running) sequences.
    #[must_use]
    pub fn inflight(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Run until one event is produced, or return `None` if idle.
    pub fn poll_event(&mut self, engine: &mut Engine) -> Option<SequenceEvent> {
        self.clock = self.clock.wrapping_add(1);
        self.try_admit(engine);

        if self.running.is_empty() {
            return None;
        }

        // Round-robin: pop front, step, push back if still running.
        let mut seq = self.running.pop_front()?;
        seq.last_used = self.clock;
        seq.status = SequenceStatus::Running;

        match self.step_one(engine, &mut seq) {
            Ok(Some(ev)) => {
                if matches!(ev, SequenceEvent::Done { .. } | SequenceEvent::Error { .. }) {
                    engine.release_seq_cache(&mut seq.seq_cache);
                    Some(ev)
                } else {
                    self.running.push_back(seq);
                    Some(ev)
                }
            }
            Ok(None) => {
                self.running.push_back(seq);
                None
            }
            Err(e) => {
                let id = seq.id;
                engine.release_seq_cache(&mut seq.seq_cache);
                Some(SequenceEvent::Error {
                    id,
                    message: e.to_string(),
                })
            }
        }
    }

    fn try_admit(&mut self, engine: &mut Engine) {
        while let Some(mut seq) = self.waiting.pop_front() {
            // Need at least one free block per layer to start / continue.
            let need = engine.spec().n_attn_layers().max(1);
            if engine.cache_free_blocks() < need {
                // Try swap of LRU running/preempted — for simplicity swap oldest running.
                if !self.try_swap_out(engine) {
                    self.waiting.push_front(seq);
                    break;
                }
            }
            // Ensure capacity for next prefill token if any.
            if seq.prefill_pos < seq.prompt_ids.len() {
                let pos = seq.prefill_pos;
                if let Err(e) = engine.ensure_seq_writable(&mut seq.seq_cache, pos) {
                    // OOM — preempt and requeue.
                    tracing::warn!(error = %e, "admit ensure_writable failed");
                    if self.try_swap_out(engine) {
                        let _ = engine.ensure_seq_writable(&mut seq.seq_cache, pos);
                    } else {
                        self.waiting.push_front(seq);
                        break;
                    }
                }
            }
            seq.status = SequenceStatus::Running;
            self.running.push_back(seq);
        }
    }

    fn try_swap_out(&mut self, engine: &mut Engine) -> bool {
        // Pick LRU from running (except we may have none finished).
        let victim_idx = self
            .running
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_used)
            .map(|(i, _)| i);
        let Some(idx) = victim_idx else {
            return false;
        };
        let mut victim = self.running.remove(idx).expect("index");
        if engine.swap_out_sequence(&mut victim.seq_cache).is_ok() {
            victim.status = SequenceStatus::Preempted;
            victim.swapped_mark();
            self.waiting.push_back(victim);
            true
        } else {
            self.running.insert(idx, victim);
            false
        }
    }

    fn step_one(
        &mut self,
        engine: &mut Engine,
        seq: &mut Sequence,
    ) -> Result<Option<SequenceEvent>> {
        // Resume from swap if needed.
        if seq.seq_cache.swapped {
            engine.swap_in_sequence(&mut seq.seq_cache)?;
        }

        // Prefill remaining prompt tokens.
        if seq.prefill_pos < seq.prompt_ids.len() {
            let tok = seq.prompt_ids[seq.prefill_pos];
            let pos = seq.prefill_pos;
            let need_logits = seq.prefill_pos + 1 == seq.prompt_ids.len();
            let logits = engine.step_sequence(&mut seq.seq_cache, tok, pos, need_logits)?;
            seq.prefill_pos += 1;
            seq.position = seq.prefill_pos;
            if need_logits {
                seq.pending_logits = Some(logits);
                seq.prompt_tokens = seq.prompt_ids.len() as u32;
                // Insert prefix into radix tree for sharing.
                engine.insert_prefix(&seq.prompt_ids, &seq.seq_cache);
            }
            return Ok(None);
        }

        // Decode
        let Some(mut logits_owned) = seq.pending_logits.take() else {
            return Ok(None);
        };
        let tok = if let Ok(work) = logits_owned.as_mut_slice::<f32>() {
            crate::sampling::sample(
                work,
                seq.params.temperature,
                seq.params.top_k,
                seq.params.top_p,
                seq.params.seed.wrapping_add(u64::from(seq.emitted)),
            )
        } else {
            let logits = logits_owned.as_slice::<f32>()?;
            let mut work = logits.to_vec();
            crate::sampling::sample(
                &mut work,
                seq.params.temperature,
                seq.params.top_k,
                seq.params.top_p,
                seq.params.seed.wrapping_add(u64::from(seq.emitted)),
            )
        };
        seq.emitted += 1;
        if seq.first_token_at.is_none() {
            seq.first_token_at = Some(std::time::Instant::now());
        }

        let stop_ids = engine.stop_token_ids_pub();
        if stop_ids.contains(&tok) {
            seq.hit_stop = true;
            return Ok(Some(self.finish_seq(seq, engine)?));
        }

        let bytes = engine.tokenizer().decode_token(tok)?;
        if !bytes.is_empty() {
            seq.full_bytes.extend_from_slice(&bytes);
            seq.byte_buf.extend_from_slice(&bytes);
        }

        let emit_live = seq.stream && !seq.has_tools;
        let mut token_ev = None;
        if emit_live {
            let chunk = flush_utf8(&mut seq.byte_buf);
            if !chunk.is_empty() {
                token_ev = Some(SequenceEvent::Token {
                    id: seq.id,
                    text: chunk,
                });
            }
        }

        if seq.emitted >= seq.params.max_tokens || seq.position + 1 >= engine.n_ctx() {
            return Ok(Some(self.finish_seq(seq, engine)?));
        }

        let next = engine.step_sequence(&mut seq.seq_cache, tok, seq.position, true)?;
        seq.position += 1;
        seq.pending_logits = Some(next);

        Ok(token_ev)
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
            prompt_ms: 0.0,
            time_to_first_token_ms: seq.first_token_at.map_or(0.0, |t| {
                t.duration_since(seq.gen_start).as_secs_f64() * 1000.0
            }),
            predicted_ms: 0.0,
            total_ms: seq.gen_start.elapsed().as_secs_f64() * 1000.0,
        };

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
    fn swapped_mark(&mut self) {
        self.seq_cache.swapped = true;
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
fn _cache_ty(_: &CacheManager) {}
