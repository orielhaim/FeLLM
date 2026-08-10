//! Dedicated OS thread that owns [`Engine`] and drains the task queue via
//! the interleaved [`Scheduler`].

use crate::state::{InferenceTask, WorkerEvent};
use fellm_runtime::{Engine, EngineSettings, Scheduler, SequenceEvent};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::mpsc;

/// Spawn the blocking inference worker with multi-sequence scheduling.
pub fn spawn_worker(
    model_path: &Path,
    settings: EngineSettings,
    mut task_rx: mpsc::Receiver<InferenceTask>,
) -> Result<std::thread::JoinHandle<()>, String> {
    let model_path = model_path.to_path_buf();
    let handle = std::thread::Builder::new()
        .name("fellm-inference".into())
        .spawn(move || {
            let mut engine = match Engine::open_with(&model_path, settings) {
                Ok(e) => e,
                Err(err) => {
                    tracing::error!(error = %err, "failed to open model");
                    while let Some(task) = task_rx.blocking_recv() {
                        let _ = task
                            .reply
                            .send(WorkerEvent::Error(format!("model load failed: {err}")));
                    }
                    return;
                }
            };
            tracing::info!("inference worker ready (paged scheduler)");
            metrics::gauge!("fellm_backend_active", "backend" => engine.backend_id()).set(1.0);
            metrics::gauge!("fellm_model_bytes").set(engine.model_bytes() as f64);
            metrics::gauge!("fellm_activation_arena_bytes")
                .set(engine.activation_arena_bytes() as f64);
            metrics::gauge!("fellm_kv_capacity_bytes").set(engine.cache_bytes() as f64);
            if let Some(memory) = engine.backend_memory_info() {
                metrics::gauge!("fellm_device_memory_total_bytes").set(memory.total_bytes as f64);
                metrics::gauge!("fellm_device_memory_available_bytes")
                    .set(memory.available_bytes as f64);
            }

            let mut scheduler = Scheduler::new();
            // seq_id → reply channel
            let mut replies: HashMap<u64, ActiveReply> = HashMap::new();

            loop {
                let total_blocks = engine.cache_total_blocks();
                let free_blocks = engine.cache_free_blocks();
                metrics::gauge!("fellm_kv_blocks_free").set(free_blocks as f64);
                metrics::gauge!("fellm_kv_blocks_used")
                    .set(total_blocks.saturating_sub(free_blocks) as f64);
                metrics::gauge!("fellm_kv_cache_utilization").set(
                    total_blocks.saturating_sub(free_blocks) as f64 / total_blocks.max(1) as f64,
                );
                let prefix = engine.prefix_cache_stats();
                metrics::counter!("fellm_prefix_hits_total").absolute(prefix.hits);
                metrics::counter!("fellm_prefix_misses_total").absolute(prefix.misses);
                metrics::counter!("fellm_prefix_hit_tokens_total").absolute(prefix.hit_tokens);
                metrics::counter!("fellm_prefix_miss_tokens_total").absolute(prefix.miss_tokens);
                metrics::counter!("fellm_prefix_evictions_total").absolute(prefix.evictions);
                metrics::counter!("fellm_prefix_evicted_tokens_total")
                    .absolute(prefix.evicted_tokens);
                metrics::counter!("fellm_prefix_tokens_saved_total").absolute(prefix.tokens_saved);
                metrics::gauge!("fellm_prefix_cached_tokens").set(prefix.cached_tokens as f64);
                metrics::gauge!("fellm_prefix_cache_blocks").set(prefix.occupied_blocks as f64);
                metrics::gauge!("fellm_prefix_cache_bytes").set(prefix.occupied_bytes as f64);
                // Cancellation is an explicit request-lifetime signal shared with
                // the async HTTP side; response-channel state is not the oracle.
                let cancelled: Vec<u64> = replies
                    .iter()
                    .filter_map(|(&id, reply)| reply.cancellation.is_cancelled().then_some(id))
                    .collect();
                for id in cancelled {
                    scheduler.cancel(&mut engine, id);
                    replies.remove(&id);
                }
                // Non-blocking admit of new tasks when possible.
                while let Ok(task) = task_rx.try_recv() {
                    admit_task(&mut engine, &mut scheduler, &mut replies, task);
                }

                // If idle, block for the next task.
                if scheduler.inflight() == 0 {
                    match task_rx.blocking_recv() {
                        Some(task) => {
                            admit_task(&mut engine, &mut scheduler, &mut replies, task);
                        }
                        None => break,
                    }
                    continue;
                }

                // Also drain any pending without blocking.
                while let Ok(task) = task_rx.try_recv() {
                    admit_task(&mut engine, &mut scheduler, &mut replies, task);
                }

                match scheduler.poll_event(&mut engine) {
                    Some(SequenceEvent::Token { id, text }) => {
                        if let Some(reply) = replies.get(&id) {
                            let _ = reply.tx.send(WorkerEvent::Token { text });
                        }
                    }
                    Some(SequenceEvent::Done {
                        id,
                        finish_reason,
                        full_text,
                        tool_calls,
                        usage,
                    }) => {
                        if let Some(reply) = replies.remove(&id) {
                            let _ = reply.tx.send(WorkerEvent::Done {
                                finish_reason,
                                full_text,
                                tool_calls,
                                usage,
                            });
                        }
                    }
                    Some(SequenceEvent::Error { id, message }) => {
                        if let Some(reply) = replies.remove(&id) {
                            let _ = reply.tx.send(WorkerEvent::Error(message));
                        }
                    }
                    None => {
                        // No progress — wait for a new task or spin briefly.
                        std::thread::yield_now();
                    }
                }
            }
            tracing::info!("inference worker shutting down");
        })
        .map_err(|e| format!("failed to spawn inference thread: {e}"))?;
    Ok(handle)
}

fn admit_task(
    engine: &mut Engine,
    scheduler: &mut Scheduler,
    replies: &mut HashMap<u64, ActiveReply>,
    task: InferenceTask,
) {
    let InferenceTask {
        messages,
        tools,
        params,
        stream,
        reply,
        cancellation,
    } = task;
    match scheduler.enqueue_chat(engine, &messages, &tools, params, stream) {
        Ok(handle) => {
            replies.insert(
                handle.id,
                ActiveReply {
                    tx: reply,
                    cancellation,
                },
            );
        }
        Err(err) => {
            let _ = reply.send(WorkerEvent::Error(err.to_string()));
        }
    }
}

struct ActiveReply {
    tx: mpsc::UnboundedSender<WorkerEvent>,
    cancellation: tokio_util::sync::CancellationToken,
}
