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

            let mut scheduler = Scheduler::new();
            // seq_id → reply channel
            let mut replies: HashMap<u64, mpsc::UnboundedSender<WorkerEvent>> = HashMap::new();

            loop {
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
                        if let Some(tx) = replies.get(&id) {
                            let _ = tx.send(WorkerEvent::Token { text });
                        }
                    }
                    Some(SequenceEvent::Done {
                        id,
                        finish_reason,
                        full_text,
                        tool_calls,
                        usage,
                    }) => {
                        if let Some(tx) = replies.remove(&id) {
                            let _ = tx.send(WorkerEvent::Done {
                                finish_reason,
                                full_text,
                                tool_calls,
                                usage,
                            });
                        }
                    }
                    Some(SequenceEvent::Error { id, message }) => {
                        if let Some(tx) = replies.remove(&id) {
                            let _ = tx.send(WorkerEvent::Error(message));
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
    replies: &mut HashMap<u64, mpsc::UnboundedSender<WorkerEvent>>,
    task: InferenceTask,
) {
    let InferenceTask {
        messages,
        tools,
        params,
        stream,
        reply,
    } = task;
    match scheduler.enqueue_chat(engine, &messages, &tools, params, stream) {
        Ok(handle) => {
            replies.insert(handle.id, reply);
        }
        Err(err) => {
            let _ = reply.send(WorkerEvent::Error(err.to_string()));
        }
    }
}
