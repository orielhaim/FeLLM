//! Dedicated OS thread that owns [`Engine`] and drains the task queue.

use crate::state::{InferenceTask, WorkerEvent};
use fellm_runtime::{
    AssistantOutput, Engine, EngineSettings, GenParams, Message, ToolDef, parse_assistant_output,
};
use std::path::Path;
use tokio::sync::mpsc;

/// Spawn the blocking inference worker.
///
/// Loads the model once, then processes tasks serially until the queue closes.
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
                    // Drain remaining senders so HTTP handlers fail cleanly.
                    while let Some(task) = task_rx.blocking_recv() {
                        let _ = task
                            .reply
                            .send(WorkerEvent::Error(format!("model load failed: {err}")));
                    }
                    return;
                }
            };
            tracing::info!("inference worker ready");

            while let Some(task) = task_rx.blocking_recv() {
                run_task(&mut engine, task);
            }
            tracing::info!("inference worker shutting down");
        })
        .map_err(|e| format!("failed to spawn inference thread: {e}"))?;
    Ok(handle)
}

fn run_task(engine: &mut Engine, task: InferenceTask) {
    let InferenceTask {
        messages,
        tools,
        params,
        stream,
        reply,
    } = task;

    let want_stream = stream;
    let has_tools = !tools.is_empty();
    // When tools are present, buffer tokens so we can emit a clean final
    // tool_calls payload instead of broken partial JSON over SSE.
    let emit_live = want_stream && !has_tools;

    if let Err(err) = generate(
        engine,
        &messages,
        &tools,
        params,
        emit_live,
        want_stream,
        has_tools,
        &reply,
    ) {
        let _ = reply.send(WorkerEvent::Error(err));
    }
}

fn generate(
    engine: &mut Engine,
    messages: &[Message],
    tools: &[ToolDef],
    params: GenParams,
    emit_live: bool,
    want_stream: bool,
    has_tools: bool,
    reply: &mpsc::UnboundedSender<WorkerEvent>,
) -> Result<(), String> {
    let stop_ids = {
        let mut s = Vec::new();
        if let Some(eos) = engine.tokenizer().eos() {
            s.push(eos);
        }
        s
    };

    let mut token_stream = engine
        .chat_with_tools(messages, tools, params)
        .map_err(|e| e.to_string())?;

    let mut byte_buf: Vec<u8> = Vec::new();
    let mut full_bytes: Vec<u8> = Vec::new();
    let mut hit_stop = false;
    let mut emitted = 0u32;

    while let Some(tok_result) = token_stream.next() {
        let tok = tok_result.map_err(|e| e.to_string())?;
        emitted = emitted.saturating_add(1);

        if stop_ids.contains(&tok) {
            hit_stop = true;
            continue;
        }

        let bytes = token_stream.decode_token(tok).map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            continue;
        }
        full_bytes.extend_from_slice(&bytes);
        byte_buf.extend_from_slice(&bytes);

        if emit_live {
            let chunk = flush_valid_utf8_prefix(&mut byte_buf);
            if !chunk.is_empty() {
                let _ = reply.send(WorkerEvent::Token { text: chunk });
            }
        }
    }

    if emit_live && !byte_buf.is_empty() {
        let rest = String::from_utf8_lossy(&byte_buf).to_string();
        if !rest.is_empty() {
            let _ = reply.send(WorkerEvent::Token { text: rest });
        }
    }

    let full_text = String::from_utf8_lossy(&full_bytes).to_string();
    let usage = token_stream.finish_stats();

    let (finish_reason, tool_calls) = if has_tools {
        match parse_assistant_output(&full_text) {
            AssistantOutput::ToolCalls(calls) => ("tool_calls".to_string(), Some(calls)),
            AssistantOutput::Text(_) => {
                let reason = if hit_stop || emitted < params.max_tokens {
                    "stop"
                } else {
                    "length"
                };
                (reason.to_string(), None)
            }
        }
    } else {
        let reason = if hit_stop || emitted < params.max_tokens {
            "stop"
        } else {
            "length"
        };
        (reason.to_string(), None)
    };

    // Tools + stream: emit content (if any) then Done with tool_calls.
    if want_stream && has_tools {
        match &tool_calls {
            Some(_) => {
                // Tool-call-only: do not stream raw JSON as content.
            }
            None if !full_text.is_empty() => {
                let _ = reply.send(WorkerEvent::Token {
                    text: full_text.clone(),
                });
            }
            None => {}
        }
    }

    let _ = reply.send(WorkerEvent::Done {
        finish_reason,
        full_text,
        tool_calls,
        usage,
    });
    Ok(())
}

fn flush_valid_utf8_prefix(buf: &mut Vec<u8>) -> String {
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
