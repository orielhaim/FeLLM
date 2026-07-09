//! Axum HTTP routes.

use crate::openai::{
    build_completion_response, completion_id, map_request, models_response, stream_content_chunk,
    stream_final_chunk, stream_role_chunk, unix_now,
};
use crate::state::{AppState, InferenceTask, WorkerEvent};
use async_openai::types::chat::CreateChatCompletionRequest;
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::Stream;
use serde_json::json;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(models_response(&state.model_id))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> Response {
    let (messages, tools, params, want_stream) = match map_request(&req, state.defaults) {
        Ok(v) => v,
        Err(err) => {
            return error_response(StatusCode::BAD_REQUEST, &err);
        }
    };

    let (reply_tx, reply_rx) = mpsc::unbounded_channel();
    let task = InferenceTask {
        messages,
        tools,
        params,
        stream: want_stream,
        reply: reply_tx,
    };

    if let Err(err) = state.task_tx.send(task).await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("inference worker unavailable: {err}"),
        );
    }

    if want_stream {
        stream_response(state.model_id.clone(), reply_rx).into_response()
    } else {
        non_stream_response(&state.model_id, reply_rx).await
    }
}

async fn non_stream_response(
    model_id: &str,
    mut reply_rx: mpsc::UnboundedReceiver<WorkerEvent>,
) -> Response {
    let mut full_text = String::new();
    let mut finish_reason = "stop".to_string();
    let mut tool_calls = None;
    let mut usage = None;

    while let Some(ev) = reply_rx.recv().await {
        match ev {
            WorkerEvent::Token { text } => {
                full_text.push_str(&text);
            }
            WorkerEvent::Done {
                finish_reason: fr,
                full_text: ft,
                tool_calls: tc,
                usage: u,
            } => {
                finish_reason = fr;
                full_text = ft;
                tool_calls = tc;
                usage = Some(u);
                break;
            }
            WorkerEvent::Error(err) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err);
            }
        }
    }

    let Some(usage) = usage else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference worker closed without a result",
        );
    };

    let body = build_completion_response(
        model_id,
        &finish_reason,
        &full_text,
        tool_calls.as_deref(),
        &usage,
    );
    Json(body).into_response()
}

fn stream_response(
    model_id: String,
    mut reply_rx: mpsc::UnboundedReceiver<WorkerEvent>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let id = completion_id();
        let created = unix_now();
        let mut sent_role = false;

        while let Some(ev) = reply_rx.recv().await {
            match ev {
                WorkerEvent::Token { text } => {
                    if !sent_role {
                        let _ = tx
                            .send(Ok(Event::default()
                                .data(json_data(&stream_role_chunk(&id, created, &model_id)))))
                            .await;
                        sent_role = true;
                    }
                    if !text.is_empty() {
                        let _ = tx
                            .send(Ok(Event::default().data(json_data(&stream_content_chunk(
                                &id, created, &model_id, &text,
                            )))))
                            .await;
                    }
                }
                WorkerEvent::Done {
                    finish_reason,
                    tool_calls,
                    ..
                } => {
                    if !sent_role {
                        let _ = tx
                            .send(Ok(Event::default()
                                .data(json_data(&stream_role_chunk(&id, created, &model_id)))))
                            .await;
                    }
                    let _ = tx
                        .send(Ok(Event::default().data(json_data(&stream_final_chunk(
                            &id,
                            created,
                            &model_id,
                            &finish_reason,
                            tool_calls.as_deref(),
                        )))))
                        .await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
                WorkerEvent::Error(err) => {
                    let payload = json!({
                        "error": { "message": err, "type": "server_error" }
                    })
                    .to_string();
                    let _ = tx.send(Ok(Event::default().data(payload))).await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
            }
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

/// Strip the `data: ` / trailing newlines from our SSE helper strings so Axum
/// can add its own framing via [`Event::data`].
fn json_data(sse_line: &str) -> String {
    let s = sse_line.trim();
    let s = s.strip_prefix("data: ").unwrap_or(s);
    s.trim_end_matches('\n').to_string()
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": null,
            "code": null,
        }
    });
    (status, Json(body)).into_response()
}
