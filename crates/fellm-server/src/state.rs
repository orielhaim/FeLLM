//! Shared server state and inference task types.

use fellm_runtime::{GenParams, GenStats, Message, ToolCall, ToolDef};
use tokio::sync::mpsc;

/// Axum application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    /// Queue into the inference worker.
    pub task_tx: mpsc::Sender<InferenceTask>,
    /// Model id advertised by `/v1/models` and completion responses.
    pub model_id: String,
    /// Default sampling params when the request omits fields.
    pub defaults: GenParams,
}

/// One generation job submitted by an HTTP handler.
pub struct InferenceTask {
    /// Chat messages (already mapped from `OpenAI`).
    pub messages: Vec<Message>,
    /// Tool definitions (may be empty).
    pub tools: Vec<ToolDef>,
    /// Sampling parameters.
    pub params: GenParams,
    /// Whether the client requested SSE streaming.
    pub stream: bool,
    /// Reply channel back to the HTTP handler.
    pub reply: mpsc::UnboundedSender<WorkerEvent>,
}

/// Events emitted by the inference worker for a single task.
#[derive(Debug)]
pub enum WorkerEvent {
    /// Decoded UTF-8 text for one or more tokens (streaming path).
    Token {
        /// Incremental text.
        text: String,
    },
    /// Generation finished successfully.
    Done {
        /// OpenAI-style finish reason: `stop`, `length`, or `tool_calls`.
        finish_reason: String,
        /// Full assistant text (before tool-call parsing).
        full_text: String,
        /// Parsed tool calls, if any.
        tool_calls: Option<Vec<ToolCall>>,
        /// Token usage stats.
        usage: GenStats,
    },
    /// Generation failed.
    Error(String),
}
