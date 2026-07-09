//! Runtime: engine glue that ties tokenizer + graph + backend + KV cache
//! into a single-request generation loop.
//!
//! Phase 1 keeps this deliberately simple:
//!   * one request at a time
//!   * contiguous KV cache (paged allocator lands in Phase 3)
//!   * synchronous loop (no tokio)
//!   * greedy or top-k/top-p sampling
//!
//! The public entry point is [`Engine`].

#![deny(missing_docs)]

pub mod engine;
pub mod executor;
pub mod hybrid_state;
pub mod kv_cache;

pub use arch_llama::parse_assistant_output;
pub use engine::{
    DEFAULT_BATCH_SIZE, DEFAULT_CTX_SIZE, DEFAULT_UBATCH_SIZE, Engine, EngineBuilder,
    EngineSettings, GenParams, GenStats, TokenStream,
};
pub use fellm_tokenizer::{AssistantOutput, Message, ToolCall, ToolDef};
pub use hybrid_state::HybridState;
pub use kv_cache::KvCache;
