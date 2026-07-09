//! Runtime: engine glue that ties tokenizer + graph + backend + paged KV cache
//! into a generation loop with optional multi-sequence scheduling.
//!
//! * Paged attention KV (`BLOCK_SIZE = 16`) with prefix sharing and CoW
//! * Fixed ShortConv state for hybrid models
//! * Interleaved single-token scheduler for concurrent requests
//!
//! The public entry point is [`Engine`].

#![deny(missing_docs)]

pub mod backend_select;
pub mod engine;
pub mod executor;
pub mod hybrid_state;
pub mod kv_cache;
pub mod paged;
pub mod sampling;
pub mod scheduler;

pub use backend_select::{BackendPreference, BackendSelect};
pub use engine::{
    DEFAULT_BATCH_SIZE, DEFAULT_CTX_SIZE, DEFAULT_UBATCH_SIZE, Engine, EngineBuilder,
    EngineSettings, GenParams, GenStats, TokenStream,
};
pub use fellm_model::{ModelSpec, parse_assistant_output};
pub use fellm_tokenizer::{AssistantOutput, Message, ToolCall, ToolDef};
pub use hybrid_state::HybridConvState;
pub use kv_cache::KvCache;
pub use paged::{BLOCK_SIZE, CacheManager, PhysicalPool, PrefixTree, SequenceCache};
pub use scheduler::{Scheduler, SequenceEvent, SequenceHandle, SequenceStatus};
