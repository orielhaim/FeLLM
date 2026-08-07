pub mod architecture;
pub mod backend_select;
pub mod block_diffusion;
pub mod compiled;
pub mod cuda_lowering;
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
