pub mod architecture;
pub mod backend_select;
pub mod block_diffusion;
pub mod compiled;
pub mod cuda_lowering;
pub mod engine;
pub mod hybrid_state;
pub mod kv_fabric;
pub mod providers;
pub mod sampling;
pub mod scheduler;

pub use backend_select::{BackendPreference, BackendSelect};
pub use engine::{
    DEFAULT_BATCH_SIZE, DEFAULT_CTX_SIZE, DEFAULT_UBATCH_SIZE, Engine, EngineBuilder,
    EngineSettings, GenParams, GenStats, TokenStream,
};
pub use fellm_model::{ModelSpec, parse_assistant_output};
pub use fellm_plugin_abi::capability::{PluginConfig, ProviderSelection};
pub use fellm_tokenizer::{AssistantOutput, Message, ToolCall, ToolDef};
pub use hybrid_state::HybridConvState;
pub use kv_fabric::{
    BLOCK_SIZE, DummyKvBuffers, FabricMetrics, KvAddressing, KvEncoding, KvEncodingPolicy,
    KvFabric, KvFabricConfig, KvGroupDesc, KvGroupKind, KvMemoryPlan, KvMode, KvPageClass,
    KvPageId, KvSequence, KvTier, PrefixCacheStats, ResidencyPolicyKind, ResidencySignals,
    STANDARD_PAGE_TOKENS,
};
pub use providers::{PreparedProviders, ProviderManager};
pub use scheduler::{
    BatchItem, BatchPlan, InteractivePolicy, Scheduler, SchedulingCandidate, SchedulingPolicy,
    SequenceEvent, SequenceHandle, SequenceId, SequenceStatus, WorkKind,
};
