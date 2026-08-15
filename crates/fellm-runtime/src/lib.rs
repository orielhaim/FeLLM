pub mod activation;
pub mod architecture;
pub mod backend_select;
pub mod block_diffusion;
pub mod compiled;
pub mod cuda_lowering;
pub mod engine;
pub mod hybrid_state;
pub mod kv_fabric;
pub mod memory_fabric;
pub mod providers;
pub mod sampling;
pub mod scheduler;
pub mod speculative;

pub use backend_select::{BackendPreference, BackendSelect};
pub use engine::{
    DEFAULT_BATCH_SIZE, DEFAULT_CTX_SIZE, DEFAULT_UBATCH_SIZE, DecodeSequence, Engine,
    EngineBuilder, EngineSettings, GenParams, GenStats, TokenStream,
};
pub use fellm_model::{ModelSpec, parse_assistant_output};
pub use fellm_plugin_abi::capability::{PluginConfig, ProviderSelection};
pub use fellm_tokenizer::{
    AssistantOutput, ChatRenderOptions, Message, ToolCall, ToolDef,
};
pub use hybrid_state::HybridConvState;
pub use kv_fabric::{
    BLOCK_SIZE, DummyKvBuffers, FabricMetrics, KvAddressing, KvEncoding, KvEncodingPolicy,
    KvFabric, KvFabricConfig, KvGroupDesc, KvGroupKind, KvMemoryPlan, KvMode, KvPageClass,
    KvPageId, KvSequence, KvTier, KvTransaction, PrefixCacheStats, ResidencyPolicyKind,
    ResidencySignals, STANDARD_PAGE_TOKENS,
};
pub use memory_fabric::{MemoryFabric, MemoryFabricConfig, MemoryFabricSnapshot};
pub use providers::{PreparedProviders, ProviderManager};
pub use scheduler::{
    BatchItem, BatchPlan, InteractivePolicy, Scheduler, SchedulingCandidate, SchedulingPolicy,
    SequenceEvent, SequenceHandle, SequenceId, SequenceStatus, WorkKind,
};
pub use speculative::{
    AdaptiveSpeculationPolicy, DraftProposal, DraftToken, GenericDraftConfig, GenericDraftRuntime,
    LinearProposal, PluginSpeculativeRuntime, ProbabilityDistribution,
    ProvisionalTargetVerification, SpeculationDecision, SpeculationMetrics, SpeculativeVerifier,
    VerificationCapacityProfile, VerificationOutcome, schedule_confident_prefixes,
};
