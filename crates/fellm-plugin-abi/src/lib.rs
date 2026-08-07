pub mod attention_dispatch;
pub mod attention_provider;
pub mod c_abi;
pub mod capability;
pub mod op;
pub mod paged_ctx;
pub mod physical_plan;
pub mod sequence_state;
pub mod tensor_ref;
pub mod traits;

pub use attention_dispatch::{
    AttentionDispatch, AttentionKernelPath, PreRopeKeyStore, attention_dispatch, pre_rope_gather,
    pre_rope_prune, pre_rope_slice, pre_rope_write, resolve_path, set_attention_dispatch,
    set_pre_rope_store, take_pre_rope_store,
};
pub use attention_provider::{
    AttentionPathKind, AttentionPrepareContext, AttentionProvider, AttentionWorkload,
    DeviceCapabilityView, PreparedAttention, attrs_from_workload, fa2_style_attention_f32,
    fa2_style_attention_paged_f32, reference_attention_f32, reference_attention_paged_f32,
};
pub use c_abi::{
    ArchitecturePluginRegistration, ArchitectureRegistryVtable, CapabilityRegistryVtable,
    DeviceHandle, HostContext, KernelRegistryVtable, PLUGIN_MAX_INPUT_DTYPES, PLUGIN_MAX_OPS,
    PLUGIN_NAME_MAX, PluginCapabilityRegistration, PluginLaunchFn, PluginManifest,
    PluginOpRegistration, abi_hash,
};
pub use capability::{
    CapabilityKind, FeatureId, FeatureSet, NegotiationError, NegotiationReport, PluginConfig,
    PreparedProviderId, ProviderDescriptor, ProviderSelection, ProviderVersion,
    negotiate_attention_and_kv_policy, negotiate_provider,
};
pub use op::{OpAttrs, OpKind};
pub use paged_ctx::{
    HostSnapshotPagedFn, PAGED_KV_ELEM_BYTES, PagedKvContext, PagedKvSnapshot, has_paged_context,
    host_snapshot_paged_kv, set_paged_context, snapshot_paged_context, with_paged_context,
    with_paged_context_mut,
};
pub use physical_plan::{
    AllocationId, ArenaSlot, DevicePtr, DeviceStepParams, DeviceTensor, MacroOpKind, PhysicalPlan,
    PlanTensorDesc, PlanTensorId, PreparedMacroOp, StorageClass,
};
pub use sequence_state::{
    AttentionKvView, FullRetentionPolicy, LayoutOwner, RetainedEntry, RetentionContext,
    RetentionPlan, RetentionStats, SequenceAttentionState, SequenceStatePolicy, StateOwnership,
    compact_layer_to_entries, gather_kv_by_positions,
};
pub use tensor_ref::{TensorMut, TensorRef};
pub use traits::{
    Architecture, ArchitectureConfig, ArchitectureProvider, Backend, BackendCapabilities,
    BackendCaps, CacheCommit, DeviceKind, DriverAction, DriverEvent, GenerationDriver,
    GenerationRequest, GraphId, GraphOutput, GraphSpec, InputBinding, InputBindings, KernelHandle,
    ModelProgram, ModelSource, StateBinding, StateBindings, TokenBatch,
};

/// A stream handle. On CPU this is always 0; on GPU backends this wraps
/// the vendor stream/queue pointer as a `u64`.
pub type StreamHandle = u64;

/// Semantic version.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiVersion {
    /// Major.
    pub major: u16,
    /// Minor.
    pub minor: u16,
    /// Patch.
    pub patch: u16,
}

/// The ABI version this crate advertises. Bump on breaking changes.
pub const ABI_VERSION: AbiVersion = AbiVersion {
    major: 0,
    minor: 7,
    patch: 0,
};
