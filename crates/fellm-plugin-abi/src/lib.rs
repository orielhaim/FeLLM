pub mod c_abi;
pub mod op;
pub mod paged_ctx;
pub mod tensor_ref;
pub mod traits;

pub use c_abi::{
    ArchitecturePluginRegistration, ArchitectureRegistryVtable, DeviceHandle, HostContext,
    KernelRegistryVtable, PLUGIN_MAX_INPUT_DTYPES, PLUGIN_MAX_OPS, PLUGIN_NAME_MAX, PluginLaunchFn,
    PluginManifest, PluginOpRegistration, abi_hash,
};
pub use op::{OpAttrs, OpKind};
pub use paged_ctx::{
    HostSnapshotPagedFn, PAGED_KV_ELEM_BYTES, PagedKvContext, PagedKvSnapshot, has_paged_context,
    host_snapshot_paged_kv, set_paged_context, snapshot_paged_context, with_paged_context,
    with_paged_context_mut,
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
    minor: 4,
    patch: 0,
};
