pub mod c_abi;
pub mod op;
pub mod paged_ctx;
pub mod tensor_ref;
pub mod traits;

pub use c_abi::{
    DeviceHandle, HostContext, KernelRegistryVtable, PLUGIN_MAX_INPUT_DTYPES, PLUGIN_MAX_OPS,
    PLUGIN_NAME_MAX, PluginLaunchFn, PluginManifest, PluginOpRegistration, abi_hash,
};
pub use op::{OpAttrs, OpKind};
pub use paged_ctx::{
    host_snapshot_paged_kv, has_paged_context, set_paged_context, snapshot_paged_context,
    with_paged_context, HostSnapshotPagedFn, PAGED_KV_ELEM_BYTES, PagedKvContext, PagedKvSnapshot,
};
pub use tensor_ref::{TensorMut, TensorRef};
pub use traits::{Architecture, Backend, BackendCaps, KernelHandle};

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
    minor: 3,
    patch: 0,
};
