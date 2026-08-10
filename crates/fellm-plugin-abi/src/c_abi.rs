//! Hard C ABI for dynamic kernel plugins.
//!
//! Registration and launch cross the `dlopen` boundary through these types.
//! Plugins must export the `_fellm_plugin_*` entry points and wrap each in
//! `catch_unwind` so panics never unwind across FFI.

use crate::op::OpAttrs;
use crate::paged_ctx::{HostSnapshotPagedFn, host_snapshot_paged_kv};
use crate::tensor_ref::{TensorMut, TensorRef};
use crate::{ABI_VERSION, AbiVersion, StreamHandle};
use core::ffi::{c_char, c_int, c_void};

/// Maximum length of a plugin name / backend id (including NUL).
pub const PLUGIN_NAME_MAX: usize = 64;
/// Maximum ops a single plugin may register in one call.
pub const PLUGIN_MAX_OPS: usize = 32;
/// Maximum input dtypes recorded per registered op.
pub const PLUGIN_MAX_INPUT_DTYPES: usize = 8;

/// Maximum length of a registered architecture identifier, including NUL.
pub const ARCHITECTURE_ID_MAX: usize = PLUGIN_NAME_MAX;

/// Opaque device / context handle (`CUcontext` cast to `u64`, or `0` on CPU).
pub type DeviceHandle = u64;

/// Host-owned context passed to the plugin at init.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HostContext {
    /// ABI version the host was built against.
    pub abi_version: AbiVersion,
    /// Vendor device / context handle (`CUcontext` as `u64`, or `0`).
    pub device_handle: DeviceHandle,
    /// Default stream / queue handle (`CUstream` as `u64`, or `0`).
    pub default_stream: StreamHandle,
    /// Optional host allocator opaque pointer (may be null).
    pub allocator_opaque: *mut c_void,
    /// Backend id the host expects this plugin to serve (NUL-terminated).
    pub backend_id: [c_char; PLUGIN_NAME_MAX],
    /// Snapshot the host process-wide paged KV arena (required for Attention/KvWrite).
    ///
    /// Plugins must use this instead of their own `PAGED_CTX` static — a `cdylib`
    /// gets a separate copy of `fellm-plugin-abi` statics.
    pub snapshot_paged: Option<HostSnapshotPagedFn>,
}

// SAFETY: HostContext is POD; pointer validity is the caller's contract.
unsafe impl Send for HostContext {}
unsafe impl Sync for HostContext {}

impl HostContext {
    /// Build a host context for `backend_id` (truncated to [`PLUGIN_NAME_MAX`] − 1).
    #[must_use]
    pub fn new(
        device_handle: DeviceHandle,
        default_stream: StreamHandle,
        allocator_opaque: *mut c_void,
        backend_id: &str,
    ) -> Self {
        let mut id = [0i8; PLUGIN_NAME_MAX];
        for (dst, src) in id
            .iter_mut()
            .zip(backend_id.bytes())
            .take(PLUGIN_NAME_MAX - 1)
        {
            *dst = src as c_char;
        }
        Self {
            abi_version: ABI_VERSION,
            device_handle,
            default_stream,
            allocator_opaque,
            backend_id: id,
            snapshot_paged: Some(host_snapshot_paged_kv),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginManifestJson {
    /// Pointer to UTF-8 JSON bytes.
    pub ptr: *const u8,
    /// Number of bytes at [`Self::ptr`].
    pub len: usize,
}

/// Hot-path kernel launch function pointer (C ABI).
///
/// Returns `0` on success, nonzero on failure.
pub type PluginLaunchFn = unsafe extern "C" fn(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    stream: StreamHandle,
) -> c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginOpRegistration {
    /// `OpKind` discriminant as `u32`.
    pub op_kind: u32,
    /// Number of valid entries in `input_dtypes`.
    pub n_input_dtypes: u32,
    /// Input dtypes as ggml codes.
    pub input_dtypes: [u32; PLUGIN_MAX_INPUT_DTYPES],
    /// Output dtype as ggml code.
    pub output_dtype: u32,
    /// Launch function for this op.
    pub launch: Option<PluginLaunchFn>,
}

#[repr(C)]
pub struct KernelRegistryVtable {
    /// Opaque host registry pointer.
    pub registry: *mut c_void,
    /// Register one op. Returns `0` on success.
    pub register_op:
        unsafe extern "C" fn(registry: *mut c_void, reg: *const PluginOpRegistration) -> c_int,
}

/// Opaque architecture provider callback types.
///
/// Architecture callbacks are deliberately strict and opaque at the C
/// boundary.  The host owns the source/config/program objects and passes
/// handles to callbacks; a plugin must implement the complete current set or
/// registration fails.  The Rust-side [`ArchitectureProvider`] trait is the
/// ergonomic contract for statically linked providers.
pub type ArchitectureProbeFn =
    unsafe extern "C" fn(source: *const c_void, config: *mut c_void) -> c_int;
/// Compile an architecture program.
pub type ArchitectureCompileFn = unsafe extern "C" fn(
    source: *const c_void,
    config: *const c_void,
    backend: *const c_void,
    program: *mut c_void,
) -> c_int;
/// Create an architecture-owned generation driver.
pub type ArchitectureCreateDriverFn = unsafe extern "C" fn(
    program: *const c_void,
    request: *const c_void,
    driver: *mut c_void,
) -> c_int;

/// One architecture provider registration record.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ArchitecturePluginRegistration {
    /// NUL-terminated architecture id.
    pub architecture_id: [c_char; ARCHITECTURE_ID_MAX],
    /// Probe callback.
    pub probe: Option<ArchitectureProbeFn>,
    /// Compile callback.
    pub compile: Option<ArchitectureCompileFn>,
    /// Generation-driver callback.
    pub create_generation_driver: Option<ArchitectureCreateDriverFn>,
}

/// Vtable for architecture-provider registration.
#[repr(C)]
pub struct ArchitectureRegistryVtable {
    /// Opaque host registry pointer.
    pub registry: *mut c_void,
    /// Register one complete architecture provider.
    pub register_architecture: unsafe extern "C" fn(
        registry: *mut c_void,
        registration: *const ArchitecturePluginRegistration,
    ) -> c_int,
}

/// Required plugin entry: report ABI version.
pub type PluginAbiVersionFn = unsafe extern "C" fn() -> AbiVersion;
/// Required plugin entry: embedded JSON manifest.
pub type PluginManifestJsonFn = unsafe extern "C" fn() -> PluginManifestJson;
/// Required plugin entry: initialize with host context.
pub type PluginInitFn = unsafe extern "C" fn(ctx: *const HostContext) -> c_int;
/// Conditional plugin entry: register kernels into the host registry.
pub type PluginRegisterKernelsFn =
    unsafe extern "C" fn(registry: *mut KernelRegistryVtable) -> c_int;
/// Conditional architecture registration entry. A plugin declaring an
/// `architecture` component must export it and register complete providers
/// using the exact current contract.
pub type PluginRegisterArchitecturesFn =
    unsafe extern "C" fn(registry: *mut ArchitectureRegistryVtable) -> c_int;
/// Conditional multi-capability registration (attention, KV policy, etc.).
pub type PluginRegisterCapabilitiesFn =
    unsafe extern "C" fn(registry: *mut CapabilityRegistryVtable) -> c_int;
/// Required plugin entry: tear down plugin state after activation.
pub type PluginShutdownFn = unsafe extern "C" fn();
/// Optional: invalidate a host f32 buffer's device mirror after a CPU write.
///
/// `ptr` is the host `*const f32`; `nbytes` is the byte length of the slice.
pub type PluginInvalidateF32Fn = unsafe extern "C" fn(ptr: *const f32, nbytes: usize);
/// Optional device-plan entry: update the one device-resident step parameter block.
pub type PluginUpdateStepParamsFn =
    unsafe extern "C" fn(params: *const crate::DeviceStepParams) -> c_int;
/// Optional device-plan entry: bind a host constant to a packed device address.
pub type PluginRegisterDeviceTensorFn =
    unsafe extern "C" fn(host_ptr: *const u8, nbytes: usize, device_ptr: u64) -> c_int;

/// Maximum free-form features listed in a C capability registration.
pub const PLUGIN_MAX_FEATURES: usize = 32;
/// Maximum metadata key/value pairs in a C capability registration.
pub const PLUGIN_MAX_META: usize = 8;
/// Maximum length of a metadata key or value including NUL.
pub const PLUGIN_META_VALUE_MAX: usize = 64;

/// One capability/provider registration filled during
/// `_fellm_plugin_register_capabilities`.
///
/// The host reconstructs a Rust [`crate::ProviderDescriptor`] from this POD
/// record. Feature ids are [`crate::FeatureId`] raw values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginCapabilityRegistration {
    /// NUL-terminated provider name (e.g. `kv.triattention`).
    pub name: [c_char; PLUGIN_NAME_MAX],
    /// [`crate::CapabilityKind`] as `u16`.
    pub capability: u16,
    /// Provider semver major.
    pub version_major: u16,
    /// Provider semver minor.
    pub version_minor: u16,
    /// Provider semver patch.
    pub version_patch: u16,
    /// Auto-selection priority.
    pub priority: i32,
    /// Number of valid `provides` entries.
    pub n_provides: u32,
    /// Feature ids this provider supplies.
    pub provides: [u32; PLUGIN_MAX_FEATURES],
    /// Number of valid `requires` entries.
    pub n_requires: u32,
    /// Feature ids this provider requires.
    pub requires: [u32; PLUGIN_MAX_FEATURES],
    /// Number of metadata pairs.
    pub n_meta: u32,
    /// Metadata keys (NUL-terminated).
    pub meta_keys: [[c_char; PLUGIN_META_VALUE_MAX]; PLUGIN_MAX_META],
    /// Metadata values (NUL-terminated).
    pub meta_values: [[c_char; PLUGIN_META_VALUE_MAX]; PLUGIN_MAX_META],
    /// Optional summary (NUL-terminated, truncated).
    pub summary: [c_char; PLUGIN_META_VALUE_MAX],
}

impl PluginCapabilityRegistration {
    /// Zeroed registration record.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            name: [0; PLUGIN_NAME_MAX],
            capability: 0,
            version_major: 0,
            version_minor: 0,
            version_patch: 0,
            priority: 0,
            n_provides: 0,
            provides: [0; PLUGIN_MAX_FEATURES],
            n_requires: 0,
            requires: [0; PLUGIN_MAX_FEATURES],
            n_meta: 0,
            meta_keys: [[0; PLUGIN_META_VALUE_MAX]; PLUGIN_MAX_META],
            meta_values: [[0; PLUGIN_META_VALUE_MAX]; PLUGIN_MAX_META],
            summary: [0; PLUGIN_META_VALUE_MAX],
        }
    }

    /// Write a NUL-terminated string into a fixed `c_char` buffer.
    pub fn write_cstr(buf: &mut [c_char], s: &str) {
        let max = buf.len().saturating_sub(1);
        for b in buf.iter_mut() {
            *b = 0;
        }
        for (dst, src) in buf.iter_mut().zip(s.bytes()).take(max) {
            *dst = src as c_char;
        }
    }
}

/// Vtable for multi-capability registration.
#[repr(C)]
pub struct CapabilityRegistryVtable {
    /// Opaque host registry pointer.
    pub registry: *mut c_void,
    /// Register one capability provider descriptor. Returns `0` on success.
    pub register_capability: unsafe extern "C" fn(
        registry: *mut c_void,
        registration: *const PluginCapabilityRegistration,
    ) -> c_int,
}

/// Symbol names resolved by the host loader.
pub mod symbols {
    /// `_fellm_plugin_abi_version`
    pub const ABI_VERSION: &[u8] = b"_fellm_plugin_abi_version\0";
    /// `_fellm_plugin_manifest_json`
    pub const MANIFEST_JSON: &[u8] = b"_fellm_plugin_manifest_json\0";
    /// `_fellm_plugin_init`
    pub const INIT: &[u8] = b"_fellm_plugin_init\0";
    /// `_fellm_plugin_register_kernels`
    pub const REGISTER_KERNELS: &[u8] = b"_fellm_plugin_register_kernels\0";
    /// `_fellm_plugin_register_capabilities`
    pub const REGISTER_CAPABILITIES: &[u8] = b"_fellm_plugin_register_capabilities\0";
    /// `_fellm_plugin_register_architectures` (optional for kernel-only plugins)
    pub const REGISTER_ARCHITECTURES: &[u8] = b"_fellm_plugin_register_architectures\0";
    /// `_fellm_plugin_shutdown`
    pub const SHUTDOWN: &[u8] = b"_fellm_plugin_shutdown\0";
    /// `_fellm_plugin_invalidate_f32` (optional)
    pub const INVALIDATE_F32: &[u8] = b"_fellm_plugin_invalidate_f32\0";
    /// `_fellm_plugin_update_step_params` (optional)
    pub const UPDATE_STEP_PARAMS: &[u8] = b"_fellm_plugin_update_step_params\0";
    /// `_fellm_plugin_register_device_tensor` (optional)
    pub const REGISTER_DEVICE_TENSOR: &[u8] = b"_fellm_plugin_register_device_tensor\0";
}
