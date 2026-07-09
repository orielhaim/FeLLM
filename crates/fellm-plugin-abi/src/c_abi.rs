//! Hard C ABI for dynamic kernel plugins.
//!
//! Registration and launch cross the `dlopen` boundary through these types.
//! Plugins must export the `_fellm_plugin_*` entry points and wrap each in
//! `catch_unwind` so panics never unwind across FFI.

use crate::op::OpAttrs;
use crate::tensor_ref::{TensorMut, TensorRef};
use crate::{ABI_VERSION, AbiVersion, StreamHandle};
use core::ffi::{c_char, c_int, c_void};

/// Maximum length of a plugin name / backend id (including NUL).
pub const PLUGIN_NAME_MAX: usize = 64;
/// Maximum ops a single plugin may register in one call.
pub const PLUGIN_MAX_OPS: usize = 32;
/// Maximum input dtypes recorded per registered op.
pub const PLUGIN_MAX_INPUT_DTYPES: usize = 8;

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
        }
    }
}

/// Static manifest returned by `_fellm_plugin_manifest` (optional but recommended).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginManifest {
    /// Plugin short name (NUL-terminated).
    pub name: [c_char; PLUGIN_NAME_MAX],
    /// Plugin semver major.
    pub version_major: u16,
    /// Plugin semver minor.
    pub version_minor: u16,
    /// Plugin semver patch.
    pub version_patch: u16,
    /// FNV-1a hash of the ABI surface the plugin was built against.
    pub abi_hash: u64,
}

impl PluginManifest {
    /// Construct a manifest with a Rust `&str` name.
    #[must_use]
    pub fn new(name: &str, major: u16, minor: u16, patch: u16, abi_hash: u64) -> Self {
        let mut buf = [0i8; PLUGIN_NAME_MAX];
        for (dst, src) in buf.iter_mut().zip(name.bytes()).take(PLUGIN_NAME_MAX - 1) {
            *dst = src as c_char;
        }
        Self {
            name: buf,
            version_major: major,
            version_minor: minor,
            version_patch: patch,
            abi_hash,
        }
    }
}

/// Stable FNV-1a hash of the current ABI version tuple.
#[must_use]
pub const fn abi_hash() -> u64 {
    // FNV-1a 64-bit over "fellm-abi-{major}.{minor}.{patch}"
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let bytes = b"fellm-abi-0.1.0";
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        i += 1;
    }
    let _ = (ABI_VERSION.major, ABI_VERSION.minor, ABI_VERSION.patch);
    hash
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

/// One op registration record filled by the plugin during `_fellm_plugin_register`.
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

/// Vtable the host passes into `_fellm_plugin_register`.
#[repr(C)]
pub struct KernelRegistryVtable {
    /// Opaque host registry pointer.
    pub registry: *mut c_void,
    /// Register one op. Returns `0` on success.
    pub register_op:
        unsafe extern "C" fn(registry: *mut c_void, reg: *const PluginOpRegistration) -> c_int,
}

/// Required plugin entry: report ABI version.
pub type PluginAbiVersionFn = unsafe extern "C" fn() -> AbiVersion;
/// Optional plugin entry: static manifest.
pub type PluginManifestFn = unsafe extern "C" fn() -> PluginManifest;
/// Required plugin entry: initialize with host context.
pub type PluginInitFn = unsafe extern "C" fn(ctx: *const HostContext) -> c_int;
/// Required plugin entry: register ops into the host registry.
pub type PluginRegisterFn = unsafe extern "C" fn(registry: *mut KernelRegistryVtable) -> c_int;
/// Required plugin entry: tear down plugin state.
pub type PluginShutdownFn = unsafe extern "C" fn();

/// Symbol names resolved by the host loader.
pub mod symbols {
    /// `_fellm_plugin_abi_version`
    pub const ABI_VERSION: &[u8] = b"_fellm_plugin_abi_version\0";
    /// `_fellm_plugin_manifest`
    pub const MANIFEST: &[u8] = b"_fellm_plugin_manifest\0";
    /// `_fellm_plugin_init`
    pub const INIT: &[u8] = b"_fellm_plugin_init\0";
    /// `_fellm_plugin_register`
    pub const REGISTER: &[u8] = b"_fellm_plugin_register\0";
    /// `_fellm_plugin_shutdown`
    pub const SHUTDOWN: &[u8] = b"_fellm_plugin_shutdown\0";
}
