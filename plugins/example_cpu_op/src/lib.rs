//! Smoke plugin: identity Cast (f32 → f32 copy) via the C plugin ABI.

use fellm_core::dtype::DType;
use fellm_plugin_abi::c_abi::{
    HostContext, KernelRegistryVtable, PluginManifestJson, PluginOpRegistration,
};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::{ABI_VERSION, AbiVersion, StreamHandle, TensorMut, TensorRef};
use std::os::raw::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};

static mut HOST_CTX: Option<HostContext> = None;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_abi_version() -> AbiVersion {
    ABI_VERSION
}

static MANIFEST_JSON: &[u8] = concat!(
    r#"{"schema":1,"id":"fellm.example.cpu-op","name":"Example CPU Op","version":"#,
    env!("CARGO_PKG_VERSION"),
    r#"","provides":[{"type":"kernels","backend":"cpu"}]}"#
)
.as_bytes();

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_manifest_json() -> PluginManifestJson {
    PluginManifestJson {
        ptr: MANIFEST_JSON.as_ptr(),
        len: MANIFEST_JSON.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_init(ctx: *const HostContext) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() {
            return -1;
        }
        // SAFETY: host guarantees valid HostContext for the call.
        let c = unsafe { *ctx };
        unsafe { HOST_CTX = Some(c) };
        0
    }));
    result.unwrap_or(-99)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_register_kernels(
    registry: *mut KernelRegistryVtable,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if registry.is_null() {
            return -1;
        }
        let vt = unsafe { &*registry };
        let mut reg = PluginOpRegistration {
            op_kind: OpKind::Cast.raw(),
            n_input_dtypes: 1,
            input_dtypes: [0; fellm_plugin_abi::PLUGIN_MAX_INPUT_DTYPES],
            output_dtype: DType::F32 as u32,
            launch: Some(launch_identity_cast),
        };
        reg.input_dtypes[0] = DType::F32 as u32;
        unsafe { (vt.register_op)(vt.registry, &raw const reg) }
    }));
    result.unwrap_or(-99)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_shutdown() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        unsafe { HOST_CTX = None };
    }));
}

unsafe extern "C" fn launch_identity_cast(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if n_inputs < 1 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return -1;
        }
        let _attrs = unsafe { &*attrs };
        let inp = unsafe { &*inputs };
        let out = unsafe { &mut *outputs };
        if inp.byte_len != out.byte_len {
            return -2;
        }
        let n = inp.byte_len as usize;
        // SAFETY: host owns both buffers for the duration of launch.
        unsafe {
            std::ptr::copy_nonoverlapping(inp.data, out.data, n);
        }
        0
    }));
    result.unwrap_or(-99)
}
