//! FeLLM CUDA kernel plugin — cuda-oxide is the only kernel path.
//!
//! Build under WSL2:
//! ```text
//! cargo oxide build --release
//! cp target/release/libcuda_kernels.so ../../plugins/dist/
//! ```
//!
//! Until oxide kernels are registered here, this plugin exports the C ABI and
//! registers **zero** ops so the host uses the embedded CPU backend (correct
//! Q4_K / attention). Never ship simplified host “stubs” that override CPU.

use fellm_plugin_abi::c_abi::{
    abi_hash, HostContext, KernelRegistryVtable, PluginManifest,
};
use fellm_plugin_abi::{AbiVersion, ABI_VERSION};
use std::os::raw::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

static mut HOST_CTX: Option<HostContext> = None;

/// Q4_K super-block size in bytes (GGUF).
pub const Q4K_BLOCK_BYTES: usize = 144;
/// Q4_K elements per super-block.
pub const Q4K_BLOCK_ELEMS: usize = 256;
/// Paged KV tokens per physical block.
pub const PAGED_BLOCK_SIZE: usize = 16;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_abi_version() -> AbiVersion {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_manifest() -> PluginManifest {
    PluginManifest::new("cuda_kernels", 0, 1, 0, abi_hash())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_init(ctx: *const HostContext) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() {
            return -1;
        }
        unsafe { HOST_CTX = Some(*ctx) };
        0
    }))
    .unwrap_or(-99)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_register(registry: *mut KernelRegistryVtable) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if registry.is_null() {
            return -1;
        }
        let _vt = unsafe { &mut *registry };
        // Register oxide #[kernel] launchers here when ready, e.g.:
        //   q4k_gemv, paged_attention, kv_write_paged
        // Until then: zero ops → host CpuBackend handles the full forward pass.
        0
    }))
    .unwrap_or(-99)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_shutdown() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        unsafe { HOST_CTX = None };
    }));
}
