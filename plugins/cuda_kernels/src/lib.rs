//! FeLLM CUDA kernel plugin — built only with cuda-oxide (Pipeline B).
//!
//! ```text
//! bash scripts/wsl-build-plugin.sh
//! ```

mod buffers;
mod launchers;
mod oxide_kernels;
mod tensor;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use fellm_core::dtype::DType;
use fellm_plugin_abi::c_abi::{
    HostContext, KernelRegistryVtable, PluginManifestJson, PluginOpRegistration,
};
use fellm_plugin_abi::op::OpKind;
use fellm_plugin_abi::{ABI_VERSION, AbiVersion, DeviceStepParams, PagedKvSnapshot, TensorRef};
use oxide_kernels::kernels;
use std::ffi::CStr;
use std::os::raw::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

pub use oxide_kernels::{
    Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q5K_BLOCK_BYTES,
    Q5K_BLOCK_ELEMS, Q6K_BLOCK_BYTES, Q6K_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};

/// Paged KV tokens per physical block.
pub const PAGED_BLOCK_SIZE: usize = 16;

static HOST_CTX: Mutex<Option<HostContext>> = Mutex::new(None);

/// Snapshot host paged KV via the callback installed in [`HostContext`].
///
/// Must not use `fellm_plugin_abi::snapshot_paged_context` — that static lives
/// in this `.so`, not in the host process that called `set_paged_context`.
pub(crate) fn host_paged_snapshot() -> Option<PagedKvSnapshot> {
    let guard = HOST_CTX.lock().ok()?;
    let ctx = guard.as_ref()?;
    let snap_fn = ctx.snapshot_paged?;
    let mut out = PagedKvSnapshot {
        arena: std::ptr::null_mut(),
        arena_len: 0,
        block_table: std::ptr::null(),
        n_block_table: 0,
        n_logical_blocks: 0,
        n_layers: 0,
        tokens_stride: 0,
        block_bytes: 0,
        block_size: 0,
        elem_bytes: 0,
        device_arena: std::ptr::null_mut(),
        device_arena_len: 0,
        device_block_table: std::ptr::null_mut(),
        n_device_block_table: 0,
        device_logical_stride: 0,
        batch_size: 0,
        row_positions: std::ptr::null(),
        row_lengths: std::ptr::null(),
        row_rope_positions: std::ptr::null(),
    };
    let rc = unsafe { snap_fn(&mut out) };
    if rc == 0 { Some(out) } else { None }
}

static OXIDE_CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
static OXIDE_STREAM: OnceLock<Arc<CudaStream>> = OnceLock::new();
static STEP_PARAMS: Mutex<Option<DeviceBuffer<u8>>> = Mutex::new(None);
static OXIDE_MODULE: OnceLock<kernels::LoadedModule> = OnceLock::new();

pub(crate) fn oxide_ctx() -> &'static Arc<CudaContext> {
    OXIDE_CTX.get_or_init(|| CudaContext::new(0).expect("cuda_kernels: CudaContext::new(0)"))
}

/// Non-blocking compute stream owned by the CUDA plan/plugin.
/// Legacy stream 0 cannot be captured by CUDA Graphs.
pub(crate) fn oxide_stream() -> &'static Arc<CudaStream> {
    OXIDE_STREAM.get_or_init(|| {
        oxide_ctx()
            .new_stream()
            .expect("cuda_kernels: create non-blocking compute stream")
    })
}

/// Borrow the stable device control allocation while preparing a kernel launch.
/// CUDA Graph capture records only its device address; replay does not lock.
pub(crate) fn with_step_params<T>(
    f: impl FnOnce(&DeviceBuffer<u8>) -> Result<T, i32>,
) -> Result<T, i32> {
    let guard = STEP_PARAMS.lock().map_err(|_| -30)?;
    let params = guard.as_ref().ok_or(-22)?;
    f(params)
}

/// Return the non-default stream used by every prepared CUDA kernel.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_device_stream() -> fellm_plugin_abi::StreamHandle {
    oxide_stream().cu_stream() as usize as fellm_plugin_abi::StreamHandle
}

/// Upload the only values permitted to vary between decode graph replays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_update_step_params(
    params: *const DeviceStepParams,
) -> c_int {
    if params.is_null() {
        return -1;
    }
    let params = unsafe { *params };
    let bytes = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(&params).cast::<u8>(),
            core::mem::size_of::<DeviceStepParams>(),
        )
    };
    let stream = oxide_stream();
    let mut guard = match STEP_PARAMS.lock() {
        Ok(guard) => guard,
        Err(_) => return -30,
    };
    let result = if let Some(buffer) = guard.as_mut() {
        buffer.copy_from_host(stream, bytes)
    } else {
        match DeviceBuffer::from_host(stream, bytes) {
            Ok(buffer) => {
                *guard = Some(buffer);
                return 0;
            }
            Err(_) => return -3,
        }
    };
    if result.is_ok() { 0 } else { -3 }
}

/// Register a stable tensor address from the host-owned Weight Fabric.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_register_device_tensor(
    host_ptr: *const u8,
    nbytes: usize,
    device_ptr: u64,
) -> c_int {
    match buffers::register_external_weight(host_ptr, nbytes, device_ptr) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

/// Configure the bounded streaming working set selected by the Memory Fabric planner.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_set_weight_cache_budget(
    bytes: u64,
    buffer_count: u32,
) -> c_int {
    match buffers::set_weight_cache_budget(bytes, buffer_count) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

/// Report the Memory Fabric capacity available to one execution group.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_weight_group_capacity() -> u64 {
    buffers::weight_group_capacity()
}

/// Enqueue the current predictive window on the plugin's CUDA stream.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_prefetch_weights(
    group_id: u64,
    weights: *const TensorRef,
    count: usize,
) -> c_int {
    if weights.is_null() && count != 0 {
        return -1;
    }
    // SAFETY: the host guarantees `weights` covers `count` descriptors for this call.
    let weights = unsafe { core::slice::from_raw_parts(weights, count) };
    let stream = oxide_stream().clone();
    if let Err(code) = buffers::prefetch_group(&stream, group_id, weights) {
        return code;
    }
    0
}

/// Return weight-tier residency and transfer telemetry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_weight_cache_metrics(
    metrics: *mut fellm_plugin_abi::c_abi::PluginWeightCacheMetrics,
) -> c_int {
    if metrics.is_null() {
        return -1;
    }
    // SAFETY: caller provided a writable metrics record.
    unsafe { metrics.write(buffers::weight_cache_metrics()) };
    0
}

/// Path of this loaded `.so` (oxide embeds PTX in `.oxart` here, not in the host exe).
fn this_plugin_path() -> PathBuf {
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const i8,
        dli_fbase: *mut c_void,
        dli_sname: *const i8,
        dli_saddr: *mut c_void,
    }
    unsafe extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }
    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    let addr = _fellm_plugin_abi_version as *const c_void;
    let rc = unsafe { dladdr(addr, &mut info) };
    assert!(
        rc != 0 && !info.dli_fname.is_null(),
        "dladdr failed for plugin"
    );
    let cstr = unsafe { CStr::from_ptr(info.dli_fname) };
    PathBuf::from(cstr.to_string_lossy().as_ref())
}

pub(crate) fn oxide_module() -> &'static kernels::LoadedModule {
    OXIDE_MODULE.get_or_init(|| {
        let ctx = oxide_ctx();
        let path = this_plugin_path();
        let bundles = cuda_core::embedded::artifact_bundles_from_binary_path(&path)
            .unwrap_or_else(|e| panic!("read .oxart from {}: {e}", path.display()));
        let bundle = bundles
            .into_iter()
            .find(|b| b.name == "cuda_kernels" || b.name == env!("CARGO_PKG_NAME"))
            .unwrap_or_else(|| panic!("no oxide artifact bundle in {}", path.display()));

        use cuda_core::embedded::ArtifactPayloadKind;
        let module = if let Some(emb) = cuda_core::embedded::EmbeddedModule::new(bundle.clone()) {
            emb.load(ctx)
                .unwrap_or_else(|e| panic!("load cubin/ptx from plugin: {e}"))
        } else if let Some(nvvm) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
            // Kernels using sin/exp emit NVVM IR; compile via libNVVM + nvJitLink.
            let arch = if bundle.target.is_empty() {
                "sm_80"
            } else {
                bundle.target.as_str()
            };
            let cubin = cuda_host::ltoir::build_cubin_from_nvvm_ir(nvvm, &bundle.name, arch)
                .unwrap_or_else(|e| panic!("NVVM→cubin for {}: {e}", bundle.name));
            ctx.load_module_from_image(&cubin)
                .unwrap_or_else(|e| panic!("load cubin: {e}"))
        } else if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
            let arch = if bundle.target.is_empty() {
                "sm_80"
            } else {
                bundle.target.as_str()
            };
            let cubin = cuda_host::ltoir::link_ltoir_to_cubin(ltoir, &bundle.name, arch)
                .unwrap_or_else(|e| panic!("LTOIR→cubin for {}: {e}", bundle.name));
            ctx.load_module_from_image(&cubin)
                .unwrap_or_else(|e| panic!("load cubin: {e}"))
        } else {
            panic!(
                "bundle '{}' has no cubin/ptx/nvvm/ltoir payload (target={})",
                bundle.name, bundle.target
            );
        };
        kernels::from_module(module).expect("bind LoadedModule")
    })
}

fn register_one(
    vt: &KernelRegistryVtable,
    op: OpKind,
    inputs: &[DType],
    output: DType,
    launch: fellm_plugin_abi::c_abi::PluginLaunchFn,
) -> c_int {
    let mut reg = PluginOpRegistration {
        op_kind: op.raw(),
        n_input_dtypes: inputs.len() as u32,
        input_dtypes: [0; fellm_plugin_abi::PLUGIN_MAX_INPUT_DTYPES],
        output_dtype: output as u32,
        launch: Some(launch),
    };
    for (i, d) in inputs.iter().enumerate() {
        reg.input_dtypes[i] = *d as u32;
    }
    unsafe { (vt.register_op)(vt.registry, &reg) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_abi_version() -> AbiVersion {
    ABI_VERSION
}

static MANIFEST_JSON: &[u8] = concat!(
    "{\"schema\":1,\"id\":\"fellm.cuda\",\"name\":\"FeLLM CUDA Backend\",\"version\":\"",
    env!("CARGO_PKG_VERSION"),
    "\",\"provides\":[{\"type\":\"backend\",\"id\":\"cuda\"},{\"type\":\"kernels\",\"backend\":\"cuda\"}]}"
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
    catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() {
            return -1;
        }
        *HOST_CTX.lock().expect("host ctx") = Some(unsafe { *ctx });
        // Module load is lazy on first kernel launch (needs CUDA/NVVM env).
        0
    }))
    .unwrap_or(-99)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_register_kernels(
    registry: *mut KernelRegistryVtable,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if registry.is_null() {
            return -1;
        }
        let vt = unsafe { &*registry };
        let f32 = DType::F32;
        let q4k = DType::Q4K;
        let q5k = DType::Q5K;
        let q5_0 = DType::Q5_0;
        let q6k = DType::Q6K;
        let q8_0 = DType::Q8_0;
        let u32 = DType::U32;

        if register_one(
            vt,
            OpKind::Sample,
            &[f32],
            f32,
            launchers::launch_sample_greedy,
        ) != 0
        {
            return -27;
        }
        if register_one(
            vt,
            OpKind::Cast,
            &[f32],
            f32,
            launchers::launch_materialize_f32,
        ) != 0
        {
            return -28;
        }

        // RmsNorm: [x f32, w f32] → f32
        if register_one(
            vt,
            OpKind::RmsNorm,
            &[f32, f32],
            f32,
            launchers::launch_rmsnorm,
        ) != 0
        {
            return -2;
        }
        // SiluGate
        if register_one(
            vt,
            OpKind::SiluGate,
            &[f32, f32],
            f32,
            launchers::launch_silu_gate,
        ) != 0
        {
            return -3;
        }
        if register_one(
            vt,
            OpKind::SigmoidGate,
            &[f32, f32],
            f32,
            launchers::launch_sigmoid_gate,
        ) != 0
            || register_one(
                vt,
                OpKind::InterleavedHeadSelect,
                &[f32],
                f32,
                launchers::launch_interleaved_head_select,
            ) != 0
        {
            return -38;
        }
        // Rope: [x, inv_freqs]
        if register_one(vt, OpKind::Rope, &[f32, f32], f32, launchers::launch_rope) != 0 {
            return -4;
        }
        // Q4_K / Q6_K MatMul + paged Attention + KvWrite (B2).
        if register_one(
            vt,
            OpKind::MatMul,
            &[q4k, f32],
            f32,
            launchers::launch_q4k_matmul,
        ) != 0
        {
            return -5;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q6k, f32],
            f32,
            launchers::launch_q6k_matmul,
        ) != 0
        {
            return -15;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q4k, f32, f32],
            f32,
            launchers::launch_q4k_matmul,
        ) != 0
        {
            return -29;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q6k, f32, f32],
            f32,
            launchers::launch_q6k_matmul,
        ) != 0
        {
            return -30;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q5k, f32],
            f32,
            launchers::launch_q5k_matmul,
        ) != 0
            || register_one(
                vt,
                OpKind::MatMul,
                &[q5k, f32, f32],
                f32,
                launchers::launch_q5k_matmul,
            ) != 0
        {
            return -37;
        }
        if register_one(
            vt,
            OpKind::GateUpSwiGlu,
            &[q4k, q4k, f32],
            f32,
            launchers::launch_q4k_gate_up_swiglu,
        ) != 0
        {
            return -31;
        }
        if register_one(
            vt,
            OpKind::GateUpSwiGlu,
            &[q5k, q5k, f32],
            f32,
            launchers::launch_q5k_gate_up_swiglu,
        ) != 0
        {
            return -40;
        }
        if register_one(
            vt,
            OpKind::GateUpSwiGlu,
            &[q8_0, q8_0, f32],
            f32,
            launchers::launch_q8_0_gate_up_swiglu,
        ) != 0
        {
            return -32;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q8_0, f32],
            f32,
            launchers::launch_q8_0_matmul,
        ) != 0
        {
            return -16;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q8_0, f32, f32],
            f32,
            launchers::launch_q8_0_matmul,
        ) != 0
        {
            return -33;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[q5_0, f32],
            f32,
            launchers::launch_q5_0_matmul,
        ) != 0
        {
            return -23;
        }
        if register_one(
            vt,
            OpKind::MatMul,
            &[DType::BF16, f32],
            f32,
            launchers::launch_bf16_matmul,
        ) != 0
        {
            return -34;
        }
        if register_one(
            vt,
            OpKind::Attention,
            &[f32, f32, f32],
            f32,
            launchers::launch_attention,
        ) != 0
        {
            return -6;
        }
        if register_one(
            vt,
            OpKind::KvWrite,
            &[f32, f32],
            f32,
            launchers::launch_kv_write,
        ) != 0
        {
            return -7;
        }
        if register_one(vt, OpKind::Add, &[f32, f32], f32, launchers::launch_add) != 0 {
            return -8;
        }
        if register_one(vt, OpKind::Mul, &[f32, f32], f32, launchers::launch_mul) != 0 {
            return -24;
        }
        if register_one(
            vt,
            OpKind::Concat,
            &[f32, f32],
            f32,
            launchers::launch_concat_f32,
        ) != 0
        {
            return -36;
        }
        if register_one(
            vt,
            OpKind::Attention,
            &[f32, f32, f32, f32, f32],
            f32,
            launchers::launch_attention,
        ) != 0
        {
            return -25;
        }
        if register_one(
            vt,
            OpKind::WeightedEmbedding,
            &[q6k, f32],
            f32,
            launchers::launch_weighted_embedding_q6k,
        ) != 0
        {
            return -26;
        }
        // Placeholder signature; the launcher dispatches on the actual weight
        // dtypes so any Q4/Q5/Q6/F32 mix of qkv/gate/out stays on device.
        if register_one(
            vt,
            OpKind::GatedDeltaNet,
            &[f32, q4k, q4k, f32, f32, f32, f32, f32, f32, q4k],
            f32,
            launchers::launch_gated_delta_net,
        ) != 0
        {
            return -39;
        }
        if register_one(
            vt,
            OpKind::Embedding,
            &[f32, u32],
            f32,
            launchers::launch_embedding_f32,
        ) != 0
        {
            return -9;
        }
        if register_one(
            vt,
            OpKind::Embedding,
            &[DType::BF16, u32],
            f32,
            launchers::launch_embedding_bf16,
        ) != 0
        {
            return -35;
        }
        if register_one(
            vt,
            OpKind::Embedding,
            &[q4k, u32],
            f32,
            launchers::launch_embedding_q4k,
        ) != 0
        {
            return -10;
        }
        if register_one(
            vt,
            OpKind::Embedding,
            &[q5k, u32],
            f32,
            launchers::launch_embedding_q5k,
        ) != 0
        {
            return -41;
        }
        if register_one(
            vt,
            OpKind::Embedding,
            &[q6k, u32],
            f32,
            launchers::launch_embedding_q6k,
        ) != 0
        {
            return -11;
        }
        if register_one(
            vt,
            OpKind::Embedding,
            &[q8_0, u32],
            f32,
            launchers::launch_embedding_q8_0,
        ) != 0
        {
            return -17;
        }
        if register_one(
            vt,
            OpKind::ShortConv,
            &[f32, q4k, f32, q4k],
            f32,
            launchers::launch_shortconv_q4k,
        ) != 0
        {
            return -18;
        }
        if register_one(
            vt,
            OpKind::MoE,
            &[f32, f32, q4k, q4k, q4k, f32],
            f32,
            launchers::launch_moe_q4k_down,
        ) != 0
        {
            return -19;
        }
        if register_one(
            vt,
            OpKind::MoE,
            &[f32, f32, q4k, q4k, q6k, f32],
            f32,
            launchers::launch_moe_q6k_down,
        ) != 0
        {
            return -20;
        }
        if register_one(
            vt,
            OpKind::MoE,
            &[f32, f32, q4k, q5_0, q4k, q4k, q5_0],
            f32,
            launchers::launch_moe_gemma_q5,
        ) != 0
        {
            return -21;
        }
        if register_one(
            vt,
            OpKind::MoE,
            &[f32, f32, q4k, q8_0, q4k, q4k, q8_0],
            f32,
            launchers::launch_moe_gemma_q8,
        ) != 0
        {
            return -22;
        }
        0
    }))
    .unwrap_or(-99)
}

/// Mark a host f32 buffer's device cache entry stale after a CPU write.
///
/// `ptr` is the host data pointer; `nbytes` is the byte length of the f32 slice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_invalidate_f32(ptr: *const f32, nbytes: usize) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if ptr.is_null() || nbytes < 4 {
            return;
        }
        let len = nbytes / 4;
        // Compiled activations may be backed by a registered external device arena rather than
        // the ordinary pointer-keyed cache. A CPU partition wrote authoritative host bytes, so
        // immediately refresh that stable device address. Ordinary cached tensors are refreshed
        // on the same stream as well, making the CPU-to-GPU ownership handoff unambiguous.
        let host = unsafe { core::slice::from_raw_parts(ptr, len) };
        let stream = oxide_stream().clone();
        if buffers::ensure_f32(&stream, host, true).is_err() {
            buffers::invalidate_f32(ptr, len);
        } else {
            // CPU arena slots can be reused immediately by the compiled schedule. Do not let an
            // asynchronous upload continue reading a host slot after ownership returns to it.
            let _ = stream.synchronize();
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_shutdown() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        *HOST_CTX.lock().expect("host ctx") = None;
    }));
}
