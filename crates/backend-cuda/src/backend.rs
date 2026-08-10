//! [`CudaBackend`]: prefers CUDA plugins, falls back to [`CpuBackend`] per op.

use crate::device::CudaDeviceState;
use crate::pinned_swap::PinnedSwapArena;
use crate::vram_pool::DeviceKvArena;
use backend_cpu::CpuBackend;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{Backend, BackendCaps, DeviceKind, KernelDescriptor, KernelHandle};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use fellm_plugin_host::PluginHost;
use std::any::Any;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};

/// Bit set on handles that route to the embedded CPU backend.
const CPU_FALLBACK_BIT: u64 = 1 << 55;
/// Bit set on handles that route to a loaded CUDA plugin.
const PLUGIN_BIT: u64 = 1 << 56;

/// CUDA execution policy. Only debug mode permits arbitrary per-operation CPU fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaExecutionMode {
    /// Every graph operation must resolve to a CUDA kernel.
    StrictCuda,
    /// Explicit model partitions may use the host; arbitrary op fallback is forbidden.
    Hybrid,
    /// Correctness bring-up with visible CPU fallback accounting.
    CpuFallbackDebug,
}

impl CudaExecutionMode {
    fn from_env() -> Result<Self> {
        match std::env::var("FELLM_CUDA_MODE")
            .unwrap_or_else(|_| "strict-cuda".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "strict" | "strict-cuda" => Ok(Self::StrictCuda),
            "hybrid" => Ok(Self::Hybrid),
            "debug" | "cpu-fallback-debug" => Ok(Self::CpuFallbackDebug),
            other => Err(FellmError::other(format!(
                "invalid FELLM_CUDA_MODE '{other}' (expected strict-cuda|hybrid|cpu-fallback-debug)"
            ))),
        }
    }
}

/// Transfer and fallback counters used by correctness assertions and benchmarks.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaExecutionMetrics {
    /// Bytes uploaded after backend initialization.
    pub h2d_bytes: u64,
    /// Bytes downloaded after backend initialization.
    pub d2h_bytes: u64,
    /// Operations executed by the CPU debug fallback.
    pub cpu_fallback_count: u64,
}

/// CUDA compute backend with per-op CPU fallback for missing GPU kernels.
pub struct CudaBackend {
    device: CudaDeviceState,
    plugins: PluginHost,
    /// Full CPU backend for ops not yet covered by CUDA plugins.
    cpu: CpuBackend,
    /// Optional VRAM KV arena (sized at model open).
    kv_arena: Mutex<Option<DeviceKvArena>>,
    /// Optional pinned swap.
    swap: Mutex<Option<PinnedSwapArena>>,
    caps: BackendCaps,
    /// Whether oxide plugin kernels are used (default on when registry non-empty).
    use_plugins: bool,
    /// Host KV arena has writes that are not yet mirrored to VRAM (prefix / swap-in).
    kv_host_dirty: AtomicBool,
    mode: CudaExecutionMode,
    h2d_bytes: AtomicU64,
    d2h_bytes: AtomicU64,
    cpu_fallback_count: AtomicU64,
}

impl CudaBackend {
    /// Update the fixed device control block before enqueueing one decode step.
    pub fn update_step_params(&self, params: &fellm_plugin_abi::DeviceStepParams) -> Result<()> {
        self.plugins.update_step_params(params)
    }

    /// Register one immutable tensor already resident in the packed model image.
    pub fn register_device_tensor(
        &self,
        host_ptr: *const u8,
        nbytes: usize,
        device_ptr: fellm_plugin_abi::DevicePtr,
    ) -> Result<()> {
        self.plugins
            .register_device_tensor(host_ptr, nbytes, device_ptr.0)
    }
    /// Device/context owner used while creating backend physical plans.
    #[must_use]
    pub fn device_state(&self) -> &CudaDeviceState {
        &self.device
    }

    /// Initialize GPU 0 and load plugins from `plugins/` (or `FELLM_PLUGIN_DIR`).
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    /// Initialize a specific device ordinal.
    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        let device = CudaDeviceState::new(ordinal)?;
        let mut plugins = PluginHost::new();
        let ctx = device.host_context();
        let dirs = [
            std::env::var_os("FELLM_PLUGIN_DIR").map(PathBuf::from),
            Some(PathBuf::from("plugins/dist")),
            Some(PathBuf::from("plugins")),
        ];
        let mut seen = std::collections::HashSet::new();
        for dir in dirs.into_iter().flatten() {
            let canon = std::fs::canonicalize(&dir).unwrap_or(dir.clone());
            if !seen.insert(canon) {
                continue;
            }
            if dir.is_dir() {
                let _ = plugins.load_dir(Some(&dir), &ctx);
            }
        }
        let cpu = CpuBackend::new();
        let mode = CudaExecutionMode::from_env()?;
        // Capability flags from compute capability — never product names.
        let (cc_major, cc_minor) = device.compute_capability();
        let has_ampere_ada = cc_major >= 8 && cc_major < 9;
        let has_hopper = cc_major >= 9 && cc_major < 10;
        let has_blackwell = cc_major >= 10;
        let caps = BackendCaps {
            device_kind: DeviceKind::Gpu,
            supports_persistent_device_state: true,
            supports_graph_capture: true,
            supports_async_execution: true,
            supports_read_only_prefix_kv: true,
            // These become true only when the corresponding native kernels exist.
            supports_grouped_moe: false,
            supports_device_sampling: false,
            supports_bidirectional_attention: plugin_ops_supports(&plugins, OpKind::Attention),
            supports_batched_quantized_gemm: plugin_ops_supports(&plugins, OpKind::MatMul),
            supports_custom_operations: true,
            compute_major: cc_major,
            compute_minor: cc_minor,
            smem_per_sm: device.smem_per_sm(),
            has_ampere_ada_features: has_ampere_ada || has_hopper || has_blackwell,
            has_hopper_features: has_hopper || has_blackwell,
            has_blackwell_features: has_blackwell,
            ..BackendCaps::default()
        };
        let plugin_ops = plugins.registry().len();
        // Default ON when registry non-empty. Opt-out: FELLM_PLUGIN_KERNELS=0.
        let use_plugins = Self::resolve_use_plugins(plugin_ops);
        tracing::info!(
            plugin_ops,
            use_plugins,
            "CUDA device up (set FELLM_PLUGIN_KERNELS=0 to disable oxide ops)"
        );
        Ok(Self {
            device,
            plugins,
            cpu,
            kv_arena: Mutex::new(None),
            swap: Mutex::new(None),
            caps,
            use_plugins,
            kv_host_dirty: AtomicBool::new(false),
            mode,
            h2d_bytes: AtomicU64::new(0),
            d2h_bytes: AtomicU64::new(0),
            cpu_fallback_count: AtomicU64::new(0),
        })
    }

    /// `FELLM_PLUGIN_KERNELS`: `0`/`false`/`off` → off; unset or any other value → on when ops exist.
    fn resolve_use_plugins(plugin_ops: usize) -> bool {
        match std::env::var_os("FELLM_PLUGIN_KERNELS") {
            Some(v) if v == "0" || v == "false" || v == "off" => false,
            Some(_) => plugin_ops > 0,
            None => plugin_ops > 0, // default ON
        }
    }

    /// Whether oxide plugin kernels are active for this backend.
    #[must_use]
    pub fn plugins_enabled(&self) -> bool {
        self.use_plugins
    }

    /// Selected CUDA execution policy.
    #[must_use]
    pub fn execution_mode(&self) -> CudaExecutionMode {
        self.mode
    }

    /// Snapshot transfer/fallback instrumentation.
    #[must_use]
    pub fn metrics(&self) -> CudaExecutionMetrics {
        CudaExecutionMetrics {
            h2d_bytes: self.h2d_bytes.load(Ordering::Relaxed),
            d2h_bytes: self.d2h_bytes.load(Ordering::Relaxed),
            cpu_fallback_count: self.cpu_fallback_count.load(Ordering::Relaxed),
        }
    }

    /// Mark host KV as needing a one-shot H2D (prefix attach / swap-in).
    pub fn mark_kv_host_dirty(&self) {
        self.kv_host_dirty.store(true, Ordering::Release);
    }

    /// One-shot full-arena H2D if host KV is dirty and plugins are active.
    ///
    /// Decode does not call this every token — plugin `KvWrite` dual-writes keep
    /// device KV coherent. Only prefix / swap-in / cold host writes set dirty.
    pub fn sync_kv_if_dirty(&self, host: &[u8]) -> Result<()> {
        if !self.use_plugins {
            return Ok(());
        }
        if !self.kv_host_dirty.load(Ordering::Acquire) {
            return Ok(());
        }
        self.sync_kv_host_to_device(host)?;
        self.kv_host_dirty.store(false, Ordering::Release);
        Ok(())
    }

    /// Device state.
    #[must_use]
    pub fn device(&self) -> &CudaDeviceState {
        &self.device
    }

    /// Allocate / replace the VRAM KV arena.
    pub fn init_kv_arena(
        &self,
        n_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
        swap_blocks: usize,
    ) -> Result<()> {
        let arena = DeviceKvArena::new(&self.device, n_blocks, n_kv_heads, head_dim)?;
        let block_bytes = arena.block_bytes();
        let swap = PinnedSwapArena::new(&self.device, swap_blocks.max(1), block_bytes)?;
        tracing::info!(
            n_blocks,
            block_bytes,
            vram_mib = arena.byte_len() / (1024 * 1024),
            "DeviceKvArena ready"
        );
        *self.kv_arena.lock().expect("kv arena lock") = Some(arena);
        *self.swap.lock().expect("swap lock") = Some(swap);
        Ok(())
    }

    /// `(device_ptr, byte_len)` for the VRAM KV arena, if initialized.
    #[must_use]
    pub fn device_kv_ptr(&self) -> Option<(*mut u8, usize)> {
        let guard = self.kv_arena.lock().expect("kv arena lock");
        let arena = guard.as_ref()?;
        #[cfg(feature = "cuda")]
        {
            Some((arena.device_ptr() as *mut u8, arena.byte_len()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = arena;
            None
        }
    }

    /// Upload host `PhysicalPool` bytes into the VRAM arena (prefix / cold start).
    pub fn sync_kv_host_to_device(&self, host: &[u8]) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let mut guard = self.kv_arena.lock().expect("kv arena lock");
            let arena = guard
                .as_mut()
                .ok_or_else(|| FellmError::other("DeviceKvArena not initialized"))?;
            if host.len() != arena.byte_len() {
                return Err(FellmError::other(format!(
                    "KV H2D size mismatch: host={} device={}",
                    host.len(),
                    arena.byte_len()
                )));
            }
            self.device
                .stream()
                .memcpy_htod(host, arena.buffer_mut())
                .map_err(|e| FellmError::other(format!("KV H2D: {e}")))?;
            self.h2d_bytes
                .fetch_add(host.len() as u64, Ordering::Relaxed);
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = host;
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Download VRAM arena into host `PhysicalPool` (swap / debug).
    pub fn sync_kv_device_to_host(&self, host: &mut [u8]) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let guard = self.kv_arena.lock().expect("kv arena lock");
            let arena = guard
                .as_ref()
                .ok_or_else(|| FellmError::other("DeviceKvArena not initialized"))?;
            if host.len() != arena.byte_len() {
                return Err(FellmError::other(format!(
                    "KV D2H size mismatch: host={} device={}",
                    host.len(),
                    arena.byte_len()
                )));
            }
            self.device
                .stream()
                .memcpy_dtoh(arena.buffer(), host)
                .map_err(|e| FellmError::other(format!("KV D2H: {e}")))?;
            self.d2h_bytes
                .fetch_add(host.len() as u64, Ordering::Relaxed);
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = host;
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Plugin host.
    #[must_use]
    pub fn plugins(&self) -> &PluginHost {
        &self.plugins
    }
}

impl Backend for CudaBackend {
    fn id(&self) -> &'static str {
        "cuda"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn memory_info(&self) -> Option<fellm_plugin_abi::DeviceMemoryInfo> {
        #[cfg(feature = "cuda")]
        {
            let (available, total) = self.device.context().mem_get_info().ok()?;
            Some(fellm_plugin_abi::DeviceMemoryInfo {
                total_bytes: total as u64,
                available_bytes: available as u64,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn synchronize(&self) -> Result<()> {
        // No GPU work when plugins are off — skip driver sync.
        if !self.use_plugins {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            self.device
                .stream()
                .synchronize()
                .map_err(|e| FellmError::other(format!("cuda synchronize: {e}")))?;
        }
        Ok(())
    }

    fn begin_step(&self) {
        self.cpu.begin_step();
    }

    fn end_step(&self) {
        self.cpu.end_step();
    }

    fn sample_device(&self, logits: TensorRef, attrs: &OpAttrs) -> Result<Option<u32>> {
        let Some(descriptor) = self.resolve_kernel(OpKind::Sample, &[DType::F32], DType::F32)
        else {
            return Ok(None);
        };
        let mut token = [0.0f32];
        let mut output = unsafe {
            TensorMut::from_raw(
                DType::F32,
                &[1],
                &[1],
                token.as_mut_ptr().cast(),
                core::mem::size_of::<f32>(),
            )
        };
        self.launch(
            descriptor.handle,
            attrs,
            &[logits],
            std::slice::from_mut(&mut output),
            self.device.stream_handle(),
        )?;
        Ok(Some(token[0] as u32))
    }

    fn materialize(&self, tensor: TensorRef, mut host: TensorMut) -> Result<()> {
        let descriptor = self
            .resolve_kernel(OpKind::Cast, &[DType::F32], DType::F32)
            .ok_or_else(|| FellmError::other("CUDA f32 materialization kernel is unavailable"))?;
        self.launch(
            descriptor.handle,
            &OpAttrs::default(),
            &[tensor],
            std::slice::from_mut(&mut host),
            self.device.stream_handle(),
        )
    }

    fn resolve_kernel(
        &self,
        op: OpKind,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<KernelDescriptor> {
        if self.use_plugins
            && let Some((h, _)) = self
                .plugins
                .registry()
                .lookup(op, input_dtypes, output_dtype)
        {
            return Some(KernelDescriptor {
                op,
                input_dtypes: input_dtypes.to_vec(),
                output_dtype,
                handle: KernelHandle(PLUGIN_BIT | h),
            });
        }
        if self.mode != CudaExecutionMode::CpuFallbackDebug {
            return None;
        }
        let desc = self.cpu.resolve_kernel(op, input_dtypes, output_dtype)?;
        Some(KernelDescriptor {
            op: desc.op,
            input_dtypes: desc.input_dtypes,
            output_dtype: desc.output_dtype,
            handle: KernelHandle(CPU_FALLBACK_BIT | desc.handle.0),
        })
    }

    fn launch(
        &self,
        handle: KernelHandle,
        attrs: &OpAttrs,
        inputs: &[TensorRef],
        outputs: &mut [TensorMut],
        stream: StreamHandle,
    ) -> Result<()> {
        if handle.0 & PLUGIN_BIT != 0 {
            let stream = if stream == 0 {
                self.device.stream_handle()
            } else {
                stream
            };
            let h = handle.0 & (PLUGIN_BIT - 1);
            return self
                .plugins
                .registry()
                .launch(h, attrs, inputs, outputs, stream);
        }
        if handle.0 & CPU_FALLBACK_BIT != 0 {
            self.cpu_fallback_count.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                op_handle = handle.0,
                "CPU fallback in cpu-fallback-debug CUDA mode"
            );
            let cpu_handle = KernelHandle(handle.0 & (CPU_FALLBACK_BIT - 1));
            let result = self.cpu.launch(cpu_handle, attrs, inputs, outputs, 0);
            if result.is_ok() && self.use_plugins {
                // CPU wrote host activations; drop stale device_valid mirrors.
                self.plugins.invalidate_f32_outputs(outputs);
            }
            return result;
        }
        Err(FellmError::other(format!(
            "unknown cuda handle {:#x}",
            handle.0
        )))
    }
}

fn plugin_ops_supports(plugins: &PluginHost, op: OpKind) -> bool {
    plugins.registry().supports_op(op)
}
