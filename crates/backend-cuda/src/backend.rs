//! [`CudaBackend`]: prefers CUDA plugins, falls back to [`CpuBackend`] per op.

use crate::device::CudaDeviceState;
use crate::graph::{GraphBucket, GraphCache};
use crate::pinned_swap::PinnedSwapArena;
use crate::vram_pool::DeviceKvArena;
use backend_cpu::CpuBackend;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{Backend, BackendCaps, KernelDescriptor, KernelHandle};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use fellm_plugin_host::PluginHost;
use std::any::Any;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Bit set on handles that route to the embedded CPU backend.
const CPU_FALLBACK_BIT: u64 = 1 << 55;
/// Bit set on handles that route to a loaded CUDA plugin.
const PLUGIN_BIT: u64 = 1 << 56;

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
    /// Bucketed CUDA graphs for decode.
    graphs: Mutex<GraphCache>,
    caps: BackendCaps,
    /// Whether oxide plugin kernels are used (default on when registry non-empty).
    use_plugins: bool,
    /// Host KV arena has writes that are not yet mirrored to VRAM (prefix / swap-in).
    kv_host_dirty: AtomicBool,
}

impl CudaBackend {
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
        let caps = cpu.capabilities();
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
            graphs: Mutex::new(GraphCache::new()),
            caps,
            use_plugins,
            kv_host_dirty: AtomicBool::new(false),
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
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = host;
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Graph cache (inference thread only).
    pub fn graphs(&self) -> &Mutex<GraphCache> {
        &self.graphs
    }

    /// Whether `FELLM_CUDA_GRAPHS=1` is set.
    #[must_use]
    pub fn graphs_enabled() -> bool {
        std::env::var_os("FELLM_CUDA_GRAPHS")
            .is_some_and(|v| v != "0" && v != "false" && v != "off")
    }

    /// Capture a decode graph for `bucket` by running `body` under stream capture.
    #[cfg(feature = "cuda")]
    pub fn capture_decode_graph<F>(&self, bucket: GraphBucket, body: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let mut graphs = self.graphs.lock().expect("graphs lock");
        graphs.capture(&self.device, bucket, body)
    }

    /// Capture decode for the bucket containing `past_len` if not already present.
    pub fn ensure_decode_graph<F>(&self, past_len: u32, max_ctx: u32, body: F) -> Result<bool>
    where
        F: FnOnce() -> Result<()>,
    {
        if !Self::graphs_enabled() {
            return Ok(false);
        }
        let bucket = GraphBucket::buckets_for_ctx(max_ctx)
            .into_iter()
            .find(|b| b.contains(past_len))
            .ok_or_else(|| FellmError::other("no graph bucket for past_len"))?;
        {
            let graphs = self.graphs.lock().expect("graphs lock");
            if graphs.has(past_len) {
                return Ok(false);
            }
        }
        #[cfg(feature = "cuda")]
        {
            self.capture_decode_graph(bucket, body)?;
            tracing::info!(
                past_len,
                min = bucket.min_past,
                max = bucket.max_past,
                "captured CUDA decode graph"
            );
            return Ok(true);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (bucket, body);
            Ok(false)
        }
    }

    /// Launch a previously captured graph for `past_len`, if one exists.
    pub fn try_launch_graph(&self, past_len: u32) -> Result<bool> {
        if !Self::graphs_enabled() {
            return Ok(false);
        }
        let graphs = self.graphs.lock().expect("graphs lock");
        graphs.launch(past_len)
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
