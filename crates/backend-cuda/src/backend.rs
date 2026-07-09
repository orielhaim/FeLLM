//! [`CudaBackend`]: prefers CUDA plugins, falls back to [`CpuBackend`] per op.

use crate::device::CudaDeviceState;
use crate::graph::GraphCache;
use crate::pinned_swap::PinnedSwapArena;
use crate::vram_pool::DeviceKvArena;
use backend_cpu::CpuBackend;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{Backend, BackendCaps, KernelDescriptor, KernelHandle};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use fellm_plugin_host::PluginHost;
use std::path::PathBuf;
use std::sync::Mutex;

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
        for dir in dirs.into_iter().flatten() {
            if dir.is_dir() {
                let _ = plugins.load_dir(Some(&dir), &ctx);
            }
        }
        let cpu = CpuBackend::new();
        let caps = cpu.capabilities();
        let plugin_ops = plugins.registry().len();
        let use_plugins = std::env::var_os("FELLM_PLUGIN_KERNELS")
            .is_some_and(|v| v != "0" && v != "false" && v != "off");
        tracing::info!(
            plugin_ops,
            use_plugins,
            "CUDA device up (ops run on CPU unless FELLM_PLUGIN_KERNELS=1 and oxide kernels are registered)"
        );
        Ok(Self {
            device,
            plugins,
            cpu,
            kv_arena: Mutex::new(None),
            swap: Mutex::new(None),
            graphs: Mutex::new(GraphCache::new()),
            caps,
        })
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
        let swap = PinnedSwapArena::new(&self.device, swap_blocks, block_bytes)?;
        *self.kv_arena.lock().expect("kv arena lock") = Some(arena);
        *self.swap.lock().expect("swap lock") = Some(swap);
        Ok(())
    }

    /// Graph cache (inference thread only).
    pub fn graphs(&self) -> &Mutex<GraphCache> {
        &self.graphs
    }

    /// Capture a decode graph for `bucket` by running `body` under stream capture.
    #[cfg(feature = "cuda")]
    pub fn capture_decode_graph<F>(&self, bucket: crate::graph::GraphBucket, body: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let mut graphs = self.graphs.lock().expect("graphs lock");
        graphs.capture(&self.device, bucket, body)
    }

    /// Launch a previously captured graph for `past_len`, if one exists.
    pub fn try_launch_graph(&self, past_len: u32) -> Result<bool> {
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

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn resolve_kernel(
        &self,
        op: OpKind,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<KernelDescriptor> {
        // Plugin kernels only when explicitly enabled. A stale/broken
        // libcuda_kernels.so must never override correct CPU Q4_K / attention.
        // Set FELLM_PLUGIN_KERNELS=1 after oxide kernels are validated.
        let use_plugins = std::env::var_os("FELLM_PLUGIN_KERNELS")
            .is_some_and(|v| v != "0" && v != "false" && v != "off");
        if use_plugins {
            if let Some((h, _)) = self
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
            return self.cpu.launch(cpu_handle, attrs, inputs, outputs, 0);
        }
        Err(FellmError::other(format!(
            "unknown cuda handle {:#x}",
            handle.0
        )))
    }
}
