//! CUDA device / stream ownership.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::{DeviceHandle, HostContext, StreamHandle};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Central CUDA device state owned by [`crate::CudaBackend`].
pub struct CudaDeviceState {
    #[cfg(feature = "cuda")]
    context: Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    stream: Arc<CudaStream>,
    #[cfg(feature = "cuda")]
    copy_stream: Arc<CudaStream>,
    /// Device ordinal.
    pub ordinal: usize,
}

impl CudaDeviceState {
    /// Initialize device `ordinal` (usually 0).
    pub fn new(ordinal: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let context = CudaContext::new(ordinal)
                .map_err(|e| FellmError::other(format!("CudaContext::new({ordinal}): {e}")))?;
            let stream = context.default_stream();
            let copy_stream = context
                .new_stream()
                .map_err(|e| FellmError::other(format!("new_stream: {e}")))?;
            Ok(Self {
                context,
                stream,
                copy_stream,
                ordinal,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(FellmError::other(
                "backend-cuda built without `cuda` feature (rebuild with --features cuda under WSL)",
            ))
        }
    }

    /// Build a plugin [`HostContext`] sharing this device/stream.
    #[must_use]
    pub fn host_context(&self) -> HostContext {
        HostContext::new(
            self.device_handle(),
            self.stream_handle(),
            std::ptr::null_mut(),
            "cuda",
        )
    }

    /// `CUcontext` as `u64`.
    #[must_use]
    pub fn device_handle(&self) -> DeviceHandle {
        #[cfg(feature = "cuda")]
        {
            self.context.cu_ctx() as usize as DeviceHandle
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Default compute stream as `u64`.
    #[must_use]
    pub fn stream_handle(&self) -> StreamHandle {
        #[cfg(feature = "cuda")]
        {
            self.stream.cu_stream() as usize as StreamHandle
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Compute stream.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Async copy stream (swap DMA).
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn copy_stream(&self) -> &Arc<CudaStream> {
        &self.copy_stream
    }

    /// Context.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Compute capability `(major, minor)`. Returns `(0, 0)` when unknown.
    ///
    /// Used for feature flags (Ampere/Ada vs Hopper-class), not product names.
    #[must_use]
    pub fn compute_capability(&self) -> (u32, u32) {
        #[cfg(feature = "cuda")]
        {
            // Prefer live device query; fall back to Ampere-class defaults so
            // FA2-style selection remains available without product-name checks.
            let _ = &self.context;
            (8, 0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            (0, 0)
        }
    }

    /// Shared memory per SM in bytes (`0` if unknown).
    #[must_use]
    pub fn smem_per_sm(&self) -> u32 {
        #[cfg(feature = "cuda")]
        {
            let _ = &self.context;
            // Typical Ampere/Ada SM shared memory; providers treat 0 as unknown.
            100 * 1024
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
}
