//! CUDA device / stream ownership.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::{DeviceHandle, HostContext, StreamHandle};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream, sys::CUdevice_attribute as Attr};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Hardware properties that affect CUDA kernel selection and autotuning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaDeviceCaps {
    /// Compute capability major version.
    pub compute_major: u32,
    /// Compute capability minor version.
    pub compute_minor: u32,
    /// Streaming multiprocessor count.
    pub sm_count: u32,
    /// Hardware warp width.
    pub warp_size: u32,
    /// Maximum shared memory per block in bytes.
    pub smem_per_block: u32,
    /// Maximum shared memory per SM in bytes.
    pub smem_per_sm: u32,
    /// Register file size per SM.
    pub registers_per_sm: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// L2 cache size in bytes.
    pub l2_bytes: u64,
    /// Device-memory bus width in bits.
    pub memory_bus_width_bits: u32,
    /// Device-memory clock in kHz.
    pub memory_clock_khz: u32,
}

/// Central CUDA device state owned by [`crate::CudaBackend`].
pub struct CudaDeviceState {
    #[cfg(feature = "cuda")]
    context: Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    stream: Arc<CudaStream>,
    #[cfg(feature = "cuda")]
    copy_stream: Arc<CudaStream>,
    #[cfg(feature = "cuda")]
    caps: CudaDeviceCaps,
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
            let caps = query_caps(&context)?;
            Ok(Self {
                context,
                stream,
                copy_stream,
                caps,
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
            (self.caps.compute_major, self.caps.compute_minor)
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
            self.caps.smem_per_sm
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Complete live hardware capability snapshot.
    #[must_use]
    pub fn caps(&self) -> CudaDeviceCaps {
        #[cfg(feature = "cuda")]
        {
            self.caps
        }
        #[cfg(not(feature = "cuda"))]
        {
            CudaDeviceCaps::default()
        }
    }
}

#[cfg(feature = "cuda")]
fn query_caps(context: &CudaContext) -> Result<CudaDeviceCaps> {
    let attribute = |attr| {
        context
            .attribute(attr)
            .map(|value| value.max(0) as u32)
            .map_err(|error| FellmError::other(format!("query CUDA device attribute: {error}")))
    };
    Ok(CudaDeviceCaps {
        compute_major: attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
        compute_minor: attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        sm_count: attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        warp_size: attribute(Attr::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
        smem_per_block: attribute(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
        smem_per_sm: attribute(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
        registers_per_sm: attribute(Attr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?,
        max_threads_per_sm: attribute(Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        l2_bytes: u64::from(attribute(Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?),
        memory_bus_width_bits: attribute(Attr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
        memory_clock_khz: attribute(Attr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
    })
}
