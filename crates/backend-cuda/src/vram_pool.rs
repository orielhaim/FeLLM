//! Device-side KV arena (fixed-size physical blocks in VRAM).

use fellm_core::error::{FellmError, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr};

/// Tokens per physical block (matches CPU [`fellm_runtime::paged::BLOCK_SIZE`]).
#[allow(dead_code)] // used when `cuda` feature is enabled
pub const BLOCK_SIZE: usize = 16;

/// Contiguous VRAM arena of fixed-size KV blocks.
pub struct DeviceKvArena {
    #[cfg(feature = "cuda")]
    buffer: CudaSlice<u8>,
    n_blocks: usize,
    block_bytes: usize,
    tokens_stride: usize,
}

impl DeviceKvArena {
    /// Allocate `n_blocks` zeroed physical blocks on the device.
    pub fn new(
        device: &crate::device::CudaDeviceState,
        n_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if n_blocks == 0 {
                return Err(FellmError::other("DeviceKvArena: n_blocks must be > 0"));
            }
            let tokens_stride = n_kv_heads.max(1) * head_dim.max(1);
            let block_elems = 2 * BLOCK_SIZE * tokens_stride;
            let block_bytes = block_elems * 4;
            let total = n_blocks
                .checked_mul(block_bytes)
                .ok_or_else(|| FellmError::other("DeviceKvArena: size overflow"))?;
            let buffer = device
                .stream()
                .alloc_zeros::<u8>(total)
                .map_err(|e| FellmError::other(format!("alloc_zeros KV arena: {e}")))?;
            Ok(Self {
                buffer,
                n_blocks,
                block_bytes,
                tokens_stride,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, n_blocks, n_kv_heads, head_dim);
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Number of physical blocks.
    #[must_use]
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Bytes per block.
    #[must_use]
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// f32 elements per token row.
    #[must_use]
    pub fn tokens_stride(&self) -> usize {
        self.tokens_stride
    }

    /// Device pointer to the arena base.
    #[cfg(feature = "cuda")]
    pub fn device_ptr(&self) -> u64 {
        let (ptr, _sync) = self.buffer.device_ptr(self.buffer.stream());
        ptr as u64
    }

    /// Total arena bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.n_blocks * self.block_bytes
    }

    /// Mutable device buffer.
    #[cfg(feature = "cuda")]
    pub fn buffer_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.buffer
    }

    /// Immutable device buffer.
    #[cfg(feature = "cuda")]
    pub fn buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }
}
