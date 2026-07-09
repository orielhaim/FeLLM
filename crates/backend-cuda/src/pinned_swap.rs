//! Pinned host swap arena for DMA KV page traffic (llama.cpp-style).

use fellm_core::error::{FellmError, Result};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, PinnedHostSlice};

/// Secondary pinned-RAM swap for preempted VRAM blocks.
pub struct PinnedSwapArena {
    #[cfg(feature = "cuda")]
    pinned: PinnedHostSlice<u8>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    block_bytes: usize,
    free_slots: Vec<u32>,
    /// physical_id → swap_slot
    map: HashMap<u32, u32>,
}

impl PinnedSwapArena {
    /// Allocate `n_slots` page-locked host blocks.
    pub fn new(
        device: &crate::device::CudaDeviceState,
        n_slots: usize,
        block_bytes: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if block_bytes == 0 {
                return Err(FellmError::other(
                    "PinnedSwapArena: block_bytes must be > 0",
                ));
            }
            let n_slots = n_slots.max(1);
            let total = n_slots
                .checked_mul(block_bytes)
                .ok_or_else(|| FellmError::other("PinnedSwapArena: size overflow"))?;
            // SAFETY: pinned allocation; freed when PinnedHostSlice drops.
            let pinned = unsafe { device.context().alloc_pinned::<u8>(total) }
                .map_err(|e| FellmError::other(format!("alloc_pinned: {e}")))?;
            let mut free_slots = Vec::with_capacity(n_slots);
            for i in (0..n_slots).rev() {
                free_slots.push(i as u32);
            }
            Ok(Self {
                pinned,
                block_bytes,
                free_slots,
                map: HashMap::new(),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, n_slots, block_bytes);
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Free swap slots remaining.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// DMA one VRAM block into a pinned slot.
    #[cfg(feature = "cuda")]
    pub fn swap_out(
        &mut self,
        device: &crate::device::CudaDeviceState,
        physical_id: u32,
        src: &CudaSlice<u8>,
        src_offset: usize,
    ) -> Result<()> {
        if self.map.contains_key(&physical_id) {
            return Ok(());
        }
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| FellmError::other("PinnedSwapArena: out of swap slots"))?;
        let dst_off = (slot as usize) * self.block_bytes;
        let view = src.slice(src_offset..src_offset + self.block_bytes);
        let pinned = self
            .pinned
            .as_mut_slice()
            .map_err(|e| FellmError::other(format!("pinned as_mut_slice: {e}")))?;
        let pinned_region = &mut pinned[dst_off..dst_off + self.block_bytes];
        device
            .copy_stream()
            .memcpy_dtoh(&view, pinned_region)
            .map_err(|e| FellmError::other(format!("swap_out dtoh: {e}")))?;
        self.map.insert(physical_id, slot);
        Ok(())
    }

    /// DMA a pinned slot back into a VRAM block.
    #[cfg(feature = "cuda")]
    pub fn swap_in(
        &mut self,
        device: &crate::device::CudaDeviceState,
        physical_id: u32,
        dst: &mut CudaSlice<u8>,
        dst_offset: usize,
    ) -> Result<()> {
        let slot = self
            .map
            .remove(&physical_id)
            .ok_or_else(|| FellmError::other("PinnedSwapArena: unknown physical id"))?;
        let src_off = (slot as usize) * self.block_bytes;
        let pinned = self
            .pinned
            .as_slice()
            .map_err(|e| FellmError::other(format!("pinned as_slice: {e}")))?;
        let host = &pinned[src_off..src_off + self.block_bytes];
        let mut dst_view = dst
            .try_slice_mut(dst_offset..dst_offset + self.block_bytes)
            .ok_or_else(|| FellmError::other("swap_in: bad dst range"))?;
        device
            .copy_stream()
            .memcpy_htod(host, &mut dst_view)
            .map_err(|e| FellmError::other(format!("swap_in htod: {e}")))?;
        self.free_slots.push(slot);
        Ok(())
    }

    /// Whether this physical id is currently in swap.
    #[must_use]
    pub fn contains(&self, physical_id: u32) -> bool {
        self.map.contains_key(&physical_id)
    }
}
