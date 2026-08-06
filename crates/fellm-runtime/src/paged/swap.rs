//! Secondary RAM swap arena for preempted sequence blocks.

use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use std::collections::HashMap;

/// Maps a physical block id → swap slot while swapped out.
pub struct SwapArena {
    arena: AlignedBuffer,
    block_bytes: usize,
    free_slots: Vec<u32>,
    map: HashMap<u32, u32>,
}

impl SwapArena {
    /// Create a swap arena holding `n_slots` block-sized regions.
    pub fn new(n_slots: usize, block_bytes: usize) -> Result<Self> {
        if block_bytes == 0 {
            return Err(FellmError::other("SwapArena: block_bytes must be > 0"));
        }
        let n_slots = n_slots.max(1);
        let total = n_slots
            .checked_mul(block_bytes)
            .ok_or_else(|| FellmError::other("SwapArena: size overflow"))?;
        let arena = AlignedBuffer::new_zeroed(total, 64);
        let mut free_slots = Vec::with_capacity(n_slots);
        for i in (0..n_slots).rev() {
            free_slots.push(i as u32);
        }
        Ok(Self {
            arena,
            block_bytes,
            free_slots,
            map: HashMap::new(),
        })
    }

    /// Free swap slots remaining.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Copy `src_bytes` (one physical block) into a swap slot; remember mapping.
    pub fn swap_out(&mut self, physical_id: u32, src_bytes: &[u8]) -> Result<()> {
        if src_bytes.len() != self.block_bytes {
            return Err(FellmError::other("SwapArena: size mismatch"));
        }
        if self.map.contains_key(&physical_id) {
            return Ok(());
        }
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| FellmError::other("SwapArena: out of swap slots"))?;
        let off = (slot as usize) * self.block_bytes;
        self.arena.as_mut_slice()[off..off + self.block_bytes].copy_from_slice(src_bytes);
        self.map.insert(physical_id, slot);
        Ok(())
    }

    /// Restore bytes for `physical_id` into `dst_bytes` and free the swap slot.
    pub fn swap_in(&mut self, physical_id: u32, dst_bytes: &mut [u8]) -> Result<()> {
        if dst_bytes.len() != self.block_bytes {
            return Err(FellmError::other("SwapArena: size mismatch"));
        }
        let slot = self
            .map
            .remove(&physical_id)
            .ok_or_else(|| FellmError::other("SwapArena: unknown physical id"))?;
        let off = (slot as usize) * self.block_bytes;
        dst_bytes.copy_from_slice(&self.arena.as_slice()[off..off + self.block_bytes]);
        self.free_slots.push(slot);
        Ok(())
    }

    /// Whether this physical id is currently in swap.
    #[must_use]
    pub fn contains(&self, physical_id: u32) -> bool {
        self.map.contains_key(&physical_id)
    }
}
