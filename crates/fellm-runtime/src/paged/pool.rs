use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use half::f16;

/// Tokens stored in one physical block.
pub const BLOCK_SIZE: usize = 16;

/// Bytes per KV element (`f16`).
pub const ELEM_BYTES: usize = 2;

/// Metadata for one physical block.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockMeta {
    /// Active sharers (sequences / prefix nodes).
    pub refcount: u32,
    /// LRU timestamp (from [`super::CacheManager::clock`]).
    pub last_used: u64,
}

/// Contiguous arena of fixed-size KV blocks.
///
/// Each physical block holds K and V for [`BLOCK_SIZE`] tokens of **one**
/// attention layer:
/// `2 × BLOCK_SIZE × n_kv_heads × head_dim` f16 elements (plus 64-byte pad).
pub struct PhysicalPool {
    arena: AlignedBuffer,
    meta: Vec<BlockMeta>,
    free: Vec<u32>,
    n_blocks: usize,
    n_layers: usize,
    tokens_stride: usize,
    block_elems: usize,
    block_bytes: usize,
}

impl PhysicalPool {
    /// Pre-allocate `n_blocks` physical blocks (shared across layers via free list).
    pub fn new(
        n_blocks: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if n_blocks == 0 {
            return Err(FellmError::other("PhysicalPool: n_blocks must be > 0"));
        }
        let tokens_stride = n_kv_heads.max(1) * head_dim.max(1);
        // K then V for BLOCK_SIZE tokens.
        let block_elems = 2 * BLOCK_SIZE * tokens_stride;
        let raw_bytes = block_elems * ELEM_BYTES;
        // Pad so each block starts on a 64-byte boundary.
        let block_bytes = (raw_bytes + 63) & !63;
        debug_assert_eq!(
            block_bytes % 64,
            0,
            "block_bytes must be 64-byte aligned (got {block_bytes})"
        );
        let total_bytes = n_blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| FellmError::other("PhysicalPool: size overflow"))?;
        let arena = AlignedBuffer::new_zeroed(total_bytes, 64);
        debug_assert_eq!(arena.as_slice().as_ptr() as usize % 64, 0);

        let mut free = Vec::with_capacity(n_blocks);
        for i in (0..n_blocks).rev() {
            free.push(i as u32);
        }
        Ok(Self {
            arena,
            meta: vec![BlockMeta::default(); n_blocks],
            free,
            n_blocks,
            n_layers,
            tokens_stride,
            block_elems,
            block_bytes,
        })
    }

    /// Number of physical blocks in the arena.
    #[must_use]
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Attention layer count this pool was sized for (informational).
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Per-token K (or V) stride in elements.
    #[must_use]
    pub fn tokens_stride(&self) -> usize {
        self.tokens_stride
    }

    /// Bytes per physical block (includes alignment padding).
    #[must_use]
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// f16 elements per physical block (payload, excluding pad).
    #[must_use]
    pub fn block_elems(&self) -> usize {
        self.block_elems
    }

    /// Free-list length.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Allocate one physical block. `O(1)`.
    pub fn alloc_block(&mut self) -> Option<u32> {
        let id = self.free.pop()?;
        self.meta[id as usize] = BlockMeta {
            refcount: 0,
            last_used: 0,
        };
        // Zero the block for safety.
        self.block_bytes_mut(id).fill(0);
        Some(id)
    }

    /// Return a block to the free list if refcount is zero. `O(1)`.
    pub fn free_block(&mut self, id: u32) {
        let idx = id as usize;
        debug_assert!(idx < self.n_blocks);
        if self.meta[idx].refcount == 0 {
            self.free.push(id);
        }
    }

    /// Increment reference count.
    pub fn inc_ref(&mut self, id: u32) {
        let m = &mut self.meta[id as usize];
        m.refcount = m.refcount.saturating_add(1);
    }

    /// Decrement reference count; free when it hits zero.
    pub fn dec_ref(&mut self, id: u32) {
        let idx = id as usize;
        let m = &mut self.meta[idx];
        m.refcount = m.refcount.saturating_sub(1);
        if m.refcount == 0 {
            self.free.push(id);
        }
    }

    /// Current refcount.
    #[must_use]
    pub fn refcount(&self, id: u32) -> u32 {
        self.meta[id as usize].refcount
    }

    /// Update LRU timestamp.
    pub fn touch(&mut self, id: u32, clock: u64) {
        self.meta[id as usize].last_used = clock;
    }

    /// LRU timestamp.
    #[must_use]
    pub fn last_used(&self, id: u32) -> u64 {
        self.meta[id as usize].last_used
    }

    /// Byte offset of a physical block in the arena.
    #[must_use]
    pub fn block_byte_offset(&self, id: u32) -> usize {
        (id as usize) * self.block_bytes
    }

    fn block_bytes_mut(&mut self, id: u32) -> &mut [u8] {
        let off = self.block_byte_offset(id);
        let end = off + self.block_bytes;
        &mut self.arena.as_mut_slice()[off..end]
    }

    /// Payload bytes for K|V (excludes trailing alignment pad).
    fn block_payload_bytes(&self) -> usize {
        self.block_elems * ELEM_BYTES
    }

    fn block_f16(&self, id: u32) -> &[f16] {
        let off = self.block_byte_offset(id);
        let bytes = &self.arena.as_slice()[off..off + self.block_payload_bytes()];
        bytemuck::cast_slice(bytes)
    }

    fn block_f16_mut(&mut self, id: u32) -> &mut [f16] {
        let off = self.block_byte_offset(id);
        let n = self.block_payload_bytes();
        let bytes = &mut self.arena.as_mut_slice()[off..off + n];
        bytemuck::cast_slice_mut(bytes)
    }

    /// Copy entire physical block `src` → `dst`.
    pub fn copy_block(&mut self, src: u32, dst: u32) {
        // Split borrows via raw offsets.
        let src_off = self.block_byte_offset(src);
        let dst_off = self.block_byte_offset(dst);
        let n = self.block_bytes;
        let arena = self.arena.as_mut_slice();
        // SAFETY: distinct blocks never overlap.
        debug_assert_ne!(src, dst);
        unsafe {
            let sp = arena.as_ptr().add(src_off);
            let dp = arena.as_mut_ptr().add(dst_off);
            core::ptr::copy_nonoverlapping(sp, dp, n);
        }
    }

    /// K row for `(physical_block, slot)` — length `tokens_stride`.
    pub fn k_row(&self, id: u32, slot: usize) -> &[f16] {
        debug_assert!(slot < BLOCK_SIZE);
        let block = self.block_f16(id);
        let base = slot * self.tokens_stride;
        &block[base..base + self.tokens_stride]
    }

    /// Mutable K row.
    pub fn k_row_mut(&mut self, id: u32, slot: usize) -> &mut [f16] {
        debug_assert!(slot < BLOCK_SIZE);
        let stride = self.tokens_stride;
        let block = self.block_f16_mut(id);
        let base = slot * stride;
        &mut block[base..base + stride]
    }

    /// V row for `(physical_block, slot)`.
    pub fn v_row(&self, id: u32, slot: usize) -> &[f16] {
        debug_assert!(slot < BLOCK_SIZE);
        let block = self.block_f16(id);
        let base = BLOCK_SIZE * self.tokens_stride + slot * self.tokens_stride;
        &block[base..base + self.tokens_stride]
    }

    /// Mutable V row.
    pub fn v_row_mut(&mut self, id: u32, slot: usize) -> &mut [f16] {
        debug_assert!(slot < BLOCK_SIZE);
        let stride = self.tokens_stride;
        let block = self.block_f16_mut(id);
        let base = BLOCK_SIZE * stride + slot * stride;
        &mut block[base..base + stride]
    }

    /// Raw arena bytes (for swap / FFI).
    #[must_use]
    pub fn arena_bytes(&self) -> &[u8] {
        self.arena.as_slice()
    }

    /// Mutable pointer to the arena base (for paged kernel context).
    #[must_use]
    pub fn arena_ptr_mut(&mut self) -> (*mut u8, usize) {
        let len = self.arena.len();
        (self.arena.as_mut_slice().as_mut_ptr(), len)
    }

    /// Copy one physical block's bytes into `dst`.
    pub fn read_block_bytes(&self, id: u32, dst: &mut [u8]) {
        let off = self.block_byte_offset(id);
        dst.copy_from_slice(&self.arena.as_slice()[off..off + self.block_bytes]);
    }

    /// Overwrite one physical block from `src`.
    pub fn write_block_bytes(&mut self, id: u32, src: &[u8]) {
        let off = self.block_byte_offset(id);
        self.arena.as_mut_slice()[off..off + self.block_bytes].copy_from_slice(src);
    }

    /// Find the least-recently-used allocated block (refcount > 0), if any.
    pub fn lru_block(&self) -> Option<u32> {
        self.meta
            .iter()
            .enumerate()
            .filter(|(_, m)| m.refcount > 0)
            .min_by_key(|(_, m)| m.last_used)
            .map(|(i, _)| i as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_roundtrip() {
        let mut pool = PhysicalPool::new(4, 2, 2, 8).unwrap();
        assert_eq!(pool.free_count(), 4);
        let a = pool.alloc_block().unwrap();
        let b = pool.alloc_block().unwrap();
        assert_ne!(a, b);
        assert_eq!(pool.free_count(), 2);
        pool.inc_ref(a);
        pool.dec_ref(a);
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn exhaustion() {
        let mut pool = PhysicalPool::new(2, 1, 1, 4).unwrap();
        assert!(pool.alloc_block().is_some());
        assert!(pool.alloc_block().is_some());
        assert!(pool.alloc_block().is_none());
    }

    #[test]
    fn block_alignment() {
        let pool = PhysicalPool::new(8, 1, 4, 64).unwrap();
        // tokens_stride = 256, block_elems = 2*16*256 = 8192, raw = 16384, pad = 16384
        assert_eq!(pool.block_bytes(), 16384);
        assert_eq!(pool.block_bytes() % 64, 0);
        for id in 0..pool.n_blocks() as u32 {
            let off = pool.block_byte_offset(id);
            assert_eq!(off % 64, 0, "block {id} offset {off}");
        }
    }

    #[test]
    fn block_bytes_halved_vs_f32() {
        // Typical: n_kv=8, head_dim=64 → stride=512, elems=2*16*512=16384
        // f32 was 65536; f16 is 32768.
        let pool = PhysicalPool::new(1, 1, 8, 64).unwrap();
        assert_eq!(pool.tokens_stride(), 512);
        assert_eq!(pool.block_elems(), 16384);
        assert_eq!(pool.block_bytes(), 32768);
    }

    #[test]
    fn k_v_row_write() {
        let mut pool = PhysicalPool::new(2, 1, 2, 4).unwrap();
        let id = pool.alloc_block().unwrap();
        pool.inc_ref(id);
        {
            let row = pool.k_row_mut(id, 3);
            for (i, x) in row.iter_mut().enumerate() {
                *x = f16::from_f32(i as f32);
            }
        }
        {
            let row = pool.v_row_mut(id, 3);
            for (i, x) in row.iter_mut().enumerate() {
                *x = f16::from_f32(100.0 + i as f32);
            }
        }
        assert_eq!(pool.k_row(id, 3)[0].to_f32(), 0.0);
        assert_eq!(pool.v_row(id, 3)[0].to_f32(), 100.0);
    }

    #[test]
    fn copy_block() {
        let mut pool = PhysicalPool::new(2, 1, 1, 4).unwrap();
        let a = pool.alloc_block().unwrap();
        let b = pool.alloc_block().unwrap();
        pool.inc_ref(a);
        pool.inc_ref(b);
        pool.k_row_mut(a, 0)[0] = f16::from_f32(42.0);
        pool.copy_block(a, b);
        assert_eq!(pool.k_row(b, 0)[0].to_f32(), 42.0);
    }
}
