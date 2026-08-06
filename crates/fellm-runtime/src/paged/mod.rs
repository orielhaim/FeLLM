//! Paged KV cache: physical pool, block tables, prefix sharing, and swap.

pub mod pool;
pub mod prefix;
pub mod swap;
pub mod table;

#[cfg(test)]
mod tests_share;

pub use pool::{BLOCK_SIZE, BlockMeta, PhysicalPool};
pub use prefix::PrefixTree;
pub use swap::SwapArena;
pub use table::{BlockTable, SequenceCache};

use fellm_core::error::{FellmError, Result};

/// Owns the shared physical pool, prefix index, and swap tier.
pub struct CacheManager {
    /// Physical block arena.
    pub pool: PhysicalPool,
    /// Radix prefix cache.
    pub prefix: PrefixTree,
    /// Secondary RAM swap.
    pub swap: SwapArena,
    /// Monotonic clock for LRU.
    pub clock: u64,
}

impl CacheManager {
    /// Create a manager sized for `n_blocks` physical blocks.
    pub fn new(
        n_blocks: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        swap_blocks: usize,
    ) -> Result<Self> {
        let pool = PhysicalPool::new(n_blocks, n_layers, n_kv_heads, head_dim)?;
        let swap = SwapArena::new(swap_blocks, pool.block_bytes())?;
        Ok(Self {
            pool,
            prefix: PrefixTree::new(),
            swap,
            clock: 0,
        })
    }

    /// Default sizing: enough blocks for `max_seq` tokens × layers × concurrency hint.
    pub fn with_capacity(
        max_seq: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        concurrency: usize,
    ) -> Result<Self> {
        let blocks_per_seq = max_seq.div_ceil(BLOCK_SIZE).max(1) * n_layers.max(1);
        let n_blocks = blocks_per_seq
            .saturating_mul(concurrency.max(1))
            .max(n_layers.max(1));
        let swap_blocks = n_blocks / 2;
        Self::new(n_blocks, n_layers, n_kv_heads, head_dim, swap_blocks)
    }

    /// Tick LRU clock.
    pub fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    /// Allocate a fresh sequence cache (empty tables).
    pub fn new_sequence(&self, max_seq: usize) -> SequenceCache {
        SequenceCache::new(self.pool.n_layers(), max_seq)
    }

    /// Free all physical blocks owned exclusively by a sequence.
    pub fn release_sequence(&mut self, seq: &mut SequenceCache) {
        for layer in 0..seq.n_layers() {
            for &phys in seq.table(layer).blocks() {
                self.pool.dec_ref(phys);
            }
        }
        seq.clear_tables();
    }

    /// Ensure `pos` is writable for every layer (allocate / `CoW` as needed).
    pub fn ensure_writable(&mut self, seq: &mut SequenceCache, pos: usize) -> Result<()> {
        if pos >= seq.max_seq {
            return Err(FellmError::other(format!(
                "paged cache: position {pos} >= max_seq {}",
                seq.max_seq
            )));
        }
        let logical = pos / BLOCK_SIZE;
        for layer in 0..seq.n_layers() {
            while seq.table(layer).num_blocks() <= logical {
                let id = self
                    .pool
                    .alloc_block()
                    .ok_or_else(|| FellmError::other("paged cache: out of physical blocks"))?;
                self.pool.inc_ref(id);
                seq.table_mut(layer).push_block(id);
            }
            let phys = seq.table(layer).block(logical);
            if self.pool.refcount(phys) > 1 {
                // Copy-on-write.
                let new_id = self
                    .pool
                    .alloc_block()
                    .ok_or_else(|| FellmError::other("paged cache: CoW alloc failed"))?;
                self.pool.copy_block(phys, new_id);
                self.pool.inc_ref(new_id);
                self.pool.dec_ref(phys);
                *seq.table_mut(layer).block_mut(logical) = new_id;
                self.pool.touch(new_id, self.clock);
            } else {
                self.pool.touch(phys, self.clock);
            }
        }
        if pos + 1 > seq.len_tokens {
            seq.len_tokens = pos + 1;
        }
        Ok(())
    }
}
