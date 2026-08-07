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

    /// Compact sequence to retained **original absolute** positions.
    ///
    /// Full lifecycle:
    /// 1. Copy retained K/V rows into new dense physical blocks (storage order)
    /// 2. Rebuild each layer's block table to only those new blocks
    /// 3. `dec_ref` all previous exclusive blocks (free list grows)
    /// 4. Set `len_tokens = retained.len()` so attention work shrinks
    /// 5. Record `original_positions` for policy / view consumers
    ///
    /// Positions in the immutable shared prefix (`seq.shared_prefix_len`) are
    /// always kept even if omitted from `retain_positions`.
    ///
    /// Returns physical blocks reclaimed into the free list.
    pub fn compact_sequence_to_positions(
        &mut self,
        seq: &mut SequenceCache,
        retain_positions: &[u32],
        block_size: usize,
    ) -> usize {
        let bs = block_size.max(1);
        let prefix = seq.shared_prefix_len as u32;
        // Always keep shared prefix + requested retain set, sorted unique.
        let mut keep: Vec<u32> = (0..prefix).collect();
        for &p in retain_positions {
            if p >= prefix {
                keep.push(p);
            }
        }
        keep.sort_unstable();
        keep.dedup();
        // Convert keep absolute positions to dense source indices under current layout.
        let prior_orig = seq.original_positions.clone();
        let was_compressed = !prior_orig.is_empty();
        let mut src_dense: Vec<usize> = Vec::with_capacity(keep.len());
        if was_compressed {
            for &abs in &keep {
                if let Some(i) = prior_orig.iter().position(|&p| p == abs) {
                    src_dense.push(i);
                }
            }
        } else {
            for &abs in &keep {
                if (abs as usize) < seq.len_tokens {
                    src_dense.push(abs as usize);
                }
            }
        }
        if src_dense.is_empty() {
            return 0;
        }

        let n_new = src_dense.len();
        let n_new_blocks = n_new.div_ceil(bs).max(1);
        let mut reclaimed = 0usize;
        // Absolute identities for each new dense slot (before mutating seq).
        let orig: Vec<u32> = src_dense
            .iter()
            .map(|&i| {
                if was_compressed {
                    prior_orig.get(i).copied().unwrap_or(i as u32)
                } else {
                    i as u32
                }
            })
            .collect();

        for layer in 0..seq.n_layers() {
            let old_blocks: Vec<u32> = seq.table(layer).blocks().to_vec();
            // Allocate new dense blocks and copy retained rows.
            let mut new_blocks = Vec::with_capacity(n_new_blocks);
            for _ in 0..n_new_blocks {
                let id = match self.pool.alloc_block() {
                    Some(id) => id,
                    None => {
                        for &nb in &new_blocks {
                            self.pool.dec_ref(nb);
                        }
                        return reclaimed;
                    }
                };
                self.pool.inc_ref(id);
                new_blocks.push(id);
            }
            for (new_i, &old_i) in src_dense.iter().enumerate() {
                let old_logical = old_i / bs;
                let old_slot = old_i % bs;
                let Some(&old_phys) = old_blocks.get(old_logical) else {
                    continue;
                };
                let new_logical = new_i / bs;
                let new_slot = new_i % bs;
                let new_phys = new_blocks[new_logical];
                let k_src: Vec<_> = self.pool.k_row(old_phys, old_slot).to_vec();
                let v_src: Vec<_> = self.pool.v_row(old_phys, old_slot).to_vec();
                self.pool
                    .k_row_mut(new_phys, new_slot)
                    .copy_from_slice(&k_src);
                self.pool
                    .v_row_mut(new_phys, new_slot)
                    .copy_from_slice(&v_src);
            }
            for &phys in &old_blocks {
                let before = self.pool.free_count();
                self.pool.dec_ref(phys);
                if self.pool.free_count() > before {
                    reclaimed += 1;
                }
            }
            seq.table_mut(layer).set_blocks(new_blocks);
        }

        seq.original_positions = orig;
        seq.len_tokens = n_new;
        reclaimed
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
