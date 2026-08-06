//! Per-sequence logical→physical block tables.

use super::pool::BLOCK_SIZE;

/// Logical→physical map for one attention layer.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Physical block ids, indexed by logical block (`token / BLOCK_SIZE`).
    blocks: Vec<u32>,
}

impl BlockTable {
    /// Empty table.
    #[must_use]
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Number of logical blocks mapped.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// All physical ids.
    #[must_use]
    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// Physical id for a logical block index.
    #[must_use]
    pub fn block(&self, logical: usize) -> u32 {
        self.blocks[logical]
    }

    /// Mutable physical id slot.
    pub fn block_mut(&mut self, logical: usize) -> &mut u32 {
        &mut self.blocks[logical]
    }

    /// Append a newly allocated physical block.
    pub fn push_block(&mut self, physical: u32) {
        self.blocks.push(physical);
    }

    /// Replace the entire mapping (e.g. after prefix attach).
    pub fn set_blocks(&mut self, blocks: Vec<u32>) {
        self.blocks = blocks;
    }

    /// Clear mappings (does not free physical blocks).
    pub fn clear(&mut self) {
        self.blocks.clear();
    }

    /// Map logical token index → (`physical_id`, `slot_in_block`).
    #[must_use]
    pub fn locate(&self, token_pos: usize) -> (u32, usize) {
        let logical = token_pos / BLOCK_SIZE;
        let slot = token_pos % BLOCK_SIZE;
        (self.blocks[logical], slot)
    }
}

/// Per-sequence paged KV state (all attention layers).
pub struct SequenceCache {
    tables: Vec<BlockTable>,
    /// Filled token count (logical).
    pub len_tokens: usize,
    /// Maximum sequence length.
    pub max_seq: usize,
    /// Optional swap slot ids per physical block currently swapped out.
    pub swapped: bool,
}

impl SequenceCache {
    /// Create empty tables for `n_layers`.
    #[must_use]
    pub fn new(n_layers: usize, max_seq: usize) -> Self {
        Self {
            tables: (0..n_layers).map(|_| BlockTable::new()).collect(),
            len_tokens: 0,
            max_seq,
            swapped: false,
        }
    }

    /// Attention layer count.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.tables.len()
    }

    /// Immutable table for a layer.
    #[must_use]
    pub fn table(&self, layer: usize) -> &BlockTable {
        &self.tables[layer]
    }

    /// Mutable table for a layer.
    pub fn table_mut(&mut self, layer: usize) -> &mut BlockTable {
        &mut self.tables[layer]
    }

    /// Clear all tables (caller must have released refs).
    pub fn clear_tables(&mut self) {
        for t in &mut self.tables {
            t.clear();
        }
        self.len_tokens = 0;
        self.swapped = false;
    }

    /// Reset logical length without freeing blocks (caller releases).
    pub fn reset_len(&mut self) {
        self.len_tokens = 0;
    }

    /// Flatten block tables into a dense `u32` vector for kernel binding:
    /// layout `[layer0_b0, layer0_b1, ..., layer1_b0, ...]`.
    ///
    /// All layers must have the same number of logical blocks.
    pub fn flatten_block_tables(&self) -> Vec<u32> {
        let n = self.tables.first().map_or(0, BlockTable::num_blocks);
        let mut out = Vec::with_capacity(n * self.tables.len());
        for t in &self.tables {
            debug_assert_eq!(t.num_blocks(), n);
            out.extend_from_slice(t.blocks());
        }
        out
    }
}
