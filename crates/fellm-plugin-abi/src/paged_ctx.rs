use half::f16;
use std::sync::{Arc, Mutex};

/// Bytes per KV element in the paged arena (`f16`).
pub const PAGED_KV_ELEM_BYTES: usize = 2;

#[derive(Clone)]
pub struct PagedKvContext {
    /// Arena bytes (K|V blocks) — uniquely borrowed for the step.
    pub arena: *mut u8,
    /// Arena length in bytes.
    pub arena_len: usize,
    /// Flattened block table: `layer * n_logical_blocks + logical_block → physical_id`.
    pub block_table: Arc<[u32]>,
    /// Logical blocks per layer (same for all layers).
    pub n_logical_blocks: usize,
    /// Attention layer count.
    pub n_layers: usize,
    /// Elements per token row (`n_kv_heads * head_dim`).
    pub tokens_stride: usize,
    /// Bytes per physical block (includes 64-byte padding).
    pub block_bytes: usize,
    /// Block size in tokens.
    pub block_size: usize,
    /// Bytes per element (always 2 for f16).
    pub elem_bytes: usize,
}

// SAFETY: The runtime guarantees the arena lives for the duration of the step
// and only one inference step runs at a time (serial Engine / scheduler).
unsafe impl Send for PagedKvContext {}
unsafe impl Sync for PagedKvContext {}

static PAGED_CTX: Mutex<Option<PagedKvContext>> = Mutex::new(None);

/// Install paged context for the process (call before graph run).
pub fn set_paged_context(ctx: Option<PagedKvContext>) {
    *PAGED_CTX.lock().expect("paged ctx lock") = ctx;
}

/// True if a paged context is currently installed.
#[must_use]
pub fn has_paged_context() -> bool {
    PAGED_CTX.lock().expect("paged ctx lock").is_some()
}

/// Clone the active paged context out of the mutex (arena pointer + block table).
///
/// Safe for read-only parallel use during attention: the arena is not mutated
/// while attention runs.
#[must_use]
pub fn snapshot_paged_context() -> Option<PagedKvContext> {
    PAGED_CTX.lock().expect("paged ctx lock").clone()
}

/// Run `f` with a shared reference to the active paged context, if any.
pub fn with_paged_context<R>(f: impl FnOnce(Option<&PagedKvContext>) -> R) -> R {
    let guard = PAGED_CTX.lock().expect("paged ctx lock");
    f(guard.as_ref())
}

impl PagedKvContext {
    /// Physical block id for `(layer, logical_block)`.
    #[must_use]
    pub fn physical(&self, layer: usize, logical: usize) -> u32 {
        self.block_table[layer * self.n_logical_blocks + logical]
    }

    /// Byte offset of a physical block in the arena.
    #[must_use]
    pub fn block_offset(&self, physical: u32) -> usize {
        (physical as usize) * self.block_bytes
    }

    #[inline]
    fn row_byte_len(&self) -> usize {
        self.tokens_stride * self.elem_bytes
    }

    /// K row for logical token `t` at `layer` (full `tokens_stride` as `f16`).
    ///
    /// # Safety
    /// Arena must outlive the returned slice; caller must not mutate concurrently.
    pub unsafe fn k_row(&self, layer: usize, t: usize) -> &[f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical(layer, logical);
        let off = self.block_offset(phys);
        let row_bytes = self.row_byte_len();
        let base = off + slot * row_bytes;
        debug_assert!(base + row_bytes <= self.arena_len);
        // SAFETY: offsets validated by pool construction and ensure_writable.
        unsafe {
            let ptr = self.arena.add(base) as *const f16;
            std::slice::from_raw_parts(ptr, self.tokens_stride)
        }
    }

    /// V row for logical token `t` at `layer`.
    ///
    /// # Safety
    /// Same as [`Self::k_row`].
    pub unsafe fn v_row(&self, layer: usize, t: usize) -> &[f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical(layer, logical);
        let off = self.block_offset(phys);
        let row_bytes = self.row_byte_len();
        let v_base = self.block_size * row_bytes;
        let base = off + v_base + slot * row_bytes;
        debug_assert!(base + row_bytes <= self.arena_len);
        unsafe {
            let ptr = self.arena.add(base) as *const f16;
            std::slice::from_raw_parts(ptr, self.tokens_stride)
        }
    }

    /// Mutable K or V row (`is_v`) as `f16`.
    ///
    /// # Safety
    /// Arena must be uniquely borrowed for mutation.
    pub unsafe fn row_mut(&self, layer: usize, t: usize, is_v: bool) -> &mut [f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical(layer, logical);
        let off = self.block_offset(phys);
        let row_bytes = self.row_byte_len();
        let v_base = if is_v { self.block_size * row_bytes } else { 0 };
        let base = off + v_base + slot * row_bytes;
        debug_assert!(base + row_bytes <= self.arena_len);
        unsafe {
            let ptr = self.arena.add(base) as *mut f16;
            std::slice::from_raw_parts_mut(ptr, self.tokens_stride)
        }
    }
}
