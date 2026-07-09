//! Process-wide paged KV context for kernels during a forward step.
//!
//! Uses a mutex (not thread-local) so parallel attention workers can read the
//! same arena/block-table installed by the runtime on the executor thread.

use std::sync::Mutex;

/// Active paged KV view for the current step (set by the runtime before `exec.run`).
pub struct PagedKvContext {
    /// Arena bytes (K|V blocks) — uniquely borrowed for the step.
    pub arena: *mut u8,
    /// Arena length in bytes.
    pub arena_len: usize,
    /// Flattened block table: `layer * n_logical_blocks + logical_block → physical_id`.
    pub block_table: Vec<u32>,
    /// Logical blocks per layer (same for all layers).
    pub n_logical_blocks: usize,
    /// Attention layer count.
    pub n_layers: usize,
    /// f32 elements per token row (`n_kv_heads * head_dim`).
    pub tokens_stride: usize,
    /// Bytes per physical block.
    pub block_bytes: usize,
    /// Block size in tokens.
    pub block_size: usize,
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

    /// K row for logical token `t` at `layer` (full tokens_stride).
    ///
    /// # Safety
    /// Arena must outlive the returned slice; caller must not mutate concurrently.
    pub unsafe fn k_row(&self, layer: usize, t: usize) -> &[f32] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical(layer, logical);
        let off = self.block_offset(phys);
        let base = off + slot * self.tokens_stride * 4;
        debug_assert!(base + self.tokens_stride * 4 <= self.arena_len);
        // SAFETY: offsets validated by pool construction and ensure_writable.
        unsafe {
            let ptr = self.arena.add(base) as *const f32;
            std::slice::from_raw_parts(ptr, self.tokens_stride)
        }
    }

    /// V row for logical token `t` at `layer`.
    ///
    /// # Safety
    /// Same as [`Self::k_row`].
    pub unsafe fn v_row(&self, layer: usize, t: usize) -> &[f32] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical(layer, logical);
        let off = self.block_offset(phys);
        let v_base = self.block_size * self.tokens_stride * 4;
        let base = off + v_base + slot * self.tokens_stride * 4;
        debug_assert!(base + self.tokens_stride * 4 <= self.arena_len);
        unsafe {
            let ptr = self.arena.add(base) as *const f32;
            std::slice::from_raw_parts(ptr, self.tokens_stride)
        }
    }

    /// Mutable K or V row (`is_v`).
    ///
    /// # Safety
    /// Arena must be uniquely borrowed for mutation.
    pub unsafe fn row_mut(&self, layer: usize, t: usize, is_v: bool) -> &mut [f32] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical(layer, logical);
        let off = self.block_offset(phys);
        let v_base = if is_v {
            self.block_size * self.tokens_stride * 4
        } else {
            0
        };
        let base = off + v_base + slot * self.tokens_stride * 4;
        debug_assert!(base + self.tokens_stride * 4 <= self.arena_len);
        unsafe {
            let ptr = self.arena.add(base) as *mut f32;
            std::slice::from_raw_parts_mut(ptr, self.tokens_stride)
        }
    }
}
