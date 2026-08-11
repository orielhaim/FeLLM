use core::ffi::c_int;
use half::f16;
use std::sync::{Arc, Mutex};

/// Bytes per KV element in the paged arena (`f16`).
pub const PAGED_KV_ELEM_BYTES: usize = 2;

/// POD snapshot of the host paged KV arena for the plugin C ABI.
///
/// Pointers remain valid only while the host's [`set_paged_context`] install
/// is live (one inference step). Plugins must not cache across launches.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PagedKvSnapshot {
    /// Host arena bytes (K|V blocks) — always set when context is installed.
    pub arena: *mut u8,
    /// Host arena length in bytes.
    pub arena_len: usize,
    /// Flattened block table pointer (`layer * n_logical + logical → phys`).
    pub block_table: *const u32,
    /// Length of `block_table`.
    pub n_block_table: usize,
    /// Logical blocks per layer.
    pub n_logical_blocks: usize,
    /// Attention layer count.
    pub n_layers: usize,
    /// Elements per token row (`n_kv_heads * head_dim`).
    pub tokens_stride: usize,
    /// Bytes per physical block.
    pub block_bytes: usize,
    /// Block size in tokens.
    pub block_size: usize,
    /// Bytes per element (2 for f16).
    pub elem_bytes: usize,
    /// Device (VRAM) arena base, or null if host-only.
    pub device_arena: *mut u8,
    /// Device arena length in bytes (`0` if host-only).
    pub device_arena_len: usize,
    /// Persistent device page-table address, or null for host-only execution.
    pub device_block_table: *mut u32,
    /// Active entries at [`Self::device_block_table`].
    pub n_device_block_table: usize,
    /// Fixed device-table entries reserved per layer.
    pub device_logical_stride: usize,
    /// Number of independently-addressed sequence rows in this physical batch.
    pub batch_size: usize,
    /// Dense KV write position for each batch row.
    pub row_positions: *const u32,
    /// Attention-visible KV length for each batch row.
    pub row_lengths: *const u32,
    /// Absolute RoPE position for each batch row.
    pub row_rope_positions: *const u32,
}

/// Host callback: fill `out` from the process-wide paged context.
///
/// Returns `0` if a context is installed, `1` if none, negative on error.
pub type HostSnapshotPagedFn = unsafe extern "C" fn(out: *mut PagedKvSnapshot) -> c_int;

/// C ABI entry the host passes to plugins via [`crate::c_abi::HostContext`].
///
/// Lives in the host binary so plugins share the same `PAGED_CTX` static.
///
/// # Safety
///
/// `out` must be non-null and point to writable storage for one
/// [`PagedKvSnapshot`]. Any pointers written into the snapshot are borrowed
/// from the currently installed context and must not be retained after that
/// context is removed.
pub unsafe extern "C" fn host_snapshot_paged_kv(out: *mut PagedKvSnapshot) -> c_int {
    if out.is_null() {
        return -1;
    }
    let guard = match PAGED_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -2,
    };
    let Some(ctx) = guard.as_ref() else {
        return 1;
    };
    // SAFETY: caller provides a valid `PagedKvSnapshot` slot.
    unsafe {
        *out = PagedKvSnapshot {
            arena: ctx.arena,
            arena_len: ctx.arena_len,
            block_table: ctx.block_table.as_ptr(),
            n_block_table: ctx.block_table.len(),
            n_logical_blocks: ctx.n_logical_blocks,
            n_layers: ctx.n_layers,
            tokens_stride: ctx.tokens_stride,
            block_bytes: ctx.block_bytes,
            block_size: ctx.block_size,
            elem_bytes: ctx.elem_bytes,
            device_arena: ctx.device_arena,
            device_arena_len: ctx.device_arena_len,
            device_block_table: ctx.device_block_table,
            n_device_block_table: ctx.n_device_block_table,
            device_logical_stride: ctx.device_logical_stride,
            batch_size: ctx.batch_size(),
            row_positions: ctx.row_positions.as_ptr(),
            row_lengths: ctx.row_lengths.as_ptr(),
            row_rope_positions: ctx.row_rope_positions.as_ptr(),
        };
    }
    0
}

impl PagedKvSnapshot {
    /// Physical block id for `(layer, logical_block)`.
    #[must_use]
    pub fn physical(&self, layer: usize, logical: usize) -> u32 {
        self.physical_for(0, layer, logical)
    }

    /// Physical block id for one batch row's `(layer, logical_block)`.
    #[must_use]
    pub fn physical_for(&self, row: usize, layer: usize, logical: usize) -> u32 {
        let idx = (row * self.n_layers + layer) * self.n_logical_blocks + logical;
        debug_assert!(idx < self.n_block_table);
        // SAFETY: host guarantees table covers all logical blocks for the step.
        unsafe { *self.block_table.add(idx) }
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
        unsafe { self.k_row_for(0, layer, t) }
    }

    /// K row for a specific physical-batch row.
    pub unsafe fn k_row_for(&self, row: usize, layer: usize, t: usize) -> &[f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical_for(row, layer, logical);
        let off = (phys as usize) * self.block_bytes;
        let row_bytes = self.row_byte_len();
        let base = off + slot * row_bytes;
        debug_assert!(base + row_bytes <= self.arena_len);
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
        unsafe { self.v_row_for(0, layer, t) }
    }

    /// V row for a specific physical-batch row.
    pub unsafe fn v_row_for(&self, row: usize, layer: usize, t: usize) -> &[f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical_for(row, layer, logical);
        let off = (phys as usize) * self.block_bytes;
        let row_bytes = self.row_byte_len();
        let v_base = self.block_size * row_bytes;
        let base = off + v_base + slot * row_bytes;
        debug_assert!(base + row_bytes <= self.arena_len);
        unsafe {
            let ptr = self.arena.add(base) as *const f16;
            std::slice::from_raw_parts(ptr, self.tokens_stride)
        }
    }
}

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
    /// Optional VRAM mirror of `arena` (null = host-only).
    pub device_arena: *mut u8,
    /// Device arena byte length.
    pub device_arena_len: usize,
    /// Persistent device page-table address.
    pub device_block_table: *mut u32,
    /// Active persistent page-table entries.
    pub n_device_block_table: usize,
    /// Fixed device-table entries reserved per layer.
    pub device_logical_stride: usize,
    /// Dense KV write position for every row in the physical batch.
    pub row_positions: Arc<[u32]>,
    /// Attention-visible KV length for every row in the physical batch.
    pub row_lengths: Arc<[u32]>,
    /// Absolute RoPE position for every row in the physical batch.
    pub row_rope_positions: Arc<[u32]>,
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

/// Run `f` with an exclusive reference to the active paged context, if any.
pub fn with_paged_context_mut<R>(f: impl FnOnce(Option<&mut PagedKvContext>) -> R) -> R {
    let mut guard = PAGED_CTX.lock().expect("paged ctx lock");
    f(guard.as_mut())
}

impl PagedKvContext {
    /// Number of independently-addressed sequence rows.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.row_positions.len().max(1)
    }

    /// Physical block id for `(layer, logical_block)`.
    #[must_use]
    pub fn physical(&self, layer: usize, logical: usize) -> u32 {
        self.physical_for(0, layer, logical)
    }

    /// Physical block id for a row in a multi-sequence batch.
    #[must_use]
    pub fn physical_for(&self, row: usize, layer: usize, logical: usize) -> u32 {
        self.block_table[(row * self.n_layers + layer) * self.n_logical_blocks + logical]
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
        unsafe { self.k_row_for(0, layer, t) }
    }

    /// K row for a specific physical-batch row.
    pub unsafe fn k_row_for(&self, row: usize, layer: usize, t: usize) -> &[f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical_for(row, layer, logical);
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
        unsafe { self.v_row_for(0, layer, t) }
    }

    /// V row for a specific physical-batch row.
    pub unsafe fn v_row_for(&self, row: usize, layer: usize, t: usize) -> &[f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical_for(row, layer, logical);
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
    pub unsafe fn row_mut(&mut self, layer: usize, t: usize, is_v: bool) -> &mut [f16] {
        unsafe { self.row_mut_for(0, layer, t, is_v) }
    }

    /// Mutable K or V row for a specific physical-batch row.
    pub unsafe fn row_mut_for(
        &mut self,
        row: usize,
        layer: usize,
        t: usize,
        is_v: bool,
    ) -> &mut [f16] {
        let logical = t / self.block_size;
        let slot = t % self.block_size;
        let phys = self.physical_for(row, layer, logical);
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
