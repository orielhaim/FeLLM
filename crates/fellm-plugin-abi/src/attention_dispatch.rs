//! Process-wide attention dispatch set at prepare/step time.
//!
//! The executor patches [`crate::OpAttrs::custom_op_id`] from this context so
//! CUDA/host launchers call the prepared path (FA2 decode/prefill, FA3, …)
//! without string lookup.

use std::sync::Mutex;

/// Prepared attention kernel path (stable numeric id for hot path).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKernelPath {
    /// Legacy / auto: warp decode for q_len=1, FA2 prefill otherwise.
    #[default]
    Auto = 0,
    /// FA2-style decode (Ampere/Ada-class work partitioning).
    Fa2Decode = 1,
    /// FA2-style prefill (query tiles × KV tiles in SRAM).
    Fa2Prefill = 2,
    /// FA3-style decode (Hopper-class async / warp-specialized schedule).
    Fa3Decode = 3,
    /// FA3-style prefill.
    Fa3Prefill = 4,
    /// Host FA2-style reference path (CPU).
    HostFa2 = 5,
}

impl AttentionKernelPath {
    /// From prepared plan handle / kernel_variant style bits.
    #[must_use]
    pub fn from_prepared(plan_handle: u64, kernel_variant: u64, is_prefill: bool) -> Self {
        let fa3 = kernel_variant & 1 == 1 || (plan_handle >> 16) == 0xFA3;
        match (fa3, is_prefill) {
            (true, true) => Self::Fa3Prefill,
            (true, false) => Self::Fa3Decode,
            (false, true) => Self::Fa2Prefill,
            (false, false) => Self::Fa2Decode,
        }
    }

    /// Encode into `OpAttrs.custom_op_id`.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Decode from `custom_op_id`.
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Fa2Decode,
            2 => Self::Fa2Prefill,
            3 => Self::Fa3Decode,
            4 => Self::Fa3Prefill,
            5 => Self::HostFa2,
            _ => Self::Auto,
        }
    }
}

/// Dispatch parameters installed for one step / request.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionDispatch {
    /// Decode path.
    pub decode: AttentionKernelPath,
    /// Prefill path.
    pub prefill: AttentionKernelPath,
    /// Q tile (Br) for FA2 prefill.
    pub q_tile: u32,
    /// KV tile (Bc) for FA2.
    pub kv_tile: u32,
    /// FA3 pipeline stages.
    pub pipeline_stages: u32,
    /// Prepared provider id (diagnostic / graph replay).
    pub provider_id: u64,
}

static ATTENTION_DISPATCH: Mutex<AttentionDispatch> = Mutex::new(AttentionDispatch {
    decode: AttentionKernelPath::Fa2Decode,
    prefill: AttentionKernelPath::Fa2Prefill,
    q_tile: 16,
    kv_tile: 16,
    pipeline_stages: 2,
    provider_id: 0,
});

/// Install dispatch for subsequent attention launches.
pub fn set_attention_dispatch(d: AttentionDispatch) {
    *ATTENTION_DISPATCH.lock().expect("attention dispatch lock") = d;
}

/// Read current dispatch (copy).
#[must_use]
pub fn attention_dispatch() -> AttentionDispatch {
    *ATTENTION_DISPATCH.lock().expect("attention dispatch lock")
}

/// Resolve kernel path for a launch given attrs.
#[must_use]
pub fn resolve_path(query_len: u32, custom_op_id: u32) -> AttentionKernelPath {
    let explicit = AttentionKernelPath::from_u32(custom_op_id);
    if explicit != AttentionKernelPath::Auto {
        return explicit;
    }
    let d = attention_dispatch();
    if query_len > 1 { d.prefill } else { d.decode }
}

// ---- Pre-RoPE key store (host-visible, device kernels may also write) ----

/// Per-layer pre-RoPE key rows for sequence-state policies (TriAttention etc.).
#[derive(Debug, Clone, Default)]
pub struct PreRopeKeyStore {
    /// Flattened `[layer][pos][kv_head * head_dim]` f32.
    pub layers: Vec<Vec<f32>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl PreRopeKeyStore {
    /// Allocate storage for `n_layers` × `max_seq` rows.
    #[must_use]
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let stride = n_kv_heads.max(1) * head_dim.max(1);
        let layers = (0..n_layers)
            .map(|_| vec![0.0f32; max_seq.max(1) * stride])
            .collect();
        Self {
            layers,
            n_kv_heads: n_kv_heads.max(1),
            head_dim: head_dim.max(1),
            max_seq: max_seq.max(1),
        }
    }

    /// Stride in elements per token.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Write one pre-RoPE K row at `(layer, pos)`.
    pub fn write_row(&mut self, layer: usize, pos: usize, row: &[f32]) {
        if layer >= self.layers.len() || pos >= self.max_seq {
            return;
        }
        let s = self.stride();
        let base = pos * s;
        let dst = &mut self.layers[layer];
        if base + s > dst.len() {
            dst.resize(base + s, 0.0);
        }
        let n = row.len().min(s);
        dst[base..base + n].copy_from_slice(&row[..n]);
    }

    /// Flatten keys for absolute positions `start..end` (inclusive-exclusive).
    #[must_use]
    pub fn slice_positions(&self, layer: usize, start: u32, end: u32) -> Vec<f32> {
        let positions: Vec<u32> = (start..end).collect();
        self.gather_positions(layer, &positions)
    }

    /// Pack pre-RoPE rows for an explicit absolute-position list (live index order).
    #[must_use]
    pub fn gather_positions(&self, layer: usize, positions: &[u32]) -> Vec<f32> {
        let s = self.stride();
        let Some(data) = self.layers.get(layer) else {
            return vec![0.0; positions.len() * s];
        };
        let mut out = Vec::with_capacity(positions.len() * s);
        for &pos in positions {
            let base = pos as usize * s;
            if base + s <= data.len() {
                out.extend_from_slice(&data[base..base + s]);
            } else {
                out.extend(std::iter::repeat_n(0.0, s));
            }
        }
        out
    }

    /// Drop all absolute rows not in `keep` (prune ghost pre-RoPE history).
    pub fn prune_to_positions(&mut self, layer: usize, keep: &[u32]) {
        if layer >= self.layers.len() {
            return;
        }
        let s = self.stride();
        let data = &self.layers[layer];
        let mut kept = std::collections::BTreeSet::new();
        for &p in keep {
            kept.insert(p);
        }
        // Zero discarded absolute slots so re-select cannot score ghosts.
        let max_pos = (data.len() / s).min(self.max_seq);
        let mut new_data = data.clone();
        for pos in 0..max_pos {
            if !kept.contains(&(pos as u32)) {
                let base = pos * s;
                if base + s <= new_data.len() {
                    new_data[base..base + s].fill(0.0);
                }
            }
        }
        self.layers[layer] = new_data;
    }
}

static PRE_ROPE: Mutex<Option<PreRopeKeyStore>> = Mutex::new(None);

/// Install pre-RoPE store for the request.
pub fn set_pre_rope_store(store: Option<PreRopeKeyStore>) {
    *PRE_ROPE.lock().expect("pre_rope lock") = store;
}

/// Write a pre-RoPE key row into the active store.
pub fn pre_rope_write(layer: usize, pos: usize, row: &[f32]) {
    if let Some(s) = PRE_ROPE.lock().expect("pre_rope lock").as_mut() {
        s.write_row(layer, pos, row);
    }
}

/// Snapshot private-suffix keys for a layer.
#[must_use]
pub fn pre_rope_slice(layer: usize, start: u32, end: u32) -> Option<Vec<f32>> {
    PRE_ROPE
        .lock()
        .expect("pre_rope lock")
        .as_ref()
        .map(|s| s.slice_positions(layer, start, end))
}

/// Pack pre-RoPE rows for a live absolute-position list.
#[must_use]
pub fn pre_rope_gather(layer: usize, positions: &[u32]) -> Option<Vec<f32>> {
    PRE_ROPE
        .lock()
        .expect("pre_rope lock")
        .as_ref()
        .map(|s| s.gather_positions(layer, positions))
}

/// Prune pre-RoPE store to live absolute positions (drop evicted ghosts).
pub fn pre_rope_prune(layer: usize, keep: &[u32]) {
    if let Some(s) = PRE_ROPE.lock().expect("pre_rope lock").as_mut() {
        s.prune_to_positions(layer, keep);
    }
}

/// Take ownership of the store (for tests / teardown).
pub fn take_pre_rope_store() -> Option<PreRopeKeyStore> {
    PRE_ROPE.lock().expect("pre_rope lock").take()
}
