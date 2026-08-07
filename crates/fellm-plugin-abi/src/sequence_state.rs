//! Sequence / KV state views and retention policy contracts.
//!
//! Separates logical token identity from physical storage so compression
//! policies can compact storage while preserving original positions.

use crate::capability::{FeatureSet, PluginConfig, PreparedProviderId, ProviderDescriptor};
use fellm_core::error::Result;
use std::collections::BTreeMap;

/// Ownership / mutability class for a region of sequence state.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateOwnership {
    /// Immutable shared prefix (radix / prefix-cache); must not be compressed
    /// in place — fork/CoW first if a request needs to mutate.
    SharedImmutable = 1,
    /// Request-private mutable state (compressible suffix, decode writes).
    RequestPrivate = 2,
    /// Copy-on-write private view of previously shared state.
    ForkedPrivate = 3,
}

/// One retained key/value entry visible to attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedEntry {
    /// Original logical / absolute sequence position.
    pub logical_pos: u32,
    /// Physical block id in the pool (or dense index when contiguous).
    pub physical_block: u32,
    /// Slot within the physical block (token index inside block).
    pub physical_slot: u16,
    /// KV head this entry belongs to (`u16::MAX` = all heads share).
    pub kv_head: u16,
    /// Layer ordinal (`u16::MAX` = shared across layers — uncommon).
    pub layer: u16,
}

/// Prepared attention view over (possibly compressed) sequence state.
///
/// Attention kernels consume this view rather than assuming dense `0..seq_len`.
#[derive(Debug, Clone, Default)]
pub struct AttentionKvView {
    /// Layer this view describes.
    pub layer: u32,
    /// Retained entries in attention order (usually chronological by logical_pos).
    pub entries: Vec<RetainedEntry>,
    /// Dense block table fallback when entries is empty and layout is dense-paged.
    /// Layout: `logical_block → physical_block` for this layer.
    pub dense_block_table: Vec<u32>,
    /// Tokens represented when using dense_block_table (full sequence length).
    pub dense_seq_len: u32,
    /// Ownership of the underlying state.
    pub ownership: Option<StateOwnership>,
}

impl AttentionKvView {
    /// Number of KV positions visible to attention.
    #[must_use]
    pub fn retained_len(&self) -> usize {
        if !self.entries.is_empty() {
            self.entries.len()
        } else {
            self.dense_seq_len as usize
        }
    }

    /// Logical positions of retained entries (empty when dense-only).
    #[must_use]
    pub fn logical_positions(&self) -> Vec<u32> {
        self.entries.iter().map(|e| e.logical_pos).collect()
    }

    /// True when the view is a non-dense retained subset.
    #[must_use]
    pub fn is_indirect(&self) -> bool {
        !self.entries.is_empty()
    }
}

/// Who owns physical layout after a retention plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutOwner {
    /// Host densified storage (`compact_sequence_to_positions`); policy must not
    /// re-derive phys from absolute positions against the new table.
    HostDensified,
    /// Policy builds views only (no host densify yet).
    #[default]
    PolicyView,
}

/// Per-request sequence attention state: physical storage + views + policy.
#[derive(Debug, Clone, Default)]
pub struct SequenceAttentionState {
    /// Request id (diagnostic).
    pub request_id: u64,
    /// Absolute generation length (RoPE cursor upper bound).
    pub logical_len: u32,
    /// Length of the immutable shared prefix portion.
    pub shared_prefix_len: u32,
    /// **Live** absolute positions that still exist in physical KV storage.
    ///
    /// Empty means uncompressed identity `0..dense_len` where `dense_len` is
    /// the first layer's `dense_seq_len` (or `logical_len` if unset).
    /// After compression this is exactly `SequenceCache.original_positions`.
    /// Policies **must** score only this set (plus shared prefix), never a
    /// ghost range of already-evicted absolute ids.
    pub live_positions: Vec<u32>,
    /// Dense storage length (attention KV length).
    pub dense_len: u32,
    /// Layout ownership for the next `apply_plan`.
    pub layout_owner: LayoutOwner,
    /// Per-layer attention views.
    pub layer_views: Vec<AttentionKvView>,
    /// Active prepared policy id (`NONE` = full retention).
    pub policy: PreparedProviderId,
    /// Free-form policy metadata (budget, last compact step, etc.).
    pub meta: BTreeMap<String, String>,
}

impl SequenceAttentionState {
    /// Construct empty state for `n_layers`.
    #[must_use]
    pub fn new(n_layers: usize) -> Self {
        Self {
            request_id: 0,
            logical_len: 0,
            shared_prefix_len: 0,
            live_positions: Vec::new(),
            dense_len: 0,
            layout_owner: LayoutOwner::PolicyView,
            layer_views: (0..n_layers)
                .map(|i| AttentionKvView {
                    layer: i as u32,
                    ownership: Some(StateOwnership::RequestPrivate),
                    ..AttentionKvView::default()
                })
                .collect(),
            policy: PreparedProviderId::NONE,
            meta: BTreeMap::new(),
        }
    }

    /// Absolute positions that still exist physically (live index contract).
    ///
    /// Sorted unique. Shared prefix ids are included when present in storage.
    #[must_use]
    pub fn live_retained_positions(&self) -> Vec<u32> {
        if !self.live_positions.is_empty() {
            let mut v = self.live_positions.clone();
            v.sort_unstable();
            v.dedup();
            return v;
        }
        // Uncompressed identity over dense storage length.
        let n = if self.dense_len > 0 {
            self.dense_len
        } else {
            self.logical_len
        };
        (0..n).collect()
    }

    /// Private (compressible) subset of [`Self::live_retained_positions`].
    #[must_use]
    pub fn live_private_positions(&self) -> Vec<u32> {
        let prefix = self.shared_prefix_len;
        self.live_retained_positions()
            .into_iter()
            .filter(|&p| p >= prefix)
            .collect()
    }

    /// View for a layer.
    #[must_use]
    pub fn view(&self, layer: usize) -> Option<&AttentionKvView> {
        self.layer_views.get(layer)
    }

    /// Mutable view for a layer.
    pub fn view_mut(&mut self, layer: usize) -> Option<&mut AttentionKvView> {
        self.layer_views.get_mut(layer)
    }
}

/// Statistics collected for a compression/retention decision.
#[derive(Debug, Clone, Default)]
pub struct RetentionStats {
    /// Number of logical positions considered.
    pub candidate_count: u32,
    /// Number of positions retained after selection.
    pub retained_count: u32,
    /// Physical blocks freed by the last compaction.
    pub blocks_reclaimed: u32,
    /// Tokens scored (may exceed retained).
    pub scored_count: u32,
}

/// Input to a retention policy evaluation.
#[derive(Debug, Clone)]
pub struct RetentionContext<'a> {
    /// Current sequence attention state.
    pub state: &'a SequenceAttentionState,
    /// Layer under consideration (`None` = all layers).
    pub layer: Option<u32>,
    /// Absolute position of the latest token (decode step).
    pub current_pos: u32,
    /// Tokens generated since last compression window.
    pub tokens_since_window: u32,
    /// Pre-RoPE key vectors for scoring: layout policy-defined.
    /// Flat `f32` storage; policy interprets with `head_dim`, `n_kv_heads`.
    pub pre_rope_keys: Option<&'a [f32]>,
    /// Head dim for pre-RoPE keys.
    pub head_dim: u32,
    /// Number of KV heads.
    pub n_kv_heads: u32,
    /// Number of query heads (for GQA aggregation).
    pub n_heads: u32,
    /// Provider-scoped config.
    pub config: &'a PluginConfig,
}

/// Plan produced by a policy: which logical positions to keep, per layer/head.
#[derive(Debug, Clone, Default)]
pub struct RetentionPlan {
    /// Retained logical (absolute) positions — **must** be a subset of the
    /// live index that existed at plan time.
    pub retain_positions: Vec<u32>,
    /// Optional per-KV-head retention (when policy is head-specific).
    pub per_kv_head: Vec<Vec<u32>>,
    /// Whether physical compaction/reclaim should run.
    pub compact: bool,
    /// When true, host already densified storage; `apply_plan` is bookkeeping only.
    pub host_densified: bool,
}

/// Sequence-state / KV retention policy provider.
pub trait SequenceStatePolicy: Send + Sync {
    /// Static descriptor.
    fn descriptor(&self) -> &ProviderDescriptor;

    /// Validate configuration.
    fn validate_config(&self, config: &PluginConfig) -> Result<()>;

    /// Features this policy requires from attention / runtime.
    fn required_features(&self) -> &FeatureSet {
        &self.descriptor().requires
    }

    /// Whether a compression window should fire at this step.
    fn should_compress(&self, ctx: &RetentionContext<'_>) -> bool;

    /// Score and select retained entries (may be pure host math).
    fn plan_retention(&self, ctx: &RetentionContext<'_>) -> Result<RetentionPlan>;

    /// Apply plan: update views. Physical reclaim is performed by the host
    /// using [`RetentionPlan`] + pool APIs; the policy must not free memory
    /// it does not own.
    fn apply_plan(
        &self,
        state: &mut SequenceAttentionState,
        plan: &RetentionPlan,
    ) -> Result<RetentionStats>;
}

/// Built-in full-retention policy (no compression).
#[derive(Debug)]
pub struct FullRetentionPolicy {
    desc: ProviderDescriptor,
}

impl FullRetentionPolicy {
    /// Construct the default full-retention policy.
    #[must_use]
    pub fn new() -> Self {
        use crate::capability::{CapabilityKind, FeatureId, ProviderVersion};
        let provides = FeatureSet::from_ids([
            FeatureId::KV_LOGICAL_POSITIONS,
            FeatureId::KV_PREFIX_PRIVATE_SPLIT,
        ]);
        let desc = ProviderDescriptor::new(
            "kv.full",
            CapabilityKind::SequenceStatePolicy,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "Retain full sequence; no compression",
        )
        .with_provides(provides)
        .with_priority(0)
        .with_meta("builtin", "true");
        Self { desc }
    }
}

impl Default for FullRetentionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceStatePolicy for FullRetentionPolicy {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.desc
    }

    fn validate_config(&self, _config: &PluginConfig) -> Result<()> {
        Ok(())
    }

    fn should_compress(&self, _ctx: &RetentionContext<'_>) -> bool {
        false
    }

    fn plan_retention(&self, ctx: &RetentionContext<'_>) -> Result<RetentionPlan> {
        Ok(RetentionPlan {
            retain_positions: ctx.state.live_retained_positions(),
            per_kv_head: Vec::new(),
            compact: false,
            host_densified: false,
        })
    }

    fn apply_plan(
        &self,
        state: &mut SequenceAttentionState,
        plan: &RetentionPlan,
    ) -> Result<RetentionStats> {
        let retained = plan.retain_positions.len() as u32;
        for view in &mut state.layer_views {
            view.entries.clear();
            view.dense_seq_len = state.dense_len.max(retained);
        }
        state.live_positions = plan.retain_positions.clone();
        Ok(RetentionStats {
            candidate_count: retained,
            retained_count: retained,
            blocks_reclaimed: 0,
            scored_count: 0,
        })
    }
}

/// Compact a dense-paged sequence view to retained logical positions.
///
/// Returns the new entries and the set of physical block ids that became
/// unreferenced **within this sequence** (host must `dec_ref` / free).
pub fn compact_layer_to_entries(
    logical_positions: &[u32],
    block_table: &[u32],
    block_size: u32,
    layer: u32,
    kv_head: u16,
) -> (Vec<RetainedEntry>, Vec<u32>) {
    let bs = block_size.max(1);
    let mut entries = Vec::with_capacity(logical_positions.len());
    let mut used_blocks = std::collections::BTreeSet::new();
    for &pos in logical_positions {
        let logical_block = pos / bs;
        let slot = (pos % bs) as u16;
        let phys = block_table
            .get(logical_block as usize)
            .copied()
            .unwrap_or(0);
        used_blocks.insert(phys);
        entries.push(RetainedEntry {
            logical_pos: pos,
            physical_block: phys,
            physical_slot: slot,
            kv_head,
            layer: layer as u16,
        });
    }
    // Blocks that appear in the table but are unused after retention.
    let mut reclaimed = Vec::new();
    for (i, &phys) in block_table.iter().enumerate() {
        // A block is reclaimable from this sequence only if no retained
        // position maps to it.
        let start = i as u32 * bs;
        let end = start + bs;
        let still_used = logical_positions.iter().any(|&p| p >= start && p < end);
        if !still_used {
            reclaimed.push(phys);
        }
    }
    let _ = used_blocks;
    (entries, reclaimed)
}

/// Gather dense K rows (f32, layout `[pos][kv_head][dim]`) for retained positions.
pub fn gather_kv_by_positions(
    dense: &[f32],
    positions: &[u32],
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let stride = n_kv_heads * head_dim;
    let mut out = Vec::with_capacity(positions.len() * stride);
    for &pos in positions {
        let base = pos as usize * stride;
        if base + stride <= dense.len() {
            out.extend_from_slice(&dense[base..base + stride]);
        } else {
            out.extend(std::iter::repeat_n(0.0, stride));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_identifies_unused_blocks() {
        // block_size=4, 3 blocks, keep positions 0,1,5 → block 2 unused
        let table = vec![10, 11, 12];
        let keep = vec![0u32, 1, 5];
        let (entries, reclaimed) = compact_layer_to_entries(&keep, &table, 4, 0, u16::MAX);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].physical_block, 10);
        assert_eq!(entries[2].physical_block, 11);
        assert!(reclaimed.contains(&12));
        assert!(!reclaimed.contains(&10));
    }

    #[test]
    fn gather_preserves_order() {
        // 3 positions, 1 kv head, dim 2
        let dense = vec![
            1.0, 2.0, // pos 0
            3.0, 4.0, // pos 1
            5.0, 6.0, // pos 2
        ];
        let g = gather_kv_by_positions(&dense, &[2, 0], 1, 2);
        assert_eq!(g, vec![5.0, 6.0, 1.0, 2.0]);
    }
}
