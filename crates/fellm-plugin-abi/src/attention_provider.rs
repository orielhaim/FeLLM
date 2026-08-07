//! Attention provider contracts.
//!
//! Semantic graphs emit generic [`crate::OpKind::Attention`]. Physical planning
//! selects an [`AttentionProvider`] once, then executes prepared handles.

use crate::capability::{FeatureSet, PluginConfig, PreparedProviderId, ProviderDescriptor};
use crate::op::OpAttrs;
use fellm_core::dtype::DType;
use fellm_core::error::Result;

/// Workload shape used to prepare an attention implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionWorkload {
    /// Number of query heads.
    pub n_heads: u32,
    /// Number of KV heads (GQA/MQA).
    pub n_kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Query sequence length (1 = decode).
    pub query_len: u32,
    /// KV sequence length (dense upper bound before retention).
    pub kv_len: u32,
    /// Element type of Q and of KV storage.
    pub dtype: DType,
    /// Causal mask when true.
    pub causal: bool,
    /// Sliding window size (`0` = unrestricted).
    pub window: u32,
    /// Paged KV when true; contiguous otherwise.
    pub paged: bool,
    /// Whether the KV view may be position-indirect / non-dense.
    pub indirect_positions: bool,
}

impl AttentionWorkload {
    /// True when this is a pure decode step (single query token).
    #[must_use]
    pub fn is_decode(&self) -> bool {
        self.query_len == 1
    }

    /// True when this is prefill / multi-token attention.
    #[must_use]
    pub fn is_prefill(&self) -> bool {
        self.query_len > 1
    }
}

/// Device / backend capability summary visible to attention providers.
#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilityView {
    /// Features exposed by the active backend / hardware.
    pub features: FeatureSet,
    /// Shared memory per SM in bytes when known.
    pub smem_per_sm: u32,
    /// Compute capability major (0 if unknown / CPU).
    pub compute_major: u32,
    /// Compute capability minor.
    pub compute_minor: u32,
}

/// Kind of prepared attention path (semantic, not product name).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionPathKind {
    /// Multi-token query path (prefill / prompt).
    Prefill = 1,
    /// Single-token decode path.
    Decode = 2,
    /// Short multi-query decode batch.
    BatchedDecode = 3,
    /// Generic fallback path.
    Generic = 4,
}

/// One prepared attention implementation ready for steady-state launch.
#[derive(Debug, Clone)]
pub struct PreparedAttention {
    /// Registry id of the provider that prepared this plan.
    pub provider: PreparedProviderId,
    /// Semantic path kind selected for this workload.
    pub path: AttentionPathKind,
    /// Backend-owned kernel / plan variant id (direct dispatch key).
    pub kernel_variant: u64,
    /// Opaque provider-local plan handle (may be a function pointer as u64).
    pub plan_handle: u64,
    /// Features actually used by this prepared plan.
    pub features_used: FeatureSet,
}

/// Context passed when preparing attention for a model/plan.
#[derive(Debug, Clone)]
pub struct AttentionPrepareContext<'a> {
    /// Workload descriptors that must be supportable (at least one path each).
    pub workloads: &'a [AttentionWorkload],
    /// Device capabilities.
    pub device: &'a DeviceCapabilityView,
    /// Provider-scoped configuration.
    pub config: &'a PluginConfig,
    /// Layer count (for multi-layer binding).
    pub n_layers: u32,
}

/// Attention provider: selected once, prepares paths, then executes handles.
pub trait AttentionProvider: Send + Sync {
    /// Static descriptor for discovery / CLI.
    fn descriptor(&self) -> &ProviderDescriptor;

    /// Validate provider-specific configuration before preparation.
    fn validate_config(&self, config: &PluginConfig) -> Result<()>;

    /// Whether this provider can handle the workload against device caps.
    fn supports(&self, workload: &AttentionWorkload, device: &DeviceCapabilityView) -> bool;

    /// Prepare implementation(s) for the given contexts.
    ///
    /// Returns one prepared plan per distinct path kind required.
    fn prepare(&self, ctx: &AttentionPrepareContext<'_>) -> Result<Vec<PreparedAttention>>;

    /// Optional: score applicability for auto-selection (higher is better).
    fn applicability(&self, workload: &AttentionWorkload, device: &DeviceCapabilityView) -> i32 {
        if self.supports(workload, device) {
            self.descriptor().priority
        } else {
            i32::MIN
        }
    }
}

/// Build default [`OpAttrs`] fields related to attention for a workload.
#[must_use]
pub fn attrs_from_workload(w: &AttentionWorkload, scale: f32) -> OpAttrs {
    OpAttrs {
        n_heads: w.n_heads,
        n_kv_heads: w.n_kv_heads,
        head_dim: w.head_dim,
        scale,
        attention_mode: u32::from(!w.causal),
        attention_window: w.window,
        query_len: w.query_len,
        kv_len: w.kv_len,
        block_size: if w.paged { 16 } else { 0 },
        ..OpAttrs::default()
    }
}

/// Host-side reference attention (FP32) for correctness testing.
///
/// Implements standard scaled-dot-product attention with optional causal mask,
/// GQA/MQA, and sliding window. Used as the numerical oracle for provider tests.
#[allow(clippy::too_many_arguments)]
pub fn reference_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_len: usize,
    kv_len: usize,
    scale: f32,
    causal: bool,
    window: usize,
) {
    assert_eq!(q.len(), n_heads * query_len * head_dim);
    assert_eq!(k.len(), n_kv_heads * kv_len * head_dim);
    assert_eq!(v.len(), n_kv_heads * kv_len * head_dim);
    assert_eq!(out.len(), n_heads * query_len * head_dim);
    assert!(n_heads.is_multiple_of(n_kv_heads.max(1)));
    let n_kv = n_kv_heads.max(1);
    let group = (n_heads / n_kv).max(1);

    for h in 0..n_heads {
        let kv_h = h / group;
        for qi in 0..query_len {
            let q_off = (h * query_len + qi) * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            // Online softmax into out row.
            let o_off = q_off;
            let out_row = &mut out[o_off..o_off + head_dim];
            out_row.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for kj in 0..kv_len {
                if causal {
                    // Query token qi attends to keys 0..=(kv_len - query_len + qi)
                    let max_k = kv_len - query_len + qi;
                    if kj > max_k {
                        continue;
                    }
                    if window > 0 && max_k.saturating_sub(kj) >= window {
                        continue;
                    }
                } else if window > 0 {
                    let dist = qi.abs_diff(kj);
                    if dist >= window {
                        continue;
                    }
                }
                let k_off = (kj * n_kv + kv_h) * head_dim;
                let k_row = &k[k_off..k_off + head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;
                let m_new = if score > m { score } else { m };
                let alpha = if m.is_finite() {
                    (m - m_new).exp()
                } else {
                    0.0
                };
                let p = (score - m_new).exp();
                if alpha != 1.0 {
                    for d in 0..head_dim {
                        out_row[d] *= alpha;
                    }
                    l *= alpha;
                }
                let v_off = (kj * n_kv + kv_h) * head_dim;
                let v_row = &v[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    out_row[d] += p * v_row[d];
                }
                l += p;
                m = m_new;
            }
            if l > 0.0 {
                let inv = 1.0 / l;
                for d in 0..head_dim {
                    out_row[d] *= inv;
                }
            }
        }
    }
}

/// FA2-style host tiled attention path: query-block × KV-tile online softmax.
///
/// Mirrors FlashAttention-2 work partitioning principles on the host for
/// correctness / benchmarking without requiring a GPU:
/// - tiles Q along the sequence (parallel across query blocks / heads)
/// - streams K/V tiles without materializing S = QK^T
/// - online softmax with FP32 accumulation
/// - GQA/MQA via shared KV heads
///
/// This is the algorithmic reference for the CUDA FA2-style provider path.
#[allow(clippy::too_many_arguments)]
pub fn fa2_style_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_len: usize,
    kv_len: usize,
    scale: f32,
    causal: bool,
    window: usize,
    // Query tile size (Br in FA2).
    q_tile: usize,
    // KV tile size (Bc in FA2).
    kv_tile: usize,
) {
    assert_eq!(q.len(), n_heads * query_len * head_dim);
    assert_eq!(k.len(), n_kv_heads * kv_len * head_dim);
    assert_eq!(v.len(), n_kv_heads * kv_len * head_dim);
    assert_eq!(out.len(), n_heads * query_len * head_dim);
    let n_kv = n_kv_heads.max(1);
    let group = (n_heads / n_kv).max(1);
    let br = q_tile.max(1);
    let bc = kv_tile.max(1);

    // Parallelism across heads and query blocks (FA2: sequence-dimension split).
    for h in 0..n_heads {
        let kv_h = h / group;
        let mut q_start = 0usize;
        while q_start < query_len {
            let q_end = (q_start + br).min(query_len);
            // Per-query online state for this block (warp-local in FA2).
            let block_q = q_end - q_start;
            let mut m = vec![f32::NEG_INFINITY; block_q];
            let mut l = vec![0.0f32; block_q];
            for qi in q_start..q_end {
                let o_off = (h * query_len + qi) * head_dim;
                out[o_off..o_off + head_dim].fill(0.0);
            }

            let mut k_start = 0usize;
            while k_start < kv_len {
                let k_end = (k_start + bc).min(kv_len);
                for (local_i, qi) in (q_start..q_end).enumerate() {
                    let q_off = (h * query_len + qi) * head_dim;
                    let q_row = &q[q_off..q_off + head_dim];
                    let o_off = q_off;
                    let out_row = &mut out[o_off..o_off + head_dim];

                    // Score tile for this query against KV tile (no full matrix).
                    let mut scores = vec![f32::NEG_INFINITY; k_end - k_start];
                    for (local_j, kj) in (k_start..k_end).enumerate() {
                        if causal {
                            let max_k = kv_len - query_len + qi;
                            if kj > max_k {
                                continue;
                            }
                            if window > 0 && max_k.saturating_sub(kj) >= window {
                                continue;
                            }
                        } else if window > 0 {
                            let dist = qi.abs_diff(kj);
                            if dist >= window {
                                continue;
                            }
                        }
                        let k_off = (kj * n_kv + kv_h) * head_dim;
                        let k_row = &k[k_off..k_off + head_dim];
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_row[d] * k_row[d];
                        }
                        scores[local_j] = score * scale;
                    }

                    let mut m_tile = f32::NEG_INFINITY;
                    for &s in &scores {
                        if s.is_finite() && s > m_tile {
                            m_tile = s;
                        }
                    }
                    if !m_tile.is_finite() {
                        continue;
                    }
                    let m_new = if m_tile > m[local_i] {
                        m_tile
                    } else {
                        m[local_i]
                    };
                    let alpha = if m[local_i].is_finite() {
                        (m[local_i] - m_new).exp()
                    } else {
                        0.0
                    };
                    if alpha != 1.0 {
                        for d in 0..head_dim {
                            out_row[d] *= alpha;
                        }
                        l[local_i] *= alpha;
                    }
                    let mut l_tile = 0.0f32;
                    for (local_j, kj) in (k_start..k_end).enumerate() {
                        let s = scores[local_j];
                        if !s.is_finite() {
                            continue;
                        }
                        let p = (s - m_new).exp();
                        l_tile += p;
                        let v_off = (kj * n_kv + kv_h) * head_dim;
                        let v_row = &v[v_off..v_off + head_dim];
                        for d in 0..head_dim {
                            out_row[d] += p * v_row[d];
                        }
                    }
                    l[local_i] += l_tile;
                    m[local_i] = m_new;
                }
                k_start = k_end;
            }

            for (local_i, qi) in (q_start..q_end).enumerate() {
                if l[local_i] > 0.0 {
                    let inv = 1.0 / l[local_i];
                    let o_off = (h * query_len + qi) * head_dim;
                    for d in 0..head_dim {
                        out[o_off + d] *= inv;
                    }
                }
            }
            q_start = q_end;
        }
    }
}

/// Reference attention over a **paged** host arena (f16 rows via gather).
///
/// `gather_kv(layer, pos, is_v, out_row)` fills one KV head-stride row.
#[allow(clippy::too_many_arguments)]
pub fn reference_attention_paged_f32(
    q: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq: usize,
    scale: f32,
    causal: bool,
    window: usize,
    mut gather_kv: impl FnMut(usize, bool, &mut [f32]),
) {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    let n_kv = n_kv_heads.max(1);
    let group = (n_heads / n_kv).max(1);
    let stride = n_kv * head_dim;
    let mut k_row = vec![0.0f32; stride];
    let mut v_row = vec![0.0f32; stride];

    for h in 0..n_heads {
        let kv_h = h / group;
        let q_row = &q[h * head_dim..(h + 1) * head_dim];
        let out_row = &mut out[h * head_dim..(h + 1) * head_dim];
        out_row.fill(0.0);
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..seq {
            if causal && window > 0 && (seq - 1).saturating_sub(t) >= window {
                continue;
            }
            gather_kv(t, false, &mut k_row);
            gather_kv(t, true, &mut v_row);
            let k = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            let v = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_row[d] * k[d];
            }
            score *= scale;
            let m_new = if score > m { score } else { m };
            let alpha = if m.is_finite() {
                (m - m_new).exp()
            } else {
                0.0
            };
            let p = (score - m_new).exp();
            if alpha != 1.0 {
                for d in 0..head_dim {
                    out_row[d] *= alpha;
                }
                l *= alpha;
            }
            for d in 0..head_dim {
                out_row[d] += p * v[d];
            }
            l += p;
            m = m_new;
        }
        if l > 0.0 {
            let inv = 1.0 / l;
            for d in 0..head_dim {
                out_row[d] *= inv;
            }
        }
    }
}

/// FA2-style host path over paged gather (same tiling as contiguous FA2).
#[allow(clippy::too_many_arguments)]
pub fn fa2_style_attention_paged_f32(
    q: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq: usize,
    scale: f32,
    causal: bool,
    window: usize,
    q_tile: usize,
    kv_tile: usize,
    mut gather_kv: impl FnMut(usize, bool, &mut [f32]),
) {
    // Gather dense then call contiguous FA2 — proves paged gather + FA2 compose.
    let stride = n_kv_heads.max(1) * head_dim;
    let mut k = vec![0.0f32; seq * stride];
    let mut v = vec![0.0f32; seq * stride];
    for t in 0..seq {
        gather_kv(t, false, &mut k[t * stride..(t + 1) * stride]);
        gather_kv(t, true, &mut v[t * stride..(t + 1) * stride]);
    }
    fa2_style_attention_f32(
        q, &k, &v, out, n_heads, n_kv_heads, head_dim, 1, seq, scale, causal, window, q_tile,
        kv_tile,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fa2_style_matches_reference_decode() {
        let n_heads = 4;
        let n_kv = 2;
        let hd = 8;
        let q_len = 1;
        let kv_len = 17;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut q = vec![0.0f32; n_heads * q_len * hd];
        let mut k = vec![0.0f32; n_kv * kv_len * hd];
        let mut v = vec![0.0f32; n_kv * kv_len * hd];
        for (i, x) in q.iter_mut().enumerate() {
            *x = ((i * 17) % 13) as f32 * 0.1 - 0.5;
        }
        for (i, x) in k.iter_mut().enumerate() {
            *x = ((i * 7) % 11) as f32 * 0.1 - 0.3;
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i * 3) % 9) as f32 * 0.05;
        }
        let mut out_ref = vec![0.0f32; n_heads * q_len * hd];
        let mut out_fa2 = vec![0.0f32; n_heads * q_len * hd];
        reference_attention_f32(
            &q,
            &k,
            &v,
            &mut out_ref,
            n_heads,
            n_kv,
            hd,
            q_len,
            kv_len,
            scale,
            true,
            0,
        );
        fa2_style_attention_f32(
            &q,
            &k,
            &v,
            &mut out_fa2,
            n_heads,
            n_kv,
            hd,
            q_len,
            kv_len,
            scale,
            true,
            0,
            4,
            8,
        );
        for (a, b) in out_ref.iter().zip(out_fa2.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch {a} vs {b}");
        }
    }

    #[test]
    fn fa2_style_matches_reference_prefill() {
        let n_heads = 2;
        let n_kv = 2;
        let hd = 4;
        let q_len = 5;
        let kv_len = 5;
        let scale = 0.5;
        let mut q = vec![0.0f32; n_heads * q_len * hd];
        let mut k = vec![0.0f32; n_kv * kv_len * hd];
        let mut v = vec![0.0f32; n_kv * kv_len * hd];
        for i in 0..q.len() {
            q[i] = (i as f32 * 0.13).sin();
            if i < k.len() {
                k[i] = (i as f32 * 0.17).cos();
                v[i] = (i as f32 * 0.11).sin() * 0.5;
            }
        }
        let mut out_ref = vec![0.0f32; n_heads * q_len * hd];
        let mut out_fa2 = vec![0.0f32; n_heads * q_len * hd];
        reference_attention_f32(
            &q,
            &k,
            &v,
            &mut out_ref,
            n_heads,
            n_kv,
            hd,
            q_len,
            kv_len,
            scale,
            true,
            0,
        );
        fa2_style_attention_f32(
            &q,
            &k,
            &v,
            &mut out_fa2,
            n_heads,
            n_kv,
            hd,
            q_len,
            kv_len,
            scale,
            true,
            0,
            2,
            3,
        );
        for (a, b) in out_ref.iter().zip(out_fa2.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch {a} vs {b}");
        }
    }
}
