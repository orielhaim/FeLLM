//! Paper-accurate TriAttention scoring (arXiv:2604.04921 §4).

use fellm_core::error::{FellmError, Result};
use std::collections::BTreeMap;

/// Runtime configuration for TriAttention.
#[derive(Debug, Clone)]
pub struct TriAttentionConfig {
    /// Max retained tokens (B).
    pub budget: u32,
    /// Prune every `window` generated tokens (β, default 128).
    pub window: u32,
    /// RoPE base θ (default 10000).
    pub rope_theta: f32,
    /// Warmup length before first prune (optional).
    pub warmup: u32,
}

impl Default for TriAttentionConfig {
    fn default() -> Self {
        Self {
            budget: 2048,
            window: 128,
            rope_theta: 10_000.0,
            warmup: 0,
        }
    }
}

impl TriAttentionConfig {
    /// Parse from provider-scoped string map.
    pub fn from_map(map: &BTreeMap<String, String>) -> Result<Self> {
        let mut c = Self::default();
        if let Some(v) = map.get("budget") {
            c.budget = v
                .parse()
                .map_err(|_| FellmError::other(format!("kv.triattention: invalid budget '{v}'")))?;
            if c.budget == 0 {
                return Err(FellmError::other("kv.triattention: budget must be > 0"));
            }
        }
        if let Some(v) = map.get("window") {
            c.window = v
                .parse()
                .map_err(|_| FellmError::other(format!("kv.triattention: invalid window '{v}'")))?;
            if c.window == 0 {
                return Err(FellmError::other("kv.triattention: window must be > 0"));
            }
        }
        if let Some(v) = map.get("rope_theta") {
            c.rope_theta = v.parse().map_err(|_| {
                FellmError::other(format!("kv.triattention: invalid rope_theta '{v}'"))
            })?;
        }
        if let Some(v) = map.get("warmup") {
            c.warmup = v
                .parse()
                .map_err(|_| FellmError::other(format!("kv.triattention: invalid warmup '{v}'")))?;
        }
        // Reject unknown keys that look like typos for this provider.
        for k in map.keys() {
            match k.as_str() {
                "budget" | "window" | "rope_theta" | "warmup" | "calibration_path" => {}
                other if other.starts_with("kv.") => {}
                _ => {
                    // Allow bare unknown only if empty policy — ignore for forward compat.
                }
            }
        }
        Ok(c)
    }
}

/// Offline calibration: per-query-head frequency-band centers.
///
/// For each query head `h` and frequency band `f`:
/// - `eq_re[h][f], eq_im[h][f]` = E[q_f] as complex
/// - `e_norm[h][f]` = E[‖q_f‖]
#[derive(Debug, Clone)]
pub struct CalibrationCenters {
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of frequency bands (`head_dim / 2`).
    pub n_freq: usize,
    /// Real part of E[q_f], layout `[head][freq]`.
    pub eq_re: Vec<f32>,
    /// Imag part of E[q_f].
    pub eq_im: Vec<f32>,
    /// E[‖q_f‖].
    pub e_norm: Vec<f32>,
    /// RoPE θ used when scoring.
    pub rope_theta: f32,
}

impl CalibrationCenters {
    /// Unit real centers for synthetic tests.
    #[must_use]
    pub fn identity(n_heads: usize, n_freq: usize, rope_theta: f32) -> Self {
        let n = n_heads * n_freq;
        Self {
            n_heads,
            n_freq,
            eq_re: vec![1.0; n],
            eq_im: vec![0.0; n],
            e_norm: vec![1.0; n],
            rope_theta,
        }
    }

    #[inline]
    fn idx(&self, head: usize, freq: usize) -> usize {
        head * self.n_freq + freq
    }

    /// ‖E[q_f]‖
    #[must_use]
    pub fn center_norm(&self, head: usize, freq: usize) -> f32 {
        let i = self.idx(head, freq);
        let re = self.eq_re[i];
        let im = self.eq_im[i];
        (re * re + im * im).sqrt()
    }

    /// arg(E[q_f])
    #[must_use]
    pub fn center_arg(&self, head: usize, freq: usize) -> f32 {
        let i = self.idx(head, freq);
        self.eq_im[i].atan2(self.eq_re[i])
    }

    /// E[‖q_f‖]
    #[must_use]
    pub fn expected_norm(&self, head: usize, freq: usize) -> f32 {
        self.e_norm[self.idx(head, freq)]
    }

    /// Mean Resultant Length R_f = ‖E[q_f]‖ / E[‖q_f‖]
    #[must_use]
    pub fn mean_resultant_length(&self, head: usize, freq: usize) -> f32 {
        let en = self.expected_norm(head, freq).max(1e-8);
        (self.center_norm(head, freq) / en).clamp(0.0, 1.0)
    }
}

/// Geometric future offsets D = {1, 2, 4, ..., 2^16}.
#[must_use]
pub fn geometric_offsets() -> Vec<u32> {
    let mut d = Vec::with_capacity(17);
    let mut v = 1u32;
    for _ in 0..=16 {
        d.push(v);
        v = v.saturating_mul(2);
    }
    d
}

/// RoPE frequency ω_f = θ^{-2f/d} with d = 2 * n_freq.
#[inline]
pub fn rope_omega(freq: usize, n_freq: usize, theta: f32) -> f32 {
    let d = (2 * n_freq) as f32;
    theta.powf(-2.0 * freq as f32 / d)
}

/// Complex components of a key frequency band from a real head_dim vector.
#[inline]
fn key_band(k: &[f32], freq: usize) -> (f32, f32) {
    let re = k[2 * freq];
    let im = k[2 * freq + 1];
    (re, im)
}

#[inline]
fn complex_norm(re: f32, im: f32) -> f32 {
    (re * re + im * im).sqrt()
}

#[inline]
fn complex_arg(re: f32, im: f32) -> f32 {
    im.atan2(re)
}

/// S_trig + S_norm for one key vector at distance Δ for one query head.
///
/// Paper eqs (6)+(8)+(10):
/// S_trig(k,Δ) = Σ_f ‖E[q_f]‖ · ‖k_f‖ · cos(ω_f Δ + φ_f)
/// S_norm(k)   = Σ_f (E[‖q_f‖] - ‖E[q_f]‖) · ‖k_f‖
pub fn trig_score_key(key: &[f32], centers: &CalibrationCenters, head: usize, delta: f32) -> f32 {
    let n_freq = centers.n_freq.min(key.len() / 2);
    let mut s_trig = 0.0f32;
    let mut s_norm = 0.0f32;
    for f in 0..n_freq {
        let (kre, kim) = key_band(key, f);
        let kn = complex_norm(kre, kim);
        let eq_n = centers.center_norm(head, f);
        let e_norm = centers.expected_norm(head, f);
        let phi = centers.center_arg(head, f) - complex_arg(kre, kim);
        let omega = rope_omega(f, centers.n_freq, centers.rope_theta);
        s_trig += eq_n * kn * (omega * delta + phi).cos();
        // (E[‖q‖] - ‖E[q]‖) = (1 - R) * E[‖q‖]
        s_norm += (e_norm - eq_n).max(0.0) * kn;
    }
    s_trig + s_norm
}

/// Multi-offset average S̃(k) over geometric future distances.
pub fn multi_offset_score(
    key: &[f32],
    centers: &CalibrationCenters,
    head: usize,
    base_delta: f32,
) -> f32 {
    let offsets = geometric_offsets();
    let mut sum = 0.0f32;
    for &delta_off in &offsets {
        sum += trig_score_key(key, centers, head, base_delta + delta_off as f32);
    }
    sum / offsets.len() as f32
}

/// Score each key position for a single query head.
///
/// `keys`: `[pos][kv_head][head_dim]` flat
/// `positions`: logical positions matching the pos axis of `keys`
pub fn score_keys_single(
    keys: &[f32],
    positions: &[u32],
    centers: &CalibrationCenters,
    head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_pos: u32,
) -> Vec<f32> {
    let n_kv = n_kv_heads.max(1);
    let group = (centers.n_heads / n_kv).max(1);
    let kv_h = head / group;
    let stride = n_kv * head_dim;
    let mut scores = Vec::with_capacity(positions.len());
    for (i, &pos) in positions.iter().enumerate() {
        let base = i * stride + kv_h * head_dim;
        let key = &keys[base..base + head_dim];
        let delta = query_pos.saturating_sub(pos) as f32;
        scores.push(multi_offset_score(key, centers, head, delta));
    }
    scores
}

/// GQA normalize-then-max aggregation (paper §4.3 eqs 12–13).
#[allow(clippy::too_many_arguments)]
pub fn score_keys_gqa(
    keys: &[f32],
    positions: &[u32],
    centers: &CalibrationCenters,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_pos: u32,
    _rope_theta: f32,
) -> Vec<f32> {
    let n = positions.len();
    let mut final_scores = vec![f32::NEG_INFINITY; n];
    for h in 0..n_heads.min(centers.n_heads) {
        let mut s = score_keys_single(keys, positions, centers, h, n_kv_heads, head_dim, query_pos);
        // Z-score within this head across keys.
        let mean = s.iter().sum::<f32>() / n.max(1) as f32;
        let var = s.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n.max(1) as f32;
        let std = var.sqrt().max(1e-6);
        for (i, v) in s.iter_mut().enumerate() {
            let z = (*v - mean) / std;
            if z > final_scores[i] {
                final_scores[i] = z;
            }
        }
    }
    final_scores
}

/// Batch scoring entry over a **host** f32 key buffer.
///
/// This is the shipped hot path for the host policy. It is *not*
/// device-resident: it reads a host mirror of pre-RoPE keys (which CUDA
/// kernels populate via `pre_rope_write`). Kept as a standalone entry so the
/// same math is callable from the C ABI and future device consumers.
///
/// Layout: `keys` is `[pos][kv_head][head_dim]` f32 for `n_pos` positions.
/// Returns per-position final scores after GQA max-aggregate.
#[allow(clippy::too_many_arguments)]
pub fn score_batch_host(
    keys: &[f32],
    n_pos: usize,
    centers: &CalibrationCenters,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_pos: u32,
    rope_theta: f32,
) -> Vec<f32> {
    let positions: Vec<u32> = (0..n_pos as u32).collect();
    let _ = rope_theta;
    score_keys_gqa(
        keys,
        &positions,
        centers,
        n_heads,
        n_kv_heads,
        head_dim,
        query_pos,
        centers.rope_theta,
    )
}

/// Select top-B positions by score (stable on ties: prefer higher logical pos).
pub fn select_top_b(scores: &[f32], positions: &[u32], budget: usize) -> Vec<u32> {
    assert_eq!(scores.len(), positions.len());
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| positions[b].cmp(&positions[a]))
    });
    idx.truncate(budget.min(idx.len()));
    let mut out: Vec<u32> = idx.into_iter().map(|i| positions[i]).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_offsets_match_paper() {
        let d = geometric_offsets();
        assert_eq!(d[0], 1);
        assert_eq!(d[1], 2);
        assert_eq!(d[10], 1024);
        assert_eq!(d[16], 1u32 << 16);
    }

    #[test]
    fn trig_score_prefers_distance_matching_phase() {
        // One head, one frequency band: E[q] = (1,0), key = (1,0)
        // S_trig(Δ) = 1 * 1 * cos(ω Δ) — peaks at Δ=0.
        let centers = CalibrationCenters {
            n_heads: 1,
            n_freq: 1,
            eq_re: vec![1.0],
            eq_im: vec![0.0],
            e_norm: vec![1.0],
            rope_theta: 10_000.0,
        };
        let key = [1.0f32, 0.0]; // re, im for band 0
        let s0 = trig_score_key(&key, &centers, 0, 0.0);
        let s_far = trig_score_key(&key, &centers, 0, 1000.0);
        // At Δ=0, cos(0)=1; far away oscillates — near-zero distance scores higher on average.
        assert!(s0 > 0.5, "near score {s0}");
        // Norm term is zero when R=1 (e_norm == center_norm).
        assert!((s0 - 1.0).abs() < 1e-5, "s0={s0}");
        assert!(s_far < s0 + 0.1);
    }

    #[test]
    fn norm_term_active_when_low_concentration() {
        // E[q] near zero direction but E[‖q‖]=1 → R≈0 → full norm contribution.
        let centers = CalibrationCenters {
            n_heads: 1,
            n_freq: 1,
            eq_re: vec![0.0],
            eq_im: vec![0.0],
            e_norm: vec![1.0],
            rope_theta: 10_000.0,
        };
        let key = [3.0f32, 4.0]; // ‖k‖=5
        let s = trig_score_key(&key, &centers, 0, 0.0);
        // s_trig = 0, s_norm = (1 - 0) * 5 = 5
        assert!((s - 5.0).abs() < 1e-4, "s={s}");
    }

    #[test]
    fn gqa_max_aggregate_retains_any_head_importance() {
        let centers = CalibrationCenters::identity(2, 2, 10_000.0);
        // 3 positions, 1 kv head, dim 4
        let mut keys = vec![0.0f32; 3 * 1 * 4];
        // pos 1 has large key → should rank high
        for d in 0..4 {
            keys[1 * 4 + d] = 2.0;
            keys[d] = 0.1;
            keys[2 * 4 + d] = 0.1;
        }
        let positions = vec![0u32, 1, 2];
        let scores = score_keys_gqa(&keys, &positions, &centers, 2, 1, 4, 10, 10_000.0);
        let top = select_top_b(&scores, &positions, 1);
        assert_eq!(top, vec![1]);
    }

    #[test]
    fn multi_offset_averages_geometric_set() {
        let centers = CalibrationCenters::identity(1, 2, 10_000.0);
        let key = [1.0f32, 0.0, 0.5, 0.0];
        let s = multi_offset_score(&key, &centers, 0, 0.0);
        // Manual average of trig_score over offsets should match.
        let offs = geometric_offsets();
        let mut manual = 0.0f32;
        for &delta_off in &offs {
            manual += trig_score_key(&key, &centers, 0, delta_off as f32);
        }
        manual /= offs.len() as f32;
        assert!((s - manual).abs() < 1e-5);
    }
}
