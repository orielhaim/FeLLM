//! Full scaled-dot-product attention over a KV cache.
//!
//! Layout convention (all f32 for Phase 1):
//!   q:      [n_heads, head_dim]                      (single query token)
//!   k:      [past_len + 1, n_kv_heads, head_dim]     (cache, contiguous)
//!   v:      [past_len + 1, n_kv_heads, head_dim]
//!   out:    [n_heads, head_dim]
//!
//! GQA: `n_heads` must be a multiple of `n_kv_heads`.

use crate::kernels::softmax::softmax_rows_inplace;

/// Compute attention for one query token.
#[allow(clippy::too_many_arguments)]
pub fn attention_step(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    past_len: usize,
    scale: f32,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert!(n_heads % n_kv_heads == 0);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    let seq = past_len + 1;
    debug_assert_eq!(k_cache.len(), seq * n_kv_heads * head_dim);
    debug_assert_eq!(v_cache.len(), seq * n_kv_heads * head_dim);
    let heads_per_kv = n_heads / n_kv_heads;

    use rayon::prelude::*;
    // Parallelize across heads.
    let out_chunks: Vec<Vec<f32>> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let kv_h = h / heads_per_kv;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            // Compute scores [seq]
            let mut scores = vec![0.0f32; seq];
            for t in 0..seq {
                let k_row = &k_cache[(t * n_kv_heads + kv_h) * head_dim
                    ..(t * n_kv_heads + kv_h + 1) * head_dim];
                let mut s = 0.0f32;
                for i in 0..head_dim {
                    s += q_head[i] * k_row[i];
                }
                scores[t] = s * scale;
            }
            // softmax
            softmax_rows_inplace(&mut scores, 1, seq, None);
            // Weighted sum over V
            let mut out_h = vec![0.0f32; head_dim];
            for t in 0..seq {
                let v_row = &v_cache[(t * n_kv_heads + kv_h) * head_dim
                    ..(t * n_kv_heads + kv_h + 1) * head_dim];
                let w = scores[t];
                for i in 0..head_dim {
                    out_h[i] += w * v_row[i];
                }
            }
            out_h
        })
        .collect();
    for (h, out_h) in out_chunks.into_iter().enumerate() {
        out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out_h);
    }
}
