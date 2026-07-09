use crate::cpu_profile::CpuHardwareProfile;
use fellm_plugin_abi as paged_ctx;
use rayon::prelude::*;

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
    profile: &CpuHardwareProfile,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert!(n_heads % n_kv_heads == 0);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    let seq = past_len + 1;
    debug_assert_eq!(k_cache.len(), seq * n_kv_heads * head_dim);
    debug_assert_eq!(v_cache.len(), seq * n_kv_heads * head_dim);
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_tile = profile.kv_tile_for(head_dim, seq);

    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_h)| {
            let kv_h = h / heads_per_kv;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            attention_head_tiled(
                q_head, k_cache, v_cache, out_h, n_kv_heads, head_dim, seq, kv_h, scale, kv_tile,
            );
        });
}

/// PagedAttention: K/V rows resolved via [`PagedKvContext`] block tables.
#[allow(clippy::too_many_arguments)]
pub fn attention_step_paged(
    q: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    past_len: usize,
    scale: f32,
    layer: usize,
    profile: &CpuHardwareProfile,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert!(n_heads % n_kv_heads == 0);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    let seq = past_len + 1;
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_tile = profile.kv_tile_for(head_dim, seq);

    // Serial over heads when using thread-local paged ctx (Rayon would need Send ctx).
    // For multi-head parallelism we gather rows into a contiguous scratch first.
    let tokens_stride = n_kv_heads * head_dim;
    let mut k_contig = vec![0.0f32; seq * tokens_stride];
    let mut v_contig = vec![0.0f32; seq * tokens_stride];
    paged_ctx::with_paged_context(|ctx| {
        let ctx = ctx.expect("paged attention requires PagedKvContext");
        for t in 0..seq {
            // SAFETY: ctx arena valid for step; rows are tokens_stride.
            let k_row = unsafe { ctx.k_row(layer, t) };
            let v_row = unsafe { ctx.v_row(layer, t) };
            let dst_k = &mut k_contig[t * tokens_stride..(t + 1) * tokens_stride];
            let dst_v = &mut v_contig[t * tokens_stride..(t + 1) * tokens_stride];
            dst_k.copy_from_slice(k_row);
            dst_v.copy_from_slice(v_row);
        }
    });

    attention_step(
        q, &k_contig, &v_contig, out, n_heads, n_kv_heads, head_dim, past_len, scale, profile,
    );
    let _ = (heads_per_kv, kv_tile);
}

#[allow(clippy::too_many_arguments)]
fn attention_head_tiled(
    q_head: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out_h: &mut [f32],
    n_kv_heads: usize,
    head_dim: usize,
    seq: usize,
    kv_h: usize,
    scale: f32,
    kv_tile: usize,
) {
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    out_h.fill(0.0);

    let mut scores = vec![0.0f32; kv_tile];
    let mut tile_start = 0usize;
    while tile_start < seq {
        let tile_end = (tile_start + kv_tile).min(seq);
        let tile_len = tile_end - tile_start;

        // Prefetch the next K/V tile while we compute this one.
        let next_start = tile_end;
        if next_start < seq {
            let next_end = (next_start + kv_tile).min(seq);
            prefetch_kv_tile(k_cache, n_kv_heads, head_dim, kv_h, next_start, next_end);
            prefetch_kv_tile(v_cache, n_kv_heads, head_dim, kv_h, next_start, next_end);
        }

        // scores = Q · K_tile^T * scale
        for (i, t) in (tile_start..tile_end).enumerate() {
            let k_row = kv_row(k_cache, n_kv_heads, head_dim, t, kv_h);
            scores[i] = dot(q_head, k_row) * scale;
        }

        // Online softmax update against this tile.
        let mut m_tile = f32::NEG_INFINITY;
        for i in 0..tile_len {
            if scores[i] > m_tile {
                m_tile = scores[i];
            }
        }
        let m_new = if m_tile > m { m_tile } else { m };
        let alpha = if m.is_finite() {
            (m - m_new).exp()
        } else {
            0.0
        };

        if alpha != 1.0 {
            for o in out_h.iter_mut() {
                *o *= alpha;
            }
            l *= alpha;
        }

        let mut l_tile = 0.0f32;
        for i in 0..tile_len {
            let p = (scores[i] - m_new).exp();
            scores[i] = p;
            l_tile += p;
            let t = tile_start + i;
            let v_row = kv_row(v_cache, n_kv_heads, head_dim, t, kv_h);
            for d in 0..head_dim {
                out_h[d] += p * v_row[d];
            }
        }
        l += l_tile;
        m = m_new;
        tile_start = tile_end;
    }

    if l > 0.0 {
        let inv = 1.0 / l;
        for o in out_h.iter_mut() {
            *o *= inv;
        }
    }
}

#[inline]
fn kv_row(cache: &[f32], n_kv_heads: usize, head_dim: usize, t: usize, kv_h: usize) -> &[f32] {
    let base = (t * n_kv_heads + kv_h) * head_dim;
    &cache[base..base + head_dim]
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let mut s = 0.0f32;
    let mut i = 0;
    // 4-wide unroll; wide/SIMD can be layered later without changing numerics.
    while i + 4 <= n {
        s += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
fn prefetch_kv_tile(
    cache: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    kv_h: usize,
    start: usize,
    end: usize,
) {
    // Touch a few cache-line starts across the tile (stride ~64 bytes / 16 f32).
    const STRIDE: usize = 16;
    for t in start..end {
        let base = (t * n_kv_heads + kv_h) * head_dim;
        let mut off = 0;
        while off < head_dim {
            let idx = base + off;
            if idx < cache.len() {
                prefetch_read(cache.as_ptr().wrapping_add(idx) as *const u8);
            }
            off += STRIDE;
        }
    }
}

#[inline]
fn prefetch_read(ptr: *const u8) {
    // SAFETY: software prefetch is a hint; the address need not be dereferenceable.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(all(target_arch = "x86", not(target_arch = "x86_64")))]
    unsafe {
        use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::aarch64::_prefetch::<
            { core::arch::aarch64::_PREFETCH_READ },
            { core::arch::aarch64::_PREFETCH_LOCALITY3 },
        >(ptr);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Naive reference attention (full scores → softmax → V) for tests.
#[cfg(test)]
pub fn attention_step_naive(
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
    use crate::kernels::softmax::softmax_rows_inplace;

    let seq = past_len + 1;
    let heads_per_kv = n_heads / n_kv_heads;
    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; seq];
        for t in 0..seq {
            let k_row = kv_row(k_cache, n_kv_heads, head_dim, t, kv_h);
            scores[t] = dot(q_head, k_row) * scale;
        }
        softmax_rows_inplace(&mut scores, 1, seq, None);
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        out_h.fill(0.0);
        for t in 0..seq {
            let v_row = kv_row(v_cache, n_kv_heads, head_dim, t, kv_h);
            let w = scores[t];
            for i in 0..head_dim {
                out_h[i] += w * v_row[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_profile::CpuHardwareProfile;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn run_case(n_heads: usize, n_kv: usize, head_dim: usize, seq: usize, seed: u64) {
        let mut rng = seed;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0
        };

        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..seq * n_kv * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..seq * n_kv * head_dim).map(|_| next()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let past = seq - 1;

        let mut out_fa = vec![0.0f32; n_heads * head_dim];
        let mut out_ref = vec![0.0f32; n_heads * head_dim];
        let profile = CpuHardwareProfile::detect();

        attention_step(
            &q,
            &k,
            &v,
            &mut out_fa,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            &profile,
        );
        attention_step_naive(
            &q,
            &k,
            &v,
            &mut out_ref,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
        );

        let err = max_abs_diff(&out_fa, &out_ref);
        assert!(
            err < 1e-4,
            "max abs err {err} for heads={n_heads} kv={n_kv} d={head_dim} seq={seq}"
        );
    }

    #[test]
    fn matches_naive_small() {
        run_case(4, 4, 32, 17, 1);
        run_case(8, 2, 64, 33, 2); // GQA
        run_case(2, 2, 16, 1, 3); // seq = 1
        run_case(4, 1, 48, 7, 4); // seq not multiple of tile
    }

    #[test]
    fn matches_naive_longer() {
        run_case(8, 8, 64, 128, 5);
        run_case(16, 4, 128, 65, 6);
    }
}
