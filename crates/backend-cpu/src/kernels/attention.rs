use crate::cpu_profile::CpuHardwareProfile;
use crate::kernels::simd_f32::{self, PulpDispatch};
use fellm_plugin_abi as paged_ctx;
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    /// One reusable score tile per Rayon worker. Attention is launched for
    /// every head and every denoising pass; allocating this tile in each head
    /// otherwise turns a small scratch buffer into steady-state allocator
    /// traffic.
    static SCORE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

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
    simd: PulpDispatch,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert!(n_heads.is_multiple_of(n_kv_heads));
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
                simd,
            );
        });
}

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
    simd: PulpDispatch,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert!(n_heads.is_multiple_of(n_kv_heads));
    debug_assert_eq!(out.len(), n_heads * head_dim);
    let seq = past_len + 1;
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_tile = profile.kv_tile_for(head_dim, seq);
    let ctx = paged_ctx::snapshot_paged_context().expect("paged attention requires PagedKvContext");
    // The shared paged arena is sized for the widest layer.  Gemma 4 mixes
    // 256- and 512-wide attention heads; each layer consumes only its own
    // prefix of a max-stride row.

    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_h)| {
            let kv_h = h / heads_per_kv;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            attention_head_paged(
                q_head, &ctx, out_h, head_dim, seq, kv_h, scale, kv_tile, layer, simd,
            );
        });
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
    simd: PulpDispatch,
) {
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    out_h.fill(0.0);

    SCORE_SCRATCH.with(|cell| {
        let mut scores = cell.borrow_mut();
        scores.resize(kv_tile, 0.0);
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
                scores[i] = simd_f32::dot_f32(q_head, k_row, simd) * scale;
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
                simd_f32::scale_f32(out_h, alpha, simd);
                l *= alpha;
            }

            let mut l_tile = 0.0f32;
            for i in 0..tile_len {
                let p = (scores[i] - m_new).exp();
                scores[i] = p;
                l_tile += p;
                let t = tile_start + i;
                let v_row = kv_row(v_cache, n_kv_heads, head_dim, t, kv_h);
                simd_f32::axpy_f32(out_h, v_row, p, simd);
            }
            l += l_tile;
            m = m_new;
            tile_start = tile_end;
        }

        if l > 0.0 {
            simd_f32::scale_f32(out_h, 1.0 / l, simd);
        }
    });
}

/// Online-softmax tiled attention reading f16 K/V rows from the paged arena.
#[allow(clippy::too_many_arguments)]
fn attention_head_paged(
    q_head: &[f32],
    ctx: &paged_ctx::PagedKvContext,
    out_h: &mut [f32],
    head_dim: usize,
    seq: usize,
    kv_h: usize,
    scale: f32,
    kv_tile: usize,
    layer: usize,
    simd: PulpDispatch,
) {
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    out_h.fill(0.0);

    SCORE_SCRATCH.with(|cell| {
        let mut scores = cell.borrow_mut();
        scores.resize(kv_tile, 0.0);
        let mut tile_start = 0usize;
        while tile_start < seq {
            let tile_end = (tile_start + kv_tile).min(seq);
            let tile_len = tile_end - tile_start;

            for (i, t) in (tile_start..tile_end).enumerate() {
                // SAFETY: arena valid for step; read-only during attention.
                let k_full = unsafe { ctx.k_row(layer, t) };
                let k_head = &k_full[kv_h * head_dim..(kv_h + 1) * head_dim];
                scores[i] = simd_f32::dot_f32_f16(q_head, k_head, simd) * scale;
            }

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
                simd_f32::scale_f32(out_h, alpha, simd);
                l *= alpha;
            }

            let mut l_tile = 0.0f32;
            for i in 0..tile_len {
                let p = (scores[i] - m_new).exp();
                scores[i] = p;
                l_tile += p;
                let t = tile_start + i;
                // SAFETY: same as K path.
                let v_full = unsafe { ctx.v_row(layer, t) };
                let v_head = &v_full[kv_h * head_dim..(kv_h + 1) * head_dim];
                simd_f32::axpy_f32_f16(out_h, v_head, p, simd);
            }
            l += l_tile;
            m = m_new;
            tile_start = tile_end;
        }

        if l > 0.0 {
            simd_f32::scale_f32(out_h, 1.0 / l, simd);
        }
    });
}

#[inline]
fn kv_row(cache: &[f32], n_kv_heads: usize, head_dim: usize, t: usize, kv_h: usize) -> &[f32] {
    let base = (t * n_kv_heads + kv_h) * head_dim;
    &cache[base..base + head_dim]
}

/// Scalar reference dot used by the naive test path.
#[cfg(test)]
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * b[i];
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
    use crate::kernels::simd_f32;
    use fellm_plugin_abi::{PagedKvContext, set_paged_context};
    use half::f16;

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
            PulpDispatch::new(),
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

    #[test]
    fn paged_matches_contig_f16_roundtrip() {
        // Build a tiny 1-block arena, write f16 K/V, compare paged vs contig (f32).
        let n_heads = 4;
        let n_kv = 2;
        let head_dim = 8;
        let seq = 5;
        let tokens_stride = n_kv * head_dim;
        let block_size = 16usize;
        let block_elems = 2 * block_size * tokens_stride;
        let raw_bytes = block_elems * 2;
        let block_bytes = (raw_bytes + 63) & !63;

        let mut rng = 42u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0
        };

        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| next()).collect();
        let k_f32: Vec<f32> = (0..seq * tokens_stride).map(|_| next()).collect();
        let v_f32: Vec<f32> = (0..seq * tokens_stride).map(|_| next()).collect();

        let mut arena = vec![0u8; block_bytes];
        {
            let elems: &mut [f16] = bytemuck::cast_slice_mut(&mut arena[..raw_bytes]);
            for t in 0..seq {
                let k_dst = &mut elems[t * tokens_stride..(t + 1) * tokens_stride];
                for (d, &s) in k_dst.iter_mut().zip(k_f32[t * tokens_stride..].iter()) {
                    *d = f16::from_f32(s);
                }
                let v_base = block_size * tokens_stride + t * tokens_stride;
                let v_dst = &mut elems[v_base..v_base + tokens_stride];
                for (d, &s) in v_dst.iter_mut().zip(v_f32[t * tokens_stride..].iter()) {
                    *d = f16::from_f32(s);
                }
            }
        }

        // Contig path uses original f32; paged uses f16 roundtrip — allow looser tol.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let past = seq - 1;
        let profile = CpuHardwareProfile::detect();

        let mut out_contig = vec![0.0f32; n_heads * head_dim];
        attention_step(
            &q,
            &k_f32,
            &v_f32,
            &mut out_contig,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            &profile,
            PulpDispatch::new(),
        );

        set_paged_context(Some(PagedKvContext {
            arena: arena.as_mut_ptr(),
            arena_len: arena.len(),
            block_table: std::sync::Arc::<[u32]>::from(vec![0]),
            n_logical_blocks: 1,
            n_layers: 1,
            tokens_stride,
            block_bytes,
            block_size,
            elem_bytes: 2,
            device_arena: std::ptr::null_mut(),
            device_arena_len: 0,
            device_block_table: std::ptr::null_mut(),
            n_device_block_table: 0,
            device_logical_stride: 0,
            row_positions: std::sync::Arc::from([past as u32]),
            row_lengths: std::sync::Arc::from([seq as u32]),
            row_rope_positions: std::sync::Arc::from([past as u32]),
        }));

        let mut out_paged = vec![0.0f32; n_heads * head_dim];
        attention_step_paged(
            &q,
            &mut out_paged,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            0,
            &profile,
            PulpDispatch::new(),
        );
        set_paged_context(None);

        let err = max_abs_diff(&out_contig, &out_paged);
        assert!(
            err < 2e-2,
            "paged vs contig max abs err {err} (f16 roundtrip)"
        );
    }

    #[test]
    fn dot_f32_f16_matches_scalar() {
        let a: Vec<f32> = (0..17).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f16> = a.iter().map(|&x| f16::from_f32(x + 0.5)).collect();
        let mut expected = 0.0f32;
        for i in 0..a.len() {
            expected += a[i] * b[i].to_f32();
        }
        let got = simd_f32::dot_f32_f16(&a, &b, PulpDispatch::new());
        assert!(
            (got - expected).abs() < 1e-5,
            "got {got} expected {expected}"
        );
    }
}
