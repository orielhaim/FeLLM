use crate::dequant::{QK_K, QK4_0, QK8_0};
use crate::kernels::vec_dot_q4k::vec_dot_q4_k_q8_k_row;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use half::f16;
use rayon::prelude::*;
use std::cell::RefCell;
use wide::f32x8;

#[derive(Clone)]
pub struct Q8KBlock {
    /// Block scale: `x[i] ≈ d * qs[i]`.
    pub d: f32,
    /// Quantized activations, one i8 per weight.
    pub qs: [i8; QK_K],
    /// Group sums: `bsums[g] = Σ qs[16*g .. 16*g+16]`.
    pub bsums: [i16; QK_K / 16],
}

impl Default for Q8KBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; QK_K],
            bsums: [0i16; QK_K / 16],
        }
    }
}

#[derive(Clone, Copy)]
struct Q80XBlock {
    d: f32,
    qs: [i8; QK8_0],
}

impl Default for Q80XBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; QK8_0],
        }
    }
}

thread_local! {
    /// Reused Q8_K activation scratch (avoids alloc-per-matmul).
    static Q8K_SCRATCH: RefCell<Vec<Q8KBlock>> = const { RefCell::new(Vec::new()) };
    /// Q8_K activations reused by the q/k/v and gate/up matvecs in one step.
    static Q8K_STEP_CACHE: RefCell<Q8KStepCache> = const { RefCell::new(Q8KStepCache::new()) };
    /// Reused Q8_0 activation scratch.
    static Q80_SCRATCH: RefCell<Vec<Q80XBlock>> = const { RefCell::new(Vec::new()) };
}

struct Q8KStepCache {
    active: bool,
    valid: bool,
    source_ptr: usize,
    source_len: usize,
    blocks: Vec<Q8KBlock>,
}

impl Q8KStepCache {
    const fn new() -> Self {
        Self {
            active: false,
            valid: false,
            source_ptr: 0,
            source_len: 0,
            blocks: Vec::new(),
        }
    }
}

/// Start the lifetime of the per-forward-step Q8_K activation cache.
pub fn begin_q8k_step_cache() {
    Q8K_STEP_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.active = true;
        cache.valid = false;
    });
}

/// End the lifetime of the per-forward-step Q8_K activation cache.
pub fn end_q8k_step_cache() {
    Q8K_STEP_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.active = false;
        cache.valid = false;
    });
}

fn with_q8k_activations<R>(x: &[f32], n: usize, f: impl FnOnce(&[Q8KBlock]) -> R) -> R {
    let use_cache = Q8K_STEP_CACHE.with(|cache| cache.borrow().active);
    if use_cache {
        Q8K_STEP_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let same_source = cache.valid
                && cache.source_ptr == x.as_ptr() as usize
                && cache.source_len == x.len();
            if !same_source {
                if cache.blocks.len() < n {
                    cache.blocks.resize(n, Q8KBlock::default());
                }
                quantize_row_q8_k(x, &mut cache.blocks[..n]);
                cache.source_ptr = x.as_ptr() as usize;
                cache.source_len = x.len();
                cache.valid = true;
            }
            f(&cache.blocks[..n])
        })
    } else {
        Q8K_SCRATCH.with(|scratch| {
            let mut xq = scratch.replace(Vec::new());
            if xq.len() < n {
                xq.resize(n, Q8KBlock::default());
            }
            quantize_row_q8_k(x, &mut xq[..n]);
            let result = f(&xq[..n]);
            scratch.replace(xq);
            result
        })
    }
}

pub fn quantize_row_q8_k(x: &[f32], out: &mut [Q8KBlock]) {
    debug_assert_eq!(x.len() % QK_K, 0);
    debug_assert_eq!(out.len(), x.len() / QK_K);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature gate above.
            unsafe { quantize_row_q8_k_avx2(x, out) };
            return;
        }
    }
    quantize_row_q8_k_scalar(x, out);
}

fn quantize_row_q8_k_scalar(x: &[f32], out: &mut [Q8KBlock]) {
    for (b, blk) in out.iter_mut().enumerate() {
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let amax = xb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if amax == 0.0 {
            blk.d = 0.0;
            blk.qs = [0i8; QK_K];
            blk.bsums = [0i16; QK_K / 16];
            continue;
        }
        let iscale = 127.0 / amax;
        for i in 0..QK_K {
            let v = (xb[i] * iscale).round() as i32;
            blk.qs[i] = v.clamp(-128, 127) as i8;
        }
        for g in 0..QK_K / 16 {
            let mut s = 0i32;
            for l in 0..16 {
                s += blk.qs[g * 16 + l] as i32;
            }
            blk.bsums[g] = s as i16;
        }
        blk.d = amax / 127.0;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_q8_k_avx2(x: &[f32], out: &mut [Q8KBlock]) {
    // SAFETY: caller gated on AVX2; Rust 2024 requires unsafe ops in unsafe fn bodies.
    unsafe {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let n_blocks = out.len();
        for b in 0..n_blocks {
            let xb = &x[b * QK_K..(b + 1) * QK_K];
            let blk = &mut out[b];

            let mut vmax = _mm256_setzero_ps();
            let sign = _mm256_set1_ps(-0.0);
            let mut i = 0usize;
            while i + 8 <= QK_K {
                let v = _mm256_loadu_ps(xb.as_ptr().add(i));
                let a = _mm256_andnot_ps(sign, v);
                vmax = _mm256_max_ps(vmax, a);
                i += 8;
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), vmax);
            let amax = tmp.iter().copied().fold(0.0f32, f32::max);
            if amax == 0.0 {
                blk.d = 0.0;
                blk.qs = [0i8; QK_K];
                blk.bsums = [0i16; QK_K / 16];
                continue;
            }
            let iscale = 127.0 / amax;
            let vscale = _mm256_set1_ps(iscale);
            i = 0;
            while i + 8 <= QK_K {
                let v = _mm256_mul_ps(_mm256_loadu_ps(xb.as_ptr().add(i)), vscale);
                let vi = _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT));
                let lo = _mm256_castsi256_si128(vi);
                let hi = _mm256_extracti128_si256(vi, 1);
                let packed16 = _mm_packs_epi32(lo, hi);
                let packed8 = _mm_packs_epi16(packed16, packed16);
                let mut bytes = [0i8; 16];
                _mm_storeu_si128(bytes.as_mut_ptr().cast(), packed8);
                for k in 0..8 {
                    blk.qs[i + k] = bytes[k];
                }
                i += 8;
            }
            for g in 0..QK_K / 16 {
                let mut s = 0i32;
                for l in 0..16 {
                    s += blk.qs[g * 16 + l] as i32;
                }
                blk.bsums[g] = s as i16;
            }
            blk.d = amax / 127.0;
        }
    }
}

/// int8·int8 dot with i32 accumulation over `n` elements.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature gate above.
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let mut acc = 0i32;
    let mut i = 0;
    while i + 8 <= n {
        for k in 0..8 {
            acc += a[i + k] as i32 * b[i + k] as i32;
        }
        i += 8;
    }
    while i < n {
        acc += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    // SAFETY: caller gated on AVX2; Rust 2024 requires unsafe ops in unsafe fn bodies.
    unsafe {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let n = a.len();
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            let p_lo = _mm256_madd_epi16(a_lo, b_lo);
            let p_hi = _mm256_madd_epi16(a_hi, b_hi);
            acc = _mm256_add_epi32(acc, p_lo);
            acc = _mm256_add_epi32(acc, p_hi);
            i += 32;
        }
        let hi = _mm256_extracti128_si256(acc, 1);
        let lo = _mm256_castsi256_si128(acc);
        let sum2 = _mm_add_epi32(lo, hi);
        let shuf = _mm_shuffle_epi32(sum2, 0b01_00_11_10);
        let sum4 = _mm_add_epi32(sum2, shuf);
        let shuf2 = _mm_shuffle_epi32(sum4, 0b10_11_00_01);
        let sum = _mm_add_epi32(sum4, shuf2);
        let mut total = _mm_cvtsi128_si32(sum);
        while i < n {
            total += a[i] as i32 * b[i] as i32;
            i += 1;
        }
        total
    }
}

/// f32 weight, f32 input -> f32 output. Row-major weight.
pub fn matvec_f32(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        *yi = dot_f32(row, x);
    });
}

/// One batched matvec descriptor.
pub struct MatDesc<'a> {
    /// Quantized weight bytes for this matrix.
    pub w: &'a [u8],
    /// Weight dtype (`Q4K` / `Q6K`).
    pub dtype: DType,
    /// Number of output rows.
    pub out_dim: usize,
    /// Number of input columns (must be a multiple of [`QK_K`]).
    pub in_dim: usize,
    /// Offset into the shared input buffer `x` where this matrix's input starts.
    pub x_off: usize,
}

pub fn matvec_quant_multi(x: &[f32], mats: &[MatDesc<'_>], y: &mut [f32]) -> Result<()> {
    let n_mats = mats.len();
    let mut y_base = Vec::with_capacity(n_mats);
    let mut total = 0usize;
    for m in mats {
        y_base.push(total);
        total += m.out_dim;
    }
    if total != y.len() {
        return Err(FellmError::other("matvec_multi: y length mismatch"));
    }

    let mut pool: Vec<Q8KBlock> = Vec::new();
    let mut xq_base: Vec<usize> = Vec::with_capacity(n_mats);
    for m in mats {
        if m.in_dim % QK_K != 0 {
            return Err(FellmError::other(format!(
                "matvec_multi: in_dim {} not multiple of {QK_K}",
                m.in_dim
            )));
        }
        if m.x_off + m.in_dim > x.len() {
            return Err(FellmError::other("matvec_multi: input slice out of bounds"));
        }
        let dup = mats
            .iter()
            .take(n_mats)
            .enumerate()
            .find(|(j, pm)| *j < xq_base.len() && pm.in_dim == m.in_dim && pm.x_off == m.x_off)
            .map(|(j, _)| xq_base[j]);
        let off = match dup {
            Some(o) => o,
            None => {
                let o = pool.len();
                let blocks = m.in_dim / QK_K;
                pool.resize(o + blocks, Q8KBlock::default());
                quantize_row_q8_k(&x[m.x_off..m.x_off + m.in_dim], &mut pool[o..o + blocks]);
                o
            }
        };
        xq_base.push(off);
    }
    let pool = &pool;

    let n_threads = rayon::current_num_threads().max(1);
    let chunk = (total / n_threads).max(16);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let row0 = ci * chunk;
            // Find the matrix containing row0 by prefix-sum walk.
            let mut mi = 0;
            while mi + 1 < n_mats && y_base[mi + 1] <= row0 {
                mi += 1;
            }
            let mut local = row0 - y_base[mi];
            for yi in y_chunk.iter_mut() {
                while mi + 1 < n_mats && local >= mats[mi].out_dim {
                    mi += 1;
                    local = 0;
                }
                let m = &mats[mi];
                let row = &m.w[local * m.dtype.byte_size(m.in_dim)
                    ..(local + 1) * m.dtype.byte_size(m.in_dim)];
                let xq = &pool[xq_base[mi]..xq_base[mi] + m.in_dim / QK_K];
                *yi = match m.dtype {
                    DType::Q4K => vec_dot_q4_k_q8_k_row(row, xq),
                    DType::Q6K => {
                        let mut acc = 0.0f32;
                        let block_bytes = DType::Q6K.bytes_per_block();
                        for (b, xb) in xq.iter().enumerate() {
                            let block = &row[b * block_bytes..(b + 1) * block_bytes];
                            acc += fused_q6_k_block(block, xb);
                        }
                        acc
                    }
                    other => unreachable!("matvec_multi: unsupported dtype {other:?}"),
                };
                local += 1;
            }
        });
    Ok(())
}

/// Quantized weight matvec with in-register fused dequant (no f32 row materialization).
pub fn matvec_quant(
    w_bytes: &[u8],
    w_dtype: DType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);
    let bytes_per_row = w_dtype.byte_size(in_dim);
    debug_assert_eq!(w_bytes.len(), out_dim * bytes_per_row);

    match w_dtype {
        DType::Q4_0 => {
            if !in_dim.is_multiple_of(QK4_0) {
                return Err(FellmError::other("Q4_0: in_dim not multiple of 32"));
            }
            matvec_q4_0(w_bytes, x, y, out_dim, in_dim, bytes_per_row);
        }
        DType::Q8_0 => {
            if !in_dim.is_multiple_of(QK8_0) {
                return Err(FellmError::other("Q8_0: in_dim not multiple of 32"));
            }
            matvec_q8_0(w_bytes, x, y, out_dim, in_dim, bytes_per_row);
        }
        DType::Q4K => {
            if !in_dim.is_multiple_of(QK_K) {
                return Err(FellmError::other("Q4_K: in_dim not multiple of 256"));
            }
            let n = in_dim / QK_K;
            with_q8k_activations(x, n, |xq| {
                matvec_q4_k(w_bytes, xq, y, out_dim, in_dim, bytes_per_row);
            });
        }
        DType::Q6K => {
            if !in_dim.is_multiple_of(QK_K) {
                return Err(FellmError::other("Q6_K: in_dim not multiple of 256"));
            }
            let n = in_dim / QK_K;
            with_q8k_activations(x, n, |xq| {
                matvec_q6_k(w_bytes, xq, y, out_dim, in_dim, bytes_per_row);
            });
        }
        other => return Err(FellmError::UnsupportedDType(other)),
    }
    Ok(())
}

fn matvec_q4_0(
    w_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
) {
    let block_bytes = DType::Q4_0.bytes_per_block();
    let n_blocks = in_dim / QK4_0;
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let base = b * block_bytes;
            let d = f16::from_bits(u16::from_le_bytes([row[base], row[base + 1]])).to_f32();
            let qs = &row[base + 2..base + 2 + 16];
            let x0 = &x[b * QK4_0..b * QK4_0 + 16];
            let x1 = &x[b * QK4_0 + 16..b * QK4_0 + 32];
            let mut sum = 0.0f32;
            let mut j = 0;
            while j + 8 <= 16 {
                let mut lo = [0.0f32; 8];
                let mut hi = [0.0f32; 8];
                for k in 0..8 {
                    let byte = qs[j + k];
                    lo[k] = ((byte & 0x0F) as i32 - 8) as f32;
                    hi[k] = ((byte >> 4) as i32 - 8) as f32;
                }
                sum += (f32x8::from(lo)
                    * f32x8::from(*<&[f32; 8]>::try_from(&x0[j..j + 8]).unwrap()))
                .reduce_add();
                sum += (f32x8::from(hi)
                    * f32x8::from(*<&[f32; 8]>::try_from(&x1[j..j + 8]).unwrap()))
                .reduce_add();
                j += 8;
            }
            acc += d * sum;
        }
        *yi = acc;
    });
}

fn matvec_q8_0(
    w_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
) {
    let block_bytes = DType::Q8_0.bytes_per_block();
    let n_blocks = in_dim / QK8_0;
    let mut xq = Q80_SCRATCH.with(|c| c.replace(Vec::new()));
    if xq.len() < n_blocks {
        xq.resize(n_blocks, Q80XBlock::default());
    }
    for b in 0..n_blocks {
        let xb = &x[b * QK8_0..(b + 1) * QK8_0];
        let amax = xb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let out = &mut xq[b];
        if amax == 0.0 {
            out.d = 0.0;
            out.qs = [0i8; QK8_0];
            continue;
        }
        let iscale = 127.0 / amax;
        for i in 0..QK8_0 {
            out.qs[i] = ((xb[i] * iscale).round() as i32).clamp(-128, 127) as i8;
        }
        out.d = amax / 127.0;
    }
    let chunk = 64usize.max(y.len() / (rayon::current_num_threads() * 4).max(1));
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let row0 = ci * chunk;
            for (j, yi) in y_chunk.iter_mut().enumerate() {
                let i = row0 + j;
                let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
                let mut acc = 0.0f32;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let d = f16::from_bits(u16::from_le_bytes([row[base], row[base + 1]])).to_f32();
                    let wqs: &[i8] = bytemuck::cast_slice(&row[base + 2..base + 2 + 32]);
                    let xb = &xq[b];
                    let dot = dot_i8(wqs, &xb.qs);
                    acc += d * xb.d * dot as f32;
                }
                *yi = acc;
            }
        });
    Q80_SCRATCH.with(|c| {
        c.replace(xq);
    });
}

fn matvec_q4_k(
    w_bytes: &[u8],
    xq: &[Q8KBlock],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
) {
    let n_blocks = in_dim / QK_K;
    debug_assert_eq!(xq.len(), n_blocks);
    // Chunk by physical-core count so each worker streams many rows over the
    // same quantized activation (llama.cpp / ggml style). Avoid oversubscription.
    let profile = crate::cpu_profile::CpuHardwareProfile::get();
    let n_threads = if y.len() >= 16_384 {
        profile.physical_cores.saturating_add(4)
    } else {
        profile.physical_cores
    }
    .max(1);
    let chunk = (y.len() / n_threads).max(16);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let row0 = ci * chunk;
            let n = y_chunk.len();
            for j in 0..n {
                let i = row0 + j;
                let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
                // Prefetch the next weight row into L2 while computing current.
                if j + 1 < n {
                    let next = w_bytes.as_ptr().wrapping_add((i + 1) * bytes_per_row);
                    #[cfg(target_arch = "x86_64")]
                    {
                        // SAFETY: prefetch is a hint; address need not be dereferenceable.
                        unsafe {
                            core::arch::x86_64::_mm_prefetch(
                                next as *const i8,
                                core::arch::x86_64::_MM_HINT_T1,
                            );
                        }
                    }
                    #[cfg(target_arch = "x86")]
                    {
                        unsafe {
                            core::arch::x86::_mm_prefetch(
                                next as *const i8,
                                core::arch::x86::_MM_HINT_T1,
                            );
                        }
                    }
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                        let _ = next;
                    }
                }
                y_chunk[j] = vec_dot_q4_k_q8_k_row(row, xq);
            }
        });
}

fn matvec_q6_k(
    w_bytes: &[u8],
    xq: &[Q8KBlock],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
) {
    let block_bytes = DType::Q6K.bytes_per_block();
    let n_blocks = in_dim / QK_K;
    let chunk = 64usize.max(y.len() / (rayon::current_num_threads() * 4).max(1));
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let row0 = ci * chunk;
            for (j, yi) in y_chunk.iter_mut().enumerate() {
                let i = row0 + j;
                let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
                let mut acc = 0.0f32;
                for b in 0..n_blocks {
                    let block = &row[b * block_bytes..(b + 1) * block_bytes];
                    acc += fused_q6_k_block(block, &xq[b]);
                }
                *yi = acc;
            }
        });
}

/// Fused Q6_K block · quantized-x (matches ggml `dequantize_row_q6_K` layout).
///
/// Uses int8·int8 dots: unpack each 6-bit weight to `q-32` (i8 range) and
/// multiply by the i8 activation, accumulating in i32 per scale group, then
/// scale by `d*sc*xd`.
fn fused_q6_k_block(block: &[u8], xb: &Q8KBlock) -> f32 {
    debug_assert_eq!(block.len(), DType::Q6K.bytes_per_block());

    let ql_all = &block[0..128];
    let qh_all = &block[128..192];
    let scales: &[i8] = bytemuck::cast_slice(&block[192..208]);
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
    let xd = xb.d;
    let xqs = &xb.qs;

    let mut acc = 0.0f32;
    let mut y_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    for _ in 0..2 {
        let ql = &ql_all[ql_off..ql_off + 64];
        let qh = &qh_all[qh_off..qh_off + 32];
        let sc = &scales[sc_off..sc_off + 8];

        // Each of the four 32-weight lanes spans two 16-elem scale groups
        // (is = l/16). Accumulate an i32 dot per (lane, group) then scale.
        let mut dots = [[0i32; 2]; 4]; // [lane][is]
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) as i32 | (((qh[l]) & 3) as i32) << 4) - 32;
            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;

            dots[0][is] += q1 * xqs[y_off + l] as i32;
            dots[1][is] += q2 * xqs[y_off + l + 32] as i32;
            dots[2][is] += q3 * xqs[y_off + l + 64] as i32;
            dots[3][is] += q4 * xqs[y_off + l + 96] as i32;
        }
        for is in 0..2 {
            acc += d * sc[is] as f32 * dots[0][is] as f32;
            acc += d * sc[is + 2] as f32 * dots[1][is] as f32;
            acc += d * sc[is + 4] as f32 * dots[2][is] as f32;
            acc += d * sc[is + 6] as f32 * dots[3][is] as f32;
        }

        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    acc * xd
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + 8 <= n {
        let av = f32x8::from(*<&[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let bv = f32x8::from(*<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        acc += av * bv;
        i += 8;
    }
    let mut s = acc.reduce_add();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::dequantize_row;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn matvec_ref(
        w_bytes: &[u8],
        w_dtype: DType,
        x: &[f32],
        y: &mut [f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        let bytes_per_row = w_dtype.byte_size(in_dim);
        for i in 0..out_dim {
            let mut scratch = vec![0.0f32; in_dim];
            let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
            dequantize_row(w_dtype, row, &mut scratch, in_dim).unwrap();
            y[i] = dot_f32(&scratch, x);
        }
    }

    fn fill_rng(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            out.push((rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0);
        }
        out
    }

    fn pack_q4_0_from_f32(weights: &[f32]) -> Vec<u8> {
        assert_eq!(weights.len() % QK4_0, 0);
        let n_blocks = weights.len() / QK4_0;
        let mut out = vec![0u8; n_blocks * DType::Q4_0.bytes_per_block()];
        for b in 0..n_blocks {
            let w = &weights[b * QK4_0..(b + 1) * QK4_0];
            let amax = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
            let d = amax / 7.0;
            let id = 1.0 / d;
            let base = b * DType::Q4_0.bytes_per_block();
            let db = f16::from_f32(d).to_bits().to_le_bytes();
            out[base] = db[0];
            out[base + 1] = db[1];
            for i in 0..16 {
                let lo = ((w[i] * id).round() as i32).clamp(-8, 7) + 8;
                let hi = ((w[i + 16] * id).round() as i32).clamp(-8, 7) + 8;
                out[base + 2 + i] = (lo as u8) | ((hi as u8) << 4);
            }
        }
        out
    }

    fn pack_q8_0_from_f32(weights: &[f32]) -> Vec<u8> {
        assert_eq!(weights.len() % QK8_0, 0);
        let n_blocks = weights.len() / QK8_0;
        let mut out = vec![0u8; n_blocks * DType::Q8_0.bytes_per_block()];
        for b in 0..n_blocks {
            let w = &weights[b * QK8_0..(b + 1) * QK8_0];
            let amax = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
            let d = amax / 127.0;
            let id = 1.0 / d;
            let base = b * DType::Q8_0.bytes_per_block();
            let db = f16::from_f32(d).to_bits().to_le_bytes();
            out[base] = db[0];
            out[base + 1] = db[1];
            for i in 0..32 {
                out[base + 2 + i] = ((w[i] * id).round() as i32).clamp(-128, 127) as u8;
            }
        }
        out
    }

    /// Build a Q4_K row with known scales (d=1, dmin=0.1, scales=1.., qs patterned).
    fn make_q4_k_row(n_blocks: usize) -> Vec<u8> {
        let mut out = vec![0u8; n_blocks * DType::Q4K.bytes_per_block()];
        for b in 0..n_blocks {
            let base = b * DType::Q4K.bytes_per_block();
            let d = f16::from_f32(1.0).to_bits().to_le_bytes();
            let dmin = f16::from_f32(0.1).to_bits().to_le_bytes();
            out[base] = d[0];
            out[base + 1] = d[1];
            out[base + 2] = dmin[0];
            out[base + 3] = dmin[1];
            // scales: j<4 → bytes 0..4 scale, 4..8 min
            for j in 0..4 {
                out[base + 4 + j] = (j as u8 + 1) & 63;
                out[base + 4 + j + 4] = (j as u8 + 2) & 63;
            }
            for j in 0..4 {
                let ls = (j as u8 + 5) & 63;
                let lm = (j as u8 + 3) & 63;
                out[base + 4 + j + 8] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                out[base + 4 + j] |= (ls >> 4) << 6;
                out[base + 4 + j + 4] |= (lm >> 4) << 6;
            }
            for i in 0..128 {
                out[base + 16 + i] = ((i * 17 + b * 3) & 0xFF) as u8;
            }
        }
        out
    }

    fn make_q6_k_row(n_blocks: usize) -> Vec<u8> {
        let mut out = vec![0u8; n_blocks * DType::Q6K.bytes_per_block()];
        for b in 0..n_blocks {
            let base = b * DType::Q6K.bytes_per_block();
            for i in 0..128 {
                out[base + i] = ((i + b) & 0xFF) as u8;
            }
            for i in 0..64 {
                out[base + 128 + i] = ((i * 3 + b) & 0xFF) as u8;
            }
            for i in 0..16 {
                out[base + 192 + i] = (i as i8).wrapping_mul(3).wrapping_add(1) as u8;
            }
            let d = f16::from_f32(0.05).to_bits().to_le_bytes();
            out[base + 208] = d[0];
            out[base + 209] = d[1];
        }
        out
    }

    fn check_fused(dtype: DType, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) {
        let mut y_fused = vec![0.0f32; out_dim];
        let mut y_ref = vec![0.0f32; out_dim];
        matvec_quant(w, dtype, x, &mut y_fused, out_dim, in_dim).unwrap();
        matvec_ref(w, dtype, x, &mut y_ref, out_dim, in_dim);
        let err = max_abs_diff(&y_fused, &y_ref);
        let scale = y_ref.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        // Q4_0 stays f32-exact; the int8-activation paths (Q8_0/Q4_K/Q6_K)
        // quantize x to 7 bits, so a small relative error is expected.
        let tol = match dtype {
            DType::Q4_0 => (1e-4_f32).max(1e-5 * scale),
            _ => (1e-3_f32).max(5e-3 * scale),
        };
        assert!(
            err < tol,
            "dtype={dtype:?} max abs err {err} tol={tol} (out={out_dim} in={in_dim})"
        );
    }

    #[test]
    fn matvec_f32_identity() {
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        matvec_f32(&w, &x, &mut y, 4, 4);
        assert_eq!(y, x);
    }

    #[test]
    fn fused_q4_0_matches_ref() {
        let out_dim = 8;
        let in_dim = 64;
        let weights = fill_rng(1, out_dim * in_dim);
        let mut packed = Vec::new();
        for r in 0..out_dim {
            packed.extend(pack_q4_0_from_f32(&weights[r * in_dim..(r + 1) * in_dim]));
        }
        let x = fill_rng(2, in_dim);
        check_fused(DType::Q4_0, &packed, &x, out_dim, in_dim);
    }

    #[test]
    fn fused_q8_0_matches_ref() {
        let out_dim = 8;
        let in_dim = 64;
        let weights = fill_rng(3, out_dim * in_dim);
        let mut packed = Vec::new();
        for r in 0..out_dim {
            packed.extend(pack_q8_0_from_f32(&weights[r * in_dim..(r + 1) * in_dim]));
        }
        let x = fill_rng(4, in_dim);
        check_fused(DType::Q8_0, &packed, &x, out_dim, in_dim);
    }

    #[test]
    fn fused_q4_k_matches_ref() {
        let out_dim = 4;
        let in_dim = 512; // 2 super-blocks
        let w = {
            let mut v = Vec::new();
            for _ in 0..out_dim {
                v.extend(make_q4_k_row(in_dim / QK_K));
            }
            v
        };
        let x = fill_rng(5, in_dim);
        check_fused(DType::Q4K, &w, &x, out_dim, in_dim);
    }

    #[test]
    fn fused_q6_k_matches_ref() {
        let out_dim = 4;
        let in_dim = 256;
        let w = {
            let mut v = Vec::new();
            for _ in 0..out_dim {
                v.extend(make_q6_k_row(in_dim / QK_K));
            }
            v
        };
        let x = fill_rng(6, in_dim);
        check_fused(DType::Q6K, &w, &x, out_dim, in_dim);
    }
}
