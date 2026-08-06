use crate::dequant::QK_K;
use crate::kernels::matmul::Q8KBlock;
use half::f16;

const KMASK1: u32 = 0x3f3f_3f3f;
const KMASK2: u32 = 0x0f0f_0f0f;
const KMASK3: u32 = 0x0303_0303;

/// Q4_K super-block size in bytes.
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// A Q4_K block decoded once for reuse across multiple activation rows.
///
/// The packed GGUF representation is intentionally kept out of the inner
/// batch loop.  `weights` contains the unsigned nibbles as signed bytes so
/// the existing AVX2 int8 dot helper can be reused; the per-group scales and
/// minima retain the exact Q4_K arithmetic.
#[derive(Clone, Copy)]
pub(crate) struct Q4KBlockCache {
    pub(crate) weights: [i8; QK_K],
    pub(crate) scales: [u8; QK_K / 32],
    pub(crate) mins: [u8; QK_K / 32],
    pub(crate) d: f32,
    pub(crate) dmin: f32,
}

impl Default for Q4KBlockCache {
    fn default() -> Self {
        Self {
            weights: [0; QK_K],
            scales: [0; QK_K / 32],
            mins: [0; QK_K / 32],
            d: 0.0,
            dmin: 0.0,
        }
    }
}

/// Decode the 12-byte packed Q4_K scales+mins into 8 scale and 8 min bytes.
#[inline]
fn decode_scales_mins(scales12: &[u8]) -> ([u8; 8], [u8; 8]) {
    debug_assert!(scales12.len() >= 12);
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes([scales12[0], scales12[1], scales12[2], scales12[3]]);
    utmp[1] = u32::from_le_bytes([scales12[4], scales12[5], scales12[6], scales12[7]]);
    utmp[2] = u32::from_le_bytes([scales12[8], scales12[9], scales12[10], scales12[11]]);

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let s0 = utmp[0].to_le_bytes();
    let s1 = utmp[1].to_le_bytes();
    let m0 = utmp[2].to_le_bytes();
    let m1 = utmp[3].to_le_bytes();
    (
        [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]],
        [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]],
    )
}

/// Same as [`decode_scales_mins`] but returns the four utmp words (ggml layout).
#[inline]
fn decode_utmp(scales12: &[u8]) -> [u32; 4] {
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes([scales12[0], scales12[1], scales12[2], scales12[3]]);
    utmp[1] = u32::from_le_bytes([scales12[4], scales12[5], scales12[6], scales12[7]]);
    utmp[2] = u32::from_le_bytes([scales12[8], scales12[9], scales12[10], scales12[11]]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    utmp
}

/// Unpack Q4_K `qs` (128 bytes) into 256 unsigned nibble weights (ggml aux8 layout).
#[inline]
fn unpack_q4_k_aux8(qs: &[u8], aux8: &mut [u8; QK_K]) {
    debug_assert!(qs.len() >= 128);
    for j in 0..4 {
        let q = &qs[j * 32..j * 32 + 32];
        let chunk = &mut aux8[j * 64..j * 64 + 64];
        let (lo, hi) = chunk.split_at_mut(32);
        for l in 0..32 {
            lo[l] = q[l] & 0x0F;
            hi[l] = q[l] >> 4;
        }
    }
}

/// Decode one Q4_K block into a reusable cache entry.
#[inline]
pub(crate) fn decode_q4_k_block_cached(block: &[u8]) -> Q4KBlockCache {
    debug_assert!(block.len() >= Q4_K_BLOCK_BYTES);
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let (scales, mins) = decode_scales_mins(&block[4..16]);
    let mut aux8 = [0u8; QK_K];
    unpack_q4_k_aux8(&block[16..144], &mut aux8);
    let mut weights = [0i8; QK_K];
    for (dst, src) in weights.iter_mut().zip(aux8) {
        *dst = src as i8;
    }
    Q4KBlockCache {
        weights,
        scales,
        mins,
        d,
        dmin,
    }
}

/// Dot a cached Q4_K block against one Q8_K activation block.
#[inline]
pub(crate) fn dot_q4_k_cached(block: &Q4KBlockCache, x: &Q8KBlock) -> f32 {
    let mut scaled = 0i32;
    for group in 0..QK_K / 32 {
        let start = group * 32;
        scaled += dot_u4_i8_scaled(
            &block.weights[start..start + 32],
            &x.qs[start..start + 32],
            block.scales[group],
        );
    }
    let mut minimum = 0i32;
    for group in 0..QK_K / 16 {
        minimum += x.bsums[group] as i32 * block.mins[group / 2] as i32;
    }
    block.d * x.d * scaled as f32 - block.dmin * x.d * minimum as f32
}

/// Dot 32 unsigned Q4 values with signed Q8 values and apply the Q4 scale.
///
/// Q4_K nibbles are in `[0, 15]`, so AVX2's unsigned-byte × signed-byte
/// multiply-add is exactly the packed arithmetic needed here.  This avoids
/// the signed-byte widening path and matches the hot loop in ggml's Q4_K
/// kernel while still allowing the decoded nibbles to be reused by every
/// canvas row.
#[inline]
fn dot_u4_i8_scaled(a: &[i8], b: &[i8], scale: u8) -> i32 {
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the runtime feature check gates the AVX2 implementation;
        // both slices contain at least 32 bytes.
        return unsafe { dot_u4_i8_scaled_avx2(a, b, scale) };
    }
    let mut sum = 0i32;
    for i in 0..32 {
        sum += a[i] as i32 * b[i] as i32 * scale as i32;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_u4_i8_scaled_avx2(a: &[i8], b: &[i8], scale: u8) -> i32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    // SAFETY: caller is gated on AVX2 and both slices have 32 bytes.
    unsafe {
        let av = _mm256_loadu_si256(a.as_ptr().cast());
        let bv = _mm256_loadu_si256(b.as_ptr().cast());
        let products = _mm256_maddubs_epi16(av, bv);
        let scaled = _mm256_madd_epi16(products, _mm256_set1_epi16(scale as i16));
        let lo = _mm256_castsi256_si128(scaled);
        let hi = _mm256_extracti128_si256(scaled, 1);
        let sum = _mm_add_epi32(lo, hi);
        let sum = _mm_hadd_epi32(sum, sum);
        _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum))
    }
}

/// Dot one Q4_K super-block (144 bytes) against one Q8_K activation block.
#[inline]
pub fn vec_dot_q4_k_q8_k(q4_block: &[u8], q8: &Q8KBlock) -> f32 {
    debug_assert!(q4_block.len() >= Q4_K_BLOCK_BYTES);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature gate above.
            return unsafe { vec_dot_q4_k_q8_k_avx2(q4_block, q8) };
        }
    }
    vec_dot_q4_k_q8_k_scalar(q4_block, q8)
}

/// Dot a full weight row (`n_blocks` × 144 bytes) against quantized activations.
#[inline]
pub fn vec_dot_q4_k_q8_k_row(row: &[u8], xq: &[Q8KBlock]) -> f32 {
    let n_blocks = xq.len();
    debug_assert_eq!(row.len(), n_blocks * Q4_K_BLOCK_BYTES);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: feature gate above.
            return unsafe { vec_dot_q4_k_q8_k_row_avx512(row, xq) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature gate above.
            return unsafe { vec_dot_q4_k_q8_k_row_avx2(row, xq) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature gate above.
            return unsafe { vec_dot_q4_k_q8_k_row_avx2_nofma(row, xq) };
        }
    }
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let base = b * Q4_K_BLOCK_BYTES;
        acc += vec_dot_q4_k_q8_k_scalar(&row[base..base + Q4_K_BLOCK_BYTES], &xq[b]);
    }
    acc
}

/// Portable path matching `ggml_vec_dot_q4_K_q8_K_generic`.
pub fn vec_dot_q4_k_q8_k_scalar(q4_block: &[u8], q8: &Q8KBlock) -> f32 {
    let d = f16::from_bits(u16::from_le_bytes([q4_block[0], q4_block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([q4_block[2], q4_block[3]])).to_f32();
    let (scales, mins) = decode_scales_mins(&q4_block[4..16]);
    let qs = &q4_block[16..144];

    let mut aux8 = [0u8; QK_K];
    unpack_q4_k_aux8(qs, &mut aux8);

    let xd = q8.d;
    let mut sumi_min = 0i32;
    for j in 0..QK_K / 16 {
        sumi_min += q8.bsums[j] as i32 * mins[j / 2] as i32;
    }
    let mut sumf = -dmin * xd * sumi_min as f32;

    let mut sumi = 0i32;
    let mut is = 0usize;
    let mut off = 0usize;
    for _ in 0..QK_K / 32 {
        let scale = scales[is] as i32;
        is += 1;
        let mut dot = 0i32;
        for l in 0..32 {
            dot += aux8[off + l] as i32 * q8.qs[off + l] as i32;
        }
        sumi += scale * dot;
        off += 32;
    }
    sumf += d * xd * sumi as f32;
    sumf
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_avx2 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__m256, __m256i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__m256, __m256i};

    /// ggml `k_shuffle` for `get_scale_shuffle_k4` — 8 × 32-byte patterns.
    #[rustfmt::skip]
    static K_SHUFFLE: [u8; 256] = [
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    ];

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        debug_assert!(i < 8);
        unsafe { _mm256_loadu_si256(K_SHUFFLE.as_ptr().add(i * 32).cast()) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn hsum_float_8(v: __m256) -> f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        let sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        _mm_cvtss_f32(sum)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn vec_dot_q4_k_q8_k_avx2(q4_block: &[u8], q8: &Q8KBlock) -> f32 {
        unsafe { vec_dot_q4_k_q8_k_row_avx2_nofma(q4_block, core::slice::from_ref(q8)) }
    }

    /// Row path with FMA (preferred) — mirrors ggml `__AVX2__` + FMA.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn vec_dot_q4_k_q8_k_row_avx2(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;

            let n_blocks = xq.len();
            let m4 = _mm256_set1_epi8(0x0F_u8 as i8);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            for i in 0..n_blocks {
                let blk = row.as_ptr().add(i * Q4_K_BLOCK_BYTES);
                let d = xq[i].d * f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32();
                // ggml: dmin = -y.d * x.dmin
                let dmin = -xq[i].d
                    * f16::from_bits(u16::from_le_bytes([*blk.add(2), *blk.add(3)])).to_f32();

                let utmp = decode_utmp(core::slice::from_raw_parts(blk.add(4), 12));
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32,
                    utmp[2] as i32,
                    utmp[1] as i32,
                    utmp[0] as i32,
                ));

                let q8sums = _mm256_loadu_si256(xq[i].bsums.as_ptr().cast());
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

                let mut q4 = blk.add(16);
                let mut q8p = xq[i].qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();

                for j in 0..QK_K / 64 {
                    let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q4bits = _mm256_loadu_si256(q4.cast());
                    q4 = q4.add(32);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                    let q8l = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));

                    let q8h = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }

                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }

            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
        }
    }

    /// Row path without FMA requirement.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn vec_dot_q4_k_q8_k_row_avx2_nofma(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;

            let n_blocks = xq.len();
            let m4 = _mm256_set1_epi8(0x0F_u8 as i8);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            for i in 0..n_blocks {
                let blk = row.as_ptr().add(i * Q4_K_BLOCK_BYTES);
                let d = xq[i].d * f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32();
                let dmin = -xq[i].d
                    * f16::from_bits(u16::from_le_bytes([*blk.add(2), *blk.add(3)])).to_f32();

                let utmp = decode_utmp(core::slice::from_raw_parts(blk.add(4), 12));
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32,
                    utmp[2] as i32,
                    utmp[1] as i32,
                    utmp[0] as i32,
                ));

                let q8sums = _mm256_loadu_si256(xq[i].bsums.as_ptr().cast());
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_add_ps(acc_m, _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)));

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

                let mut q4 = blk.add(16);
                let mut q8p = xq[i].qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();

                for j in 0..QK_K / 64 {
                    let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q4bits = _mm256_loadu_si256(q4.cast());
                    q4 = q4.add(32);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                    let q8l = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));

                    let q8h = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }

                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)),
                );
            }

            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86_avx2::{
    vec_dot_q4_k_q8_k_avx2, vec_dot_q4_k_q8_k_row_avx2, vec_dot_q4_k_q8_k_row_avx2_nofma,
};

/// AVX-512 path: same numerics as AVX2, but processes 128 weights per inner step
/// with 512-bit loads (maddubs still on 256-bit halves — no `_mm512_maddubs_epi16`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_avx512 {
    use super::x86_avx2::{get_scale_shuffle_k4, hsum_float_8};
    use super::*;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx2",
        enable = "fma"
    )]
    pub(super) unsafe fn vec_dot_q4_k_q8_k_row_avx512(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe {
            let n_blocks = xq.len();
            let m4 = _mm256_set1_epi8(0x0F_u8 as i8);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            for i in 0..n_blocks {
                let blk = row.as_ptr().add(i * Q4_K_BLOCK_BYTES);
                let d = xq[i].d * f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32();
                let dmin = -xq[i].d
                    * f16::from_bits(u16::from_le_bytes([*blk.add(2), *blk.add(3)])).to_f32();

                let utmp = decode_utmp(core::slice::from_raw_parts(blk.add(4), 12));
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32,
                    utmp[2] as i32,
                    utmp[1] as i32,
                    utmp[0] as i32,
                ));

                let q8sums = _mm256_loadu_si256(xq[i].bsums.as_ptr().cast());
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

                let mut q4 = blk.add(16);
                let mut q8p = xq[i].qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();

                // Two 64-weight chunks per iteration via 512-bit loads → 256-bit halves.
                for j in 0..QK_K / 128 {
                    let q4_512 = _mm512_loadu_si512(q4 as *const __m512i);
                    q4 = q4.add(64);
                    let q8a_512 = _mm512_loadu_si512(q8p as *const __m512i);
                    q8p = q8p.add(64);
                    let q8b_512 = _mm512_loadu_si512(q8p as *const __m512i);
                    q8p = q8p.add(64);

                    let q4_lo = _mm512_castsi512_si256(q4_512);
                    let q4_hi = _mm512_extracti64x4_epi64(q4_512, 1);
                    let q8a_lo = _mm512_castsi512_si256(q8a_512);
                    let q8a_hi = _mm512_extracti64x4_epi64(q8a_512, 1);
                    let q8b_lo = _mm512_castsi512_si256(q8b_512);
                    let q8b_hi = _mm512_extracti64x4_epi64(q8b_512, 1);

                    // Chunk 0 (weights 0..63): q8a = [q8lo0 | q8hi0].
                    let scale_l0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(4 * j));
                    let scale_h0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(4 * j + 1));
                    let q4l0 = _mm256_and_si256(q4_lo, m4);
                    let q4h0 = _mm256_and_si256(_mm256_srli_epi16(q4_lo, 4), m4);
                    let p0 = _mm256_add_epi32(
                        _mm256_madd_epi16(scale_l0, _mm256_maddubs_epi16(q4l0, q8a_lo)),
                        _mm256_madd_epi16(scale_h0, _mm256_maddubs_epi16(q4h0, q8a_hi)),
                    );

                    // Chunk 1 (weights 64..127): q8b = [q8lo1 | q8hi1].
                    let scale_l1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(4 * j + 2));
                    let scale_h1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(4 * j + 3));
                    let q4l1 = _mm256_and_si256(q4_hi, m4);
                    let q4h1 = _mm256_and_si256(_mm256_srli_epi16(q4_hi, 4), m4);
                    let p1 = _mm256_add_epi32(
                        _mm256_madd_epi16(scale_l1, _mm256_maddubs_epi16(q4l1, q8b_lo)),
                        _mm256_madd_epi16(scale_h1, _mm256_maddubs_epi16(q4h1, q8b_hi)),
                    );

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
                }

                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }

            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86_avx512::vec_dot_q4_k_q8_k_row_avx512;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::{dequantize_row, get_scale_min_k4};
    use fellm_core::dtype::DType;

    fn make_q4_k_block() -> Vec<u8> {
        let mut out = vec![0u8; DType::Q4K.bytes_per_block()];
        let d = f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin = f16::from_f32(0.1).to_bits().to_le_bytes();
        out[0] = d[0];
        out[1] = d[1];
        out[2] = dmin[0];
        out[3] = dmin[1];
        for j in 0..4 {
            out[4 + j] = (j as u8 + 1) & 63;
            out[4 + j + 4] = (j as u8 + 2) & 63;
        }
        for j in 0..4 {
            let ls = (j as u8 + 5) & 63;
            let lm = (j as u8 + 3) & 63;
            out[4 + j + 8] = (ls & 0x0F) | ((lm & 0x0F) << 4);
            out[4 + j] |= (ls >> 4) << 6;
            out[4 + j + 4] |= (lm >> 4) << 6;
        }
        for i in 0..128 {
            out[16 + i] = ((i * 17) & 0xFF) as u8;
        }
        out
    }

    #[test]
    fn decode_scales_mins_matches_get_scale_min_k4() {
        let block = make_q4_k_block();
        let scales12 = &block[4..16];
        let (scales, mins) = decode_scales_mins(scales12);
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, scales12);
            assert_eq!(scales[j], sc, "scale[{j}]");
            assert_eq!(mins[j], m, "min[{j}]");
        }
    }

    #[test]
    fn vec_dot_q4_k_matches_dequant_dot() {
        let block = make_q4_k_block();
        let mut x = [0.0f32; QK_K];
        let mut rng = 42u64;
        for v in &mut x {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0;
        }
        let mut xq = [Q8KBlock::default()];
        crate::kernels::matmul::quantize_row_q8_k(&x, &mut xq);

        let got = vec_dot_q4_k_q8_k(&block, &xq[0]);
        let got_scalar = vec_dot_q4_k_q8_k_scalar(&block, &xq[0]);
        assert!(
            (got - got_scalar).abs() < 1e-3,
            "avx2/scalar mismatch: {got} vs {got_scalar}"
        );

        let mut w = [0.0f32; QK_K];
        dequantize_row(DType::Q4K, &block, &mut w, QK_K).unwrap();
        let mut ref_dot = 0.0f32;
        for i in 0..QK_K {
            ref_dot += w[i] * (xq[0].d * xq[0].qs[i] as f32);
        }
        let err = (got - ref_dot).abs();
        let scale = ref_dot.abs().max(1.0);
        assert!(
            err < 1e-3 * scale + 1e-3,
            "vec_dot={got} ref={ref_dot} err={err}"
        );
    }

    #[test]
    fn row_dot_matches_block_sum() {
        let mut row = Vec::new();
        row.extend(make_q4_k_block());
        let mut b2 = make_q4_k_block();
        b2[16] = 0xAB;
        row.extend(b2);
        let mut x = [0.0f32; QK_K * 2];
        let mut rng = 7u64;
        for v in &mut x {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0;
        }
        let mut xq = vec![Q8KBlock::default(); 2];
        crate::kernels::matmul::quantize_row_q8_k(&x, &mut xq);
        let row_sum = vec_dot_q4_k_q8_k_row(&row, &xq);
        let block_sum = vec_dot_q4_k_q8_k_scalar(&row[0..144], &xq[0])
            + vec_dot_q4_k_q8_k_scalar(&row[144..288], &xq[1]);
        assert!(
            (row_sum - block_sum).abs() < 1e-2,
            "row={row_sum} blocks={block_sum}"
        );
    }
}
