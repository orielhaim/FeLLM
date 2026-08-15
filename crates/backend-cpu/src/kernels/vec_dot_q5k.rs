use crate::dequant::QK_K;
use crate::kernels::matmul::Q8KBlock;
use crate::kernels::vec_dot_q4k::{decode_scales_mins, decode_utmp};
use half::f16;
use std::sync::OnceLock;

type RowDotFn = unsafe fn(&[u8], &[Q8KBlock]) -> f32;
type Rows4DotFn = unsafe fn([&[u8]; 4], &[Q8KBlock]) -> [f32; 4];

/// Q5_K super-block size in bytes (`block_q5_K`).
pub const Q5_K_BLOCK_BYTES: usize = 176;

fn resolve_row_dot() -> RowDotFn {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return vec_dot_q5_k_q8_k_row_avx512;
        }
        if is_x86_feature_detected!("avxvnni")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return vec_dot_q5_k_q8_k_row_vnni;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return vec_dot_q5_k_q8_k_row_avx2;
        }
        if is_x86_feature_detected!("avx2") {
            return vec_dot_q5_k_q8_k_row_avx2_nofma;
        }
    }
    vec_dot_q5_k_q8_k_row_scalar_ptr
}

fn resolve_rows4_dot() -> Rows4DotFn {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avxvnni")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return vec_dot_q5_k_q8_k_4rows_vnni;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return vec_dot_q5_k_q8_k_4rows_avx2;
        }
        if is_x86_feature_detected!("avx2") {
            return vec_dot_q5_k_q8_k_4rows_avx2_nofma;
        }
    }
    vec_dot_q5_k_q8_k_4rows_scalar_ptr
}

fn row_dot_fn() -> RowDotFn {
    static FN: OnceLock<RowDotFn> = OnceLock::new();
    *FN.get_or_init(resolve_row_dot)
}

fn rows4_dot_fn() -> Rows4DotFn {
    static FN: OnceLock<Rows4DotFn> = OnceLock::new();
    *FN.get_or_init(resolve_rows4_dot)
}

unsafe fn vec_dot_q5_k_q8_k_row_scalar_ptr(row: &[u8], xq: &[Q8KBlock]) -> f32 {
    let n_blocks = xq.len();
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let base = b * Q5_K_BLOCK_BYTES;
        acc += vec_dot_q5_k_q8_k_scalar(&row[base..base + Q5_K_BLOCK_BYTES], &xq[b]);
    }
    acc
}

unsafe fn vec_dot_q5_k_q8_k_4rows_scalar_ptr(rows: [&[u8]; 4], xq: &[Q8KBlock]) -> [f32; 4] {
    unsafe {
        [
            vec_dot_q5_k_q8_k_row_scalar_ptr(rows[0], xq),
            vec_dot_q5_k_q8_k_row_scalar_ptr(rows[1], xq),
            vec_dot_q5_k_q8_k_row_scalar_ptr(rows[2], xq),
            vec_dot_q5_k_q8_k_row_scalar_ptr(rows[3], xq),
        ]
    }
}

/// A Q5_K block decoded once for reuse across multiple activation rows.
#[derive(Clone, Copy)]
pub(crate) struct Q5KBlockCache {
    pub(crate) weights: [i8; QK_K],
    pub(crate) scales: [u8; QK_K / 32],
    pub(crate) mins: [u8; QK_K / 32],
    pub(crate) d: f32,
    pub(crate) dmin: f32,
}

impl Default for Q5KBlockCache {
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

#[inline]
fn unpack_q5_k_aux8(qs: &[u8], qh: &[u8], aux8: &mut [u8; QK_K]) {
    debug_assert!(qs.len() >= 128);
    debug_assert!(qh.len() >= 32);
    let mut m = 1u8;
    let mut a = 0usize;
    let mut q_off = 0usize;
    for _ in 0..QK_K / 64 {
        for l in 0..32 {
            aux8[a + l] = (qs[q_off + l] & 0x0F) + if qh[l] & m != 0 { 16 } else { 0 };
        }
        a += 32;
        m <<= 1;
        for l in 0..32 {
            aux8[a + l] = (qs[q_off + l] >> 4) + if qh[l] & m != 0 { 16 } else { 0 };
        }
        a += 32;
        m <<= 1;
        q_off += 32;
    }
}

#[inline]
pub(crate) fn decode_q5_k_block_cached(block: &[u8]) -> Q5KBlockCache {
    debug_assert!(block.len() >= Q5_K_BLOCK_BYTES);
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let (scales, mins) = decode_scales_mins(&block[4..16]);
    let mut aux8 = [0u8; QK_K];
    unpack_q5_k_aux8(&block[48..176], &block[16..48], &mut aux8);
    let mut weights = [0i8; QK_K];
    for (dst, src) in weights.iter_mut().zip(aux8) {
        *dst = src as i8;
    }
    Q5KBlockCache {
        weights,
        scales,
        mins,
        d,
        dmin,
    }
}

#[inline]
pub(crate) fn dot_q5_k_cached(block: &Q5KBlockCache, x: &Q8KBlock) -> f32 {
    let mut scaled = 0i32;
    for group in 0..QK_K / 32 {
        let start = group * 32;
        let mut dot = 0i32;
        for l in 0..32 {
            dot += block.weights[start + l] as i32 * x.qs[start + l] as i32;
        }
        scaled += dot * block.scales[group] as i32;
    }
    let mut minimum = 0i32;
    for group in 0..QK_K / 16 {
        minimum += x.bsums[group] as i32 * block.mins[group / 2] as i32;
    }
    block.d * x.d * scaled as f32 - block.dmin * x.d * minimum as f32
}

#[inline]
pub fn vec_dot_q5_k_q8_k_row(row: &[u8], xq: &[Q8KBlock]) -> f32 {
    let n_blocks = xq.len();
    debug_assert_eq!(row.len(), n_blocks * Q5_K_BLOCK_BYTES);
    unsafe { row_dot_fn()(row, xq) }
}

#[inline]
pub fn vec_dot_q5_k_q8_k_4rows(rows: [&[u8]; 4], xq: &[Q8KBlock]) -> [f32; 4] {
    let n_blocks = xq.len();
    for row in &rows {
        debug_assert_eq!(row.len(), n_blocks * Q5_K_BLOCK_BYTES);
    }
    unsafe { rows4_dot_fn()(rows, xq) }
}

/// Portable path matching `ggml_vec_dot_q5_K_q8_K_generic`.
pub fn vec_dot_q5_k_q8_k_scalar(q5_block: &[u8], q8: &Q8KBlock) -> f32 {
    let d = f16::from_bits(u16::from_le_bytes([q5_block[0], q5_block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([q5_block[2], q5_block[3]])).to_f32();
    let (scales, mins) = decode_scales_mins(&q5_block[4..16]);
    let mut aux8 = [0u8; QK_K];
    unpack_q5_k_aux8(&q5_block[48..176], &q5_block[16..48], &mut aux8);

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
    use crate::kernels::vec_dot_q4k::x86_avx2::{get_scale_shuffle_k4, hsum_float_8};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_row_avx2(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe {
            let n_blocks = xq.len();
            let m4 = _mm256_set1_epi8(0x0F_u8 as i8);
            let mone = _mm256_set1_epi8(1);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            for i in 0..n_blocks {
                let blk = row.as_ptr().add(i * Q5_K_BLOCK_BYTES);
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

                let hbits = _mm256_loadu_si256(blk.add(16).cast());
                let mut hmask = mone;
                let mut bit = 0u32;
                let mut q5 = blk.add(48);
                let mut q8p = xq[i].qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();

                for j in 0..QK_K / 64 {
                    let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q5bits = _mm256_loadu_si256(q5.cast());
                    q5 = q5.add(32);
                    let q5l = _mm256_and_si256(q5bits, m4);
                    let q5h0 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit as i32)),
                        4,
                    );
                    let q5_0 = _mm256_add_epi8(q5l, q5h0);
                    bit += 1;
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q5n = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                    let q5h1 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit as i32)),
                        4,
                    );
                    let q5_1 = _mm256_add_epi8(q5n, q5h1);
                    bit += 1;
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q8l = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q5_0, q8l));

                    let q8h = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q5_1, q8h));

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }

                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }

            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_row_avx2_nofma(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe {
            let n_blocks = xq.len();
            let m4 = _mm256_set1_epi8(0x0F_u8 as i8);
            let mone = _mm256_set1_epi8(1);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            for i in 0..n_blocks {
                let blk = row.as_ptr().add(i * Q5_K_BLOCK_BYTES);
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

                let hbits = _mm256_loadu_si256(blk.add(16).cast());
                let mut hmask = mone;
                let mut bit = 0u32;
                let mut q5 = blk.add(48);
                let mut q8p = xq[i].qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();

                for j in 0..QK_K / 64 {
                    let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q5bits = _mm256_loadu_si256(q5.cast());
                    q5 = q5.add(32);
                    let q5l = _mm256_and_si256(q5bits, m4);
                    let q5h0 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit as i32)),
                        4,
                    );
                    let q5_0 = _mm256_add_epi8(q5l, q5h0);
                    bit += 1;
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q5n = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                    let q5h1 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit as i32)),
                        4,
                    );
                    let q5_1 = _mm256_add_epi8(q5n, q5h1);
                    bit += 1;
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q8l = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q5_0, q8l));

                    let q8h = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q5_1, q8h));

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

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_4rows_avx2(
        rows: [&[u8]; 4],
        xq: &[Q8KBlock],
    ) -> [f32; 4] {
        unsafe {
            [
                vec_dot_q5_k_q8_k_row_avx2(rows[0], xq),
                vec_dot_q5_k_q8_k_row_avx2(rows[1], xq),
                vec_dot_q5_k_q8_k_row_avx2(rows[2], xq),
                vec_dot_q5_k_q8_k_row_avx2(rows[3], xq),
            ]
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_4rows_avx2_nofma(
        rows: [&[u8]; 4],
        xq: &[Q8KBlock],
    ) -> [f32; 4] {
        unsafe {
            [
                vec_dot_q5_k_q8_k_row_avx2_nofma(rows[0], xq),
                vec_dot_q5_k_q8_k_row_avx2_nofma(rows[1], xq),
                vec_dot_q5_k_q8_k_row_avx2_nofma(rows[2], xq),
                vec_dot_q5_k_q8_k_row_avx2_nofma(rows[3], xq),
            ]
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86_avx2::{
    vec_dot_q5_k_q8_k_4rows_avx2, vec_dot_q5_k_q8_k_4rows_avx2_nofma, vec_dot_q5_k_q8_k_row_avx2,
    vec_dot_q5_k_q8_k_row_avx2_nofma,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_vnni {
    use super::*;
    use crate::kernels::vec_dot_q4k::x86_avx2::hsum_float_8;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2", enable = "avxvnni", enable = "fma")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_row_vnni(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe {
            let n_blocks = xq.len();
            let m4 = _mm256_set1_epi8(0x0F_u8 as i8);
            let mone = _mm256_set1_epi8(1);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();

            for i in 0..n_blocks {
                let blk = row.as_ptr().add(i * Q5_K_BLOCK_BYTES);
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
                let scales_i32 = _mm256_cvtepu16_epi32(sc128);

                let hbits = _mm256_loadu_si256(blk.add(16).cast());
                let mut hmask = mone;
                let mut bit = 0u32;
                let mut q5 = blk.add(48);
                let mut q8p = xq[i].qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();

                for j in 0..QK_K / 64 {
                    let s_lo =
                        _mm256_permutevar8x32_epi32(scales_i32, _mm256_set1_epi32((2 * j) as i32));
                    let s_hi = _mm256_permutevar8x32_epi32(
                        scales_i32,
                        _mm256_set1_epi32((2 * j + 1) as i32),
                    );

                    let q5bits = _mm256_loadu_si256(q5.cast());
                    q5 = q5.add(32);
                    let q5l = _mm256_and_si256(q5bits, m4);
                    let q5h0 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit as i32)),
                        4,
                    );
                    let q5_0 = _mm256_add_epi8(q5l, q5h0);
                    bit += 1;
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q5n = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                    let q5h1 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit as i32)),
                        4,
                    );
                    let q5_1 = _mm256_add_epi8(q5n, q5h1);
                    bit += 1;
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q8l = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p_lo = _mm256_mullo_epi32(
                        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q5_0, q8l),
                        s_lo,
                    );

                    let q8h = _mm256_loadu_si256(q8p.cast());
                    q8p = q8p.add(32);
                    let p_hi = _mm256_mullo_epi32(
                        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q5_1, q8h),
                        s_hi,
                    );

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p_lo, p_hi));
                }

                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }

            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
        }
    }

    #[target_feature(enable = "avx2", enable = "avxvnni", enable = "fma")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_4rows_vnni(
        rows: [&[u8]; 4],
        xq: &[Q8KBlock],
    ) -> [f32; 4] {
        unsafe {
            [
                vec_dot_q5_k_q8_k_row_vnni(rows[0], xq),
                vec_dot_q5_k_q8_k_row_vnni(rows[1], xq),
                vec_dot_q5_k_q8_k_row_vnni(rows[2], xq),
                vec_dot_q5_k_q8_k_row_vnni(rows[3], xq),
            ]
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86_vnni::{vec_dot_q5_k_q8_k_4rows_vnni, vec_dot_q5_k_q8_k_row_vnni};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_avx512 {
    use super::*;

    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx2",
        enable = "fma"
    )]
    pub(super) unsafe fn vec_dot_q5_k_q8_k_row_avx512(row: &[u8], xq: &[Q8KBlock]) -> f32 {
        unsafe { super::x86_avx2::vec_dot_q5_k_q8_k_row_avx2(row, xq) }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86_avx512::vec_dot_q5_k_q8_k_row_avx512;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::{QK_K, dequantize_row};
    use crate::kernels::matmul::quantize_row_q8_k;
    use fellm_core::dtype::DType;

    fn make_q5_k_block() -> Vec<u8> {
        let mut out = vec![0u8; Q5_K_BLOCK_BYTES];
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
        for i in 0..32 {
            out[16 + i] = ((i * 13) & 0xFF) as u8;
        }
        for i in 0..128 {
            out[48 + i] = ((i * 17) & 0xFF) as u8;
        }
        out
    }

    #[test]
    fn vec_dot_q5_k_matches_dequant_dot() {
        let block = make_q5_k_block();
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32) * 0.01 - 1.2).collect();
        let mut xq = vec![Q8KBlock::default(); 1];
        quantize_row_q8_k(&x, &mut xq);
        let got = vec_dot_q5_k_q8_k_scalar(&block, &xq[0]);
        let row = vec_dot_q5_k_q8_k_row(&block, &xq);
        let mut w = vec![0.0f32; QK_K];
        dequantize_row(DType::Q5K, &block, &mut w, QK_K).unwrap();
        let expected: f32 = w
            .iter()
            .zip(xq[0].qs.iter())
            .map(|(&weight, &q)| weight * f32::from(q))
            .sum::<f32>()
            * xq[0].d;
        let err = (got - expected).abs();
        let scale = expected.abs().max(1.0);
        assert!(err < 1e-4 * scale, "scalar err={err} got={got} exp={expected}");
        assert!((row - got).abs() < 1e-4 * scale, "row err got={row} scalar={got}");
    }

    fn cuda_q5k_gemv_block(block: &[u8], x: &[f32]) -> f32 {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let (scales, mins) = crate::kernels::vec_dot_q4k::decode_scales_mins(&block[4..16]);
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut acc = 0.0f32;
        for chunk in 0..4 {
            let qbase = chunk * 32;
            let low_bit = 1u8 << (chunk * 2);
            let high_bit = 2u8 << (chunk * 2);
            let mut low_dot = 0.0f32;
            let mut low_sum = 0.0f32;
            let mut high_dot = 0.0f32;
            let mut high_sum = 0.0f32;
            for lane in 0..32 {
                let low_x = x[chunk * 64 + lane];
                let high_x = x[chunk * 64 + 32 + lane];
                let low = (qs[qbase + lane] & 0x0f)
                    + if qh[lane] & low_bit != 0 { 16 } else { 0 };
                let high = (qs[qbase + lane] >> 4)
                    + if qh[lane] & high_bit != 0 { 16 } else { 0 };
                low_dot += low as f32 * low_x;
                low_sum += low_x;
                high_dot += high as f32 * high_x;
                high_sum += high_x;
            }
            let low_group = chunk * 2;
            let high_group = low_group + 1;
            acc += d * scales[low_group] as f32 * low_dot
                - dmin * mins[low_group] as f32 * low_sum;
            acc += d * scales[high_group] as f32 * high_dot
                - dmin * mins[high_group] as f32 * high_sum;
        }
        acc
    }

    #[test]
    fn cuda_q5k_gemv_formula_matches_dequant() {
        let block = make_q5_k_block();
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32) * 0.01 - 1.2).collect();
        let mut w = vec![0.0f32; QK_K];
        dequantize_row(DType::Q5K, &block, &mut w, QK_K).unwrap();
        let expected: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = cuda_q5k_gemv_block(&block, &x);
        let err = (got - expected).abs();
        assert!(err < 1e-3 * expected.abs().max(1.0), "err={err} got={got} exp={expected}");
    }
}
