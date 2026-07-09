//! Matmul / matvec.
//!
//! Computes `y[i] = sum_j W[i,j] * x[j]` where W is `[out_dim, in_dim]`.
//! Quantized weights are fused: each block is unpacked in-register and
//! immediately FMAd into the accumulator — no f32 row scratch in RAM.

use crate::dequant::{get_scale_min_k4, QK4_0, QK8_0, QK_K};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use half::f16;
use rayon::prelude::*;
use wide::f32x8;

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
            if in_dim % QK4_0 != 0 {
                return Err(FellmError::other("Q4_0: in_dim not multiple of 32"));
            }
            matvec_q4_0(w_bytes, x, y, out_dim, in_dim, bytes_per_row);
        }
        DType::Q8_0 => {
            if in_dim % QK8_0 != 0 {
                return Err(FellmError::other("Q8_0: in_dim not multiple of 32"));
            }
            matvec_q8_0(w_bytes, x, y, out_dim, in_dim, bytes_per_row);
        }
        DType::Q4K => {
            if in_dim % QK_K != 0 {
                return Err(FellmError::other("Q4_K: in_dim not multiple of 256"));
            }
            matvec_q4_k(w_bytes, x, y, out_dim, in_dim, bytes_per_row);
        }
        DType::Q6K => {
            if in_dim % QK_K != 0 {
                return Err(FellmError::other("Q6_K: in_dim not multiple of 256"));
            }
            matvec_q6_k(w_bytes, x, y, out_dim, in_dim, bytes_per_row);
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
            let mut sum_lo = 0.0f32;
            let mut sum_hi = 0.0f32;
            for j in 0..16 {
                let byte = qs[j];
                sum_lo += ((byte & 0x0F) as i32 - 8) as f32 * x0[j];
                sum_hi += ((byte >> 4) as i32 - 8) as f32 * x1[j];
            }
            acc += d * (sum_lo + sum_hi);
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
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let base = b * block_bytes;
            let d = f16::from_bits(u16::from_le_bytes([row[base], row[base + 1]])).to_f32();
            let qs = &row[base + 2..base + 2 + 32];
            let xb = &x[b * QK8_0..(b + 1) * QK8_0];
            let mut sum = 0.0f32;
            let mut j = 0;
            while j + 8 <= 32 {
                let mut wv = [0.0f32; 8];
                for k in 0..8 {
                    wv[k] = (qs[j + k] as i8) as f32;
                }
                sum += (f32x8::from(wv) * f32x8::from(*<&[f32; 8]>::try_from(&xb[j..j + 8]).unwrap()))
                    .reduce_add();
                j += 8;
            }
            while j < 32 {
                sum += (qs[j] as i8) as f32 * xb[j];
                j += 1;
            }
            acc += d * sum;
        }
        *yi = acc;
    });
}

fn matvec_q4_k(
    w_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
) {
    let block_bytes = DType::Q4K.bytes_per_block();
    let n_blocks = in_dim / QK_K;
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let base = b * block_bytes;
            let d = f16::from_bits(u16::from_le_bytes([row[base], row[base + 1]])).to_f32();
            let dmin = f16::from_bits(u16::from_le_bytes([row[base + 2], row[base + 3]])).to_f32();
            let scales_bytes = &row[base + 4..base + 4 + 12];
            let qs = &row[base + 16..base + 16 + 128];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            let mut is = 0usize;
            let mut q_off = 0usize;
            let mut y_off = 0usize;
            for _ in 0..4 {
                let (sc0, m0) = get_scale_min_k4(is, scales_bytes);
                let (sc1, m1) = get_scale_min_k4(is + 1, scales_bytes);
                let d1 = d * sc0 as f32;
                let m1f = dmin * m0 as f32;
                let d2 = d * sc1 as f32;
                let m2f = dmin * m1 as f32;
                let q = &qs[q_off..q_off + 32];
                let x_lo = &x_block[y_off..y_off + 32];
                let x_hi = &x_block[y_off + 32..y_off + 64];

                let mut sum_q_lo = 0.0f32;
                let mut sum_x_lo = 0.0f32;
                let mut sum_q_hi = 0.0f32;
                let mut sum_x_hi = 0.0f32;
                for l in 0..32 {
                    let lo = (q[l] & 0x0F) as f32;
                    let hi = (q[l] >> 4) as f32;
                    sum_q_lo += lo * x_lo[l];
                    sum_x_lo += x_lo[l];
                    sum_q_hi += hi * x_hi[l];
                    sum_x_hi += x_hi[l];
                }
                // w = d*sc*nibble - dmin*m  →  contrib = d*sc*(n·x) - dmin*m*(Σx)
                acc += d1 * sum_q_lo - m1f * sum_x_lo;
                acc += d2 * sum_q_hi - m2f * sum_x_hi;

                q_off += 32;
                y_off += 64;
                is += 2;
            }
        }
        *yi = acc;
    });
}

fn matvec_q6_k(
    w_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
) {
    let block_bytes = DType::Q6K.bytes_per_block();
    let n_blocks = in_dim / QK_K;
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let block = &row[b * block_bytes..(b + 1) * block_bytes];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];
            acc += fused_q6_k_block(block, x_block);
        }
        *yi = acc;
    });
}

/// Fused Q6_K block · x (matches ggml `dequantize_row_q6_K` layout).
fn fused_q6_k_block(block: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(block.len(), DType::Q6K.bytes_per_block());
    debug_assert_eq!(x.len(), QK_K);

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales: &[i8] = bytemuck::cast_slice(&block[192..208]);
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

    let mut acc = 0.0f32;
    let mut y_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    for _ in 0..2 {
        let ql = &ql[ql_off..ql_off + 64];
        let qh = &qh[qh_off..qh_off + 32];
        let sc = &scales[sc_off..sc_off + 8];
        let xb = &x[y_off..y_off + 128];

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;

            acc += d * sc[is] as f32 * q1 as f32 * xb[l];
            acc += d * sc[is + 2] as f32 * q2 as f32 * xb[l + 32];
            acc += d * sc[is + 4] as f32 * q3 as f32 * xb[l + 64];
            acc += d * sc[is + 6] as f32 * q4 as f32 * xb[l + 96];
        }

        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    acc
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
        let scale = y_ref
            .iter()
            .map(|v| v.abs())
            .fold(1.0f32, f32::max);
        // Accumulation order can differ slightly from dequant-then-dot.
        let tol = (1e-4_f32).max(1e-5 * scale);
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
            packed.extend(pack_q4_0_from_f32(
                &weights[r * in_dim..(r + 1) * in_dim],
            ));
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
            packed.extend(pack_q8_0_from_f32(
                &weights[r * in_dim..(r + 1) * in_dim],
            ));
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
