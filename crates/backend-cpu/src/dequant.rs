//! GGUF k-quant dequantization.
//!
//! Correctness reference: `ggml-quants.c` from ggml-org/ggml.
//!
//! We support the common legacy and K-quant rows used by the bundled models.

use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use half::f16;

/// Number of weights per super-block for k-quants.
pub const QK_K: usize = 256;
/// Number of weights per legacy block.
pub const QK4_0: usize = 32;
/// Number of weights per Q8_0 block.
pub const QK8_0: usize = 32;

/// Dequantize `n_elements` values from `src` into `dst`.
pub fn dequantize_row(dtype: DType, src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    if dst.len() < n_elements {
        return Err(FellmError::other("dst too small"));
    }
    match dtype {
        DType::F32 => {
            let src_f32: &[f32] = bytemuck::cast_slice(&src[..n_elements * 4]);
            dst[..n_elements].copy_from_slice(&src_f32[..n_elements]);
            Ok(())
        }
        DType::F16 => {
            let src_f16: &[f16] = bytemuck::cast_slice(&src[..n_elements * 2]);
            for i in 0..n_elements {
                dst[i] = src_f16[i].to_f32();
            }
            Ok(())
        }
        DType::BF16 => {
            let src_u16: &[u16] = bytemuck::cast_slice(&src[..n_elements * 2]);
            for i in 0..n_elements {
                dst[i] = f32::from_bits((u32::from(src_u16[i])) << 16);
            }
            Ok(())
        }
        DType::Q4_0 => dequantize_q4_0(src, dst, n_elements),
        DType::Q5_0 => dequantize_q5_0(src, dst, n_elements),
        DType::Q8_0 => dequantize_q8_0(src, dst, n_elements),
        DType::Q2K => dequantize_q2_k(src, dst, n_elements),
        DType::Q4K => dequantize_q4_k(src, dst, n_elements),
        DType::Q5K => dequantize_q5_k(src, dst, n_elements),
        DType::Q6K => dequantize_q6_k(src, dst, n_elements),
        DType::IQ2XS | DType::IQ2XXS | DType::IQ3XXS | DType::IQ3S | DType::MXFP4 => {
            crate::iq::dequantize_row(dtype, src, dst, n_elements)
        }
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

// --- Q5_K super-block ---
// Layout: fp16 d, fp16 dmin, 12 packed scale/min bytes, 32 high-bit bytes,
// then 128 low-nibble bytes. This follows ggml's block_q5_K exactly.
fn dequantize_q5_k(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    if !n_elements.is_multiple_of(QK_K) {
        return Err(FellmError::other("Q5_K: n_elements not multiple of 256"));
    }
    let block_bytes = DType::Q5K.bytes_per_block();
    let n_blocks = n_elements / QK_K;
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q5_K: src too small"));
    }
    for block_index in 0..n_blocks {
        let base = block_index * block_bytes;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([src[base + 2], src[base + 3]])).to_f32();
        let scales = &src[base + 4..base + 16];
        let qh = &src[base + 16..base + 48];
        let qs = &src[base + 48..base + 176];
        let out = &mut dst[block_index * QK_K..(block_index + 1) * QK_K];
        let mut scale_index = 0;
        let mut low_offset = 0;
        let mut output_offset = 0;
        let mut low_high_bit = 1_u8;
        let mut high_high_bit = 2_u8;
        for _ in 0..4 {
            let (low_scale, low_min) = get_scale_min_k4(scale_index, scales);
            let (high_scale, high_min) = get_scale_min_k4(scale_index + 1, scales);
            let low = &qs[low_offset..low_offset + 32];
            for index in 0..32 {
                let quant =
                    (low[index] & 0x0f) + if qh[index] & low_high_bit != 0 { 16 } else { 0 };
                out[output_offset + index] =
                    d * f32::from(low_scale) * f32::from(quant) - dmin * f32::from(low_min);
            }
            for index in 0..32 {
                let quant = (low[index] >> 4)
                    + if qh[index] & high_high_bit != 0 {
                        16
                    } else {
                        0
                    };
                out[output_offset + 32 + index] =
                    d * f32::from(high_scale) * f32::from(quant) - dmin * f32::from(high_min);
            }
            scale_index += 2;
            low_offset += 32;
            output_offset += 64;
            low_high_bit <<= 2;
            high_high_bit <<= 2;
        }
    }
    Ok(())
}

// --- Q5_0 ---
// Block layout: fp16 d (2), 4-byte high-bit mask, 16 low-nibble bytes.
// Formula: w[i] = d * (((high_bit << 4) | low_nibble) - 16).
fn dequantize_q5_0(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    if !n_elements.is_multiple_of(QK4_0) {
        return Err(FellmError::other("Q5_0: n_elements not multiple of 32"));
    }
    let block_bytes = DType::Q5_0.bytes_per_block();
    let n_blocks = n_elements / QK4_0;
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q5_0: src too small"));
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qh = u32::from_le_bytes([src[base + 2], src[base + 3], src[base + 4], src[base + 5]]);
        let qs = &src[base + 6..base + 22];
        let out = &mut dst[b * QK4_0..(b + 1) * QK4_0];
        for i in 0..16 {
            let byte = qs[i];
            let hi0 = ((qh >> i) & 1) as i32;
            let hi1 = ((qh >> (i + 16)) & 1) as i32;
            out[i] = d * ((((hi0 << 4) | i32::from(byte & 0x0F)) - 16) as f32);
            out[i + 16] = d * ((((hi1 << 4) | i32::from(byte >> 4)) - 16) as f32);
        }
    }
    Ok(())
}

// --- Q4_0 ---
// Block layout: fp16 d (2 bytes) + 16 bytes of 4-bit weights (32 weights).
// Formula: w[i] = d * (nibble - 8)
fn dequantize_q4_0(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK4_0;
    if !n_elements.is_multiple_of(QK4_0) {
        return Err(FellmError::other("Q4_0: n_elements not multiple of 32"));
    }
    let block_bytes = DType::Q4_0.bytes_per_block();
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q4_0: src too small"));
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qs = &src[base + 2..base + 2 + 16];
        let out = &mut dst[b * QK4_0..(b + 1) * QK4_0];
        for i in 0..16 {
            let byte = qs[i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            out[i] = d * lo as f32;
            out[i + 16] = d * hi as f32;
        }
    }
    Ok(())
}

// --- Q8_0 ---
// Block layout: fp16 d (2 bytes) + 32 int8 weights.
// Formula: w[i] = d * qs[i]
fn dequantize_q8_0(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK8_0;
    if !n_elements.is_multiple_of(QK8_0) {
        return Err(FellmError::other("Q8_0: n_elements not multiple of 32"));
    }
    let block_bytes = DType::Q8_0.bytes_per_block();
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q8_0: src too small"));
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qs = &src[base + 2..base + 2 + 32];
        let out = &mut dst[b * QK8_0..(b + 1) * QK8_0];
        for i in 0..32 {
            out[i] = d * (qs[i] as i8) as f32;
        }
    }
    Ok(())
}

// --- Q4_K super-block ---
// Layout (144 bytes, 256 weights) — matches ggml `block_q4_K`:
//   fp16 d (2)
//   fp16 dmin (2)
//   12 bytes packed scales+mins (6-bit each, 8 pairs)
//   128 bytes of 4-bit weights
//
// Memory layout of qs (from ggml-quants.c `dequantize_row_q4_K`):
//   For each group of 32 qs bytes (covers 64 weights):
//     low  nibbles → 32 consecutive weights with scale/min pair `is`
//     high nibbles → next 32 consecutive weights with scale/min pair `is+1`
//   Then advance qs by 32 and is by 2. Four such groups cover 256 weights.
fn dequantize_q2_k(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    if !n_elements.is_multiple_of(QK_K) {
        return Err(FellmError::other("Q2_K: n_elements not multiple of 256"));
    }
    let block_bytes = DType::Q2K.bytes_per_block();
    let n_blocks = n_elements / QK_K;
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q2_K: src too small"));
    }
    for block_index in 0..n_blocks {
        let base = block_index * block_bytes;
        let scales = &src[base..base + 16];
        let qs = &src[base + 16..base + 80];
        let d = f16::from_bits(u16::from_le_bytes([src[base + 80], src[base + 81]])).to_f32();
        let min = f16::from_bits(u16::from_le_bytes([src[base + 82], src[base + 83]])).to_f32();
        let out = &mut dst[block_index * QK_K..(block_index + 1) * QK_K];
        let mut y = 0usize;
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0xF);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    out[y + l] = dl * f32::from((qs[q_off + l] >> shift) & 3) - ml;
                }
                y += 16;
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0xF);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    out[y + l] = dl * f32::from((qs[q_off + 16 + l] >> shift) & 3) - ml;
                }
                y += 16;
                shift += 2;
            }
            q_off += 32;
        }
    }
    Ok(())
}

fn dequantize_q4_k(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK_K;
    if !n_elements.is_multiple_of(QK_K) {
        return Err(FellmError::other("Q4_K: n_elements not multiple of 256"));
    }
    let block_bytes = DType::Q4K.bytes_per_block();
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q4_K: src too small"));
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([src[base + 2], src[base + 3]])).to_f32();
        let scales_bytes = &src[base + 4..base + 4 + 12];
        let qs = &src[base + 16..base + 16 + 128];
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y_off = 0usize;
        // Four groups of 64 weights each (32 qs bytes per group).
        for _ in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales_bytes);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales_bytes);
            let d1 = d * sc0 as f32;
            let m1f = dmin * m0 as f32;
            let d2 = d * sc1 as f32;
            let m2f = dmin * m1 as f32;
            let q = &qs[q_off..q_off + 32];
            for l in 0..32 {
                out[y_off + l] = d1 * (q[l] & 0x0F) as f32 - m1f;
            }
            for l in 0..32 {
                out[y_off + 32 + l] = d2 * (q[l] >> 4) as f32 - m2f;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
        }
    }
    Ok(())
}

#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert!(j < 8);
    debug_assert!(q.len() >= 12);
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// --- Q6_K super-block ---
// Layout (210 bytes, 256 weights) — matches ggml `block_q6_K`:
//   128 bytes ql (low 4 bits of each 6-bit weight)
//   64  bytes qh (high 2 bits of each 6-bit weight, packed 4 per byte)
//   16  bytes scales (int8, one per sub-block-of-16)
//   2   bytes d (fp16)
//
// Formula: w[i] = d * scale[i/16] * (q[i] - 32)  where q is 6-bit unsigned.
fn dequantize_q6_k(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK_K;
    if !n_elements.is_multiple_of(QK_K) {
        return Err(FellmError::other("Q6_K: n_elements not multiple of 256"));
    }
    let block_bytes = DType::Q6K.bytes_per_block();
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q6_K: src too small"));
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        dequantize_q6_k_block(
            &src[base..base + block_bytes],
            &mut dst[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}

/// Scalar Q6_K block dequant matching ggml-quants.c `dequantize_row_q6_K`.
fn dequantize_q6_k_block(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales: &[i8] = bytemuck::cast_slice(&block[192..208]);
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

    let mut y_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    for _ in 0..2 {
        // two halves of 128 weights each
        let ql = &ql[ql_off..ql_off + 64];
        let qh = &qh[qh_off..qh_off + 32];
        let sc = &scales[sc_off..sc_off + 8];
        let out = &mut out[y_off..y_off + 128];

        for l in 0..32 {
            let is = l / 16; // 0 or 1
            let q1 = ((ql[l] & 0xF) as i32 | ((qh[l] & 3) as i32) << 4) - 32;
            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;

            out[l] = d * sc[is] as f32 * q1 as f32;
            out[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
            out[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
            out[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
        }

        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5_k_matches_ggml_reference_fixture() {
        let source = (0..DType::Q5K.bytes_per_block())
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect::<Vec<_>>();
        let mut output = [0.0_f32; QK_K];
        dequantize_row(DType::Q5K, &source, &mut output, QK_K).unwrap();
        let expected = [
            -2_645_366.25,
            -2_645_472.0,
            -2_645_389.75,
            -2_645_432.75,
            -2_645_350.5,
            -2_645_456.25,
            -2_645_374.0,
            -2_645_417.25,
        ];
        assert_eq!(&output[..expected.len()], &expected);
        assert_eq!(
            output.iter().map(|&value| f64::from(value)).sum::<f64>(),
            -438_199_440.1875
        );
    }

    #[test]
    fn q4_0_roundtrip_zero() {
        // Zero block: d=0, all nibbles = 8 (which after -8 gives 0)
        let mut block = vec![0u8; 18];
        // d = 0 (fp16 zero)
        block[0] = 0;
        block[1] = 0;
        // nibbles all 0x88 (both 8)
        for i in 2..18 {
            block[i] = 0x88;
        }
        let mut out = vec![0f32; 32];
        dequantize_q4_0(&block, &mut out, 32).unwrap();
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn q8_0_scale_1() {
        // d = 1.0, qs = 0..32 (as i8)
        let d = f16::from_f32(1.0);
        let d_bits = d.to_bits().to_le_bytes();
        let mut block = vec![0u8; 34];
        block[0] = d_bits[0];
        block[1] = d_bits[1];
        for i in 0..32 {
            block[2 + i] = i as u8;
        }
        let mut out = vec![0f32; 32];
        dequantize_q8_0(&block, &mut out, 32).unwrap();
        for i in 0..32 {
            assert!((out[i] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn q4_k_matches_ggml_nibble_layout() {
        // Build one Q4_K block with d=1, dmin=0, scales all 1, and known qs.
        // ggml layout: for each 32-byte qs chunk, low nibbles → 32 weights with
        // scale[is], high nibbles → next 32 weights with scale[is+1].
        let mut block = vec![0u8; 144];
        let d = f16::from_f32(1.0).to_bits().to_le_bytes();
        block[0] = d[0];
        block[1] = d[1];
        // dmin = 0 already
        // scales: j<4 live in bytes 0..4 (scale) and 4..8 (min). Set scale=1.
        for j in 0..4 {
            block[4 + j] = 1; // scale[j] = 1
            block[4 + j + 4] = 0; // min[j] = 0
        }
        // scale[4..8]=1, min[4..8]=0 packed into bytes 8..12 with high bits in 0..8
        for j in 0..4 {
            block[4 + j + 8] = 1; // low 4 bits of scale[j+4]
        }

        // First qs group (32 bytes): low nibble = 3, high nibble = 7 → 0x73
        for i in 0..32 {
            block[16 + i] = 0x73;
        }
        // Remaining qs zeroed → weights 64..256 = 0

        let mut out = vec![0f32; 256];
        dequantize_q4_k(&block, &mut out, 256).unwrap();

        // First 32 weights: d1 * 3 = 3
        for i in 0..32 {
            assert!((out[i] - 3.0).abs() < 1e-5, "out[{i}] = {} want 3", out[i]);
        }
        // Next 32 weights: d2 * 7 = 7
        for i in 32..64 {
            assert!((out[i] - 7.0).abs() < 1e-5, "out[{i}] = {} want 7", out[i]);
        }
        // Rest zero (scale may be 0 for remaining groups if packing left them 0)
        for i in 64..256 {
            assert!(out[i].abs() < 1e-5, "out[{i}] = {} want 0", out[i]);
        }
    }

    #[test]
    fn get_scale_min_k4_matches_ggml() {
        // Mimic ggml packing for j in 0..8 with distinct scale/min values.
        let mut scales_packed = [0u8; 12];
        let ls = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let lm = [10u8, 11, 12, 13, 14, 15, 16, 17];
        for j in 0..8 {
            if j < 4 {
                scales_packed[j] = ls[j];
                scales_packed[j + 4] = lm[j];
            } else {
                scales_packed[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
                scales_packed[j - 4] |= (ls[j] >> 4) << 6;
                scales_packed[j] |= (lm[j] >> 4) << 6;
            }
        }
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &scales_packed);
            assert_eq!(sc, ls[j], "scale[{j}]");
            assert_eq!(m, lm[j], "min[{j}]");
        }
    }
}
