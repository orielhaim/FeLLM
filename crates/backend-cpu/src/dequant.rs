//! GGUF k-quant dequantization.
//!
//! Correctness reference: `ggml-quants.c` from ggml-org/ggml.
//!
//! We support Q4_0, Q8_0, Q4_K, Q6_K in Phase 1. The rest return an error.

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
        DType::Q8_0 => dequantize_q8_0(src, dst, n_elements),
        DType::Q4K => dequantize_q4_k(src, dst, n_elements),
        DType::Q6K => dequantize_q6_k(src, dst, n_elements),
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

// --- Q4_0 ---
// Block layout: fp16 d (2 bytes) + 16 bytes of 4-bit weights (32 weights).
// Formula: w[i] = d * (nibble - 8)
fn dequantize_q4_0(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK4_0;
    if n_elements % QK4_0 != 0 {
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
    if n_elements % QK8_0 != 0 {
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
// Layout (144 bytes, 256 weights):
//   fp16 d (2)
//   fp16 dmin (2)
//   12 bytes packed scales+mins (6-bit each, 8 pairs)
//   128 bytes of 4-bit weights (256 weights, 8 sub-blocks of 32)
//
// The scales/mins layout is the classic ggml packed format.
fn dequantize_q4_k(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK_K;
    if n_elements % QK_K != 0 {
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

        // Unpack 8 (scale, min) 6-bit pairs.
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        get_scale_min_k4(scales_bytes, &mut scales, &mut mins);

        let out = &mut dst[b * QK_K..(b + 1) * QK_K];
        for j in 0..8 {
            let sc = d * scales[j] as f32;
            let m = dmin * mins[j] as f32;
            let sub = &qs[j * 16..(j + 1) * 16]; // 16 bytes = 32 weights (4-bit each)
            let out_sub = &mut out[j * 32..(j + 1) * 32];
            // Low nibbles first 16, high nibbles next 16.
            for i in 0..16 {
                let byte = sub[i];
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                out_sub[i] = sc * lo - m;
                out_sub[i + 16] = sc * hi - m;
            }
        }
    }
    Ok(())
}

/// Extract 8 6-bit scales and 8 6-bit mins from the 12-byte packed field.
///
/// Layout (from ggml-quants.c `get_scale_min_k4`):
///   bytes 0..4  = low 4 bits of scales (4 pairs => 8 nibbles)
///   bytes 4..8  = low 4 bits of mins   (4 pairs => 8 nibbles)
///   bytes 8..12 = high 2 bits of scale[i] and mins[i] interleaved
///
/// See ggml reference for the exact bit layout. This implementation
/// follows it verbatim.
fn get_scale_min_k4(bytes: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    // First 4 pairs
    for j in 0..4 {
        scales[j] = bytes[j] & 63;
        mins[j] = bytes[j + 4] & 63;
    }
    // Next 4 pairs: combine low 4 bits from bytes 8..12 with high 2 bits
    // taken from the top of bytes 0..8.
    for j in 0..4 {
        scales[j + 4] = (bytes[j + 8] & 0x0F) | ((bytes[j] >> 6) << 4);
        mins[j + 4] = (bytes[j + 8] >> 4) | ((bytes[j + 4] >> 6) << 4);
    }
}

// --- Q6_K super-block ---
// Layout (210 bytes, 256 weights):
//   128 bytes ql (low 4 bits of each 6-bit weight)
//   64  bytes qh (high 2 bits of each 6-bit weight, packed 4 per byte)
//   16  bytes scales (int8, one per sub-block-of-16)
//   2   bytes d (fp16)
//
// Formula: w[i] = d * scale[i/16] * (q[i] - 32)  where q is 6-bit unsigned.
fn dequantize_q6_k(src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    let n_blocks = n_elements / QK_K;
    if n_elements % QK_K != 0 {
        return Err(FellmError::other("Q6_K: n_elements not multiple of 256"));
    }
    let block_bytes = DType::Q6K.bytes_per_block();
    if src.len() < n_blocks * block_bytes {
        return Err(FellmError::other("Q6_K: src too small"));
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let ql = &src[base..base + 128];
        let qh = &src[base + 128..base + 128 + 64];
        let scales: &[i8] = bytemuck::cast_slice(&src[base + 192..base + 192 + 16]);
        let d = f16::from_bits(u16::from_le_bytes([src[base + 208], src[base + 209]])).to_f32();

        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        // Process in 2 halves of 128 weights each.
        // Reference layout (from ggml-quants.c):
        //   For l in 0..32:
        //     q1 = ((ql[l]        & 0xF) | ((qh[l]       & 3) << 4)) - 32
        //     q2 = ((ql[l + 32]   & 0xF) | ((qh[l] >> 2) & 3) << 4)) - 32
        //     q3 = ((ql[l]  >> 4)         | ((qh[l] >> 4) & 3) << 4)) - 32
        //     q4 = ((ql[l+32] >> 4)       | ((qh[l] >> 6) & 3) << 4)) - 32
        //   out[l]      = d * sc[0] * q1
        //   out[l+32]   = d * sc[2] * q2
        //   out[l+64]   = d * sc[4] * q3
        //   out[l+96]   = d * sc[6] * q4
        // ... repeat for second half with ql[64..128], qh[32..64], scales[8..].
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            for l in 0..32 {
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32
                    - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32
                    - 32;
                let out_base = half * 128;
                out[out_base + l] = d * scales[sc_off + 0] as f32 * q1 as f32;
                out[out_base + l + 32] = d * scales[sc_off + 2] as f32 * q2 as f32;
                out[out_base + l + 64] = d * scales[sc_off + 4] as f32 * q3 as f32;
                out[out_base + l + 96] = d * scales[sc_off + 6] as f32 * q4 as f32;
                // Sub-blocks 1,3,5,7 (16-wide) still need coverage:
                // The k-quant scale index changes every 16 weights, so above
                // we've applied scales[0,2,4,6] to the first 16 of each 32-run.
                // For weights [16..32] within each 32-run we need scales[1,3,5,7].
                // Correct approach below.
            }
        }
        // Rewrite using the exact ggml pattern:
        // Reset and redo carefully.
        for j in 0..QK_K {
            // no-op placeholder; the actual filling above is close-but-not-quite.
        }
        // Overwrite with a correct scalar reference:
        dequantize_q6_k_scalar(&src[base..base + 210], out);
    }
    Ok(())
}

/// Straight-line scalar reference for Q6_K, matching ggml exactly.
///
/// This is invoked as an overwrite by [`dequantize_q6_k`] to keep the fast
/// path simple and prove-correct at once. Optimize later.
fn dequantize_q6_k_scalar(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales: &[i8] = bytemuck::cast_slice(&block[192..208]);
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

    // Follow the ggml-quants.c logic exactly:
    //   for n in [0, 128) step 128:  (only one iteration; block is 256 weights but processed in 2 halves)
    // Simpler: iterate over 256 output positions using the closed form.
    //
    // Weight index i in [0, 256). Sub-block-of-16 index s = i / 16.
    // The packing groups weights into 4 lanes of 32 within each 128-half.
    // Let's compute by mimicking ggml's dequantize_row_q6_K:
    //
    // for l in 0..32:
    //   is = l / 16  (0 or 1)  -> scale index base within the half
    //   for h in 0..2:
    //     for k in 0..2:  (which nibble of ql / which pair of qh bits)
    //       ...
    //
    // Deriving this from scratch is error-prone; use the canonical layout:

    for n in 0..2 {
        // two halves of 128 weights each
        let ql = &ql[n * 64..(n + 1) * 64];
        let qh = &qh[n * 32..(n + 1) * 32];
        let sc = &scales[n * 8..(n + 1) * 8];
        let out = &mut out[n * 128..(n + 1) * 128];

        for l in 0..32 {
            let is = l / 16; // 0 or 1
            let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;

            out[l] = d * sc[is] as f32 * q1 as f32;
            out[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
            out[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
            out[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
