//! IQ / MXFP4 row dequantization matching ggml-quants.c.

use crate::dequant::QK_K;
use crate::iq_tables::{
    IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_FP4,
};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use half::f16;

pub fn dequantize_row(dtype: DType, src: &[u8], dst: &mut [f32], n_elements: usize) -> Result<()> {
    match dtype {
        DType::IQ2XXS => dequantize_iq2_xxs(src, dst, n_elements),
        DType::IQ2XS => dequantize_iq2_xs(src, dst, n_elements),
        DType::IQ3XXS => dequantize_iq3_xxs(src, dst, n_elements),
        DType::IQ3S => dequantize_iq3_s(src, dst, n_elements),
        DType::MXFP4 => dequantize_mxfp4(src, dst, n_elements),
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

fn dequantize_iq2_xxs(src: &[u8], dst: &mut [f32], n: usize) -> Result<()> {
    if !n.is_multiple_of(QK_K) {
        return Err(FellmError::other("IQ2_XXS: n not multiple of 256"));
    }
    let bpb = DType::IQ2XXS.bytes_per_block();
    let nb = n / QK_K;
    if src.len() < nb * bpb {
        return Err(FellmError::other("IQ2_XXS: src too small"));
    }
    let mut y = 0usize;
    for i in 0..nb {
        let base = i * bpb;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qs = &src[base + 2..base + 66];
        for ib32 in 0..QK_K / 32 {
            let off = 8 * ib32;
            let lo = u32::from_le_bytes(qs[off..off + 4].try_into().unwrap());
            let hi = u32::from_le_bytes(qs[off + 4..off + 8].try_into().unwrap());
            let db = d * (0.5 + (hi >> 28) as f32) * 0.25;
            let codes = lo.to_le_bytes();
            for l in 0..4 {
                let grid = IQ2XXS_GRID[codes[l] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((hi >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    dst[y] = db * f32::from(grid[j]) * sign;
                    y += 1;
                }
            }
        }
    }
    Ok(())
}

fn dequantize_iq2_xs(src: &[u8], dst: &mut [f32], n: usize) -> Result<()> {
    if !n.is_multiple_of(QK_K) {
        return Err(FellmError::other("IQ2_XS: n not multiple of 256"));
    }
    let bpb = DType::IQ2XS.bytes_per_block();
    let nb = n / QK_K;
    if src.len() < nb * bpb {
        return Err(FellmError::other("IQ2_XS: src too small"));
    }
    let mut y = 0usize;
    for i in 0..nb {
        let base = i * bpb;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qs = &src[base + 2..base + 66];
        let scales = &src[base + 66..base + 74];
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + f32::from(scales[ib32] & 0xf)) * 0.25;
            let db1 = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
            for l in 0..4 {
                let packed = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
                let grid = IQ2XS_GRID[(packed & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(packed >> 9) as usize];
                let db = if l < 2 { db0 } else { db1 };
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    dst[y] = db * f32::from(grid[j]) * sign;
                    y += 1;
                }
            }
        }
    }
    Ok(())
}

fn dequantize_iq3_xxs(src: &[u8], dst: &mut [f32], n: usize) -> Result<()> {
    if !n.is_multiple_of(QK_K) {
        return Err(FellmError::other("IQ3_XXS: n not multiple of 256"));
    }
    let bpb = DType::IQ3XXS.bytes_per_block();
    let nb = n / QK_K;
    if src.len() < nb * bpb {
        return Err(FellmError::other("IQ3_XXS: src too small"));
    }
    let mut y = 0usize;
    for i in 0..nb {
        let base = i * bpb;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qs = &src[base + 2..base + 2 + 96];
        let scales_and_signs = &qs[QK_K / 4..];
        let mut qs_off = 0usize;
        for ib32 in 0..QK_K / 32 {
            let aux32 = u32::from_le_bytes([
                scales_and_signs[4 * ib32],
                scales_and_signs[4 * ib32 + 1],
                scales_and_signs[4 * ib32 + 2],
                scales_and_signs[4 * ib32 + 3],
            ]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let grid1 = IQ3XXS_GRID[qs[qs_off + 2 * l] as usize].to_le_bytes();
                let grid2 = IQ3XXS_GRID[qs[qs_off + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y] = db * f32::from(grid1[j]) * s0;
                    dst[y + 4] = db * f32::from(grid2[j]) * s1;
                    y += 1;
                }
                y += 4;
            }
            qs_off += 8;
        }
    }
    Ok(())
}

fn dequantize_iq3_s(src: &[u8], dst: &mut [f32], n: usize) -> Result<()> {
    if !n.is_multiple_of(QK_K) {
        return Err(FellmError::other("IQ3_S: n not multiple of 256"));
    }
    let bpb = DType::IQ3S.bytes_per_block();
    let nb = n / QK_K;
    if src.len() < nb * bpb {
        return Err(FellmError::other("IQ3_S: src too small"));
    }
    let mut y = 0usize;
    for i in 0..nb {
        let base = i * bpb;
        let d = f16::from_bits(u16::from_le_bytes([src[base], src[base + 1]])).to_f32();
        let qs_all = &src[base + 2..base + 2 + 64];
        let qh = &src[base + 66..base + 74];
        let signs_all = &src[base + 74..base + 106];
        let scales = &src[base + 106..base + 110];
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut qh_off = 0usize;
        for ib32 in (0..QK_K / 32).step_by(2) {
            let db1 = d * (1.0 + 2.0 * f32::from(scales[ib32 / 2] & 0xf));
            let db2 = d * (1.0 + 2.0 * f32::from(scales[ib32 / 2] >> 4));
            for (db, qh_byte) in [(db1, qh[qh_off]), (db2, qh[qh_off + 1])] {
                for l in 0..4 {
                    let g1 = qs_all[qs_off + 2 * l] as usize
                        | ((qh_byte as usize) << (8 - 2 * l) & 256);
                    let g2 = qs_all[qs_off + 2 * l + 1] as usize
                        | ((qh_byte as usize) << (7 - 2 * l) & 256);
                    let grid1 = IQ3S_GRID[g1].to_le_bytes();
                    let grid2 = IQ3S_GRID[g2].to_le_bytes();
                    let signs = signs_all[signs_off + l];
                    for j in 0..4 {
                        let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        dst[y] = db * f32::from(grid1[j]) * s0;
                        dst[y + 4] = db * f32::from(grid2[j]) * s1;
                        y += 1;
                    }
                    y += 4;
                }
                qs_off += 8;
                signs_off += 4;
            }
            qh_off += 2;
        }
    }
    Ok(())
}

fn dequantize_mxfp4(src: &[u8], dst: &mut [f32], n: usize) -> Result<()> {
    const QK: usize = 32;
    if !n.is_multiple_of(QK) {
        return Err(FellmError::other("MXFP4: n not multiple of 32"));
    }
    let bpb = DType::MXFP4.bytes_per_block();
    let nb = n / QK;
    if src.len() < nb * bpb {
        return Err(FellmError::other("MXFP4: src too small"));
    }
    for i in 0..nb {
        let base = i * bpb;
        let d = e8m0_to_fp32_half(src[base]);
        let qs = &src[base + 1..base + 17];
        for j in 0..QK / 2 {
            let x0 = f32::from(KVALUES_FP4[(qs[j] & 0x0F) as usize]);
            let x1 = f32::from(KVALUES_FP4[(qs[j] >> 4) as usize]);
            dst[i * QK + j] = x0 * d;
            dst[i * QK + j + QK / 2] = x1 * d;
        }
    }
    Ok(())
}

fn e8m0_to_fp32_half(e: u8) -> f32 {
    f32::from_bits(u32::from(e) << 23) * 0.5
}
