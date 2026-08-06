//! Embedding lookup: copy row `token_id` from a (possibly quantized) matrix.

use crate::dequant::dequantize_row;
use fellm_core::dtype::DType;
use fellm_core::error::Result;
use rayon::prelude::*;

/// Gather row `token_id` from `w_bytes` (row-major, dtype `w_dtype`, `[vocab, dim]`)
/// into `out` as f32.
pub fn embedding_row(
    w_bytes: &[u8],
    w_dtype: DType,
    vocab: usize,
    dim: usize,
    token_id: u32,
    out: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(out.len(), dim);
    let _ = vocab;
    let bytes_per_row = w_dtype.byte_size(dim);
    let row = &w_bytes[token_id as usize * bytes_per_row..(token_id as usize + 1) * bytes_per_row];
    dequantize_row(w_dtype, row, out, dim)
}

/// Compute DiffusionGemma self-conditioning embeddings.
///
/// Each row is softmaxed over the vocabulary and used as a distribution over
/// the tied token-embedding table.  The first denoising pass can pass an
/// all-zero row; that is detected before touching the vocabulary matrix.
pub fn weighted_embedding(
    w_bytes: &[u8],
    w_dtype: DType,
    logits: &[f32],
    out: &mut [f32],
    rows: usize,
    vocab: usize,
    dim: usize,
) -> Result<()> {
    if logits.len() != rows * vocab || out.len() != rows * dim {
        return Err(fellm_core::error::FellmError::other(
            "weighted embedding: shape mismatch",
        ));
    }
    let bytes_per_row = w_dtype.byte_size(dim);
    let finite_counts: Vec<usize> = logits
        .par_chunks_exact(vocab)
        .map(|row| row.iter().filter(|value| value.is_finite()).count())
        .collect();

    if logits
        .par_chunks_exact(vocab)
        .all(|row| row.iter().all(|value| *value == 0.0))
    {
        out.fill(0.0);
        return Ok(());
    }

    // Sparse self-conditioning is the normal optimized path. Decode only the
    // selected vocabulary rows instead of traversing the full tied table for
    // every canvas position.
    if finite_counts.iter().copied().max().unwrap_or(0) <= 4096 {
        out.par_chunks_exact_mut(dim)
            .zip(logits.par_chunks_exact(vocab))
            .try_for_each(|(out_row, logit_row)| -> Result<()> {
                let mut selected: Vec<(usize, f32)> = logit_row
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, value)| value.is_finite())
                    .collect();
                if selected.is_empty() {
                    out_row.fill(0.0);
                    return Ok(());
                }
                let max = selected
                    .iter()
                    .map(|(_, value)| *value)
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum = selected
                    .iter_mut()
                    .map(|(_, value)| {
                        *value = (*value - max).exp();
                        *value
                    })
                    .sum::<f32>();
                out_row.fill(0.0);
                if !sum.is_finite() || sum <= 0.0 {
                    return Ok(());
                }
                let mut decoded = vec![0.0f32; dim];
                for (token, weight) in selected {
                    let start = token * bytes_per_row;
                    dequantize_row(
                        w_dtype,
                        &w_bytes[start..start + bytes_per_row],
                        &mut decoded,
                        dim,
                    )?;
                    let probability = weight / sum;
                    for (dst, value) in out_row.iter_mut().zip(&decoded) {
                        *dst += probability * value;
                    }
                }
                Ok(())
            })?;
        return Ok(());
    }

    let stats: Vec<(f32, f32)> = logits
        .par_chunks_exact(vocab)
        .map(|logit_row| {
            if logit_row.iter().all(|value| *value == 0.0) {
                return (0.0, 0.0);
            }
            let max = logit_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = logit_row
                .iter()
                .map(|value| (*value - max).exp())
                .sum::<f32>();
            (max, if sum.is_finite() { sum } else { 0.0 })
        })
        .collect();
    out.fill(0.0);

    // Decode each vocabulary tile once and reuse it for every canvas row.
    // The previous implementation decoded the entire tied table once per
    // canvas row, multiplying the quantized-weight overhead by 256.
    const TILE_ROWS: usize = 256;
    for tile_start in (0..vocab).step_by(TILE_ROWS) {
        let tile_end = (tile_start + TILE_ROWS).min(vocab);
        let tile_width = tile_end - tile_start;
        let mut decoded = vec![0.0f32; tile_width * dim];
        decoded.par_chunks_exact_mut(dim).enumerate().try_for_each(
            |(local, row)| -> Result<()> {
                let token = tile_start + local;
                let start = token * bytes_per_row;
                dequantize_row(w_dtype, &w_bytes[start..start + bytes_per_row], row, dim)
            },
        )?;
        out.par_chunks_exact_mut(dim)
            .zip(logits.par_chunks_exact(vocab))
            .zip(stats.par_iter())
            .for_each(|((out_row, logit_row), &(max, sum))| {
                if sum <= 0.0 {
                    return;
                }
                for local in 0..tile_width {
                    let probability = (logit_row[tile_start + local] - max).exp() / sum;
                    if probability <= 0.0 {
                        continue;
                    }
                    let source = &decoded[local * dim..(local + 1) * dim];
                    for (dst, value) in out_row.iter_mut().zip(source) {
                        *dst += probability * value;
                    }
                }
            });
    }
    Ok(())
}

/// Compute self-conditioning embeddings from packed `(token_id, logit)`
/// candidates.  Each row has `top_k` pairs, so the input shape is
/// `[rows, 2 * top_k]`.  Token ids are exactly representable in F32 for the
/// vocabulary sizes used by DiffusionGemma.
pub fn weighted_embedding_topk(
    w_bytes: &[u8],
    w_dtype: DType,
    packed: &[f32],
    out: &mut [f32],
    rows: usize,
    top_k: usize,
    vocab: usize,
    dim: usize,
) -> Result<()> {
    if top_k == 0 || packed.len() != rows * top_k * 2 || out.len() != rows * dim {
        return Err(fellm_core::error::FellmError::other(
            "weighted embedding top-k: shape mismatch",
        ));
    }
    let bytes_per_row = w_dtype.byte_size(dim);
    out.par_chunks_exact_mut(dim)
        .zip(packed.par_chunks_exact(top_k * 2))
        .try_for_each(|(out_row, candidates)| -> Result<()> {
            let mut max = f32::NEG_INFINITY;
            let mut active = 0usize;
            for pair in candidates.chunks_exact(2) {
                if pair[1].is_finite() {
                    max = max.max(pair[1]);
                    active += 1;
                }
            }
            out_row.fill(0.0);
            if active == 0 {
                return Ok(());
            }
            let mut sum = 0.0f32;
            for pair in candidates.chunks_exact(2) {
                if pair[1].is_finite() {
                    sum += (pair[1] - max).exp();
                }
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Ok(());
            }
            let mut decoded = vec![0.0f32; dim];
            for pair in candidates.chunks_exact(2) {
                let token = pair[0] as usize;
                if !pair[1].is_finite() || token >= vocab {
                    continue;
                }
                let start = token * bytes_per_row;
                dequantize_row(
                    w_dtype,
                    &w_bytes[start..start + bytes_per_row],
                    &mut decoded,
                    dim,
                )?;
                let probability = (pair[1] - max).exp() / sum;
                for (dst, value) in out_row.iter_mut().zip(&decoded) {
                    *dst += probability * value;
                }
            }
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_embedding_zero_and_uniform_rows() {
        let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let logits = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut output = vec![9.0f32; 4];
        weighted_embedding(
            bytemuck::cast_slice(&weights),
            DType::F32,
            &logits[..3],
            &mut output[..2],
            1,
            3,
            2,
        )
        .unwrap();
        assert!(output[..2].iter().all(|value| value.is_finite()));
        assert!(output[0] > 1.0 && output[1] > 2.0);

        weighted_embedding(
            bytemuck::cast_slice(&weights),
            DType::F32,
            &logits[3..],
            &mut output[2..],
            1,
            3,
            2,
        )
        .unwrap();
        assert_eq!(&output[2..], &[0.0, 0.0]);
    }

    #[test]
    fn weighted_embedding_topk_matches_dense_selection() {
        let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let packed = [0.0f32, 1.0, 1.0, 3.0];
        let mut output = vec![0.0f32; 2];
        weighted_embedding_topk(
            bytemuck::cast_slice(&weights),
            DType::F32,
            &packed,
            &mut output,
            1,
            2,
            3,
            2,
        )
        .unwrap();
        let p0 = 1.0f32 / (1.0 + 2.0f32.exp());
        let p1 = 1.0 - p0;
        assert!((output[0] - (p0 * 1.0 + p1 * 3.0)).abs() < 1e-6);
        assert!((output[1] - (p0 * 2.0 + p1 * 4.0)).abs() < 1e-6);
    }
}
