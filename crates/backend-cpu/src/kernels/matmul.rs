//! Matmul.
//!
//! Computes `y[i] = sum_j W[i,j] * x[j]` where W is `[out_dim, in_dim]`.
//! `x` and `y` are row vectors. This is the "matvec" pattern that dominates
//! LLM decoding.
//!
//! For prefill (`batch` tokens), we call this in a loop per token; a proper
//! GEMM path via `faer` is used when the batch is > 1.

use crate::dequant::dequantize_row;
use fellm_core::dtype::DType;
use fellm_core::error::Result;

/// f32 weight, f32 input -> f32 output. Row-major weight.
pub fn matvec_f32(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);
    // Parallelize across rows.
    use rayon::prelude::*;
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut acc = 0.0f32;
        // Chunked inner accumulation to give the autovectorizer a clear pattern.
        let mut j = 0;
        while j + 8 <= in_dim {
            let mut s = [0.0f32; 8];
            for k in 0..8 {
                s[k] = row[j + k] * x[j + k];
            }
            acc += s.iter().sum::<f32>();
            j += 8;
        }
        while j < in_dim {
            acc += row[j] * x[j];
            j += 1;
        }
        *yi = acc;
    });
}

/// Quantized weight (any dequantizable dtype), f32 input -> f32 output.
///
/// Dequantizes one row at a time and does the dot product without
/// materializing the full weight matrix as f32. Uses per-thread scratch.
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

    use rayon::prelude::*;
    y.par_iter_mut()
        .enumerate()
        .try_for_each(|(i, yi)| -> Result<()> {
            // Per-task dequant scratch.
            let mut scratch = vec![0.0f32; in_dim];
            let row = &w_bytes[i * bytes_per_row..(i + 1) * bytes_per_row];
            dequantize_row(w_dtype, row, &mut scratch, in_dim)?;
            let mut acc = 0.0f32;
            for j in 0..in_dim {
                acc += scratch[j] * x[j];
            }
            *yi = acc;
            Ok(())
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_f32_identity() {
        // 4x4 identity * [1,2,3,4] = [1,2,3,4]
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        matvec_f32(&w, &x, &mut y, 4, 4);
        assert_eq!(y, x);
    }
}
