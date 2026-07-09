//! Embedding lookup: copy row `token_id` from a (possibly quantized) matrix.

use crate::dequant::dequantize_row;
use fellm_core::dtype::DType;
use fellm_core::error::Result;

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
