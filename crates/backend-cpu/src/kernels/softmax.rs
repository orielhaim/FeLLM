//! Softmax with optional causal mask.

/// Softmax over the last dim of `x` (contiguous rows).
///
/// If `causal_len` is `Some(l)`, positions >= l are masked to -inf before
/// softmax. `x` is a flat buffer of `[n_rows, row_len]`.
pub fn softmax_rows_inplace(x: &mut [f32], n_rows: usize, row_len: usize, causal_len: Option<usize>) {
  debug_assert_eq!(x.len(), n_rows * row_len);
  for r in 0..n_rows {
      let row = &mut x[r * row_len..(r + 1) * row_len];
      if let Some(cl) = causal_len {
          for i in cl..row_len {
              row[i] = f32::NEG_INFINITY;
          }
      }
      // stable softmax
      let mut m = f32::NEG_INFINITY;
      for &v in row.iter() {
          if v > m {
              m = v;
          }
      }
      let mut sum = 0.0f32;
      for v in row.iter_mut() {
          let e = (*v - m).exp();
          *v = e;
          sum += e;
      }
      let inv = 1.0 / sum;
      for v in row.iter_mut() {
          *v *= inv;
      }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn softmax_sums_to_one() {
      let mut x = vec![1.0, 2.0, 3.0, 4.0];
      softmax_rows_inplace(&mut x, 1, 4, None);
      let s: f32 = x.iter().sum();
      assert!((s - 1.0).abs() < 1e-6);
  }
}
