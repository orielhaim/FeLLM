//! Rotary position embeddings.
//!
//! Rotates pairs (x[2i], x[2i+1]) of `rope_dim` elements of each head,
//! using externally supplied `inv_freqs` of length `rope_dim / 2`. This lets
//! architectures apply any scaling scheme (Llama 3, YaRN, linear, none) by
//! precomputing the frequencies at model-load time.

/// Apply RoPE in place to `x` given precomputed inverse frequencies.
///
/// `x` has layout `[n_heads * head_dim]`. Within each head, pairs
/// `(x[2i], x[2i+1])` for `i in 0..rope_dim/2` are rotated by
/// `pos * inv_freqs[i]`.
pub fn rope_inplace_with_freqs(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
    inv_freqs: &[f32],
    split_half: bool,
) {
    rope_inplace_with_freqs_ex(
        x, n_heads, head_dim, rope_dim, pos, inv_freqs, split_half, false,
    )
}

/// `tail` rotates the last `rope_dim` elements of each head (DeepSeek MLA).
pub fn rope_inplace_with_freqs_ex(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
    inv_freqs: &[f32],
    split_half: bool,
    tail: bool,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(rope_dim <= head_dim);
    debug_assert_eq!(rope_dim % 2, 0);
    debug_assert_eq!(inv_freqs.len(), rope_dim / 2);
    let pf = pos as f32;
    let start = if tail { head_dim - rope_dim } else { 0 };
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        let rot = &mut head[start..start + rope_dim];
        for i in 0..(rope_dim / 2) {
            let theta = pf * inv_freqs[i];
            let (s, c) = theta.sin_cos();
            let first = if split_half { i } else { 2 * i };
            let second = if split_half {
                i + rope_dim / 2
            } else {
                2 * i + 1
            };
            let a = rot[first];
            let b = rot[second];
            rot[first] = a * c - b * s;
            rot[second] = a * s + b * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_zero_position_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        let inv_freqs = vec![1.0f32, 0.5, 0.25, 0.125];
        rope_inplace_with_freqs(&mut x, 1, 8, 8, 0, &inv_freqs, false);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
