//! Rotary position embeddings.
//!
//! Rotates pairs (x[2i], x[2i+1]) of the first `rope_dim` elements of each head.

use core::f32::consts::PI;

/// Compute the rotation for `head_dim` at position `pos`, in-place across
/// `[n_heads, head_dim]` rows.
pub fn rope_inplace(
    x: &mut [f32], // [n_heads * head_dim]
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
    rope_base: f32,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(rope_dim <= head_dim);
    debug_assert!(rope_dim % 2 == 0);
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in (0..rope_dim).step_by(2) {
            let theta_scale = -(i as f32) / rope_dim as f32;
            let freq = rope_base.powf(theta_scale);
            let theta = pos as f32 * freq;
            let (s, c) = theta.sin_cos();
            let x0 = head[i];
            let x1 = head[i + 1];
            head[i] = x0 * c - x1 * s;
            head[i + 1] = x0 * s + x1 * c;
        }
    }
    let _ = PI;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_zero_position_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        rope_inplace(&mut x, 1, 8, 8, 0, 10000.0);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
