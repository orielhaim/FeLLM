//! SwiGLU: silu(gate) * up.

/// silu(x) = x * sigmoid(x).
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `out[i] = silu(gate[i]) * up[i]`.
pub fn silu_gate(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_at_zero() {
        assert!(silu(0.0).abs() < 1e-6);
    }
}
