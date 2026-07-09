//! RoPE inverse-frequency computation from [`crate::probe::ModelSpec`].

use crate::probe::{ModelSpec, RopeScalingType};

/// Precompute inverse frequencies for RoPE (`rope_dim / 2` values).
#[must_use]
pub fn compute_rope_inv_freqs(spec: &ModelSpec) -> Vec<f32> {
    let half = spec.rope_dim / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        let exp = -((2 * i) as f32) / spec.rope_dim as f32;
        let base_freq = spec.rope_base.powf(exp);
        let scaled = match spec.rope_scaling_type {
            RopeScalingType::None => base_freq,
            RopeScalingType::Linear => base_freq / spec.rope_scaling_factor,
            RopeScalingType::Llama3 => llama3_scale_freq(
                base_freq,
                spec.rope_scaling_factor,
                spec.rope_low_freq_factor,
                spec.rope_high_freq_factor,
                spec.rope_original_ctx.max(1) as f32,
            ),
        };
        out.push(scaled);
    }
    out
}

fn llama3_scale_freq(
    freq: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    old_context_len: f32,
) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let wavelen = two_pi / freq;
    let low_wavelen = old_context_len / low_freq_factor;
    let high_wavelen = old_context_len / high_freq_factor;
    if wavelen < high_wavelen {
        freq
    } else if wavelen > low_wavelen {
        freq / factor
    } else {
        let smooth =
            (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        (1.0 - smooth) * (freq / factor) + smooth * freq
    }
}
