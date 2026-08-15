//! One-token Qwen3.5 Gated DeltaNet kernel.

use crate::kernels::shortconv::matvec_weight;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};

#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetDimensions {
    pub model: usize,
    pub inner: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub state_size: usize,
    pub conv_kernel: usize,
    pub norm_epsilon: f32,
}

pub struct GatedDeltaNetWeights<'a> {
    pub qkv: (&'a [u8], DType),
    pub z: (&'a [u8], DType),
    pub beta: (&'a [u8], DType),
    pub alpha: (&'a [u8], DType),
    pub dt_bias: &'a [f32],
    pub decay: &'a [f32],
    pub conv: &'a [f32],
    pub norm: &'a [f32],
    pub output: (&'a [u8], DType),
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

/// Update convolution and recurrent matrix state and emit one model-width row.
pub fn gated_delta_net_decode(
    x: &[f32],
    weights: &GatedDeltaNetWeights<'_>,
    conv_state: &mut [f32],
    recurrent_state: &mut [f32],
    output: &mut [f32],
    dimensions: GatedDeltaNetDimensions,
) -> Result<()> {
    let d = dimensions;
    if d.model == 0
        || d.inner == 0
        || d.key_heads == 0
        || d.value_heads == 0
        || d.state_size == 0
        || d.conv_kernel == 0
        || d.inner != d.value_heads * d.state_size
        || !d.value_heads.is_multiple_of(d.key_heads)
    {
        return Err(FellmError::other("gated_delta_net: invalid dimensions"));
    }
    let qk_width = d.key_heads * d.state_size;
    let mixed_width = 2 * qk_width + d.inner;
    if x.len() != d.model
        || output.len() != d.model
        || weights.dt_bias.len() != d.value_heads
        || weights.decay.len() != d.value_heads
        || weights.conv.len() != mixed_width * d.conv_kernel
        || weights.norm.len() != d.state_size
        || conv_state.len() != mixed_width * d.conv_kernel.saturating_sub(1)
        || recurrent_state.len() != d.value_heads * d.state_size * d.state_size
    {
        return Err(FellmError::other("gated_delta_net: tensor shape mismatch"));
    }

    let mut mixed = vec![0.0; mixed_width];
    let mut z = vec![0.0; d.inner];
    let mut beta = vec![0.0; d.value_heads];
    let mut alpha = vec![0.0; d.value_heads];
    matvec_weight(
        weights.qkv.0,
        weights.qkv.1,
        x,
        &mut mixed,
        mixed_width,
        d.model,
    )?;
    matvec_weight(weights.z.0, weights.z.1, x, &mut z, d.inner, d.model)?;
    matvec_weight(
        weights.beta.0,
        weights.beta.1,
        x,
        &mut beta,
        d.value_heads,
        d.model,
    )?;
    matvec_weight(
        weights.alpha.0,
        weights.alpha.1,
        x,
        &mut alpha,
        d.value_heads,
        d.model,
    )?;

    let history = d.conv_kernel - 1;
    let mut convolved = vec![0.0; mixed_width];
    for channel in 0..mixed_width {
        let kernel = &weights.conv[channel * d.conv_kernel..(channel + 1) * d.conv_kernel];
        let mut value = mixed[channel] * kernel[history];
        for time in 0..history {
            value += conv_state[channel * history + time] * kernel[time];
        }
        convolved[channel] = silu(value);
    }
    for channel in 0..mixed_width {
        for time in 0..history {
            conv_state[channel * history + time] = if time + 1 < history {
                conv_state[channel * history + time + 1]
            } else {
                mixed[channel]
            };
        }
    }

    let (queries, rest) = convolved.split_at_mut(qk_width);
    let (keys, values) = rest.split_at_mut(qk_width);
    for vector in queries
        .chunks_exact_mut(d.state_size)
        .chain(keys.chunks_exact_mut(d.state_size))
    {
        let magnitude = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt() as f32;
        let inverse = magnitude.max(d.norm_epsilon).recip();
        for value in vector {
            *value *= inverse;
        }
    }

    let scale = (d.state_size as f32).sqrt().recip();
    let mut recurrent_output = vec![0.0; d.inner];
    for value_head in 0..d.value_heads {
        // ggml repeat semantics cycle the source head index (`h % H_k`),
        // rather than grouping consecutive value heads per key head.
        let key_head = value_head % d.key_heads;
        let q = &queries[key_head * d.state_size..(key_head + 1) * d.state_size];
        let k = &keys[key_head * d.state_size..(key_head + 1) * d.state_size];
        let v = &values[value_head * d.state_size..(value_head + 1) * d.state_size];
        let state_offset = value_head * d.state_size * d.state_size;
        let state = &mut recurrent_state[state_offset..state_offset + d.state_size * d.state_size];
        let retention = (weights.decay[value_head]
            * softplus(alpha[value_head] + weights.dt_bias[value_head]))
        .exp();
        let update_rate = 1.0 / (1.0 + (-beta[value_head]).exp());
        for value_index in 0..d.state_size {
            let row = &mut state[value_index * d.state_size..(value_index + 1) * d.state_size];
            for cell in &mut *row {
                *cell *= retention;
            }
            let predicted = row
                .iter()
                .zip(k)
                .map(|(&cell, &key)| cell * key)
                .sum::<f32>();
            let delta = (v[value_index] - predicted) * update_rate;
            for (cell, &key) in row.iter_mut().zip(k) {
                *cell += delta * key;
            }
            recurrent_output[value_head * d.state_size + value_index] = row
                .iter()
                .zip(q)
                .map(|(&cell, &query)| cell * query)
                .sum::<f32>()
                * scale;
        }
    }

    for head in 0..d.value_heads {
        let row = &mut recurrent_output[head * d.state_size..(head + 1) * d.state_size];
        let inverse_rms = (row.iter().map(|value| value * value).sum::<f32>()
            / d.state_size as f32
            + d.norm_epsilon)
            .sqrt()
            .recip();
        for (index, value) in row.iter_mut().enumerate() {
            let flat = head * d.state_size + index;
            *value *= inverse_rms * weights.norm[index] * silu(z[flat]);
        }
    }
    matvec_weight(
        weights.output.0,
        weights.output.1,
        &recurrent_output,
        output,
        d.model,
        d.inner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_state_changes_output_across_tokens() {
        let dimensions = GatedDeltaNetDimensions {
            model: 2,
            inner: 2,
            key_heads: 1,
            value_heads: 1,
            state_size: 2,
            conv_kernel: 1,
            norm_epsilon: 1e-6,
        };
        let qkv: [f32; 12] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let identity: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let scalar_projection: [f32; 2] = [0.0, 0.0];
        let conv: [f32; 6] = [1.0; 6];
        let weights = GatedDeltaNetWeights {
            qkv: (bytemuck::cast_slice(&qkv), DType::F32),
            z: (bytemuck::cast_slice(&identity), DType::F32),
            beta: (bytemuck::cast_slice(&scalar_projection), DType::F32),
            alpha: (bytemuck::cast_slice(&scalar_projection), DType::F32),
            dt_bias: &[0.0],
            decay: &[0.0],
            conv: &conv,
            norm: &[1.0, 1.0],
            output: (bytemuck::cast_slice(&identity), DType::F32),
        };
        let mut conv_state = [];
        let mut recurrent_state = [0.0; 4];
        let mut first = [0.0; 2];
        gated_delta_net_decode(
            &[1.0, 2.0],
            &weights,
            &mut conv_state,
            &mut recurrent_state,
            &mut first,
            dimensions,
        )
        .unwrap();
        let checkpoint = recurrent_state;
        let mut second = [0.0; 2];
        gated_delta_net_decode(
            &[1.0, 2.0],
            &weights,
            &mut conv_state,
            &mut recurrent_state,
            &mut second,
            dimensions,
        )
        .unwrap();
        assert!(first.iter().all(|value| value.is_finite()));
        assert_ne!(checkpoint, [0.0; 4]);
        assert_ne!(first, second);
    }
}
