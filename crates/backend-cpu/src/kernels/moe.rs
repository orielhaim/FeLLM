//! Mixture-of-Experts decode kernel.

use crate::dequant::QK_K;
use crate::kernels::{matmul, swiglu::silu_gate};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};

fn expert_slice(
    w_bytes: &[u8],
    w_dtype: DType,
    expert: usize,
    elements_per_expert: usize,
) -> Result<&[u8]> {
    let bytes_per_expert = w_dtype.byte_size(elements_per_expert);
    let start = expert * bytes_per_expert;
    let end = start + bytes_per_expert;
    w_bytes
        .get(start..end)
        .ok_or_else(|| FellmError::other("moe: expert weight slice out of bounds"))
}

fn matvec_weight(
    w_bytes: &[u8],
    w_dtype: DType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    match w_dtype {
        DType::F32 => {
            let w: &[f32] = bytemuck::try_cast_slice(w_bytes)
                .map_err(|e| FellmError::other(format!("moe: f32 cast: {e:?}")))?;
            matmul::matvec_f32(w, x, y, out_dim, in_dim);
            Ok(())
        }
        DType::Q4_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            matmul::matvec_quant(w_bytes, w_dtype, x, y, out_dim, in_dim)
        }
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

/// Run one-token MoE.
#[allow(clippy::too_many_arguments)]
pub fn moe_decode(
    x: &[f32],
    gate_inp: &[f32],
    gate_exps_bytes: &[u8],
    gate_exps_dtype: DType,
    up_exps_bytes: &[u8],
    up_exps_dtype: DType,
    down_exps_bytes: &[u8],
    down_exps_dtype: DType,
    bias: Option<&[f32]>,
    y: &mut [f32],
    n_experts: usize,
    n_expert_used: usize,
    n_ff: usize,
    n_embd: usize,
    gating_func: u32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
) -> Result<()> {
    if n_experts == 0 || n_expert_used == 0 || n_ff == 0 || n_embd == 0 {
        return Err(FellmError::other("moe: bad dimensions"));
    }
    if x.len() != n_embd || y.len() != n_embd {
        return Err(FellmError::other(format!(
            "moe: x/y len mismatch (x={}, y={}, n_embd={n_embd})",
            x.len(),
            y.len()
        )));
    }
    if gate_inp.len() != n_experts * n_embd {
        return Err(FellmError::other(format!(
            "moe: router len {} != {}",
            gate_inp.len(),
            n_experts * n_embd
        )));
    }
    if let Some(b) = bias
        && b.len() < n_experts
    {
        return Err(FellmError::other("moe: bias shorter than n_experts"));
    }

    let mut logits = vec![0.0f32; n_experts];
    matmul::matvec_f32(gate_inp, x, &mut logits, n_experts, n_embd);
    if let Some(b) = bias {
        for e in 0..n_experts {
            logits[e] += b[e];
        }
    }

    let mut scored = match gating_func {
        2 => logits
            .into_iter()
            .enumerate()
            .map(|(e, v)| (e, 1.0 / (1.0 + (-v).exp())))
            .collect::<Vec<_>>(),
        _ => {
            let max_logit = logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let mut denom = 0.0f32;
            let mut scores = Vec::with_capacity(n_experts);
            for (e, logit) in logits.into_iter().enumerate() {
                let score = (logit - max_logit).exp();
                denom += score;
                scores.push((e, score));
            }
            if denom > 0.0 {
                for (_, score) in &mut scores {
                    *score /= denom;
                }
            }
            scores
        }
    };

    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let k = n_expert_used.min(n_experts);
    let selected = &mut scored[..k];
    if norm_topk_prob {
        let sum = selected.iter().map(|(_, score)| *score).sum::<f32>();
        if sum > 0.0 {
            for (_, score) in selected.iter_mut() {
                *score /= sum;
            }
        }
    }
    let scale = if routed_scaling_factor == 0.0 {
        1.0
    } else {
        routed_scaling_factor
    };

    y.fill(0.0);
    let mut hidden = vec![0.0f32; k * n_ff];
    let mut expert_out = vec![0.0f32; k * n_embd];
    let ffn_elems = n_ff * n_embd;

    let can_batch = n_embd % QK_K == 0 && n_ff % QK_K == 0;
    if !can_batch {
        let mut gate = vec![0.0f32; n_ff];
        let mut up = vec![0.0f32; n_ff];
        for &(expert, score) in selected.iter() {
            let gate_w = expert_slice(gate_exps_bytes, gate_exps_dtype, expert, ffn_elems)?;
            let up_w = expert_slice(up_exps_bytes, up_exps_dtype, expert, ffn_elems)?;
            let down_w = expert_slice(down_exps_bytes, down_exps_dtype, expert, ffn_elems)?;
            matvec_weight(gate_w, gate_exps_dtype, x, &mut gate, n_ff, n_embd)?;
            matvec_weight(up_w, up_exps_dtype, x, &mut up, n_ff, n_embd)?;
            silu_gate(&gate, &up, &mut hidden[..n_ff]);
            matvec_weight(
                down_w,
                down_exps_dtype,
                &hidden[..n_ff],
                &mut expert_out[..n_embd],
                n_embd,
                n_ff,
            )?;
            let weight = score * scale;
            for i in 0..n_embd {
                y[i] += weight * expert_out[i];
            }
        }
        return Ok(());
    }

    let mut y_gu = vec![0.0f32; k * 2 * n_ff];
    {
        let mut mats: Vec<matmul::MatDesc> = Vec::with_capacity(2 * k);
        for &(expert, _) in selected.iter() {
            let gate_w = expert_slice(gate_exps_bytes, gate_exps_dtype, expert, ffn_elems)?;
            let up_w = expert_slice(up_exps_bytes, up_exps_dtype, expert, ffn_elems)?;
            mats.push(matmul::MatDesc {
                w: gate_w,
                dtype: gate_exps_dtype,
                out_dim: n_ff,
                in_dim: n_embd,
                x_off: 0,
            });
            mats.push(matmul::MatDesc {
                w: up_w,
                dtype: up_exps_dtype,
                out_dim: n_ff,
                in_dim: n_embd,
                x_off: 0,
            });
        }
        matmul::matvec_quant_multi(x, &mats, &mut y_gu)?;
        for e in 0..k {
            let g = &y_gu[e * 2 * n_ff..e * 2 * n_ff + n_ff];
            let u = &y_gu[e * 2 * n_ff + n_ff..e * 2 * n_ff + 2 * n_ff];
            let hid = &mut hidden[e * n_ff..(e + 1) * n_ff];
            silu_gate(g, u, hid);
        }
    }

    {
        let mut mats: Vec<matmul::MatDesc> = Vec::with_capacity(k);
        for (e, &(expert, _)) in selected.iter().enumerate() {
            let down_w = expert_slice(down_exps_bytes, down_exps_dtype, expert, ffn_elems)?;
            mats.push(matmul::MatDesc {
                w: down_w,
                dtype: down_exps_dtype,
                out_dim: n_embd,
                in_dim: n_ff,
                x_off: e * n_ff,
            });
        }
        matmul::matvec_quant_multi(&hidden, &mats, &mut expert_out)?;
    }

    for (e, &(_, score)) in selected.iter().enumerate() {
        let weight = score * scale;
        for i in 0..n_embd {
            y[i] += weight * expert_out[e * n_embd + i];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_decode_f32_output_is_finite() {
        let n_experts = 4;
        let n_expert_used = 2;
        let n_embd = 4;
        let n_ff = 3;
        let x = [0.5f32, -1.0, 2.0, 0.25];
        let mut router = vec![0.0f32; n_experts * n_embd];
        for e in 0..n_experts {
            router[e * n_embd + (e % n_embd)] = 1.0 + e as f32 * 0.1;
        }
        let gate = vec![0.25f32; n_experts * n_ff * n_embd];
        let up = vec![0.5f32; n_experts * n_ff * n_embd];
        let down = vec![0.125f32; n_experts * n_embd * n_ff];
        let bias = [0.0f32, 0.1, -0.2, 0.3];
        let mut y = vec![0.0f32; n_embd];

        moe_decode(
            &x,
            &router,
            bytemuck::cast_slice(&gate),
            DType::F32,
            bytemuck::cast_slice(&up),
            DType::F32,
            bytemuck::cast_slice(&down),
            DType::F32,
            Some(&bias),
            &mut y,
            n_experts,
            n_expert_used,
            n_ff,
            n_embd,
            2,
            1.0,
            true,
        )
        .unwrap();

        assert!(y.iter().all(|v| v.is_finite()));
    }
}
