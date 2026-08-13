//! Mixture-of-Experts decode kernel.

use crate::dequant::QK_K;
use crate::kernels::{matmul, swiglu::silu_gate};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use rayon::prelude::*;
use std::cell::RefCell;
use std::time::Instant;

thread_local! {
    /// Persistent routing and shared-expert scratch for one backend worker.
    /// The buffers grow to the largest canvas seen by that worker and are
    /// reused by subsequent denoising passes.
    static GEMMA_BATCH_SCRATCH: RefCell<GemmaBatchScratch> =
        const { RefCell::new(GemmaBatchScratch::new()) };
    /// Scratch used by one routed expert at a time on a Rayon worker.
    static GEMMA_WORKER_SCRATCH: RefCell<GemmaWorkerScratch> =
        const { RefCell::new(GemmaWorkerScratch::new()) };
}

struct GemmaBatchScratch {
    groups: Vec<Vec<(usize, f32)>>,
    routed_logits: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<(usize, f32)>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_hidden: Vec<f32>,
    shared_out: Vec<f32>,
    routed_out: Vec<Vec<f32>>,
}

impl GemmaBatchScratch {
    const fn new() -> Self {
        Self {
            groups: Vec::new(),
            routed_logits: Vec::new(),
            logits: Vec::new(),
            scores: Vec::new(),
            shared_gate: Vec::new(),
            shared_up: Vec::new(),
            shared_hidden: Vec::new(),
            shared_out: Vec::new(),
            routed_out: Vec::new(),
        }
    }
}

struct GemmaWorkerScratch {
    group_x: Vec<f32>,
    group_gate: Vec<f32>,
    group_up: Vec<f32>,
    group_hidden: Vec<f32>,
}

impl GemmaWorkerScratch {
    const fn new() -> Self {
        Self {
            group_x: Vec::new(),
            group_gate: Vec::new(),
            group_up: Vec::new(),
            group_hidden: Vec::new(),
        }
    }
}

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
        DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            matmul::matvec_quant(w_bytes, w_dtype, x, y, out_dim, in_dim)
        }
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

fn matmul_weight_batch(
    w_bytes: &[u8],
    w_dtype: DType,
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    if x.len() != rows * in_dim || y.len() != rows * out_dim {
        return Err(FellmError::other("moe: batched weight shape mismatch"));
    }
    match w_dtype {
        DType::F32 => {
            let weights: &[f32] = bytemuck::try_cast_slice(w_bytes)
                .map_err(|e| FellmError::other(format!("moe: f32 cast: {e:?}")))?;
            matmul::matmul_f32_batch(weights, x, y, rows, out_dim, in_dim)
        }
        DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            matmul::matmul_quant_batch(w_bytes, w_dtype, x, y, rows, out_dim, in_dim)
        }
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

fn matmul_weight_batch_serial(
    w_bytes: &[u8],
    w_dtype: DType,
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    if x.len() != rows * in_dim || y.len() != rows * out_dim {
        return Err(FellmError::other("moe: batched weight shape mismatch"));
    }
    match w_dtype {
        DType::F32 => {
            let weights: &[f32] = bytemuck::try_cast_slice(w_bytes)
                .map_err(|e| FellmError::other(format!("moe: f32 cast: {e:?}")))?;
            matmul::matmul_f32_batch_serial(weights, x, y, rows, out_dim, in_dim)
        }
        DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            matmul::matmul_quant_batch_serial(w_bytes, w_dtype, x, y, rows, out_dim, in_dim)
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
    fellm_plugin_abi::record_expert_route(selected.iter().map(|(expert, _)| *expert as u32));
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

/// Run the Gemma 4 MoE layout used by DiffusionGemma.
///
/// Inputs use one packed `[expert, 2 * ff, hidden]` gate/up tensor plus a
/// dense shared SwiGLU expert.  This is kept as a backend operation rather
/// than an architecture-specific graph op.
#[allow(clippy::too_many_arguments)]
pub fn moe_decode_gemma(
    x: &[f32],
    gate_inp: &[f32],
    gate_up_bytes: &[u8],
    gate_up_dtype: DType,
    down_bytes: &[u8],
    down_dtype: DType,
    shared_gate_bytes: &[u8],
    shared_gate_dtype: DType,
    shared_up_bytes: &[u8],
    shared_up_dtype: DType,
    shared_down_bytes: &[u8],
    shared_down_dtype: DType,
    bias: Option<&[f32]>,
    y: &mut [f32],
    n_experts: usize,
    n_expert_used: usize,
    n_ff: usize,
    shared_ff: usize,
    n_embd: usize,
    gating_func: u32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
) -> Result<()> {
    if x.len() != n_embd || y.len() != n_embd || gate_inp.len() != n_experts * n_embd {
        return Err(FellmError::other(
            "moe gemma: activation/router dimensions mismatch",
        ));
    }
    let packed = 2 * n_ff * n_embd;
    let down_elems = n_ff * n_embd;
    let mut router = vec![0.0f32; n_experts];
    matmul::matvec_f32(gate_inp, x, &mut router, n_experts, n_embd);
    if let Some(bias) = bias {
        for (logit, b) in router.iter_mut().zip(bias.iter().copied()) {
            *logit += b;
        }
    }
    let mut scores: Vec<(usize, f32)> = if gating_func == 2 {
        router
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i, 1.0 / (1.0 + (-v).exp())))
            .collect()
    } else {
        let max = router.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut values: Vec<(usize, f32)> = router
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i, (v - max).exp()))
            .collect();
        let sum = values.iter().map(|(_, v)| *v).sum::<f32>();
        if sum > 0.0 {
            for (_, v) in &mut values {
                *v /= sum;
            }
        }
        values
    };
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = n_expert_used.min(n_experts);
    let selected = &mut scores[..k];
    fellm_plugin_abi::record_expert_route(selected.iter().map(|(expert, _)| *expert as u32));
    if norm_topk_prob {
        let sum = selected.iter().map(|(_, v)| *v).sum::<f32>();
        if sum > 0.0 {
            for (_, v) in selected.iter_mut() {
                *v /= sum;
            }
        }
    }
    let scale = if routed_scaling_factor == 0.0 {
        1.0
    } else {
        routed_scaling_factor
    };
    y.fill(0.0);
    let mut gate = vec![0.0f32; n_ff];
    let mut up = vec![0.0f32; n_ff];
    let mut hidden = vec![0.0f32; n_ff];
    let mut expert_out = vec![0.0f32; n_embd];
    for &(expert, score) in selected.iter() {
        let packed_bytes = expert_slice(gate_up_bytes, gate_up_dtype, expert, packed)?;
        let row_bytes = gate_up_dtype.byte_size(n_ff * n_embd);
        let gate_bytes = &packed_bytes[..row_bytes];
        let up_bytes = &packed_bytes[row_bytes..row_bytes * 2];
        let down = expert_slice(down_bytes, down_dtype, expert, down_elems)?;
        matvec_weight(gate_bytes, gate_up_dtype, x, &mut gate, n_ff, n_embd)?;
        matvec_weight(up_bytes, gate_up_dtype, x, &mut up, n_ff, n_embd)?;
        silu_gate(&gate, &up, &mut hidden);
        matvec_weight(down, down_dtype, &hidden, &mut expert_out, n_embd, n_ff)?;
        for i in 0..n_embd {
            y[i] += score * scale * expert_out[i];
        }
    }

    // Dense shared expert is always active.
    let mut shared_hidden = vec![0.0f32; shared_ff];
    let mut shared_gate_vec = vec![0.0f32; shared_ff];
    let mut shared_up_vec = vec![0.0f32; shared_ff];
    matvec_weight(
        shared_gate_bytes,
        shared_gate_dtype,
        x,
        &mut shared_gate_vec,
        shared_ff,
        n_embd,
    )?;
    matvec_weight(
        shared_up_bytes,
        shared_up_dtype,
        x,
        &mut shared_up_vec,
        shared_ff,
        n_embd,
    )?;
    silu_gate(&shared_gate_vec, &shared_up_vec, &mut shared_hidden);
    let mut shared_out = vec![0.0f32; n_embd];
    matvec_weight(
        shared_down_bytes,
        shared_down_dtype,
        &shared_hidden,
        &mut shared_out,
        n_embd,
        shared_ff,
    )?;
    for i in 0..n_embd {
        y[i] += shared_out[i];
    }
    Ok(())
}

/// Token-batched Gemma MoE reference path.
///
/// Routing is computed for the complete `[tokens, hidden]` input, assignments
/// are grouped by expert, and each expert's packed gate/up and down matrices
/// are traversed once per group.  Quantized weights use the same dequantizing
/// matvec helpers as the scalar path; f32 callers can replace this function at
/// backend registration time with a grouped GEMM implementation.
#[allow(clippy::too_many_arguments)]
pub fn moe_decode_gemma_batch(
    x: &[f32],
    gate_inp: &[f32],
    gate_up_bytes: &[u8],
    gate_up_dtype: DType,
    down_bytes: &[u8],
    down_dtype: DType,
    shared_gate_bytes: &[u8],
    shared_gate_dtype: DType,
    shared_up_bytes: &[u8],
    shared_up_dtype: DType,
    shared_down_bytes: &[u8],
    shared_down_dtype: DType,
    bias: Option<&[f32]>,
    y: &mut [f32],
    tokens: usize,
    n_experts: usize,
    n_expert_used: usize,
    n_ff: usize,
    shared_ff: usize,
    n_embd: usize,
    gating_func: u32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
) -> Result<()> {
    let router_row_len = n_experts * n_embd;
    let shared_router = gate_inp.len() == router_row_len;
    if x.len() != tokens * n_embd
        || y.len() != tokens * n_embd
        || (!shared_router && gate_inp.len() != tokens * router_row_len)
    {
        return Err(FellmError::other(format!(
            "moe batch: activation/router dimensions mismatch x={} y={} router={} tokens={} n_embd={} n_experts={}",
            x.len(),
            y.len(),
            gate_inp.len(),
            tokens,
            n_embd,
            n_experts
        )));
    }
    y.fill(0.0);
    let packed = 2 * n_ff * n_embd;
    let down_elems = n_ff * n_embd;
    let k = n_expert_used.min(n_experts);
    let scale = if routed_scaling_factor == 0.0 {
        1.0
    } else {
        routed_scaling_factor
    };
    let profile = std::env::var_os("FELLM_PROFILE_OPS").is_some();
    let started = profile.then(Instant::now);
    GEMMA_BATCH_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();
        if scratch.groups.len() != n_experts {
            scratch.groups = (0..n_experts).map(|_| Vec::new()).collect();
            scratch.routed_out = (0..n_experts).map(|_| Vec::new()).collect();
        }
        let GemmaBatchScratch {
            groups,
            routed_logits,
            logits,
            scores,
            shared_gate,
            shared_up,
            shared_hidden,
            shared_out,
            routed_out,
        } = &mut *scratch;
        let average_group = (tokens * k / n_experts.max(1)).max(1);
        for group in groups.iter_mut() {
            group.clear();
            if group.capacity() < average_group {
                group.reserve(average_group - group.capacity());
            }
        }

        // The Gemma router is shared by the whole canvas.  Compute all routing
        // logits as one output-tiled GEMM so the router weights are traversed once
        // for the complete batch instead of once per canvas row.
        routed_logits.resize(tokens * n_experts, 0.0);
        if shared_router {
            matmul::matmul_f32_batch(gate_inp, x, routed_logits, tokens, n_experts, n_embd)?;
        }

        logits.resize(n_experts, 0.0);
        if scores.capacity() < n_experts {
            let capacity = scores.capacity();
            scores.reserve(n_experts - capacity);
        }
        for token in 0..tokens {
            let x_row = &x[token * n_embd..(token + 1) * n_embd];
            let router_row = if shared_router {
                gate_inp
            } else {
                &gate_inp[token * router_row_len..(token + 1) * router_row_len]
            };
            if shared_router {
                logits.copy_from_slice(&routed_logits[token * n_experts..(token + 1) * n_experts]);
            } else {
                matmul::matvec_f32(router_row, x_row, logits, n_experts, n_embd);
            }
            if let Some(bias) = bias {
                let offset = token * n_experts;
                for e in 0..n_experts {
                    let bias_index = if bias.len() == n_experts {
                        e
                    } else {
                        offset + e
                    };
                    logits[e] += bias.get(bias_index).copied().unwrap_or(0.0);
                }
            }
            scores.clear();
            if gating_func == 2 {
                scores.extend(
                    logits
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(e, v)| (e, 1.0 / (1.0 + (-v).exp()))),
                );
            } else {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                scores.extend(
                    logits
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(e, v)| (e, (v - max).exp())),
                );
                let sum = scores.iter().map(|(_, v)| *v).sum::<f32>();
                if sum > 0.0 {
                    for (_, v) in scores.iter_mut() {
                        *v /= sum;
                    }
                }
            }
            // Only the configured top-k is needed. A full sort of all 128
            // experts for every canvas row is pure routing overhead.
            let ranking =
                |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
            if k < scores.len() {
                let _ = scores.select_nth_unstable_by(k, ranking);
            }
            scores[..k].sort_unstable_by(ranking);
            let selected = &mut scores[..k];
            fellm_plugin_abi::record_expert_route(
                selected.iter().map(|(expert, _)| *expert as u32),
            );
            if norm_topk_prob {
                let sum = selected.iter().map(|(_, v)| *v).sum::<f32>();
                if sum > 0.0 {
                    for (_, v) in selected.iter_mut() {
                        *v /= sum;
                    }
                }
            }
            for &(expert, score) in selected.iter() {
                groups[expert].push((token, score * scale));
            }
        }

        let route_ms = started
            .as_ref()
            .map(|start| start.elapsed().as_secs_f64() * 1000.0);

        // Shared expert runs over the complete canvas as one batched projection.
        // This avoids 256 independent expert launches and lets the quantized
        // matmul path parallelize over rows.
        shared_gate.resize(tokens * shared_ff, 0.0);
        shared_up.resize(tokens * shared_ff, 0.0);
        shared_hidden.resize(tokens * shared_ff, 0.0);
        shared_out.resize(tokens * n_embd, 0.0);
        let shared_started = profile.then(Instant::now);
        matmul_weight_batch(
            shared_gate_bytes,
            shared_gate_dtype,
            x,
            shared_gate,
            tokens,
            shared_ff,
            n_embd,
        )?;
        matmul_weight_batch(
            shared_up_bytes,
            shared_up_dtype,
            x,
            shared_up,
            tokens,
            shared_ff,
            n_embd,
        )?;
        silu_gate(shared_gate, shared_up, shared_hidden);
        matmul_weight_batch(
            shared_down_bytes,
            shared_down_dtype,
            shared_hidden,
            shared_out,
            tokens,
            n_embd,
            shared_ff,
        )?;
        for (dst, src) in y.iter_mut().zip(shared_out.iter().copied()) {
            *dst += src;
        }
        let shared_ms = shared_started
            .as_ref()
            .map(|start| start.elapsed().as_secs_f64() * 1000.0);
        let routed_started = profile.then(Instant::now);
        let nonempty_experts = groups
            .iter()
            .filter(|assignments| !assignments.is_empty())
            .count();
        // A canvas normally activates only a small subset of experts.  Keeping
        // every expert completely serial leaves most of the CPU pool idle; in
        // that case let each grouped GEMM use the pool's remaining workers.
        // When many experts are active, the outer expert scheduler is already
        // wide enough and serial inner kernels avoid nested scheduling.
        let parallel_grouped_gemm =
            nonempty_experts.saturating_mul(2) < rayon::current_num_threads().max(1);

        // Run routed experts concurrently.  Each expert owns disjoint output
        // scratch, so the expensive projections can use the CPU pool without
        // locking the shared canvas.  The reduction is deterministic and remains
        // on the caller thread.
        let routed: Result<()> = groups
            .par_iter()
            .zip(routed_out.par_iter_mut())
            .enumerate()
            .filter(|(_, (assignments, _))| !assignments.is_empty())
            .map(|(expert, (assignments, expert_out))| -> Result<()> {
                let packed_bytes = expert_slice(gate_up_bytes, gate_up_dtype, expert, packed)?;
                let row_bytes = gate_up_dtype.byte_size(n_ff * n_embd);
                let gate_bytes = &packed_bytes[..row_bytes];
                let up_bytes = &packed_bytes[row_bytes..row_bytes * 2];
                let down = expert_slice(down_bytes, down_dtype, expert, down_elems)?;
                let count = assignments.len();
                let mut run_group = |worker: &mut GemmaWorkerScratch| -> Result<()> {
                    let GemmaWorkerScratch {
                        group_x,
                        group_gate,
                        group_up,
                        group_hidden,
                    } = worker;
                    group_x.resize(count * n_embd, 0.0);
                    group_gate.resize(count * n_ff, 0.0);
                    group_up.resize(count * n_ff, 0.0);
                    group_hidden.resize(count * n_ff, 0.0);
                    for (index, &(token, _)) in assignments.iter().enumerate() {
                        group_x[index * n_embd..(index + 1) * n_embd]
                            .copy_from_slice(&x[token * n_embd..(token + 1) * n_embd]);
                    }
                    expert_out.resize(count * n_embd, 0.0);
                    let run_batch = if parallel_grouped_gemm {
                        matmul_weight_batch
                    } else {
                        matmul_weight_batch_serial
                    };
                    run_batch(
                        gate_bytes,
                        gate_up_dtype,
                        group_x,
                        group_gate,
                        count,
                        n_ff,
                        n_embd,
                    )?;
                    run_batch(
                        up_bytes,
                        gate_up_dtype,
                        group_x,
                        group_up,
                        count,
                        n_ff,
                        n_embd,
                    )?;
                    silu_gate(group_gate, group_up, group_hidden);
                    run_batch(
                        down,
                        down_dtype,
                        group_hidden,
                        expert_out,
                        count,
                        n_embd,
                        n_ff,
                    )
                };
                GEMMA_WORKER_SCRATCH.with(|worker_cell| -> Result<()> {
                    if let Ok(mut worker) = worker_cell.try_borrow_mut() {
                        run_group(&mut worker)
                    } else {
                        // A non-Q4 fallback kernel can re-enter Rayon on the
                        // same worker. Do not panic or serialize the whole
                        // batch behind a RefCell; use a one-off fallback only
                        // for that rare recursive case.
                        let mut worker = GemmaWorkerScratch::new();
                        run_group(&mut worker)
                    }
                })
            })
            .collect();
        routed?;
        if let Some(start) = started {
            let assignment_count = groups.iter().map(Vec::len).sum::<usize>();
            let singleton_groups = groups
                .iter()
                .filter(|assignments| assignments.len() == 1)
                .count();
            tracing::info!(
                target = "fellm::profile",
                tokens,
                n_experts,
                nonempty_experts,
                assignment_count,
                singleton_groups,
                route_ms = route_ms.unwrap_or_default(),
                shared_ms = shared_ms.unwrap_or_default(),
                routed_ms = routed_started
                    .as_ref()
                    .map(|start| start.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or_default(),
                total_ms = start.elapsed().as_secs_f64() * 1000.0,
                "gemma MoE batch profile"
            );
        }
        for (assignments, group_out) in groups.iter().zip(routed_out.iter()) {
            for (index, &(token, weight)) in assignments.iter().enumerate() {
                let dst = &mut y[token * n_embd..(token + 1) * n_embd];
                let src = &group_out[index * n_embd..(index + 1) * n_embd];
                for (value, contribution) in dst.iter_mut().zip(src) {
                    *value += weight * contribution;
                }
            }
        }
        Ok(())
    })
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

    #[test]
    fn gemma_batch_matches_scalar_reference_for_f32_weights() {
        let tokens = 3;
        let n_experts = 3;
        let n_expert_used = 2;
        let n_embd = 4;
        let n_ff = 2;
        let shared_ff = 3;
        let x = vec![
            0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
        ];
        let mut router = vec![0.0f32; tokens * n_experts * n_embd];
        for token in 0..tokens {
            for expert in 0..n_experts {
                router[(token * n_experts + expert) * n_embd + (expert % n_embd)] =
                    0.5 + expert as f32;
            }
        }
        let packed = vec![0.2f32; n_experts * 2 * n_ff * n_embd];
        let down = vec![0.1f32; n_experts * n_embd * n_ff];
        let shared_gate = vec![0.3f32; shared_ff * n_embd];
        let shared_up = vec![0.4f32; shared_ff * n_embd];
        let shared_down = vec![0.2f32; n_embd * shared_ff];
        let mut batch = vec![0.0f32; tokens * n_embd];
        moe_decode_gemma_batch(
            &x,
            &router,
            bytemuck::cast_slice(&packed),
            DType::F32,
            bytemuck::cast_slice(&down),
            DType::F32,
            bytemuck::cast_slice(&shared_gate),
            DType::F32,
            bytemuck::cast_slice(&shared_up),
            DType::F32,
            bytemuck::cast_slice(&shared_down),
            DType::F32,
            None,
            &mut batch,
            tokens,
            n_experts,
            n_expert_used,
            n_ff,
            shared_ff,
            n_embd,
            1,
            1.0,
            true,
        )
        .unwrap();
        for token in 0..tokens {
            let mut one = vec![0.0f32; n_embd];
            moe_decode_gemma(
                &x[token * n_embd..(token + 1) * n_embd],
                &router[token * n_experts * n_embd..(token + 1) * n_experts * n_embd],
                bytemuck::cast_slice(&packed),
                DType::F32,
                bytemuck::cast_slice(&down),
                DType::F32,
                bytemuck::cast_slice(&shared_gate),
                DType::F32,
                bytemuck::cast_slice(&shared_up),
                DType::F32,
                bytemuck::cast_slice(&shared_down),
                DType::F32,
                None,
                &mut one,
                n_experts,
                n_expert_used,
                n_ff,
                shared_ff,
                n_embd,
                1,
                1.0,
                true,
            )
            .unwrap();
            for (a, b) in batch[token * n_embd..(token + 1) * n_embd].iter().zip(one) {
                assert!((a - b).abs() < 1e-5, "batch {a} scalar {b}");
            }
        }
    }
}
