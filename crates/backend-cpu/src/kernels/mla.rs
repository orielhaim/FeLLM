//! Multi-head latent attention with compressed KV, sinks, and indexed sparse scores.

use crate::kernels::{matmul, norm, rope, softmax};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

struct CompressLayer {
    kv: Vec<f32>,
    score: Vec<f32>,
    blocks: Vec<f32>,
    lid_kv: Vec<f32>,
    lid_score: Vec<f32>,
    lid_blocks: Vec<f32>,
}

fn compress_layers() -> &'static Mutex<HashMap<u32, CompressLayer>> {
    static LAYERS: OnceLock<Mutex<HashMap<u32, CompressLayer>>> = OnceLock::new();
    LAYERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn empty_layer() -> CompressLayer {
    CompressLayer {
        kv: Vec::new(),
        score: Vec::new(),
        blocks: Vec::new(),
        lid_kv: Vec::new(),
        lid_score: Vec::new(),
        lid_blocks: Vec::new(),
    }
}

/// Optional compressor / lightning-indexer weights.
#[derive(Clone, Copy)]
pub struct MlaExtras<'a> {
    pub compress_kv: Option<(&'a [u8], DType)>,
    pub compress_gate: Option<(&'a [u8], DType)>,
    pub compress_ape: Option<&'a [f32]>,
    pub compress_norm: Option<&'a [f32]>,
    pub compress_state_dim: usize,
    pub indexer_q_b: Option<(&'a [u8], DType)>,
    pub indexer_proj: Option<(&'a [u8], DType)>,
    pub indexer_comp_kv: Option<(&'a [u8], DType)>,
    pub indexer_comp_gate: Option<(&'a [u8], DType)>,
    pub indexer_comp_ape: Option<&'a [f32]>,
    pub indexer_comp_norm: Option<&'a [f32]>,
    pub indexer_state_dim: usize,
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_top_k: usize,
}

fn softmax_feature_pool(
    kv: &[f32],
    score: &[f32],
    state_dim: usize,
    start: usize,
    n: usize,
    feat_off: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    if n == 0 {
        return;
    }
    let mut logits = vec![0.0f32; n];
    for d in 0..head_dim {
        for t in 0..n {
            logits[t] = score[(start + t) * state_dim + feat_off + d];
        }
        softmax::softmax_rows_inplace(&mut logits, 1, n, None);
        for t in 0..n {
            out[d] += logits[t] * kv[(start + t) * state_dim + feat_off + d];
        }
    }
}

pub(crate) fn compress_block(
    kv: &[f32],
    score: &[f32],
    state_dim: usize,
    n_tokens: usize,
    ratio: usize,
    head_dim: usize,
    overlap: bool,
    norm_w: Option<&[f32]>,
    rms_eps: f32,
    rope_dim: usize,
    inv_freqs: &[f32],
) -> Result<Vec<f32>> {
    let start = n_tokens - ratio;
    let mut acc = vec![0.0f32; head_dim];
    if overlap && state_dim >= 2 * head_dim {
        let mut logits = vec![0.0f32; 2 * ratio];
        for d in 0..head_dim {
            for t in 0..ratio {
                if start >= ratio {
                    let prev = start - ratio + t;
                    logits[t] = score[prev * state_dim + d];
                } else {
                    logits[t] = 0.0;
                }
                logits[ratio + t] = score
                    .get((start + t) * state_dim + head_dim + d)
                    .copied()
                    .unwrap_or(0.0);
            }
            softmax::softmax_rows_inplace(&mut logits, 1, 2 * ratio, None);
            for t in 0..ratio {
                if start >= ratio {
                    let prev = start - ratio + t;
                    acc[d] += logits[t] * kv[prev * state_dim + d];
                }
                acc[d] += logits[ratio + t] * kv[(start + t) * state_dim + head_dim + d];
            }
        }
    } else {
        softmax_feature_pool(kv, score, state_dim, start, ratio, 0, head_dim, &mut acc);
    }
    if let Some(cn) = norm_w {
        let mut out = vec![0.0f32; head_dim];
        norm::rmsnorm_row(&acc, cn, rms_eps, &mut out);
        acc = out;
    }
    let block_pos = ((n_tokens / ratio).saturating_sub(1) * ratio) as u32;
    rope::rope_inplace_with_freqs_ex(
        &mut acc, 1, head_dim, rope_dim, block_pos, inv_freqs, false, true,
    );
    Ok(acc)
}

fn mla_trace_layer(layer_ord: u32) -> bool {
    matches!(layer_ord, 0 | 1 | 8 | 9)
}

fn mla_trace(layer_ord: u32, name: &str, values: &[f32]) -> bool {
    if !mla_trace_layer(layer_ord) {
        return false;
    }
    let mut finite = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &value in values {
        if !value.is_finite() {
            continue;
        }
        finite += 1;
        min = min.min(value);
        max = max.max(value);
        sum += f64::from(value);
        sum_sq += f64::from(value) * f64::from(value);
    }
    let n = values.len();
    let l2 = sum_sq.sqrt() as f32;
    let rms = if n == 0 { 0.0 } else { l2 / (n as f32).sqrt() };
    let mean = if finite == 0 {
        f32::NAN
    } else {
        (sum / finite as f64) as f32
    };
    if finite == 0 {
        min = f32::NAN;
        max = f32::NAN;
    }
    let head: Vec<String> = values
        .iter()
        .take(8)
        .map(|v| format!("{v:.6}"))
        .collect();
    tracing::debug!(
        layer_ord,
        name,
        len = n,
        finite,
        min,
        max,
        mean,
        l2,
        rms,
        first8 = %head.join(","),
        "mla intermediate"
    );
    let exploded = finite < n || max.abs() > 1.0e3 || min.abs() > 1.0e3 && finite > 0;
    if exploded {
        tracing::error!(
            layer_ord,
            name,
            finite,
            len = n,
            min,
            max,
            rms,
            "mla intermediate is non-finite or exploded"
        );
    }
    exploded
}

fn mla_trace_proj(
    layer_ord: u32,
    name: &str,
    w: &[u8],
    dtype: DType,
    x: &[f32],
    y: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    if !mla_trace_layer(layer_ord) {
        return;
    }
    let expected = dtype.byte_size(out_dim.saturating_mul(in_dim));
    let x_rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let y_rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len().max(1) as f32).sqrt();
    let mut w_max = 0.0f32;
    let mut w_rms = 0.0f32;
    if let Ok(()) = (|| -> Result<()> {
        let row_bytes = dtype.byte_size(in_dim);
        if w.len() < row_bytes {
            return Ok(());
        }
        let mut row = vec![0.0f32; in_dim];
        crate::dequant::dequantize_row(dtype, &w[..row_bytes], &mut row, in_dim)?;
        w_max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        w_rms = (row.iter().map(|v| v * v).sum::<f32>() / in_dim.max(1) as f32).sqrt();
        Ok(())
    })() {}
    tracing::debug!(
        layer_ord,
        name,
        dtype = %dtype,
        in_dim,
        out_dim,
        weight_bytes = w.len(),
        expected_bytes = expected,
        input_rms = x_rms,
        output_rms = y_rms,
        first_row_weight_rms = w_rms,
        first_row_weight_max = w_max,
        "mla projection"
    );
}

pub(crate) fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len()).max(1);
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

#[allow(clippy::too_many_arguments)]
pub fn mla_decode(
    x: &[f32],
    q_a: &[u8],
    q_a_dtype: DType,
    q_a_norm: &[f32],
    q_b: &[u8],
    q_b_dtype: DType,
    kv_w: &[u8],
    kv_dtype: DType,
    kv_norm: &[f32],
    wo_a: &[u8],
    wo_a_dtype: DType,
    wo_b: &[u8],
    wo_b_dtype: DType,
    sinks: &[f32],
    inv_freqs: &[f32],
    k_out: &mut [f32],
    c_out: &mut [f32],
    y: &mut [f32],
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_lora: usize,
    o_groups: usize,
    o_lora: usize,
    position: u32,
    past_len: u32,
    window: usize,
    rms_eps: f32,
    compress_ratio: usize,
    extras: MlaExtras<'_>,
    layer_ord: u32,
) -> Result<()> {
    let mut qr = vec![0.0f32; q_lora];
    matmul::matvec_quant(q_a, q_a_dtype, x, &mut qr, q_lora, d_model)?;
    mla_trace_proj(layer_ord, "q_a", q_a, q_a_dtype, x, &qr, q_lora, d_model);
    if mla_trace(layer_ord, "input", x) {
        tracing::error!(layer_ord, "mla stop: input");
    }
    if mla_trace(layer_ord, "q_a", &qr) {
        tracing::error!(layer_ord, "mla stop: q_a");
    }
    let mut qr_n = vec![0.0f32; q_lora];
    norm::rmsnorm_row(&qr, q_a_norm, rms_eps, &mut qr_n);
    let _ = mla_trace(layer_ord, "q_a_norm", &qr_n);
    let q_len = n_heads * head_dim;
    let mut q = vec![0.0f32; q_len];
    matmul::matvec_quant(q_b, q_b_dtype, &qr_n, &mut q, q_len, q_lora)?;
    mla_trace_proj(layer_ord, "q_b", q_b, q_b_dtype, &qr_n, &q, q_len, q_lora);
    let _ = mla_trace(layer_ord, "q_b", &q);
    let ones_h = vec![1.0f32; head_dim];
    let mut qn = vec![0.0f32; q_len];
    for (src, dst) in q.chunks_exact(head_dim).zip(qn.chunks_exact_mut(head_dim)) {
        norm::rmsnorm_row(src, &ones_h, rms_eps, dst);
    }
    q = qn;
    let _ = mla_trace(layer_ord, "q_head_norm", &q);
    rope::rope_inplace_with_freqs_ex(
        &mut q, n_heads, head_dim, rope_dim, position, inv_freqs, false, true,
    );
    let _ = mla_trace(layer_ord, "q_rope", &q);

    let mut kv = vec![0.0f32; head_dim];
    matmul::matvec_quant(kv_w, kv_dtype, x, &mut kv, head_dim, d_model)?;
    mla_trace_proj(layer_ord, "kv", kv_w, kv_dtype, x, &kv, head_dim, d_model);
    let _ = mla_trace(layer_ord, "kv", &kv);
    let mut kvn = vec![0.0f32; head_dim];
    norm::rmsnorm_row(&kv, kv_norm, rms_eps, &mut kvn);
    kv = kvn;
    let _ = mla_trace(layer_ord, "kv_a_norm", &kv);
    rope::rope_inplace_with_freqs_ex(
        &mut kv, 1, head_dim, rope_dim, position, inv_freqs, false, true,
    );
    let _ = mla_trace(layer_ord, "kv_rope", &kv);

    let pos = position as usize;
    let row = pos * head_dim;
    if row + head_dim <= k_out.len() {
        k_out[row..row + head_dim].copy_from_slice(&kv);
    }

    let ratio = compress_ratio;
    let overlap = extras.compress_state_dim >= 2 * head_dim && ratio > 0 && ratio <= 8;
    let idx_heads = extras.indexer_heads.max(1);
    let idx_dim = extras.indexer_head_dim.max(1);
    let idx_len = idx_heads * idx_dim;

    let mut q_idx = vec![0.0f32; idx_len];
    let mut idx_weights = vec![0.0f32; idx_heads];
    if let (Some((iqb, iqb_dt)), Some((iproj, ip_dt))) = (extras.indexer_q_b, extras.indexer_proj) {
        matmul::matvec_quant(iqb, iqb_dt, &qr_n, &mut q_idx, idx_len, q_lora)?;
        for h in 0..idx_heads {
            let head = &mut q_idx[h * idx_dim..(h + 1) * idx_dim];
            rope::rope_inplace_with_freqs_ex(
                head, 1, idx_dim, rope_dim.min(idx_dim), position, inv_freqs, false, true,
            );
        }
        matmul::matvec_quant(iproj, ip_dt, x, &mut idx_weights, idx_heads, d_model)?;
        let scale = 1.0 / ((idx_dim * idx_heads) as f32).sqrt();
        for w in &mut idx_weights {
            *w *= scale;
        }
    }

    if ratio > 0
        && let (Some((ckv, ck_dt)), Some((cgate, cg_dt))) =
            (extras.compress_kv, extras.compress_gate)
    {
        let state_dim = extras.compress_state_dim.max(head_dim);
        let mut token_kv = vec![0.0f32; state_dim];
        let mut token_score = vec![0.0f32; state_dim];
        matmul::matvec_quant(ckv, ck_dt, x, &mut token_kv, state_dim, d_model)?;
        matmul::matvec_quant(cgate, cg_dt, x, &mut token_score, state_dim, d_model)?;
        if let Some(ape) = extras.compress_ape {
            let ape_rows = ape.len() / state_dim.max(1);
            if ape_rows > 0 {
                let row_i = pos % ape_rows;
                let ape_row = &ape[row_i * state_dim..(row_i + 1) * state_dim];
                for (s, &a) in token_score.iter_mut().zip(ape_row) {
                    *s += a;
                }
            }
        }
        let mut lid_kv_t = vec![0.0f32; extras.indexer_state_dim.max(idx_dim)];
        let mut lid_score_t = vec![0.0f32; extras.indexer_state_dim.max(idx_dim)];
        let have_lid = extras.indexer_comp_kv.is_some() && extras.indexer_comp_gate.is_some();
        if let (Some((lkv, ldt)), Some((lg, lgd))) =
            (extras.indexer_comp_kv, extras.indexer_comp_gate)
        {
            let ldim = extras.indexer_state_dim.max(idx_dim);
            lid_kv_t.resize(ldim, 0.0);
            lid_score_t.resize(ldim, 0.0);
            matmul::matvec_quant(lkv, ldt, x, &mut lid_kv_t, ldim, d_model)?;
            matmul::matvec_quant(lg, lgd, x, &mut lid_score_t, ldim, d_model)?;
            if let Some(ape) = extras.indexer_comp_ape {
                let ape_rows = ape.len() / ldim.max(1);
                if ape_rows > 0 {
                    let row_i = pos % ape_rows;
                    let ape_row = &ape[row_i * ldim..(row_i + 1) * ldim];
                    for (s, &a) in lid_score_t.iter_mut().zip(ape_row) {
                        *s += a;
                    }
                }
            }
        }

        let mut guard = compress_layers()
            .lock()
            .map_err(|_| FellmError::other("mla compress lock"))?;
        if pos == 0 {
            guard.remove(&layer_ord);
        }
        let layer = guard.entry(layer_ord).or_insert_with(empty_layer);
        layer.kv.extend_from_slice(&token_kv);
        layer.score.extend_from_slice(&token_score);
        if have_lid {
            layer.lid_kv.extend_from_slice(&lid_kv_t);
            layer.lid_score.extend_from_slice(&lid_score_t);
        }
        let n_tokens = layer.kv.len() / state_dim.max(1);
        if n_tokens > 0 && n_tokens.is_multiple_of(ratio) {
            let acc = compress_block(
                &layer.kv,
                &layer.score,
                state_dim,
                n_tokens,
                ratio,
                head_dim,
                overlap,
                extras.compress_norm,
                rms_eps,
                rope_dim,
                inv_freqs,
            )?;
            layer.blocks.extend_from_slice(&acc);
            if have_lid {
                let ldim = extras.indexer_state_dim.max(idx_dim);
                let lid_overlap = ldim >= 2 * idx_dim;
                let lid_acc = compress_block(
                    &layer.lid_kv,
                    &layer.lid_score,
                    ldim,
                    n_tokens,
                    ratio,
                    idx_dim,
                    lid_overlap,
                    extras.indexer_comp_norm,
                    rms_eps,
                    rope_dim.min(idx_dim),
                    inv_freqs,
                )?;
                layer.lid_blocks.extend_from_slice(&lid_acc);
            }
        }
        let n_blocks = layer.blocks.len() / head_dim.max(1);
        let copy = n_blocks * head_dim;
        if copy <= c_out.len() {
            c_out[..copy].copy_from_slice(&layer.blocks[..copy]);
        }
    }

    let kv_len = (past_len as usize + 1).min(k_out.len() / head_dim.max(1));
    let start = if window > 0 {
        kv_len.saturating_sub(window)
    } else {
        0
    };
    let n_comp = if ratio > 0 && c_out.len() >= head_dim {
        c_out.len() / head_dim
    } else {
        0
    };

    let mut selected: Vec<usize> = (0..n_comp).collect();
    if n_comp > 0 && extras.indexer_q_b.is_some() && extras.indexer_top_k > 0 {
        let guard = compress_layers()
            .lock()
            .map_err(|_| FellmError::other("mla indexer lock"))?;
        if let Some(layer) = guard.get(&layer_ord) {
            let n_lid = layer.lid_blocks.len() / idx_dim.max(1);
            let n = n_lid.min(n_comp);
            if n > 0 {
                let mut scores = vec![0.0f32; n];
                for b in 0..n {
                    let kh = &layer.lid_blocks[b * idx_dim..(b + 1) * idx_dim];
                    let mut s = 0.0f32;
                    for h in 0..idx_heads {
                        let qh = &q_idx[h * idx_dim..(h + 1) * idx_dim];
                        let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                        s += dot.max(0.0) * idx_weights[h];
                    }
                    scores[b] = s;
                }
                selected = top_k_indices(&scores, extras.indexer_top_k);
            }
        }
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; q_len];
    let mut traced_scores = false;
    for h in 0..n_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = Vec::with_capacity(kv_len - start + selected.len() + 1);
        for t in start..kv_len {
            let kh = &k_out[t * head_dim..(t + 1) * head_dim];
            let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
            scores.push(dot * scale);
        }
        for &b in &selected {
            let kh = &c_out[b * head_dim..(b + 1) * head_dim];
            let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
            scores.push(dot * scale);
        }
        if h < sinks.len() {
            scores.push(sinks[h]);
        }
        if !traced_scores && mla_trace_layer(layer_ord) {
            traced_scores = true;
            let _ = mla_trace(layer_ord, "scores_scaled", &scores);
        }
        let n_scores = scores.len();
        softmax::softmax_rows_inplace(&mut scores, 1, n_scores, None);
        if h == 0 {
            let _ = mla_trace(layer_ord, "softmax", &scores);
        }
        let mut acc = vec![0.0f32; head_dim];
        for (i, t) in (start..kv_len).enumerate() {
            let kh = &k_out[t * head_dim..(t + 1) * head_dim];
            let w = scores[i];
            for (a, &v) in acc.iter_mut().zip(kh) {
                *a += w * v;
            }
        }
        let raw = kv_len - start;
        for (i, &b) in selected.iter().enumerate() {
            let kh = &c_out[b * head_dim..(b + 1) * head_dim];
            let w = scores[raw + i];
            for (a, &v) in acc.iter_mut().zip(kh) {
                *a += w * v;
            }
        }
        attn_out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&acc);
    }
    let mut neg = inv_freqs.to_vec();
    for v in &mut neg {
        *v = -*v;
    }
    rope::rope_inplace_with_freqs_ex(
        &mut attn_out, n_heads, head_dim, rope_dim, position, &neg, false, true,
    );
    let _ = mla_trace(layer_ord, "attn_out", &attn_out);

    let group_dim = n_heads * head_dim / o_groups.max(1);
    let group_elems = o_lora * group_dim;
    let group_bytes = wo_a_dtype.byte_size(group_elems);
    let mut oa = vec![0.0f32; o_groups * o_lora];
    for g in 0..o_groups {
        let xin = &attn_out[g * group_dim..(g + 1) * group_dim];
        let w_off = g * group_bytes;
        if w_off + group_bytes > wo_a.len() {
            return Err(FellmError::other(format!(
                "mla wo_a group {g}: offset {w_off}+{group_bytes} > {}",
                wo_a.len()
            )));
        }
        let slice = &wo_a[w_off..w_off + group_bytes];
        matmul::matvec_quant(
            slice,
            wo_a_dtype,
            xin,
            &mut oa[g * o_lora..(g + 1) * o_lora],
            o_lora,
            group_dim,
        )?;
        if g == 0 {
            mla_trace_proj(
                layer_ord,
                "attn_output_a_g0",
                slice,
                wo_a_dtype,
                xin,
                &oa[g * o_lora..(g + 1) * o_lora],
                o_lora,
                group_dim,
            );
        }
    }
    let _ = mla_trace(layer_ord, "attn_output_a", &oa);
    matmul::matvec_quant(wo_b, wo_b_dtype, &oa, y, d_model, o_groups * o_lora)?;
    mla_trace_proj(
        layer_ord,
        "attn_output_b",
        wo_b,
        wo_b_dtype,
        &oa,
        y,
        d_model,
        o_groups * o_lora,
    );
    let _ = mla_trace(layer_ord, "attn_output_b", y);
    Ok(())
}

impl<'a> Default for MlaExtras<'a> {
    fn default() -> Self {
        Self {
            compress_kv: None,
            compress_gate: None,
            compress_ape: None,
            compress_norm: None,
            compress_state_dim: 0,
            indexer_q_b: None,
            indexer_proj: None,
            indexer_comp_kv: None,
            indexer_comp_gate: None,
            indexer_comp_ape: None,
            indexer_comp_norm: None,
            indexer_state_dim: 0,
            indexer_heads: 0,
            indexer_head_dim: 0,
            indexer_top_k: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(xs: &[f32]) -> Vec<f32> {
        let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = xs.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.into_iter().map(|v| v / s).collect()
    }

    fn eye(n: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    fn stacked_eye(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        for r in 0..rows {
            w[r * cols + (r % cols)] = 1.0;
        }
        w
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn csa_first_block_zero_pads_prev_and_uses_per_feature_softmax() {
        let head_dim = 2;
        let ratio = 2;
        let state_dim = 4;
        let kv = vec![
            1.0, 2.0, 10.0, 20.0, // t0: prev half unused, cur half
            3.0, 4.0, 30.0, 40.0, // t1
        ];
        let score = vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0];
        let out = compress_block(
            &kv, &score, state_dim, 2, ratio, head_dim, true, None, 1e-6, 0, &[],
        )
        .unwrap();
        let mut expect = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            let logits = [0.0, 0.0, score[d + 2], score[state_dim + d + 2]];
            let w = softmax(&logits);
            expect[d] = w[2] * kv[d + 2] + w[3] * kv[state_dim + d + 2];
        }
        close(&out, &expect);
    }

    #[test]
    fn csa_second_block_reads_prev_first_half() {
        let head_dim = 2;
        let ratio = 2;
        let state_dim = 4;
        let kv = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0,
        ];
        let score = vec![
            0.1, 0.2, 1.0, 2.0, 0.3, 0.4, 3.0, 4.0, 0.5, 0.6, 5.0, 6.0, 0.7, 0.8, 7.0, 8.0,
        ];
        let out = compress_block(
            &kv, &score, state_dim, 4, ratio, head_dim, true, None, 1e-6, 0, &[],
        )
        .unwrap();
        let mut expect = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            let logits = [
                score[d],
                score[state_dim + d],
                score[2 * state_dim + head_dim + d],
                score[3 * state_dim + head_dim + d],
            ];
            let w = softmax(&logits);
            expect[d] = w[0] * kv[d]
                + w[1] * kv[state_dim + d]
                + w[2] * kv[2 * state_dim + head_dim + d]
                + w[3] * kv[3 * state_dim + head_dim + d];
        }
        close(&out, &expect);
    }

    #[test]
    fn hca_pools_ratio_tokens_per_feature() {
        let head_dim = 2;
        let ratio = 2;
        let kv = vec![1.0, 10.0, 3.0, 30.0];
        let score = vec![1.0, 2.0, 3.0, 4.0];
        let out = compress_block(&kv, &score, head_dim, 2, ratio, head_dim, false, None, 1e-6, 0, &[])
            .unwrap();
        let mut expect = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            let logits = [score[d], score[head_dim + d]];
            let w = softmax(&logits);
            expect[d] = w[0] * kv[d] + w[1] * kv[head_dim + d];
        }
        close(&out, &expect);
    }

    #[test]
    fn lightning_top_k_discards_low_scores() {
        let scores = [0.1, 5.0, 0.2, 4.0];
        let idx = top_k_indices(&scores, 2);
        assert_eq!(idx, vec![1, 3]);
        let all = top_k_indices(&scores, 8);
        assert_eq!(all.len(), 4);
    }

    fn run_tokens(
        xs: &[Vec<f32>],
        extras_fn: impl Fn() -> MlaExtras<'static>,
        ratio: usize,
        top_k: usize,
        layer_ord: u32,
    ) -> Vec<Vec<f32>> {
        let d = 2;
        let qa = eye(d);
        let ones = vec![1.0f32; d];
        let mut k_out = vec![0.0f32; xs.len() * d];
        let mut c_out = vec![0.0f32; xs.len() * d];
        let qa_b = qa.clone();
        let mut ys = Vec::new();
        for (pos, x) in xs.iter().enumerate() {
            let mut extras = extras_fn();
            extras.indexer_top_k = top_k;
            let mut y = vec![0.0f32; d];
            mla_decode(
                x,
                bytemuck::cast_slice(&qa),
                DType::F32,
                &ones,
                bytemuck::cast_slice(&qa_b),
                DType::F32,
                bytemuck::cast_slice(&qa),
                DType::F32,
                &ones,
                bytemuck::cast_slice(&qa),
                DType::F32,
                bytemuck::cast_slice(&qa),
                DType::F32,
                &[],
                &[],
                &mut k_out,
                &mut c_out,
                &mut y,
                d,
                1,
                d,
                0,
                d,
                1,
                d,
                pos as u32,
                pos as u32,
                128,
                1e-6,
                ratio,
                extras,
                layer_ord,
            )
            .unwrap();
            ys.push(y);
        }
        ys
    }

    #[test]
    fn first_completed_csa_block_is_visible() {
        let d = 2;
        let ck_store: &'static [f32] = Box::leak(stacked_eye(4, d).into_boxed_slice());
        let extras = || MlaExtras {
            compress_kv: Some((bytemuck::cast_slice(ck_store), DType::F32)),
            compress_gate: Some((bytemuck::cast_slice(ck_store), DType::F32)),
            compress_state_dim: 4,
            ..MlaExtras::default()
        };
        let xs = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let ys = run_tokens(&xs, extras, 2, 0, 11);
        assert_eq!(ys.len(), 2);
        assert!(ys[1].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn indexer_top_k_changes_output_when_blocks_exceed_k() {
        let d = 2;
        let ck: &'static [f32] = Box::leak(stacked_eye(4, d).into_boxed_slice());
        let iq: &'static [f32] = Box::leak(eye(d).into_boxed_slice());
        let ip: &'static [f32] = Box::leak(vec![1.0f32, 0.0].into_boxed_slice());
        let extras = || MlaExtras {
            compress_kv: Some((bytemuck::cast_slice(ck), DType::F32)),
            compress_gate: Some((bytemuck::cast_slice(ck), DType::F32)),
            compress_state_dim: 4,
            indexer_q_b: Some((bytemuck::cast_slice(iq), DType::F32)),
            indexer_proj: Some((bytemuck::cast_slice(ip), DType::F32)),
            indexer_comp_kv: Some((bytemuck::cast_slice(ck), DType::F32)),
            indexer_comp_gate: Some((bytemuck::cast_slice(ck), DType::F32)),
            indexer_state_dim: 4,
            indexer_heads: 1,
            indexer_head_dim: 2,
            indexer_top_k: 1,
            ..MlaExtras::default()
        };
        let xs: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let mut x = vec![0.0f32; 2];
                x[i % 2] = (i as f32) + 1.0;
                x
            })
            .collect();
        let y_all = run_tokens(&xs, extras, 2, 8, 21);
        let y_k = run_tokens(&xs, extras, 2, 1, 22);
        assert_ne!(y_all.last().unwrap(), y_k.last().unwrap());
    }
}
