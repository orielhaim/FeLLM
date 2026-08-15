//! Hyper-connection residual mixing (pre / post / head + Sinkhorn).

use crate::kernels::{matmul, norm};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `mode` 0 = pre (hc*d -> d), 1 = post (d + hc*d -> hc*d), 2 = head (hc*d -> d).
pub fn hyper_connection(
    x: &[f32],
    residual: Option<&[f32]>,
    hc_fn: &[u8],
    hc_fn_dtype: DType,
    hc_scale: &[f32],
    hc_base: &[f32],
    y: &mut [f32],
    d_model: usize,
    hc: usize,
    mode: u32,
    sinkhorn_iters: u32,
    eps: f32,
    rms_eps: f32,
) -> Result<()> {
    let hc_dim = hc * d_model;
    let x_stride = if mode == 1 { d_model } else { hc_dim };
    let y_stride = if mode == 1 { hc_dim } else { d_model };
    let rows = if y_stride == 0 {
        1
    } else {
        (y.len() / y_stride).max(1)
    };
    if rows > 1 {
        for row in 0..rows {
            let residual_row = residual.map(|values| &values[row * hc_dim..(row + 1) * hc_dim]);
            hyper_connection(
                &x[row * x_stride..(row + 1) * x_stride],
                residual_row,
                hc_fn,
                hc_fn_dtype,
                hc_scale,
                hc_base,
                &mut y[row * y_stride..(row + 1) * y_stride],
                d_model,
                hc,
                mode,
                sinkhorn_iters,
                eps,
                rms_eps,
            )?;
        }
        return Ok(());
    }
    if x.len() < hc_dim && mode != 1 {
        return Err(FellmError::other("deepseek4 hc: x width"));
    }
    let src = if mode == 1 {
        residual.ok_or_else(|| FellmError::other("deepseek4 hc post missing residual"))?
    } else {
        x
    };
    if src.len() != hc_dim {
        return Err(FellmError::other("deepseek4 hc: residual width"));
    }
    let mix_dim = (2 + hc) * hc;
    let ones = vec![1.0f32; hc_dim];
    let mut flat_norm = vec![0.0f32; hc_dim];
    norm::rmsnorm_row(src, &ones, rms_eps, &mut flat_norm);
    let mut mixes = vec![0.0f32; mix_dim];
    let mix_filled;
    match hc_fn_dtype {
        DType::F32 => {
            let w: &[f32] = bytemuck::cast_slice(hc_fn);
            if w.len() == mix_dim.saturating_mul(hc_dim) {
                matmul::matvec_f32(w, &flat_norm, &mut mixes, mix_dim, hc_dim);
                mix_filled = mix_dim;
            } else if w.len() == hc.saturating_mul(hc_dim) {
                matmul::matvec_f32(w, &flat_norm, &mut mixes[..hc], hc, hc_dim);
                mix_filled = hc;
            } else if w.len() == mix_dim.saturating_mul(d_model) {
                let mut reduced = vec![0.0f32; d_model];
                for h in 0..hc {
                    for (dst, &v) in reduced.iter_mut().zip(&src[h * d_model..(h + 1) * d_model]) {
                        *dst += v;
                    }
                }
                for v in &mut reduced {
                    *v /= hc as f32;
                }
                let mut reduced_norm = vec![0.0f32; d_model];
                let ones_d = vec![1.0f32; d_model];
                norm::rmsnorm_row(&reduced, &ones_d, rms_eps, &mut reduced_norm);
                matmul::matvec_f32(w, &reduced_norm, &mut mixes, mix_dim, d_model);
                mix_filled = mix_dim;
            } else {
                return Err(FellmError::other(format!(
                    "deepseek4 hc fn: w={} mix={mix_dim} hc_dim={hc_dim} d_model={d_model}",
                    w.len()
                )));
            }
        }
        other => {
            if hc_fn.len() == other.byte_size(mix_dim.saturating_mul(hc_dim)) {
                matmul::matvec_quant(hc_fn, other, &flat_norm, &mut mixes, mix_dim, hc_dim)?;
                mix_filled = mix_dim;
            } else if hc_fn.len() == other.byte_size(hc.saturating_mul(hc_dim)) {
                matmul::matvec_quant(hc_fn, other, &flat_norm, &mut mixes[..hc], hc, hc_dim)?;
                mix_filled = hc;
            } else {
                return Err(FellmError::other(format!(
                    "deepseek4 hc fn quant: bytes={} dtype={other:?}",
                    hc_fn.len()
                )));
            }
        }
    }

    let scale_pre = hc_scale.first().copied().unwrap_or(1.0);
    let scale_post = hc_scale.get(1).copied().unwrap_or(1.0);
    let scale_comb = hc_scale.get(2).copied().unwrap_or(1.0);
    let full_mix = mix_filled >= mix_dim;

    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    for i in 0..hc {
        pre[i] = sigmoid(mixes[i] * scale_pre + hc_base.get(i).copied().unwrap_or(0.0)) + eps;
        post[i] = if full_mix {
            sigmoid(mixes[hc + i] * scale_post + hc_base.get(hc + i).copied().unwrap_or(0.0)) * 2.0
        } else {
            1.0
        };
    }
    let mut comb = vec![0.0f32; hc * hc];
    if full_mix {
        for i in 0..hc * hc {
            comb[i] = mixes[2 * hc + i] * scale_comb + hc_base.get(2 * hc + i).copied().unwrap_or(0.0);
        }
        sinkhorn(&mut comb, hc, sinkhorn_iters.max(1), eps);
    } else {
        // Reduced `[hc, hc_dim]` exports only produce the pre logits. Keep a residual
        // identity mix so streams are not zeroed by sinkhorn-on-zeros.
        for i in 0..hc {
            comb[i * hc + i] = 1.0;
        }
    }

    match mode {
        0 => {
            if y.len() != d_model {
                return Err(FellmError::other("deepseek4 hc pre: y width"));
            }
            y.fill(0.0);
            for h in 0..hc {
                let row = &src[h * d_model..(h + 1) * d_model];
                for (out, &v) in y.iter_mut().zip(row) {
                    *out += v * pre[h];
                }
            }
        }
        1 => {
            if y.len() != hc_dim || x.len() != d_model {
                return Err(FellmError::other("deepseek4 hc post: shape"));
            }
            for dst in 0..hc {
                let out = &mut y[dst * d_model..(dst + 1) * d_model];
                for (o, &v) in out.iter_mut().zip(x.iter()) {
                    *o = v * post[dst];
                }
                for src_h in 0..hc {
                    let res = &src[src_h * d_model..(src_h + 1) * d_model];
                    let w = comb[dst + src_h * hc];
                    for (o, &v) in out.iter_mut().zip(res) {
                        *o += v * w;
                    }
                }
            }
        }
        _ => {
            if y.len() != d_model {
                return Err(FellmError::other("deepseek4 hc head: y width"));
            }
            y.fill(0.0);
            for h in 0..hc {
                let row = &src[h * d_model..(h + 1) * d_model];
                for (out, &v) in y.iter_mut().zip(row) {
                    *out += v * pre[h];
                }
            }
        }
    }
    Ok(())
}

fn sinkhorn(comb: &mut [f32], hc: usize, iters: u32, eps: f32) {
    // Softmax over the destination axis (inner dim), then Sinkhorn-Knopp.
    for src in 0..hc {
        let row = &mut comb[src * hc..(src + 1) * hc];
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            max = max.max(v);
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    for v in comb.iter_mut() {
        *v += eps;
    }
    for dst in 0..hc {
        let mut s = eps;
        for src in 0..hc {
            s += comb[dst + src * hc];
        }
        for src in 0..hc {
            comb[dst + src * hc] /= s;
        }
    }
    for _ in 1..iters.max(1) {
        for src in 0..hc {
            let mut s = eps;
            for dst in 0..hc {
                s += comb[dst + src * hc];
            }
            for dst in 0..hc {
                comb[dst + src * hc] /= s;
            }
        }
        for dst in 0..hc {
            let mut s = eps;
            for src in 0..hc {
                s += comb[dst + src * hc];
            }
            for src in 0..hc {
                comb[dst + src * hc] /= s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_core::dtype::DType;

    fn eye_rows(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        let n = rows.min(cols);
        for i in 0..n {
            w[i * cols + i] = 1.0;
        }
        w
    }

    #[test]
    fn hc_pre_sums_streams() {
        let d = 2;
        let hc = 2;
        let x = vec![1.0f32, 0.0, 0.0, 1.0];
        let w = eye_rows(hc, hc * d);
        let scale = [1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; 3 * hc];
        let mut y = vec![0.0f32; d];
        hyper_connection(
            &x,
            None,
            bytemuck::cast_slice(&w),
            DType::F32,
            &scale,
            &base,
            &mut y,
            d,
            hc,
            0,
            1,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn hc_post_keeps_hc_width() {
        let d = 2;
        let hc = 2;
        let residual = vec![1.0f32, 0.0, 0.0, 1.0];
        let x = vec![0.5f32, 0.25];
        let mix_dim = (2 + hc) * hc;
        let w = vec![0.0f32; mix_dim * hc * d];
        let scale = [1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; mix_dim];
        let mut y = vec![0.0f32; hc * d];
        hyper_connection(
            &x,
            Some(&residual),
            bytemuck::cast_slice(&w),
            DType::F32,
            &scale,
            &base,
            &mut y,
            d,
            hc,
            1,
            4,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert_eq!(y.len(), 4);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}

