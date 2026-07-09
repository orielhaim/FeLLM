//! ShortConv decode kernel.

use crate::kernels::matmul;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};

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
                .map_err(|e| FellmError::other(format!("shortconv: f32 cast: {e:?}")))?;
            matmul::matvec_f32(w, x, y, out_dim, in_dim);
            Ok(())
        }
        DType::Q4_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            matmul::matvec_quant(w_bytes, w_dtype, x, y, out_dim, in_dim)
        }
        other => Err(FellmError::UnsupportedDType(other)),
    }
}

/// Run one-token ShortConv.
///
/// State layout is time-major `[d_conv, n_embd]`, indexed as `t * n_embd + c`.
pub fn shortconv_decode(
    x: &[f32],
    in_proj_bytes: &[u8],
    in_proj_dtype: DType,
    conv_w: &[f32],
    out_proj_bytes: &[u8],
    out_proj_dtype: DType,
    state: &mut [f32],
    y: &mut [f32],
    n_embd: usize,
    l_cache: usize,
) -> Result<()> {
    if n_embd == 0 || l_cache < 1 {
        return Err(FellmError::other("shortconv: bad dimensions"));
    }
    if x.len() != n_embd || y.len() != n_embd {
        return Err(FellmError::other(format!(
            "shortconv: x/y len mismatch (x={}, y={}, n_embd={n_embd})",
            x.len(),
            y.len()
        )));
    }
    if conv_w.len() != n_embd * l_cache {
        return Err(FellmError::other(format!(
            "shortconv: conv len {} != {}",
            conv_w.len(),
            n_embd * l_cache
        )));
    }
    let d_conv = l_cache - 1;
    if state.len() != d_conv * n_embd {
        return Err(FellmError::other(format!(
            "shortconv: state len {} != {}",
            state.len(),
            d_conv * n_embd
        )));
    }

    let mut bcx = vec![0.0f32; 3 * n_embd];
    matvec_weight(
        in_proj_bytes,
        in_proj_dtype,
        x,
        &mut bcx,
        3 * n_embd,
        n_embd,
    )?;

    let (b, rest) = bcx.split_at(n_embd);
    let (c_gate, x_proj) = rest.split_at(n_embd);
    let mut bx = vec![0.0f32; n_embd];
    for i in 0..n_embd {
        bx[i] = b[i] * x_proj[i];
    }

    let mut y_pre = vec![0.0f32; n_embd];
    for c in 0..n_embd {
        let mut conv_out = bx[c] * conv_w[c * l_cache + d_conv];
        for t in 0..d_conv {
            conv_out += state[t * n_embd + c] * conv_w[c * l_cache + t];
        }
        y_pre[c] = c_gate[c] * conv_out;
    }

    for t in 0..d_conv {
        for c in 0..n_embd {
            state[t * n_embd + c] = if t + 1 < d_conv {
                state[(t + 1) * n_embd + c]
            } else {
                bx[c]
            };
        }
    }

    matvec_weight(out_proj_bytes, out_proj_dtype, &y_pre, y, n_embd, n_embd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortconv_decode_shifts_state() {
        let n_embd = 4;
        let l_cache = 3;
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut in_proj = vec![0.0f32; 3 * n_embd * n_embd];
        for i in 0..n_embd {
            in_proj[i * n_embd + i] = 1.0;
            in_proj[(n_embd + i) * n_embd + i] = 1.0;
            in_proj[(2 * n_embd + i) * n_embd + i] = 1.0;
        }
        let mut conv = vec![0.0f32; n_embd * l_cache];
        for c in 0..n_embd {
            conv[c * l_cache + l_cache - 1] = 1.0;
        }
        let mut out_proj = vec![0.0f32; n_embd * n_embd];
        for i in 0..n_embd {
            out_proj[i * n_embd + i] = 1.0;
        }
        let mut state = vec![0.0f32; (l_cache - 1) * n_embd];
        state[..n_embd].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
        state[n_embd..].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        let mut y = vec![0.0f32; n_embd];

        shortconv_decode(
            &x,
            bytemuck::cast_slice(&in_proj),
            DType::F32,
            &conv,
            bytemuck::cast_slice(&out_proj),
            DType::F32,
            &mut state,
            &mut y,
            n_embd,
            l_cache,
        )
        .unwrap();

        assert_eq!(y, [1.0, 8.0, 27.0, 64.0]);
        assert_eq!(&state[..n_embd], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&state[n_embd..], &[1.0, 4.0, 9.0, 16.0]);
    }
}
