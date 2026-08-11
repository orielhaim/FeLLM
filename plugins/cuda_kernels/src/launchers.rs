//! FFI launch wrappers: activation-cached H2D → oxide kernel → optional D2H.

use crate::buffers;
use crate::tensor::{bytes_slice, dims, f32_slice, f32_slice_mut, u32_slice};
use crate::{
    Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q5_0_BLOCK_ELEMS, Q6K_BLOCK_BYTES, Q6K_BLOCK_ELEMS,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS, host_paged_snapshot, oxide_ctx, oxide_module, oxide_stream,
    with_step_params,
};
use cuda_core::{DeviceBuffer, LaunchConfig};
use fellm_core::dtype::DType;
use fellm_plugin_abi::op::OpAttrs;
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use half::f16;
use std::sync::OnceLock;

static FORCE_SYNC_OUTPUTS: OnceLock<bool> = OnceLock::new();

fn cfg_1d(n: u32) -> LaunchConfig {
    LaunchConfig::for_num_elems(n.max(1))
}

fn cfg_warp_blocks(blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// FA2 prefill: grid (q_tiles, n_heads), 4 warps, SMEM for K/V tiles.
fn cfg_fa2_prefill(q_tiles: u32, n_heads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (q_tiles.max(1), n_heads.max(1), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0, // static SharedArray in kernel
    }
}

/// FA2/FA3 decode: one block per head, 4 warps, SMEM tiles.
fn cfg_fa_decode(n_heads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_heads.max(1), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn cfg_mmvq_blocks(blocks: u32, _n_blocks: u32) -> LaunchConfig {
    let warps = 4;
    LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (warps * 32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn cfg_block_256() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

type LaunchResult = Result<(), i32>;

fn run(body: impl FnOnce() -> LaunchResult) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => 0,
        Ok(Err(c)) => c,
        Err(_) => -99,
    }
}

/// Greedy sampling from device-resident logits; downloads one f32 token id.
pub unsafe extern "C" fn launch_sample_greedy(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if inputs.is_null() || outputs.is_null() || n_inputs < 1 || n_outputs < 1 {
            return Err(-1);
        }
        let logits = f32_slice(unsafe { &*inputs })?;
        let out = f32_slice_mut(unsafe { &mut *outputs })?;
        if logits.is_empty() || out.len() != 1 {
            return Err(-2);
        }
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let input_key = buffers::ensure_f32(&stream, logits, false)?;
        let output_key = buffers::ensure_f32_out(&stream, out)?;
        let (input, _) = buffers::take_f32(input_key)?;
        let (mut output, _) = buffers::take_f32(output_key)?;
        let result = unsafe {
            oxide_module()
                .argmax_token(
                    &stream,
                    cfg_block_256(),
                    &input,
                    logits.len() as u32,
                    &mut output,
                )
                .map_err(|_| -4)
        };
        buffers::put_f32(input_key, input, true)?;
        buffers::put_f32(output_key, output, false)?;
        result?;
        finish_out(&stream, output_key, out, true)
    })
}

/// Explicitly materialize a cached f32 device tensor into host storage.
pub unsafe extern "C" fn launch_materialize_f32(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if inputs.is_null() || outputs.is_null() || n_inputs < 1 || n_outputs < 1 {
            return Err(-1);
        }
        let source = f32_slice(unsafe { &*inputs })?;
        let host = f32_slice_mut(unsafe { &mut *outputs })?;
        if source.len() != host.len() {
            return Err(-2);
        }
        let stream = oxide_stream().clone();
        let key = buffers::ensure_f32(&stream, source, false)?;
        buffers::download_to(&stream, key, host)
    })
}

/// Finish a launch: mark output device-valid; D2H only when `sync_host` (e.g. lm_head logits).
fn finish_out(
    stream: &cuda_core::CudaStream,
    out_key: buffers::BufferKey,
    out: &mut [f32],
    sync_host: bool,
) -> LaunchResult {
    buffers::mark_valid(out_key)?;
    let force_sync = *FORCE_SYNC_OUTPUTS.get_or_init(|| {
        std::env::var_os("FELLM_CUDA_SYNC_OUTPUTS")
            .is_some_and(|value| value != "0" && value != "false" && value != "off")
    });
    if sync_host || force_sync {
        buffers::download_to(stream, out_key, out)?;
    }
    Ok(())
}

/// `out = silu(gate) * up`
pub unsafe extern "C" fn launch_silu_gate(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let gate_t = unsafe { &*inputs };
        let up_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        let gate = f32_slice(gate_t)?;
        let up = f32_slice(up_t)?;
        let out = f32_slice_mut(out_t)?;
        if gate.len() != up.len() || gate.len() != out.len() {
            return Err(-2);
        }
        let n = gate.len();
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let g_key = buffers::ensure_f32(&stream, gate, false)?;
        let u_key = buffers::ensure_f32(&stream, up, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (g, _) = buffers::take_f32(g_key)?;
        let (u, _) = buffers::take_f32(u_key)?;
        let (mut o, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = unsafe {
            module
                .silu_gate(&stream, cfg_1d(n as u32), &g, &u, &mut o)
                .map_err(|_| -4)
        };
        buffers::put_f32(g_key, g, true)?;
        buffers::put_f32(u_key, u, true)?;
        buffers::put_f32(o_key, o, false)?;
        rc?;
        finish_out(&stream, o_key, out, false)
    })
}

/// RMSNorm (row or grouped by head_dim).
pub unsafe extern "C" fn launch_rmsnorm(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || attrs.is_null() || inputs.is_null() || outputs.is_null()
        {
            return Err(-1);
        }
        let attrs = unsafe { &*attrs };
        let x_t = unsafe { &*inputs };
        let w_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        let x = f32_slice(x_t)?;
        let w = f32_slice(w_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() != out.len() {
            return Err(-2);
        }
        if w.is_empty() || x.len() % w.len() != 0 {
            return Err(-2);
        }
        let group_n = w.len() as u32;
        let n_groups = (x.len() / w.len()) as u32;
        let group_stride = group_n;

        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let w_key = buffers::ensure_f32(&stream, w, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (wd, _) = buffers::take_f32(w_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let launch = LaunchConfig {
            grid_dim: (n_groups, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rc = unsafe {
            module
                .rmsnorm_group(
                    &stream,
                    launch,
                    &xd,
                    &wd,
                    attrs.eps,
                    group_n,
                    group_stride,
                    &mut od,
                )
                .map_err(|_| -4)
        };
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(w_key, wd, true)?;
        buffers::put_f32(o_key, od, false)?;
        rc?;
        finish_out(&stream, o_key, out, false)
    })
}

/// RoPE with precomputed inv_freqs.
pub unsafe extern "C" fn launch_rope(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || attrs.is_null() || inputs.is_null() || outputs.is_null()
        {
            return Err(-1);
        }
        let attrs = unsafe { &*attrs };
        let x_t = unsafe { &*inputs };
        let f_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        let x = f32_slice(x_t)?;
        let inv = f32_slice(f_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() != out.len() {
            return Err(-2);
        }
        let batch = host_paged_snapshot().filter(|snapshot| snapshot.batch_size > 1);
        let n = x.len();
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        if attrs.custom_op_id == 1 {
            let row_width = attrs.n_heads.max(1) as usize * attrs.head_dim.max(1) as usize;
            if row_width > 0 && n >= row_width {
                for row in 0..n / row_width {
                    let position = batch
                        .as_ref()
                        .map_or(attrs.position + row as u32, |snapshot| unsafe {
                            *snapshot.row_rope_positions.add(row)
                        });
                    fellm_plugin_abi::pre_rope_write(
                        attrs.layer_ord as usize,
                        position as usize,
                        &x[row * row_width..(row + 1) * row_width],
                    );
                }
            }
        }
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let f_key = buffers::ensure_f32(&stream, inv, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (fd, _) = buffers::take_f32(f_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = if let Some(snapshot) = batch {
            let positions = unsafe {
                std::slice::from_raw_parts(snapshot.row_rope_positions, snapshot.batch_size)
            };
            let p_key = buffers::ensure_u32(&stream, positions)?;
            let pd = buffers::take_u32(p_key)?;
            let result = unsafe {
                module
                    .rope_batch(
                        &stream,
                        cfg_1d(n as u32),
                        &xd,
                        &fd,
                        &pd,
                        attrs.n_heads,
                        attrs.head_dim,
                        attrs.rope_dim,
                        n as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            };
            buffers::put_u32(p_key, pd)?;
            result
        } else {
            with_step_params(|params| unsafe {
                module
                    .rope_controlled(
                        &stream,
                        cfg_1d(n as u32),
                        &xd,
                        &fd,
                        params,
                        attrs.n_heads,
                        attrs.head_dim,
                        attrs.rope_dim,
                        n as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            })
        };
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(f_key, fd, true)?;
        buffers::put_f32(o_key, od, false)?;
        rc?;
        finish_out(&stream, o_key, out, false)
    })
}

/// Elementwise add: `out = a + b`.
pub unsafe extern "C" fn launch_add(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let a_t = unsafe { &*inputs };
        let b_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        let a = f32_slice(a_t)?;
        let b = f32_slice(b_t)?;
        let out = f32_slice_mut(out_t)?;
        if a.len() != b.len() || a.len() != out.len() {
            return Err(-2);
        }
        let n = a.len();
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let a_key = buffers::ensure_f32(&stream, a, false)?;
        let b_key = buffers::ensure_f32(&stream, b, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (ad, _) = buffers::take_f32(a_key)?;
        let (bd, _) = buffers::take_f32(b_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = unsafe {
            module
                .add_f32(&stream, cfg_1d(n as u32), &ad, &bd, &mut od)
                .map_err(|_| -4)
        };
        buffers::put_f32(a_key, ad, true)?;
        buffers::put_f32(b_key, bd, true)?;
        buffers::put_f32(o_key, od, false)?;
        rc?;
        finish_out(&stream, o_key, out, false)
    })
}

pub unsafe extern "C" fn launch_mul(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let a = f32_slice(unsafe { &*inputs })?;
        let b = f32_slice(unsafe { &*inputs.add(1) })?;
        let out = f32_slice_mut(unsafe { &mut *outputs })?;
        if a.len() != b.len() || a.len() != out.len() {
            return Err(-2);
        }
        let stream = oxide_stream().clone();
        let ak = buffers::ensure_f32(&stream, a, false)?;
        let bk = buffers::ensure_f32(&stream, b, false)?;
        let ok = buffers::ensure_f32_out(&stream, out)?;
        let (ad, _) = buffers::take_f32(ak)?;
        let (bd, _) = buffers::take_f32(bk)?;
        let (mut od, _) = buffers::take_f32(ok)?;
        let rc = unsafe {
            oxide_module()
                .mul_f32(&stream, cfg_1d(a.len() as u32), &ad, &bd, &mut od)
                .map_err(|_| -4)
        };
        buffers::put_f32(ak, ad, true)?;
        buffers::put_f32(bk, bd, true)?;
        buffers::put_f32(ok, od, false)?;
        rc?;
        finish_out(&stream, ok, out, false)
    })
}

/// F32 embedding: weight `[vocab, dim]` × token id → row.
pub unsafe extern "C" fn launch_embedding_f32(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let tok_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let dim = wdims[1] as usize;
        let table = f32_slice(w_t)?;
        let ids = u32_slice(tok_t)?;
        if ids.is_empty() {
            return Err(-2);
        }
        let out = f32_slice_mut(out_t)?;
        if out.len() != ids.len() * dim
            || table.len() < (ids.iter().copied().max().unwrap_or(0) as usize + 1) * dim
        {
            return Err(-2);
        }
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        // Weight table: cache via ensure_f32 (stable host ptr).
        let w_key = buffers::ensure_f32(&stream, table, false)?;
        let i_key = buffers::ensure_u32(&stream, ids)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (wd, _) = buffers::take_f32(w_key)?;
        let id = buffers::take_u32(i_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = unsafe {
            module
                .embedding_f32_rows(
                    &stream,
                    cfg_1d(out.len() as u32),
                    &wd,
                    &id,
                    ids.len() as u32,
                    dim as u32,
                    &mut od,
                )
                .map_err(|_| -4)
        };
        buffers::put_f32(w_key, wd, true)?;
        buffers::put_u32(i_key, id)?;
        buffers::put_f32(o_key, od, false)?;
        rc?;
        finish_out(&stream, o_key, out, false)
    })
}

/// Q4_K embedding: dequantize one weight row on device.
pub unsafe extern "C" fn launch_embedding_q4k(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let tok_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        if w_t.dtype != DType::Q4K as u32 {
            return Err(-10);
        }
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let dim = wdims[1] as usize;
        if dim % (Q4K_BLOCK_ELEMS as usize) != 0 {
            return Err(-2);
        }
        let n_blocks = (dim / Q4K_BLOCK_ELEMS as usize) as u32;
        let wb = bytes_slice(w_t);
        let ids = u32_slice(tok_t)?;
        if ids.is_empty() {
            return Err(-2);
        }
        let out = f32_slice_mut(out_t)?;
        if out.len() != ids.len() * dim {
            return Err(-2);
        }
        let row_bytes = n_blocks as usize * Q4K_BLOCK_BYTES as usize;
        let need = (ids.iter().copied().max().unwrap_or(0) as usize + 1) * row_bytes;
        if wb.len() < need {
            return Err(-2);
        }

        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let mut batch_ids = None;
        let rc = buffers::with_weight(w_key, |wd| {
            if ids.len() == 1 {
                with_step_params(|params| unsafe {
                    module
                        .embedding_q4k_row(
                            &stream,
                            cfg_1d(dim as u32),
                            wd,
                            params,
                            dim as u32,
                            n_blocks,
                            &mut od,
                        )
                        .map_err(|_| -4)
                })
            } else {
                let key = buffers::ensure_u32(&stream, ids)?;
                let device = buffers::take_u32(key)?;
                let result = unsafe {
                    module
                        .embedding_q4k_rows(
                            &stream,
                            cfg_1d(out.len() as u32),
                            wd,
                            &device,
                            ids.len() as u32,
                            dim as u32,
                            n_blocks,
                            &mut od,
                        )
                        .map_err(|_| -4)
                };
                batch_ids = Some((key, device));
                result
            }
        });
        if let Some((key, device)) = batch_ids {
            buffers::put_u32(key, device)?;
        }
        buffers::put_f32(o_key, od, false)?;
        rc??;
        finish_out(&stream, o_key, out, false)
    })
}

/// Q4_K matvec: weight [out,in] Q4K × x [in] f32 → y [out] f32.
pub unsafe extern "C" fn launch_q4k_matmul(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let x_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        if w_t.dtype != DType::Q4K as u32 {
            return Err(-10);
        }
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let out_dim = wdims[0] as u32;
        let in_dim = wdims[1] as usize;
        if in_dim % (Q4K_BLOCK_ELEMS as usize) != 0 {
            return Err(-2);
        }
        let n_blocks = (in_dim / Q4K_BLOCK_ELEMS as usize) as u32;
        let wb = bytes_slice(w_t);
        let expect = out_dim as usize * n_blocks as usize * Q4K_BLOCK_BYTES as usize;
        if wb.len() < expect {
            return Err(-2);
        }
        let x = f32_slice(x_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() % in_dim != 0 {
            return Err(-2);
        }
        let rows = x.len() / in_dim;
        if out.len() != rows * out_dim as usize {
            return Err(-2);
        }
        let residual = if n_inputs >= 3 {
            let values = f32_slice(unsafe { &*inputs.add(2) })?;
            if values.len() != out.len() {
                return Err(-2);
            }
            Some(values)
        } else {
            None
        };

        let stream = oxide_stream().clone();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let residual_key = residual
            .map(|values| buffers::ensure_f32(&stream, values, false))
            .transpose()?;
        let mut residual_device = residual_key.map(buffers::take_f32).transpose()?;
        let module = oxide_module();
        let rc = buffers::with_q8_activation(
            &stream,
            x_key,
            x.len(),
            |qx, x_scales| unsafe {
                module
                    .quantize_q8_32(
                        &stream,
                        cfg_warp_blocks(x.len().div_ceil(32) as u32),
                        &xd,
                        qx,
                        x_scales,
                    )
                    .map_err(|_| -4)
            },
            |qx, x_scales| {
                buffers::with_weight(w_key, |wd| unsafe {
                    if rows > 0 {
                        module
                            .q4k_q8_gemv_multiwarp(
                                &stream,
                                cfg_mmvq_blocks(out_dim * rows as u32, n_blocks),
                                wd,
                                qx,
                                x_scales,
                                residual_device.as_ref().map_or(&xd, |(buffer, _)| buffer),
                                u32::from(residual_device.is_some()),
                                out_dim,
                                n_blocks,
                                rows as u32,
                                &mut od,
                            )
                            .map_err(|_| -4)
                    } else {
                        module
                            .q4k_gemv_row(
                                &stream,
                                cfg_1d(out_dim * rows as u32),
                                wd,
                                &xd,
                                out_dim,
                                n_blocks,
                                rows as u32,
                                &mut od,
                            )
                            .map_err(|_| -4)
                    }
                })
            },
        );
        buffers::put_f32(x_key, xd, true)?;
        if let (Some(key), Some((buffer, valid))) = (residual_key, residual_device.take()) {
            buffers::put_f32(key, buffer, valid)?;
        }
        buffers::put_f32(o_key, od, false)?;
        rc???;
        // Logits stay resident. Sampling downloads one token id; non-device
        // sampling invokes an explicit materialization boundary.
        finish_out(&stream, o_key, out, false)
    })
}

/// Packed Q4_K gate/up projection with a fused SwiGLU epilogue.
pub unsafe extern "C" fn launch_q4k_gate_up_swiglu(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs != 3 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let gate_t = unsafe { &*inputs };
        let up_t = unsafe { &*inputs.add(1) };
        let x_t = unsafe { &*inputs.add(2) };
        let out_t = unsafe { &mut *outputs };
        if gate_t.dtype != DType::Q4K as u32 || up_t.dtype != DType::Q4K as u32 {
            return Err(-10);
        }
        let gate_dims = dims(gate_t);
        if gate_dims.len() != 2 || dims(up_t) != gate_dims {
            return Err(-2);
        }
        let out_dim = gate_dims[0] as u32;
        let in_dim = gate_dims[1] as usize;
        if in_dim % Q4K_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let x = f32_slice(x_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() != in_dim || out.len() != out_dim as usize {
            return Err(-2);
        }
        let n_blocks = (in_dim / Q4K_BLOCK_ELEMS as usize) as u32;
        let gate_bytes = bytes_slice(gate_t);
        let up_bytes = bytes_slice(up_t);
        let stream = oxide_stream().clone();
        let gate_key = buffers::ensure_weight(&stream, gate_bytes)?;
        let up_key = buffers::ensure_weight(&stream, up_bytes)?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let out_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (mut od, _) = buffers::take_f32(out_key)?;
        let module = oxide_module();
        let rc = buffers::with_q8_activation(
            &stream,
            x_key,
            x.len(),
            |qx, scales| unsafe {
                module
                    .quantize_q8_32(
                        &stream,
                        cfg_warp_blocks(x.len().div_ceil(32) as u32),
                        &xd,
                        qx,
                        scales,
                    )
                    .map_err(|_| -4)
            },
            |qx, scales| {
                buffers::with_weight(gate_key, |gate| {
                    buffers::with_weight(up_key, |up| unsafe {
                        module
                            .q4k_gate_up_swiglu_multiwarp(
                                &stream,
                                cfg_mmvq_blocks(out_dim, n_blocks),
                                gate,
                                up,
                                qx,
                                scales,
                                out_dim,
                                n_blocks,
                                &mut od,
                            )
                            .map_err(|_| -4)
                    })
                })
            },
        );
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(out_key, od, false)?;
        rc????;
        finish_out(&stream, out_key, out, false)
    })
}

/// Q8_0 gate/up projection with a single ABI dispatch and fused device SwiGLU.
pub unsafe extern "C" fn launch_q8_0_gate_up_swiglu(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs != 3 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let gate_t = unsafe { &*inputs };
        let up_t = unsafe { &*inputs.add(1) };
        let x_t = unsafe { &*inputs.add(2) };
        let out_t = unsafe { &mut *outputs };
        if gate_t.dtype != DType::Q8_0 as u32 || up_t.dtype != DType::Q8_0 as u32 {
            return Err(-10);
        }
        let shape = dims(gate_t);
        if shape.len() != 2 || dims(up_t) != shape {
            return Err(-2);
        }
        let out_dim = shape[0] as usize;
        let in_dim = shape[1] as usize;
        if in_dim % Q8_0_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let n_blocks = in_dim / Q8_0_BLOCK_ELEMS as usize;
        let x = f32_slice(x_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() != in_dim || out.len() != out_dim {
            return Err(-2);
        }
        let stream = oxide_stream().clone();
        let gate_key = buffers::ensure_weight(&stream, bytes_slice(gate_t))?;
        let up_key = buffers::ensure_weight(&stream, bytes_slice(up_t))?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let out_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (mut od, _) = buffers::take_f32(out_key)?;
        let mut gate_out = buffers::take_scratch_f32(&stream, out_dim)?;
        let mut up_out = buffers::take_scratch_f32(&stream, out_dim)?;
        let module = oxide_module();
        let rc = buffers::with_weight(gate_key, |gate| {
            buffers::with_weight(up_key, |up| unsafe {
                module
                    .q8_0_gemm_warp(
                        &stream,
                        cfg_warp_blocks(out_dim as u32),
                        gate,
                        &xd,
                        out_dim as u32,
                        n_blocks as u32,
                        1,
                        &mut gate_out,
                    )
                    .map_err(|_| -4)?;
                module
                    .q8_0_gemm_warp(
                        &stream,
                        cfg_warp_blocks(out_dim as u32),
                        up,
                        &xd,
                        out_dim as u32,
                        n_blocks as u32,
                        1,
                        &mut up_out,
                    )
                    .map_err(|_| -4)?;
                module
                    .silu_gate(&stream, cfg_1d(out_dim as u32), &gate_out, &up_out, &mut od)
                    .map_err(|_| -4)
            })
        });
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(out_key, od, false)?;
        buffers::put_scratch_f32(gate_out)?;
        buffers::put_scratch_f32(up_out)?;
        rc???;
        finish_out(&stream, out_key, out, false)
    })
}

/// LFM ShortConv decode entirely on-device: two Q4_K projections around a
/// fused recurrent convolution/state-update kernel.
pub unsafe extern "C" fn launch_shortconv_q4k(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if attrs.is_null() || inputs.is_null() || outputs.is_null() || n_inputs < 4 || n_outputs < 2
        {
            return Err(-1);
        }
        let attrs = unsafe { &*attrs };
        let x_t = unsafe { &*inputs };
        let in_w_t = unsafe { &*inputs.add(1) };
        let conv_t = unsafe { &*inputs.add(2) };
        let out_w_t = unsafe { &*inputs.add(3) };
        let y_t = unsafe { &mut *outputs };
        let state_t = unsafe { &mut *outputs.add(1) };
        if in_w_t.dtype != DType::Q4K as u32 || out_w_t.dtype != DType::Q4K as u32 {
            return Err(-10);
        }
        let n = attrs.n_embd as usize;
        let l_cache = attrs.shortconv_l_cache as usize;
        if n == 0 || l_cache == 0 || n % Q4K_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let x = f32_slice(x_t)?;
        let conv = f32_slice(conv_t)?;
        let y = f32_slice_mut(y_t)?;
        let state = f32_slice_mut(state_t)?;
        if !x.len().is_multiple_of(n)
            || y.len() != x.len()
            || conv.len() != n * l_cache
            || state.len() != x.len() / n * (l_cache - 1) * n
        {
            return Err(-2);
        }
        let rows = x.len() / n;
        let in_w = bytes_slice(in_w_t);
        let out_w = bytes_slice(out_w_t);
        let blocks = (n / Q4K_BLOCK_ELEMS as usize) as u32;
        let row_bytes = blocks as usize * Q4K_BLOCK_BYTES as usize;
        if in_w.len() < 3 * n * row_bytes || out_w.len() < n * row_bytes {
            return Err(-2);
        }

        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let module = oxide_module();
        let in_w_key = buffers::ensure_weight(&stream, in_w)?;
        let out_w_key = buffers::ensure_weight(&stream, out_w)?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let conv_key = buffers::ensure_f32(&stream, conv, false)?;
        // Scheduler batches repack request-owned recurrent state into row slots,
        // so the host buffer is authoritative at every batch boundary.
        let state_key = buffers::ensure_f32(&stream, state, true)?;
        let y_key = buffers::ensure_f32_out(&stream, y)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (convd, _) = buffers::take_f32(conv_key)?;
        let (mut stated, _) = buffers::take_f32(state_key)?;
        let (mut yd, _) = buffers::take_f32(y_key)?;
        let mut bcx = buffers::take_scratch_f32(&stream, 3 * n * rows)?;
        let mut y_pre = buffers::take_scratch_f32(&stream, n * rows)?;

        let first = buffers::with_weight(in_w_key, |wd| unsafe {
            module
                .q4k_gemv_row(
                    &stream,
                    cfg_1d((3 * n * rows) as u32),
                    wd,
                    &xd,
                    (3 * n) as u32,
                    blocks,
                    rows as u32,
                    &mut bcx,
                )
                .map_err(|_| -4)
        });
        first??;
        unsafe {
            module
                .shortconv_mix_rows(
                    &stream,
                    cfg_1d((n * rows) as u32),
                    &bcx,
                    &convd,
                    &mut stated,
                    n as u32,
                    l_cache as u32,
                    rows as u32,
                    &mut y_pre,
                )
                .map_err(|_| -4)?;
        }
        let last = buffers::with_weight(out_w_key, |wd| unsafe {
            module
                .q4k_gemv_row(
                    &stream,
                    cfg_1d((n * rows) as u32),
                    wd,
                    &y_pre,
                    n as u32,
                    blocks,
                    rows as u32,
                    &mut yd,
                )
                .map_err(|_| -4)
        });
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(conv_key, convd, true)?;
        buffers::put_f32(state_key, stated, true)?;
        buffers::put_f32(y_key, yd, false)?;
        buffers::put_scratch_f32(bcx)?;
        buffers::put_scratch_f32(y_pre)?;
        last??;
        buffers::mark_valid(state_key)?;
        finish_out(&stream, state_key, state, false)?;
        finish_out(&stream, y_key, y, false)
    })
}

unsafe fn launch_moe_routed(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    down_q6k: bool,
) -> i32 {
    run(|| {
        if attrs.is_null() || inputs.is_null() || outputs.is_null() || n_inputs < 6 || n_outputs < 1
        {
            return Err(-1);
        }
        let attrs = unsafe { &*attrs };
        let x_t = unsafe { &*inputs };
        let router_t = unsafe { &*inputs.add(1) };
        let gate_t = unsafe { &*inputs.add(2) };
        let up_t = unsafe { &*inputs.add(3) };
        let down_t = unsafe { &*inputs.add(4) };
        let bias_t = unsafe { &*inputs.add(5) };
        let out_t = unsafe { &mut *outputs };
        let n = attrs.n_embd as usize;
        let experts = attrs.n_experts as usize;
        let top_k = attrs.n_expert_used as usize;
        let gate_dims = dims(gate_t);
        if n == 0 || experts == 0 || top_k == 0 || gate_dims.len() != 3 {
            return Err(-2);
        }
        let ff = gate_dims[1] as usize;
        if ff == 0 || n % Q4K_BLOCK_ELEMS as usize != 0 || ff % Q4K_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let x = f32_slice(x_t)?;
        let router = f32_slice(router_t)?;
        let bias = f32_slice(bias_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() % n != 0 || router.len() != experts * n || bias.len() < experts {
            return Err(-2);
        }
        let tokens = x.len() / n;
        if out.len() != tokens * n {
            return Err(-2);
        }
        let gate_w = bytes_slice(gate_t);
        let up_w = bytes_slice(up_t);
        let down_w = bytes_slice(down_t);

        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let module = oxide_module();
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let router_key = buffers::ensure_f32(&stream, router, false)?;
        let bias_key = buffers::ensure_f32(&stream, bias, false)?;
        let gate_key = buffers::ensure_weight(&stream, gate_w)?;
        let up_key = buffers::ensure_weight(&stream, up_w)?;
        let down_key = buffers::ensure_weight(&stream, down_w)?;
        let out_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (routerd, _) = buffers::take_f32(router_key)?;
        let (biasd, _) = buffers::take_f32(bias_key)?;
        let (mut outd, _) = buffers::take_f32(out_key)?;
        let assignments = tokens * top_k;
        let mut ids = buffers::take_scratch_u32(&stream, assignments)?;
        let mut scores = buffers::take_scratch_f32(&stream, assignments)?;
        let mut gate_out = buffers::take_scratch_f32(&stream, assignments * ff)?;
        let mut up_out = buffers::take_scratch_f32(&stream, assignments * ff)?;
        let mut hidden = buffers::take_scratch_f32(&stream, assignments * ff)?;
        let mut expert_out = buffers::take_scratch_f32(&stream, assignments * n)?;
        let routed_scale = if attrs.routed_scaling_factor == 0.0 {
            1.0
        } else {
            attrs.routed_scaling_factor
        };
        unsafe {
            module
                .moe_route_topk(
                    &stream,
                    cfg_1d(tokens as u32),
                    &xd,
                    &routerd,
                    &biasd,
                    tokens as u32,
                    n as u32,
                    experts as u32,
                    top_k as u32,
                    attrs.expert_gating_func,
                    attrs.norm_topk_prob,
                    routed_scale,
                    &mut ids,
                    &mut scores,
                )
                .map_err(|_| -4)?;
        }
        let gate_launch = buffers::with_weight(gate_key, |wd| unsafe {
            module
                .moe_q4k_project(
                    &stream,
                    cfg_1d((assignments * ff) as u32),
                    wd,
                    &xd,
                    &ids,
                    tokens as u32,
                    top_k as u32,
                    ff as u32,
                    n as u32,
                    ff as u32,
                    0,
                    0,
                    &mut gate_out,
                )
                .map_err(|_| -4)
        });
        gate_launch??;
        let up_launch = buffers::with_weight(up_key, |wd| unsafe {
            module
                .moe_q4k_project(
                    &stream,
                    cfg_1d((assignments * ff) as u32),
                    wd,
                    &xd,
                    &ids,
                    tokens as u32,
                    top_k as u32,
                    ff as u32,
                    n as u32,
                    ff as u32,
                    0,
                    0,
                    &mut up_out,
                )
                .map_err(|_| -4)
        });
        up_launch??;
        unsafe {
            module
                .silu_gate(
                    &stream,
                    cfg_1d((assignments * ff) as u32),
                    &gate_out,
                    &up_out,
                    &mut hidden,
                )
                .map_err(|_| -4)?;
        }
        let down_launch = buffers::with_weight(down_key, |wd| unsafe {
            if down_q6k {
                module
                    .moe_q6k_project(
                        &stream,
                        cfg_1d((assignments * n) as u32),
                        wd,
                        &hidden,
                        &ids,
                        tokens as u32,
                        top_k as u32,
                        n as u32,
                        ff as u32,
                        &mut expert_out,
                    )
                    .map_err(|_| -4)
            } else {
                module
                    .moe_q4k_project(
                        &stream,
                        cfg_1d((assignments * n) as u32),
                        wd,
                        &hidden,
                        &ids,
                        tokens as u32,
                        top_k as u32,
                        n as u32,
                        ff as u32,
                        n as u32,
                        0,
                        1,
                        &mut expert_out,
                    )
                    .map_err(|_| -4)
            }
        });
        down_launch??;
        unsafe {
            module
                .moe_weighted_reduce(
                    &stream,
                    cfg_1d((tokens * n) as u32),
                    &expert_out,
                    &scores,
                    tokens as u32,
                    top_k as u32,
                    n as u32,
                    &mut outd,
                )
                .map_err(|_| -4)?;
        }
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(router_key, routerd, true)?;
        buffers::put_f32(bias_key, biasd, true)?;
        buffers::put_f32(out_key, outd, false)?;
        buffers::put_scratch_u32(ids)?;
        buffers::put_scratch_f32(scores)?;
        buffers::put_scratch_f32(gate_out)?;
        buffers::put_scratch_f32(up_out)?;
        buffers::put_scratch_f32(hidden)?;
        buffers::put_scratch_f32(expert_out)?;
        finish_out(&stream, out_key, out, false)
    })
}

pub unsafe extern "C" fn launch_moe_q4k_down(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    unsafe { launch_moe_routed(attrs, inputs, n_inputs, outputs, n_outputs, false) }
}

pub unsafe extern "C" fn launch_moe_q6k_down(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    unsafe { launch_moe_routed(attrs, inputs, n_inputs, outputs, n_outputs, true) }
}

unsafe fn launch_moe_gemma(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    shared_q5: bool,
) -> i32 {
    run(|| {
        if attrs.is_null() || inputs.is_null() || outputs.is_null() || n_inputs < 7 || n_outputs < 1
        {
            return Err(-1);
        }
        let a = unsafe { &*attrs };
        let ts: Vec<&TensorRef> = (0..7).map(|i| unsafe { &*inputs.add(i) }).collect();
        let out_t = unsafe { &mut *outputs };
        let n = a.n_embd as usize;
        let experts = a.n_experts as usize;
        let top_k = a.n_expert_used as usize;
        let pd = dims(ts[2]);
        let sd = dims(ts[4]);
        if n == 0 || experts == 0 || top_k == 0 || pd.len() != 3 || sd.len() != 2 {
            return Err(-2);
        }
        let ff = pd[1] as usize / 2;
        let shared_ff = sd[0] as usize;
        if ff == 0
            || shared_ff == 0
            || n % Q4K_BLOCK_ELEMS as usize != 0
            || ff % Q5_0_BLOCK_ELEMS as usize != 0
            || shared_ff % Q5_0_BLOCK_ELEMS as usize != 0
        {
            return Err(-2);
        }
        let x = f32_slice(ts[0])?;
        let router = f32_slice(ts[1])?;
        let out = f32_slice_mut(out_t)?;
        if x.len() % n != 0 || router.len() != experts * n {
            return Err(-2);
        }
        let tokens = x.len() / n;
        if out.len() != tokens * n {
            return Err(-2);
        }
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let module = oxide_module();
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let router_key = buffers::ensure_f32(&stream, router, false)?;
        let out_key = buffers::ensure_f32_out(&stream, out)?;
        let weight_keys: Result<Vec<usize>, i32> = ts[2..7]
            .iter()
            .map(|t| buffers::ensure_weight(&stream, bytes_slice(t)))
            .collect();
        let wk = weight_keys?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (routerd, _) = buffers::take_f32(router_key)?;
        let (mut outd, _) = buffers::take_f32(out_key)?;
        let assignments = tokens * top_k;
        let zero_bias = buffers::take_scratch_f32(&stream, experts)?;
        let mut ids = buffers::take_scratch_u32(&stream, assignments)?;
        let mut counts = buffers::take_scratch_u32(&stream, experts)?;
        let mut offsets = buffers::take_scratch_u32(&stream, experts)?;
        let mut cursors = buffers::take_scratch_u32(&stream, experts)?;
        let mut order = buffers::take_scratch_u32(&stream, assignments)?;
        let mut scores = buffers::take_scratch_f32(&stream, assignments)?;
        let mut gate = buffers::take_scratch_f32(&stream, assignments * ff)?;
        let mut up = buffers::take_scratch_f32(&stream, assignments * ff)?;
        let mut hidden = buffers::take_scratch_f32(&stream, assignments * ff)?;
        let mut expert_out = buffers::take_scratch_f32(&stream, assignments * n)?;
        let mut sg = buffers::take_scratch_f32(&stream, tokens * shared_ff)?;
        let mut su = buffers::take_scratch_f32(&stream, tokens * shared_ff)?;
        let mut sh = buffers::take_scratch_f32(&stream, tokens * shared_ff)?;
        let mut so = buffers::take_scratch_f32(&stream, tokens * n)?;
        let scale = if a.routed_scaling_factor == 0.0 {
            1.0
        } else {
            a.routed_scaling_factor
        };
        unsafe {
            module
                .moe_route_topk(
                    &stream,
                    cfg_1d(tokens as u32),
                    &xd,
                    &routerd,
                    &zero_bias,
                    tokens as u32,
                    n as u32,
                    experts as u32,
                    top_k as u32,
                    a.expert_gating_func,
                    a.norm_topk_prob,
                    scale,
                    &mut ids,
                    &mut scores,
                )
                .map_err(|_| -4)?;
        }
        unsafe {
            module
                .fill_u32(&stream, cfg_1d(experts as u32), 0, &mut counts)
                .map_err(|_| -4)?;
        }
        let counts_atomic = counts.cast_elem::<cuda_device::atomic::DeviceAtomicU32>();
        unsafe {
            module
                .moe_count_assignments(&stream, cfg_1d(assignments as u32), &ids, &counts_atomic)
                .map_err(|_| -4)?;
        }
        counts = counts_atomic.cast_elem::<u32>();
        unsafe {
            module
                .moe_prefix_offsets(&stream, cfg_1d(1), &counts, &mut offsets, &mut cursors)
                .map_err(|_| -4)?;
        }
        let cursors_atomic = cursors.cast_elem::<cuda_device::atomic::DeviceAtomicU32>();
        unsafe {
            module
                .moe_scatter_assignments(
                    &stream,
                    cfg_1d(assignments as u32),
                    &ids,
                    &offsets,
                    &cursors_atomic,
                    &mut order,
                )
                .map_err(|_| -4)?;
        }
        cursors = cursors_atomic.cast_elem::<u32>();
        let routed = |row_offset: usize, dst: &mut DeviceBuffer<f32>| {
            buffers::with_weight(wk[0], |wd| unsafe {
                if tokens > 1 {
                    module
                        .moe_q4k_project_warp(
                            &stream,
                            cfg_warp_blocks((assignments * ff) as u32),
                            wd,
                            &xd,
                            &ids,
                            &order,
                            tokens as u32,
                            top_k as u32,
                            ff as u32,
                            n as u32,
                            (2 * ff) as u32,
                            row_offset as u32,
                            0,
                            dst,
                        )
                        .map_err(|_| -4)
                } else {
                    module
                        .moe_q4k_project(
                            &stream,
                            cfg_1d((assignments * ff) as u32),
                            wd,
                            &xd,
                            &ids,
                            tokens as u32,
                            top_k as u32,
                            ff as u32,
                            n as u32,
                            (2 * ff) as u32,
                            row_offset as u32,
                            0,
                            dst,
                        )
                        .map_err(|_| -4)
                }
            })
        };
        routed(0, &mut gate)??;
        routed(ff, &mut up)??;
        unsafe {
            module
                .silu_gate(
                    &stream,
                    cfg_1d((assignments * ff) as u32),
                    &gate,
                    &up,
                    &mut hidden,
                )
                .map_err(|_| -4)?;
        }
        let down = buffers::with_weight(wk[1], |wd| unsafe {
            if shared_q5 {
                if tokens > 1 {
                    module
                        .moe_q5_0_project_warp(
                            &stream,
                            cfg_warp_blocks((assignments * n) as u32),
                            wd,
                            &hidden,
                            &ids,
                            &order,
                            tokens as u32,
                            top_k as u32,
                            n as u32,
                            ff as u32,
                            &mut expert_out,
                        )
                        .map_err(|_| -4)
                } else {
                    module
                        .moe_q5_0_project(
                            &stream,
                            cfg_1d((assignments * n) as u32),
                            wd,
                            &hidden,
                            &ids,
                            tokens as u32,
                            top_k as u32,
                            n as u32,
                            ff as u32,
                            &mut expert_out,
                        )
                        .map_err(|_| -4)
                }
            } else {
                if tokens > 1 {
                    module
                        .moe_q8_0_project_warp(
                            &stream,
                            cfg_warp_blocks((assignments * n) as u32),
                            wd,
                            &hidden,
                            &ids,
                            &order,
                            tokens as u32,
                            top_k as u32,
                            n as u32,
                            ff as u32,
                            &mut expert_out,
                        )
                        .map_err(|_| -4)
                } else {
                    module
                        .moe_q8_0_project(
                            &stream,
                            cfg_1d((assignments * n) as u32),
                            wd,
                            &hidden,
                            &ids,
                            tokens as u32,
                            top_k as u32,
                            n as u32,
                            ff as u32,
                            &mut expert_out,
                        )
                        .map_err(|_| -4)
                }
            }
        });
        down??;
        unsafe {
            module
                .moe_weighted_reduce(
                    &stream,
                    cfg_1d((tokens * n) as u32),
                    &expert_out,
                    &scores,
                    tokens as u32,
                    top_k as u32,
                    n as u32,
                    &mut outd,
                )
                .map_err(|_| -4)?;
        }
        let shared_quant =
            |key, dst: &mut DeviceBuffer<f32>, od: usize, inp: &DeviceBuffer<f32>, id: usize| {
                buffers::with_weight(key, |wd| unsafe {
                    if shared_q5 {
                        if tokens > 1 {
                            module
                                .q5_0_gemm_warp(
                                    &stream,
                                    cfg_warp_blocks((tokens * od) as u32),
                                    wd,
                                    inp,
                                    od as u32,
                                    (id / Q5_0_BLOCK_ELEMS as usize) as u32,
                                    tokens as u32,
                                    dst,
                                )
                                .map_err(|_| -4)
                        } else {
                            module
                                .q5_0_gemm_element(
                                    &stream,
                                    cfg_1d((tokens * od) as u32),
                                    wd,
                                    inp,
                                    od as u32,
                                    (id / Q5_0_BLOCK_ELEMS as usize) as u32,
                                    tokens as u32,
                                    dst,
                                )
                                .map_err(|_| -4)
                        }
                    } else {
                        if tokens > 1 {
                            module
                                .q8_0_gemm_warp(
                                    &stream,
                                    cfg_warp_blocks((tokens * od) as u32),
                                    wd,
                                    inp,
                                    od as u32,
                                    (id / Q8_0_BLOCK_ELEMS as usize) as u32,
                                    tokens as u32,
                                    dst,
                                )
                                .map_err(|_| -4)
                        } else {
                            module
                                .q8_0_gemm_element(
                                    &stream,
                                    cfg_1d((tokens * od) as u32),
                                    wd,
                                    inp,
                                    od as u32,
                                    (id / Q8_0_BLOCK_ELEMS as usize) as u32,
                                    tokens as u32,
                                    dst,
                                )
                                .map_err(|_| -4)
                        }
                    }
                })
            };
        let shared_gate = buffers::with_weight(wk[2], |wd| unsafe {
            if tokens > 1 {
                module
                    .q4k_gemm_warp(
                        &stream,
                        cfg_warp_blocks((tokens * shared_ff) as u32),
                        wd,
                        &xd,
                        shared_ff as u32,
                        (n / Q4K_BLOCK_ELEMS as usize) as u32,
                        tokens as u32,
                        &mut sg,
                    )
                    .map_err(|_| -4)
            } else {
                module
                    .q4k_gemv_row(
                        &stream,
                        cfg_1d((tokens * shared_ff) as u32),
                        wd,
                        &xd,
                        shared_ff as u32,
                        (n / Q4K_BLOCK_ELEMS as usize) as u32,
                        tokens as u32,
                        &mut sg,
                    )
                    .map_err(|_| -4)
            }
        });
        shared_gate??;
        let shared_up = buffers::with_weight(wk[3], |wd| unsafe {
            if tokens > 1 {
                module
                    .q4k_gemm_warp(
                        &stream,
                        cfg_warp_blocks((tokens * shared_ff) as u32),
                        wd,
                        &xd,
                        shared_ff as u32,
                        (n / Q4K_BLOCK_ELEMS as usize) as u32,
                        tokens as u32,
                        &mut su,
                    )
                    .map_err(|_| -4)
            } else {
                module
                    .q4k_gemv_row(
                        &stream,
                        cfg_1d((tokens * shared_ff) as u32),
                        wd,
                        &xd,
                        shared_ff as u32,
                        (n / Q4K_BLOCK_ELEMS as usize) as u32,
                        tokens as u32,
                        &mut su,
                    )
                    .map_err(|_| -4)
            }
        });
        shared_up??;
        unsafe {
            module
                .silu_gate(
                    &stream,
                    cfg_1d((tokens * shared_ff) as u32),
                    &sg,
                    &su,
                    &mut sh,
                )
                .map_err(|_| -4)?;
        }
        shared_quant(wk[4], &mut so, n, &sh, shared_ff)??;
        unsafe {
            module
                .add_in_place_f32(&stream, cfg_1d((tokens * n) as u32), &so, &mut outd)
                .map_err(|_| -4)?;
        }
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(router_key, routerd, true)?;
        buffers::put_f32(out_key, outd, false)?;
        buffers::put_scratch_f32(zero_bias)?;
        buffers::put_scratch_u32(ids)?;
        buffers::put_scratch_u32(counts)?;
        buffers::put_scratch_u32(offsets)?;
        buffers::put_scratch_u32(cursors)?;
        buffers::put_scratch_u32(order)?;
        buffers::put_scratch_f32(scores)?;
        buffers::put_scratch_f32(gate)?;
        buffers::put_scratch_f32(up)?;
        buffers::put_scratch_f32(hidden)?;
        buffers::put_scratch_f32(expert_out)?;
        buffers::put_scratch_f32(sg)?;
        buffers::put_scratch_f32(su)?;
        buffers::put_scratch_f32(sh)?;
        buffers::put_scratch_f32(so)?;
        finish_out(&stream, out_key, out, false)
    })
}

pub unsafe extern "C" fn launch_moe_gemma_q5(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    unsafe { launch_moe_gemma(attrs, inputs, n_inputs, outputs, n_outputs, true) }
}
pub unsafe extern "C" fn launch_moe_gemma_q8(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    unsafe { launch_moe_gemma(attrs, inputs, n_inputs, outputs, n_outputs, false) }
}

/// Q6_K matvec: weight [out,in] Q6K × x [in] f32 → y [out] f32.
pub unsafe extern "C" fn launch_q6k_matmul(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let x_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        if w_t.dtype != DType::Q6K as u32 {
            return Err(-10);
        }
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let out_dim = wdims[0] as u32;
        let in_dim = wdims[1] as usize;
        if in_dim % (Q6K_BLOCK_ELEMS as usize) != 0 {
            return Err(-2);
        }
        let n_blocks = (in_dim / Q6K_BLOCK_ELEMS as usize) as u32;
        let wb = bytes_slice(w_t);
        let expect = out_dim as usize * n_blocks as usize * Q6K_BLOCK_BYTES as usize;
        if wb.len() < expect {
            return Err(-2);
        }
        let x = f32_slice(x_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() % in_dim != 0 {
            return Err(-2);
        }
        let rows = x.len() / in_dim;
        if out.len() != rows * out_dim as usize {
            return Err(-2);
        }
        let residual = if n_inputs >= 3 {
            let values = f32_slice(unsafe { &*inputs.add(2) })?;
            if values.len() != out.len() {
                return Err(-2);
            }
            Some(values)
        } else {
            None
        };

        let stream = oxide_stream().clone();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let residual_key = residual
            .map(|values| buffers::ensure_f32(&stream, values, false))
            .transpose()?;
        let mut residual_device = residual_key.map(buffers::take_f32).transpose()?;
        let module = oxide_module();
        let rc = buffers::with_q8_activation(
            &stream,
            x_key,
            x.len(),
            |qx, x_scales| unsafe {
                module
                    .quantize_q8_32(
                        &stream,
                        cfg_warp_blocks(x.len().div_ceil(32) as u32),
                        &xd,
                        qx,
                        x_scales,
                    )
                    .map_err(|_| -4)
            },
            |qx, x_scales| {
                buffers::with_weight(w_key, |wd| unsafe {
                    module
                        .q6k_q8_gemv_multiwarp(
                            &stream,
                            cfg_mmvq_blocks(out_dim * rows as u32, n_blocks),
                            wd,
                            qx,
                            x_scales,
                            residual_device.as_ref().map_or(&xd, |(buffer, _)| buffer),
                            u32::from(residual_device.is_some()),
                            out_dim,
                            n_blocks,
                            rows as u32,
                            &mut od,
                        )
                        .map_err(|_| -4)
                })
            },
        );
        buffers::put_f32(x_key, xd, true)?;
        if let (Some(key), Some((buffer, valid))) = (residual_key, residual_device.take()) {
            buffers::put_f32(key, buffer, valid)?;
        }
        buffers::put_f32(o_key, od, false)?;
        rc???;
        finish_out(&stream, o_key, out, false)
    })
}

/// Q6_K embedding: dequantize one weight row on device.
pub unsafe extern "C" fn launch_embedding_q6k(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let tok_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        if w_t.dtype != DType::Q6K as u32 {
            return Err(-10);
        }
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let dim = wdims[1] as usize;
        if dim % (Q6K_BLOCK_ELEMS as usize) != 0 {
            return Err(-2);
        }
        let n_blocks = (dim / Q6K_BLOCK_ELEMS as usize) as u32;
        let wb = bytes_slice(w_t);
        let ids = u32_slice(tok_t)?;
        if ids.is_empty() {
            return Err(-2);
        }
        let out = f32_slice_mut(out_t)?;
        if out.len() != ids.len() * dim {
            return Err(-2);
        }
        let row_bytes = n_blocks as usize * Q6K_BLOCK_BYTES as usize;
        let need = (ids.iter().copied().max().unwrap_or(0) as usize + 1) * row_bytes;
        if wb.len() < need {
            return Err(-2);
        }

        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let mut batch_ids = None;
        let rc = buffers::with_weight(w_key, |wd| {
            if ids.len() == 1 {
                with_step_params(|params| unsafe {
                    module
                        .embedding_q6k_row(
                            &stream,
                            cfg_1d(dim as u32),
                            wd,
                            params,
                            dim as u32,
                            n_blocks,
                            &mut od,
                        )
                        .map_err(|_| -4)
                })
            } else {
                let key = buffers::ensure_u32(&stream, ids)?;
                let device = buffers::take_u32(key)?;
                let result = unsafe {
                    module
                        .embedding_q6k_rows(
                            &stream,
                            cfg_1d((ids.len() * dim) as u32),
                            wd,
                            &device,
                            ids.len() as u32,
                            dim as u32,
                            n_blocks,
                            &mut od,
                        )
                        .map_err(|_| -4)
                };
                batch_ids = Some((key, device));
                result
            }
        });
        if let Some((key, device)) = batch_ids {
            buffers::put_u32(key, device)?;
        }
        buffers::put_f32(o_key, od, false)?;
        rc??;
        finish_out(&stream, o_key, out, false)
    })
}

/// Q8_0 batched matrix multiplication without a global dequantization buffer.
pub unsafe extern "C" fn launch_q8_0_matmul(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let x_t = unsafe { &*inputs.add(1) };
        let residual_t = (n_inputs >= 3).then(|| unsafe { &*inputs.add(2) });
        let out_t = unsafe { &mut *outputs };
        if w_t.dtype != DType::Q8_0 as u32 {
            return Err(-10);
        }
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let out_dim = wdims[0] as usize;
        let in_dim = wdims[1] as usize;
        if in_dim % Q8_0_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let n_blocks = in_dim / Q8_0_BLOCK_ELEMS as usize;
        let wb = bytes_slice(w_t);
        if wb.len() < out_dim * n_blocks * Q8_0_BLOCK_BYTES as usize {
            return Err(-2);
        }
        let x = f32_slice(x_t)?;
        let out = f32_slice_mut(out_t)?;
        if x.len() % in_dim != 0 {
            return Err(-2);
        }
        let rows = x.len() / in_dim;
        if out.len() != rows * out_dim {
            return Err(-2);
        }
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let residual_key = residual_t
            .map(|tensor| buffers::ensure_f32(&stream, f32_slice(tensor)?, false))
            .transpose()?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let mut residual_device = residual_key.map(buffers::take_f32).transpose()?;
        let rc = buffers::with_weight(w_key, |wd| unsafe {
            if rows > 0 {
                oxide_module()
                    .q8_0_gemm_warp(
                        &stream,
                        cfg_warp_blocks((rows * out_dim) as u32),
                        wd,
                        &xd,
                        out_dim as u32,
                        n_blocks as u32,
                        rows as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            } else {
                oxide_module()
                    .q8_0_gemm_element(
                        &stream,
                        cfg_1d((rows * out_dim) as u32),
                        wd,
                        &xd,
                        out_dim as u32,
                        n_blocks as u32,
                        rows as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            }
        });
        buffers::put_f32(x_key, xd, true)?;
        rc??;
        if let Some((residual, _)) = residual_device.as_ref() {
            unsafe {
                oxide_module()
                    .add_inplace_f32(&stream, cfg_1d(out.len() as u32), residual, &mut od)
                    .map_err(|_| -4)?;
            }
        }
        if let (Some(key), Some((buffer, valid))) = (residual_key, residual_device.take()) {
            buffers::put_f32(key, buffer, valid)?;
        }
        buffers::put_f32(o_key, od, false)?;
        finish_out(&stream, o_key, out, false)
    })
}

pub unsafe extern "C" fn launch_q5_0_matmul(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let wt = unsafe { &*inputs };
        let xt = unsafe { &*inputs.add(1) };
        if wt.dtype != DType::Q5_0 as u32 {
            return Err(-10);
        }
        let d = dims(wt);
        if d.len() < 2 {
            return Err(-2);
        }
        let odim = d[0] as usize;
        let idim = d[1] as usize;
        if idim % Q5_0_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let blocks = idim / Q5_0_BLOCK_ELEMS as usize;
        let wb = bytes_slice(wt);
        let x = f32_slice(xt)?;
        let out = f32_slice_mut(unsafe { &mut *outputs })?;
        if x.len() % idim != 0 || wb.len() < odim * blocks * 22 {
            return Err(-2);
        }
        let rows = x.len() / idim;
        if out.len() != rows * odim {
            return Err(-2);
        }
        let stream = oxide_stream().clone();
        let wk = buffers::ensure_weight(&stream, wb)?;
        let xk = buffers::ensure_f32(&stream, x, false)?;
        let ok = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(xk)?;
        let (mut od, _) = buffers::take_f32(ok)?;
        let rc = buffers::with_weight(wk, |wd| unsafe {
            if rows > 0 {
                oxide_module()
                    .q5_0_gemm_warp(
                        &stream,
                        cfg_warp_blocks((rows * odim) as u32),
                        wd,
                        &xd,
                        odim as u32,
                        blocks as u32,
                        rows as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            } else {
                oxide_module()
                    .q5_0_gemm_element(
                        &stream,
                        cfg_1d((rows * odim) as u32),
                        wd,
                        &xd,
                        odim as u32,
                        blocks as u32,
                        rows as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            }
        });
        buffers::put_f32(xk, xd, true)?;
        buffers::put_f32(ok, od, false)?;
        rc??;
        finish_out(&stream, ok, out, odim >= 16384)
    })
}

pub unsafe extern "C" fn launch_weighted_embedding_q6k(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let wt = unsafe { &*inputs };
        let pt = unsafe { &*inputs.add(1) };
        let wd = dims(wt);
        let pd = dims(pt);
        if wt.dtype != DType::Q6K as u32 || wd.len() != 2 || pd.len() != 2 {
            return Err(-2);
        }
        let vocab = wd[0] as usize;
        let dim = wd[1] as usize;
        let rows = pd[0] as usize;
        let slots = pd[1] as usize;
        if slots == vocab || slots % 2 != 0 {
            return Err(-2);
        }
        let top_k = slots / 2;
        let packed = f32_slice(pt)?;
        let out = f32_slice_mut(unsafe { &mut *outputs })?;
        if out.len() != rows * dim {
            return Err(-2);
        }
        let stream = oxide_stream().clone();
        let wk = buffers::ensure_weight(&stream, bytes_slice(wt))?;
        let pk = buffers::ensure_f32(&stream, packed, false)?;
        let ok = buffers::ensure_f32_out(&stream, out)?;
        let (packed_d, _) = buffers::take_f32(pk)?;
        let (mut od, _) = buffers::take_f32(ok)?;
        let rc = buffers::with_weight(wk, |w| unsafe {
            oxide_module()
                .weighted_embedding_q6k_topk(
                    &stream,
                    cfg_1d((rows * dim) as u32),
                    w,
                    &packed_d,
                    rows as u32,
                    top_k as u32,
                    dim as u32,
                    vocab as u32,
                    &mut od,
                )
                .map_err(|_| -4)
        });
        buffers::put_f32(pk, packed_d, true)?;
        buffers::put_f32(ok, od, false)?;
        rc??;
        finish_out(&stream, ok, out, false)
    })
}

/// Direct Q8_0 embedding lookup.
pub unsafe extern "C" fn launch_embedding_q8_0(
    _attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 2 || n_outputs < 1 || inputs.is_null() || outputs.is_null() {
            return Err(-1);
        }
        let w_t = unsafe { &*inputs };
        let tok_t = unsafe { &*inputs.add(1) };
        let out_t = unsafe { &mut *outputs };
        if w_t.dtype != DType::Q8_0 as u32 {
            return Err(-10);
        }
        let wdims = dims(w_t);
        if wdims.len() < 2 {
            return Err(-2);
        }
        let dim = wdims[1] as usize;
        if dim % Q8_0_BLOCK_ELEMS as usize != 0 {
            return Err(-2);
        }
        let n_blocks = dim / Q8_0_BLOCK_ELEMS as usize;
        let ids = u32_slice(tok_t)?;
        if ids.is_empty() {
            return Err(-2);
        }
        let out = f32_slice_mut(out_t)?;
        if out.len() != ids.len() * dim {
            return Err(-2);
        }
        let wb = bytes_slice(w_t);
        if wb.len()
            < (ids.iter().copied().max().unwrap_or(0) as usize + 1)
                * n_blocks
                * Q8_0_BLOCK_BYTES as usize
        {
            return Err(-2);
        }
        let _ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let i_key = buffers::ensure_u32(&stream, ids)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let id = buffers::take_u32(i_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let rc = buffers::with_weight(w_key, |wd| unsafe {
            oxide_module()
                .embedding_q8_0_rows(
                    &stream,
                    cfg_1d(out.len() as u32),
                    wd,
                    &id,
                    ids.len() as u32,
                    dim as u32,
                    n_blocks as u32,
                    &mut od,
                )
                .map_err(|_| -4)
        });
        buffers::put_u32(i_key, id)?;
        buffers::put_f32(o_key, od, false)?;
        rc??;
        finish_out(&stream, o_key, out, false)
    })
}

/// Attention: device-paged when VRAM arena is set; else host-gather + contiguous.
pub unsafe extern "C" fn launch_attention(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 1 || n_outputs < 1 || attrs.is_null() || inputs.is_null() || outputs.is_null()
        {
            return Err(-1);
        }
        let attrs = unsafe { &*attrs };
        let q_t = unsafe { &*inputs };
        let out_t = unsafe { &mut *outputs };
        let q = f32_slice(q_t)?;
        let out = f32_slice_mut(out_t)?;
        let n_heads = attrs.n_heads.max(1) as usize;
        let n_kv = attrs.n_kv_heads.max(1) as usize;
        let head_dim = attrs.head_dim as usize;
        let past = attrs.past_len as usize;
        let seq = past + 1;
        let scale = if attrs.scale > 0.0 {
            attrs.scale
        } else {
            1.0 / (head_dim as f32).sqrt()
        };
        if attrs.attention_mode == 1 && n_inputs >= 5 {
            let rows = attrs.query_len.max(1) as usize;
            if q.len() != rows * n_heads * head_dim || out.len() != q.len() {
                return Err(-2);
            }
            let kc = f32_slice(unsafe { &*inputs.add(3) })?;
            let vc = f32_slice(unsafe { &*inputs.add(4) })?;
            if kc.len() < rows * n_kv * head_dim || vc.len() < rows * n_kv * head_dim {
                return Err(-2);
            }
            let ctx = oxide_ctx();
            let stream = oxide_stream().clone();
            let module = oxide_module();
            let snap = host_paged_snapshot().ok_or(-20)?;
            if snap.device_arena.is_null() || snap.device_arena_len == 0 {
                return Err(-21);
            }
            let table = unsafe { std::slice::from_raw_parts(snap.block_table, snap.n_block_table) };
            let qk = buffers::ensure_f32(&stream, q, false)?;
            let kk = buffers::ensure_f32(&stream, kc, false)?;
            let vk = buffers::ensure_f32(&stream, vc, false)?;
            let tk = buffers::ensure_block_table(&stream, table)?;
            let ok = buffers::ensure_f32_out(&stream, out)?;
            let (qd, _) = buffers::take_f32(qk)?;
            let (kd, _) = buffers::take_f32(kk)?;
            let (vd, _) = buffers::take_f32(vk)?;
            let td = buffers::take_u32(tk)?;
            let (mut od, _) = buffers::take_f32(ok)?;
            let arena = unsafe {
                buffers::wrap_device_bytes(
                    snap.device_arena,
                    snap.device_arena_len,
                    std::sync::Arc::clone(ctx),
                )
            };
            let rc = unsafe {
                module
                    .attention_canvas_paged_heads(
                        &stream,
                        cfg_1d((rows * n_heads) as u32),
                        &qd,
                        &arena,
                        &td,
                        &kd,
                        &vd,
                        rows as u32,
                        past as u32,
                        n_heads as u32,
                        n_kv as u32,
                        head_dim as u32,
                        scale,
                        attrs.layer_ord,
                        snap.n_logical_blocks as u32,
                        snap.block_size as u32,
                        snap.block_bytes as u32,
                        snap.tokens_stride as u32,
                        &mut od,
                    )
                    .map_err(|_| -4)
            };
            buffers::release_wrap(arena);
            buffers::put_f32(qk, qd, true)?;
            buffers::put_f32(kk, kd, true)?;
            buffers::put_f32(vk, vd, true)?;
            buffers::put_u32(tk, td)?;
            buffers::put_f32(ok, od, false)?;
            rc?;
            return finish_out(&stream, ok, out, false);
        }
        if attrs.block_size > 0
            && let Some(snap) = host_paged_snapshot()
            && snap.batch_size > 1
        {
            let width = n_heads * head_dim;
            if q.len() != snap.batch_size * width || out.len() != q.len() {
                return Err(-2);
            }
            if snap.device_arena.is_null() || snap.device_arena_len == 0 {
                return Err(-21);
            }
            let ctx = oxide_ctx();
            let stream = oxide_stream().clone();
            let table = unsafe { std::slice::from_raw_parts(snap.block_table, snap.n_block_table) };
            let lengths = unsafe { std::slice::from_raw_parts(snap.row_lengths, snap.batch_size) };
            let q_key = buffers::ensure_f32(&stream, q, false)?;
            let table_key = buffers::ensure_block_table(&stream, table)?;
            let lengths_key = buffers::ensure_u32(&stream, lengths)?;
            let out_key = buffers::ensure_f32_out(&stream, out)?;
            let (qd, _) = buffers::take_f32(q_key)?;
            let table_device = buffers::take_u32(table_key)?;
            let lengths_device = buffers::take_u32(lengths_key)?;
            let (mut out_device, _) = buffers::take_f32(out_key)?;
            let arena = unsafe {
                buffers::wrap_device_bytes(
                    snap.device_arena,
                    snap.device_arena_len,
                    std::sync::Arc::clone(ctx),
                )
            };
            let result = unsafe {
                oxide_module()
                    .attention_paged_batch_heads(
                        &stream,
                        cfg_1d((snap.batch_size * n_heads) as u32),
                        &qd,
                        &arena,
                        &table_device,
                        &lengths_device,
                        snap.batch_size as u32,
                        n_heads as u32,
                        n_kv as u32,
                        head_dim as u32,
                        scale,
                        attrs.layer_ord,
                        snap.n_layers as u32,
                        snap.n_logical_blocks as u32,
                        snap.block_size as u32,
                        snap.block_bytes as u32,
                        snap.tokens_stride as u32,
                        &mut out_device,
                    )
                    .map_err(|_| -4)
            };
            buffers::release_wrap(arena);
            buffers::put_f32(q_key, qd, true)?;
            buffers::put_u32(table_key, table_device)?;
            buffers::put_u32(lengths_key, lengths_device)?;
            buffers::put_f32(out_key, out_device, false)?;
            result?;
            return finish_out(&stream, out_key, out, false);
        }
        if q.len() != n_heads * head_dim || out.len() != n_heads * head_dim {
            eprintln!(
                "cuda_kernels: attention shape mismatch q={} out={} expected={} heads={} head_dim={} mode={}",
                q.len(),
                out.len(),
                n_heads * head_dim,
                n_heads,
                head_dim,
                attrs.attention_mode
            );
            return Err(-2);
        }

        let ctx = oxide_ctx();
        let stream = oxide_stream().clone();
        let module = oxide_module();

        // B2 device-paged path.
        if attrs.block_size > 0 {
            let snap = host_paged_snapshot().ok_or(-20)?;
            if !snap.device_arena.is_null() && snap.device_arena_len > 0 {
                let table =
                    unsafe { std::slice::from_raw_parts(snap.block_table, snap.n_block_table) };
                let q_key = buffers::ensure_f32(&stream, q, false)?;
                let o_key = buffers::ensure_f32_out(&stream, out)?;
                let (qd, _) = buffers::take_f32(q_key)?;
                let persistent_table = !snap.device_block_table.is_null()
                    && snap.n_device_block_table >= table.len();
                let device_n_logical = if persistent_table {
                    snap.device_logical_stride
                } else {
                    snap.n_logical_blocks
                } as u32;
                let (td, t_key) = if persistent_table {
                    (
                        unsafe {
                            buffers::wrap_device_u32(
                                snap.device_block_table,
                                snap.n_device_block_table,
                                std::sync::Arc::clone(ctx),
                            )
                        },
                        None,
                    )
                } else {
                    let key = buffers::ensure_block_table(&stream, table)?;
                    (buffers::take_u32(key)?, Some(key))
                };
                let (mut od, _) = buffers::take_f32(o_key)?;
                // SAFETY: host DeviceKvArena lives for the step; we release without free.
                let arena = unsafe {
                    buffers::wrap_device_bytes(
                        snap.device_arena,
                        snap.device_arena_len,
                        std::sync::Arc::clone(ctx),
                    )
                };
                let q_len = attrs.query_len.max(1);
                let path = fellm_plugin_abi::resolve_path(q_len, attrs.custom_op_id);
                let dispatch = fellm_plugin_abi::attention_dispatch();
                let launch_rc = match path {
                        fellm_plugin_abi::AttentionKernelPath::Fa3Decode
                        | fellm_plugin_abi::AttentionKernelPath::Fa3Prefill => with_step_params(|params| unsafe {
                            module.attention_fa3_decode_paged(
                                &stream,
                                cfg_fa_decode(n_heads as u32),
                                &qd,
                                &arena,
                                &td,
                                params,
                                n_heads as u32,
                                n_kv as u32,
                                head_dim as u32,
                                scale,
                                attrs.layer_ord,
                                device_n_logical,
                                snap.block_size as u32,
                                snap.block_bytes as u32,
                                snap.tokens_stride as u32,
                                dispatch.pipeline_stages.max(2),
                                &mut od,
                            ).map_err(|_| -4)
                        }),
                        fellm_plugin_abi::AttentionKernelPath::Fa2Prefill if q_len > 1 => {
                            let br = dispatch.q_tile.max(4);
                            let q_tiles = q_len.div_ceil(br);
                            unsafe { module.attention_fa2_prefill_paged(
                                &stream,
                                cfg_fa2_prefill(q_tiles, n_heads as u32),
                                &qd,
                                &arena,
                                &td,
                                n_heads as u32,
                                n_kv as u32,
                                head_dim as u32,
                                q_len,
                                seq as u32,
                                scale,
                                attrs.layer_ord,
                                    device_n_logical,
                                snap.block_size as u32,
                                snap.block_bytes as u32,
                                snap.tokens_stride as u32,
                                if attrs.attention_mode == 0 { 1 } else { 0 },
                                attrs.attention_window,
                                br,
                                &mut od,
                            ).map_err(|_| -4) }
                        }
                        fellm_plugin_abi::AttentionKernelPath::Fa2Decode
                        | fellm_plugin_abi::AttentionKernelPath::Fa2Prefill
                        | fellm_plugin_abi::AttentionKernelPath::HostFa2
                        | fellm_plugin_abi::AttentionKernelPath::Auto => {
                            with_step_params(|params| unsafe { module.attention_fa2_decode_paged(
                                &stream,
                                cfg_fa_decode(n_heads as u32),
                                &qd,
                                &arena,
                                &td,
                                params,
                                n_heads as u32,
                                n_kv as u32,
                                head_dim as u32,
                                scale,
                                attrs.layer_ord,
                                    device_n_logical,
                                snap.block_size as u32,
                                snap.block_bytes as u32,
                                snap.tokens_stride as u32,
                                &mut od,
                            ).map_err(|_| -4) })
                        }
                };
                buffers::release_wrap(arena);
                buffers::put_f32(q_key, qd, true)?;
                if let Some(key) = t_key {
                    buffers::put_u32(key, td)?;
                } else {
                    buffers::release_wrap_u32(td);
                }
                buffers::put_f32(o_key, od, false)?;
                launch_rc?;
                return finish_out(&stream, o_key, out, false);
            }

            // Host-gather fallback (no VRAM arena).
            let layer = attrs.layer_ord as usize;
            let kv_elems = seq * n_kv * head_dim;
            let mut k = vec![0.0f32; kv_elems];
            let mut v = vec![0.0f32; kv_elems];
            for t in 0..seq {
                let kr = unsafe { snap.k_row(layer, t) };
                let vr = unsafe { snap.v_row(layer, t) };
                let dst_k = &mut k[t * n_kv * head_dim..(t + 1) * n_kv * head_dim];
                let dst_v = &mut v[t * n_kv * head_dim..(t + 1) * n_kv * head_dim];
                for (d, &s) in dst_k.iter_mut().zip(kr.iter()) {
                    *d = s.to_f32();
                }
                for (d, &s) in dst_v.iter_mut().zip(vr.iter()) {
                    *d = s.to_f32();
                }
            }
            let q_key = buffers::ensure_f32(&stream, q, false)?;
            // Gathered K/V are ephemeral — upload once without long-lived cache key reuse.
            let kd = DeviceBuffer::from_host(&stream, &k).map_err(|_| -3)?;
            let vd = DeviceBuffer::from_host(&stream, &v).map_err(|_| -3)?;
            let o_key = buffers::ensure_f32_out(&stream, out)?;
            let (qd, _) = buffers::take_f32(q_key)?;
            let (mut od, _) = buffers::take_f32(o_key)?;
            let rc = unsafe {
                module
                    .attention_heads(
                        &stream,
                        cfg_1d(n_heads as u32),
                        &qd,
                        &kd,
                        &vd,
                        n_heads as u32,
                        n_kv as u32,
                        head_dim as u32,
                        seq as u32,
                        scale,
                        &mut od,
                    )
                    .map_err(|_| -4)
            };
            buffers::put_f32(q_key, qd, true)?;
            buffers::put_f32(o_key, od, false)?;
            rc?;
            return finish_out(&stream, o_key, out, false);
        }

        if n_inputs < 3 {
            return Err(-1);
        }
        let kv_elems = seq * n_kv * head_dim;
        let k_t = unsafe { &*inputs.add(1) };
        let v_t = unsafe { &*inputs.add(2) };
        let k = f32_slice(k_t)?;
        let v = f32_slice(v_t)?;
        if k.len() < kv_elems || v.len() < kv_elems {
            return Err(-2);
        }
        let q_key = buffers::ensure_f32(&stream, q, false)?;
        let k_key = buffers::ensure_f32(&stream, &k[..kv_elems], false)?;
        let v_key = buffers::ensure_f32(&stream, &v[..kv_elems], false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (qd, _) = buffers::take_f32(q_key)?;
        let (kd, _) = buffers::take_f32(k_key)?;
        let (vd, _) = buffers::take_f32(v_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let rc = unsafe {
            module
                .attention_heads(
                    &stream,
                    cfg_1d(n_heads as u32),
                    &qd,
                    &kd,
                    &vd,
                    n_heads as u32,
                    n_kv as u32,
                    head_dim as u32,
                    seq as u32,
                    scale,
                    &mut od,
                )
                .map_err(|_| -4)
        };
        buffers::put_f32(q_key, qd, true)?;
        buffers::put_f32(k_key, kd, true)?;
        buffers::put_f32(v_key, vd, true)?;
        buffers::put_f32(o_key, od, false)?;
        rc?;
        finish_out(&stream, o_key, out, false)
    })
}

/// Paged KvWrite: write directly to the device-owned f16 arena.
///
/// The host arena is updated only for a host-only backend. CUDA prefix
/// persistence and swap explicitly snapshot device KV at those boundaries.
pub unsafe extern "C" fn launch_kv_write(
    attrs: *const OpAttrs,
    inputs: *const TensorRef,
    n_inputs: u32,
    outputs: *mut TensorMut,
    n_outputs: u32,
    _stream: StreamHandle,
) -> i32 {
    run(|| {
        if n_inputs < 1 || n_outputs < 1 || attrs.is_null() || inputs.is_null() || outputs.is_null()
        {
            return Err(-1);
        }
        let attrs = unsafe { &*attrs };
        let row_t = unsafe { &*inputs };
        let row = f32_slice(row_t)?;
        let pos = attrs.position as usize;
        let is_v = attrs.kv_slot != 0;

        if attrs.block_size == 0 {
            return Err(-21);
        }
        let snap = host_paged_snapshot().ok_or(-20)?;
        if snap.batch_size > 1 {
            if row.len() != snap.batch_size * snap.tokens_stride {
                return Err(-2);
            }
            let positions =
                unsafe { std::slice::from_raw_parts(snap.row_positions, snap.batch_size) };
            if snap.device_arena.is_null() || snap.device_arena_len == 0 {
                for (batch_row, (&position, values)) in positions
                    .iter()
                    .zip(row.chunks_exact(snap.tokens_stride))
                    .enumerate()
                {
                    let logical = position as usize / snap.block_size;
                    let slot = position as usize % snap.block_size;
                    let phys =
                        snap.physical_for(batch_row, attrs.layer_ord as usize, logical) as usize;
                    let row_bytes = snap.tokens_stride * snap.elem_bytes;
                    let v_base = if is_v { snap.block_size * row_bytes } else { 0 };
                    let base = phys * snap.block_bytes + v_base + slot * row_bytes;
                    if base + row_bytes > snap.arena_len {
                        return Err(-22);
                    }
                    let destination = unsafe {
                        std::slice::from_raw_parts_mut(
                            snap.arena.add(base) as *mut f16,
                            snap.tokens_stride,
                        )
                    };
                    for (destination, &source) in destination.iter_mut().zip(values) {
                        *destination = f16::from_f32(source);
                    }
                }
                return Ok(());
            }
            let ctx = oxide_ctx();
            let stream = oxide_stream().clone();
            let table = unsafe { std::slice::from_raw_parts(snap.block_table, snap.n_block_table) };
            let row_key = buffers::ensure_f32(&stream, row, false)?;
            let table_key = buffers::ensure_block_table(&stream, table)?;
            let position_key = buffers::ensure_u32(&stream, positions)?;
            let (row_device, _) = buffers::take_f32(row_key)?;
            let table_device = buffers::take_u32(table_key)?;
            let position_device = buffers::take_u32(position_key)?;
            let mut arena = unsafe {
                buffers::wrap_device_bytes(
                    snap.device_arena,
                    snap.device_arena_len,
                    std::sync::Arc::clone(ctx),
                )
            };
            let result = unsafe {
                oxide_module()
                    .kv_write_batch(
                        &stream,
                        cfg_1d(row.len() as u32),
                        &row_device,
                        &mut arena,
                        &table_device,
                        &position_device,
                        snap.batch_size as u32,
                        attrs.layer_ord,
                        snap.n_layers as u32,
                        u32::from(is_v),
                        snap.n_logical_blocks as u32,
                        snap.tokens_stride as u32,
                        snap.block_size as u32,
                        snap.block_bytes as u32,
                    )
                    .map_err(|_| -4)
            };
            buffers::release_wrap(arena);
            buffers::put_f32(row_key, row_device, true)?;
            buffers::put_u32(table_key, table_device)?;
            buffers::put_u32(position_key, position_device)?;
            result?;
            return Ok(());
        }
        if row.len() > snap.tokens_stride {
            eprintln!(
                "cuda_kernels: kv_write shape mismatch row={} stride={} layer={} position={}",
                row.len(),
                snap.tokens_stride,
                attrs.layer_ord,
                attrs.position
            );
            return Err(-2);
        }

        // Resolve physical location from the block table (same as attention).
        let table = unsafe { std::slice::from_raw_parts(snap.block_table, snap.n_block_table) };
        let need = (attrs.layer_ord as usize + 1) * snap.n_logical_blocks.max(1);
        if table.is_empty()
            || table.len() < need
            || snap.n_logical_blocks == 0
            || snap.block_bytes == 0
            || snap.block_size == 0
        {
            eprintln!(
                "cuda_kernels: kv_write bad table len={} need={} n_logical={} layer={} block_bytes={}",
                table.len(),
                need,
                snap.n_logical_blocks,
                attrs.layer_ord,
                snap.block_bytes
            );
            return Err(-23);
        }
        let logical = pos / snap.block_size;
        let slot = pos % snap.block_size;
        if logical >= snap.n_logical_blocks {
            return Err(-24);
        }
        let phys = snap.physical(attrs.layer_ord as usize, logical) as usize;
        let row_bytes = snap.tokens_stride * snap.elem_bytes;
        let v_base = if is_v { snap.block_size * row_bytes } else { 0 };
        let base = phys * snap.block_bytes + v_base + slot * row_bytes;

        let stream = oxide_stream().clone();
        let ctx = oxide_ctx();

        let r_key = buffers::ensure_f32(&stream, row, false)?;
        // Host-only path. A CUDA decode never touches the host KV arena here.
        if snap.device_arena.is_null() || snap.device_arena_len == 0 {
            if base + row_bytes > snap.arena_len {
                return Err(-22);
            }
            let dst = unsafe {
                std::slice::from_raw_parts_mut(snap.arena.add(base) as *mut f16, snap.tokens_stride)
            };
            for (d, &s) in dst.iter_mut().zip(row.iter()) {
                *d = f16::from_f32(s);
            }
        }

        // Device arena: oxide scatter from device-resident activations.
        if !snap.device_arena.is_null() && snap.device_arena_len > 0 {
            let module = oxide_module();
            let (rd, _) = buffers::take_f32(r_key)?;
            let persistent_table = !snap.device_block_table.is_null()
                && snap.n_device_block_table >= table.len();
            let device_n_logical = if persistent_table {
                snap.device_logical_stride
            } else {
                snap.n_logical_blocks
            } as u32;
            let (td, t_key) = if persistent_table {
                (
                    unsafe {
                        buffers::wrap_device_u32(
                            snap.device_block_table,
                            snap.n_device_block_table,
                            std::sync::Arc::clone(ctx),
                        )
                    },
                    None,
                )
            } else {
                let key = buffers::ensure_block_table(&stream, table)?;
                (buffers::take_u32(key)?, Some(key))
            };
            let arena = unsafe {
                buffers::wrap_device_bytes(
                    snap.device_arena,
                    snap.device_arena_len,
                    std::sync::Arc::clone(ctx),
                )
            };
            let mut arena_mut = arena;
            let launch_rc = with_step_params(|params| unsafe {
                module
                    .kv_write_row(
                    &stream,
                    cfg_1d(row.len() as u32),
                    &rd,
                    &mut arena_mut,
                    &td,
                    params,
                    attrs.layer_ord,
                    u32::from(is_v),
                        device_n_logical,
                    snap.tokens_stride as u32,
                    row.len() as u32,
                    snap.block_size as u32,
                    snap.block_bytes as u32,
                )
                    .map_err(|_| -4)
            });
            buffers::release_wrap(arena_mut);
            buffers::put_f32(r_key, rd, true)?;
            if let Some(key) = t_key {
                buffers::put_u32(key, td)?;
            } else {
                buffers::release_wrap_u32(td);
            }
            launch_rc?;
        }
        Ok(())
    })
}
