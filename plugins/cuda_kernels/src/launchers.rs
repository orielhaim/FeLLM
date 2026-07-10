//! FFI launch wrappers: activation-cached H2D → oxide kernel → D2H.

use crate::buffers;
use crate::tensor::{bytes_slice, dims, f32_slice, f32_slice_mut, u32_slice};
use crate::{host_paged_snapshot, oxide_ctx, oxide_module, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS};
use cuda_core::{DeviceBuffer, LaunchConfig};
use fellm_core::dtype::DType;
use fellm_plugin_abi::op::OpAttrs;
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use half::f16;

fn cfg_1d(n: u32) -> LaunchConfig {
    LaunchConfig::for_num_elems(n.max(1))
}

type LaunchResult = Result<(), i32>;

fn run(body: impl FnOnce() -> LaunchResult) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => 0,
        Ok(Err(c)) => c,
        Err(_) => -99,
    }
}

/// Finish a launch: mark output valid and always D2H for numerical parity with
/// the host graph (sampling + any CPU fallback). Allocation reuse still avoids
/// per-op `cudaMalloc`; skip-H2D when `device_valid` cuts redundant uploads.
fn finish_out(
    stream: &cuda_core::CudaStream,
    out_key: usize,
    out: &mut [f32],
    _sync_host: bool,
) -> LaunchResult {
    buffers::mark_valid(out_key)?;
    buffers::download_to(stream, out_key, out)?;
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
        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
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
        let head_dim = attrs.head_dim as usize;
        let n_heads = attrs.n_heads as usize;
        let (group_n, n_groups, group_stride) =
            if head_dim > 0 && n_heads > 0 && x.len() == n_heads * head_dim && w.len() == head_dim {
                (head_dim as u32, n_heads as u32, head_dim as u32)
            } else {
                if w.len() != x.len() {
                    return Err(-2);
                }
                (x.len() as u32, 1u32, x.len() as u32)
            };

        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
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
        let n = x.len();
        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let f_key = buffers::ensure_f32(&stream, inv, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (fd, _) = buffers::take_f32(f_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = unsafe {
            module
                .rope(
                    &stream,
                    cfg_1d(n as u32),
                    &xd,
                    &fd,
                    attrs.n_heads,
                    attrs.head_dim,
                    attrs.rope_dim,
                    attrs.position as f32,
                    &mut od,
                )
                .map_err(|_| -4)
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
        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
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
        let token_id = ids[0];
        let out = f32_slice_mut(out_t)?;
        if out.len() != dim || table.len() < (token_id as usize + 1) * dim {
            return Err(-2);
        }
        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
        // Weight table: cache via ensure_f32 (stable host ptr).
        let w_key = buffers::ensure_f32(&stream, table, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (wd, _) = buffers::take_f32(w_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = unsafe {
            module
                .embedding_f32(
                    &stream,
                    cfg_1d(dim as u32),
                    &wd,
                    token_id,
                    dim as u32,
                    &mut od,
                )
                .map_err(|_| -4)
        };
        buffers::put_f32(w_key, wd, true)?;
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
        let token_id = ids[0];
        let out = f32_slice_mut(out_t)?;
        if out.len() != dim {
            return Err(-2);
        }
        let row_bytes = n_blocks as usize * Q4K_BLOCK_BYTES as usize;
        let need = (token_id as usize + 1) * row_bytes;
        if wb.len() < need {
            return Err(-2);
        }

        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = buffers::with_weight(w_key, |wd| unsafe {
            module
                .embedding_q4k_row(
                    &stream,
                    cfg_1d(dim as u32),
                    wd,
                    token_id,
                    dim as u32,
                    n_blocks,
                    &mut od,
                )
                .map_err(|_| -4)
        });
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
        if x.len() != in_dim || out.len() != out_dim as usize {
            return Err(-2);
        }

        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
        let w_key = buffers::ensure_weight(&stream, wb)?;
        let x_key = buffers::ensure_f32(&stream, x, false)?;
        let o_key = buffers::ensure_f32_out(&stream, out)?;
        let (xd, _) = buffers::take_f32(x_key)?;
        let (mut od, _) = buffers::take_f32(o_key)?;
        let module = oxide_module();
        let rc = buffers::with_weight(w_key, |wd| unsafe {
            module
                .q4k_gemv_row(&stream, cfg_1d(out_dim), wd, &xd, out_dim, n_blocks, &mut od)
                .map_err(|_| -4)
        });
        buffers::put_f32(x_key, xd, true)?;
        buffers::put_f32(o_key, od, false)?;
        rc??;
        finish_out(&stream, o_key, out, true)
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
        if q.len() != n_heads * head_dim || out.len() != n_heads * head_dim {
            return Err(-2);
        }

        let ctx = oxide_ctx();
        let stream = ctx.default_stream();
        let module = oxide_module();

        // B2 device-paged path.
        if attrs.block_size > 0 {
            let snap = host_paged_snapshot().ok_or(-20)?;
            if !snap.device_arena.is_null() && snap.device_arena_len > 0 {
                let table = unsafe {
                    std::slice::from_raw_parts(snap.block_table, snap.n_block_table)
                };
                let q_key = buffers::ensure_f32(&stream, q, false)?;
                let t_key = buffers::ensure_block_table(&stream, table)?;
                let o_key = buffers::ensure_f32_out(&stream, out)?;
                let (qd, _) = buffers::take_f32(q_key)?;
                let td = buffers::take_u32(t_key)?;
                let (mut od, _) = buffers::take_f32(o_key)?;
                // SAFETY: host DeviceKvArena lives for the step; we release without free.
                let arena = unsafe {
                    buffers::wrap_device_bytes(
                        snap.device_arena,
                        snap.device_arena_len,
                        std::sync::Arc::clone(ctx),
                    )
                };
                let launch_rc = unsafe {
                    module.attention_paged_heads(
                        &stream,
                        cfg_1d(n_heads as u32),
                        &qd,
                        &arena,
                        &td,
                        n_heads as u32,
                        n_kv as u32,
                        head_dim as u32,
                        seq as u32,
                        scale,
                        attrs.layer_ord,
                        snap.n_logical_blocks as u32,
                        snap.block_size as u32,
                        snap.block_bytes as u32,
                        snap.tokens_stride as u32,
                        &mut od,
                    )
                };
                buffers::release_wrap(arena);
                buffers::put_f32(q_key, qd, true)?;
                buffers::put_u32(t_key, td)?;
                buffers::put_f32(o_key, od, false)?;
                launch_rc.map_err(|_| -4)?;
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

/// Paged KvWrite: dual-write host arena + device arena (f32 row → f16).
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
        if row.len() != snap.tokens_stride {
            return Err(-2);
        }

        // Host dual-write (keeps PhysicalPool coherent for next-step H2D / prefix).
        {
            let logical = pos / snap.block_size;
            let slot = pos % snap.block_size;
            let phys = snap.physical(attrs.layer_ord as usize, logical) as usize;
            let row_bytes = snap.tokens_stride * snap.elem_bytes;
            let v_base = if is_v {
                snap.block_size * row_bytes
            } else {
                0
            };
            let base = phys * snap.block_bytes + v_base + slot * row_bytes;
            if base + row_bytes > snap.arena_len {
                return Err(-22);
            }
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    snap.arena.add(base) as *mut f16,
                    snap.tokens_stride,
                )
            };
            for (d, &s) in dst.iter_mut().zip(row.iter()) {
                *d = f16::from_f32(s);
            }
        }

        // Device write when VRAM arena is available.
        if !snap.device_arena.is_null() && snap.device_arena_len > 0 {
            let ctx = oxide_ctx();
            let stream = ctx.default_stream();
            let module = oxide_module();
            let table = unsafe {
                std::slice::from_raw_parts(snap.block_table, snap.n_block_table)
            };
            let r_key = buffers::ensure_f32(&stream, row, false)?;
            let t_key = buffers::ensure_block_table(&stream, table)?;
            let (rd, _) = buffers::take_f32(r_key)?;
            let td = buffers::take_u32(t_key)?;
            let arena = unsafe {
                buffers::wrap_device_bytes(
                    snap.device_arena,
                    snap.device_arena_len,
                    std::sync::Arc::clone(ctx),
                )
            };
            let mut arena_mut = arena;
            let launch_rc = unsafe {
                module.kv_write_row(
                    &stream,
                    cfg_1d(snap.tokens_stride as u32),
                    &rd,
                    &mut arena_mut,
                    &td,
                    attrs.layer_ord,
                    attrs.position,
                    u32::from(is_v),
                    snap.n_logical_blocks as u32,
                    snap.tokens_stride as u32,
                    snap.block_size as u32,
                    snap.block_bytes as u32,
                )
            };
            buffers::release_wrap(arena_mut);
            buffers::put_f32(r_key, rd, true)?;
            buffers::put_u32(t_key, td)?;
            launch_rc.map_err(|_| -4)?;
        }
        Ok(())
    })
}
