//! FFI launch wrappers: H2D → oxide kernel → D2H for host-resident tensors.

use crate::buffers;
use crate::tensor::{bytes_slice, dims, f32_slice, f32_slice_mut};
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
        let g = DeviceBuffer::from_host(&stream, gate).map_err(|_| -3)?;
        let u = DeviceBuffer::from_host(&stream, up).map_err(|_| -3)?;
        let mut o = DeviceBuffer::<f32>::zeroed(&stream, n).map_err(|_| -3)?;
        let module = oxide_module();
        unsafe {
            module
                .silu_gate(&stream, cfg_1d(n as u32), &g, &u, &mut o)
                .map_err(|_| -4)?;
        }
        let host = o.to_host_vec(&stream).map_err(|_| -5)?;
        out.copy_from_slice(&host);
        Ok(())
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
        let xd = DeviceBuffer::from_host(&stream, x).map_err(|_| -3)?;
        let wd = DeviceBuffer::from_host(&stream, w).map_err(|_| -3)?;
        let mut od = DeviceBuffer::<f32>::zeroed(&stream, x.len()).map_err(|_| -3)?;
        let module = oxide_module();
        let launch = LaunchConfig {
            grid_dim: (n_groups, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
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
                .map_err(|_| -4)?;
        }
        let host = od.to_host_vec(&stream).map_err(|_| -5)?;
        out.copy_from_slice(&host);
        Ok(())
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
        let xd = DeviceBuffer::from_host(&stream, x).map_err(|_| -3)?;
        let fd = DeviceBuffer::from_host(&stream, inv).map_err(|_| -3)?;
        let mut od = DeviceBuffer::<f32>::zeroed(&stream, n).map_err(|_| -3)?;
        let module = oxide_module();
        unsafe {
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
                .map_err(|_| -4)?;
        }
        let host = od.to_host_vec(&stream).map_err(|_| -5)?;
        out.copy_from_slice(&host);
        Ok(())
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
        let wkey = buffers::ensure_weight(&stream, wb)?;
        let xd = DeviceBuffer::from_host(&stream, x).map_err(|_| -3)?;
        let mut od = DeviceBuffer::<f32>::zeroed(&stream, out_dim as usize).map_err(|_| -3)?;
        let module = oxide_module();
        buffers::with_weight(wkey, |wd| {
            unsafe {
                module
                    .q4k_gemv_row(
                        &stream,
                        cfg_1d(out_dim),
                        wd,
                        &xd,
                        out_dim,
                        n_blocks,
                        &mut od,
                    )
                    .map_err(|_| -4)
            }
        })??;
        let host = od.to_host_vec(&stream).map_err(|_| -5)?;
        out.copy_from_slice(&host);
        Ok(())
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
                let qd = DeviceBuffer::from_host(&stream, q).map_err(|_| -3)?;
                let td = DeviceBuffer::from_host(&stream, table).map_err(|_| -3)?;
                let mut od =
                    DeviceBuffer::<f32>::zeroed(&stream, out.len()).map_err(|_| -3)?;
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
                launch_rc.map_err(|_| -4)?;
                let host = od.to_host_vec(&stream).map_err(|_| -5)?;
                out.copy_from_slice(&host);
                return Ok(());
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
            let qd = DeviceBuffer::from_host(&stream, q).map_err(|_| -3)?;
            let kd = DeviceBuffer::from_host(&stream, &k).map_err(|_| -3)?;
            let vd = DeviceBuffer::from_host(&stream, &v).map_err(|_| -3)?;
            let mut od = DeviceBuffer::<f32>::zeroed(&stream, out.len()).map_err(|_| -3)?;
            unsafe {
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
                    .map_err(|_| -4)?;
            }
            let host = od.to_host_vec(&stream).map_err(|_| -5)?;
            out.copy_from_slice(&host);
            return Ok(());
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
        let qd = DeviceBuffer::from_host(&stream, q).map_err(|_| -3)?;
        let kd = DeviceBuffer::from_host(&stream, &k[..kv_elems]).map_err(|_| -3)?;
        let vd = DeviceBuffer::from_host(&stream, &v[..kv_elems]).map_err(|_| -3)?;
        let mut od = DeviceBuffer::<f32>::zeroed(&stream, out.len()).map_err(|_| -3)?;
        unsafe {
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
                .map_err(|_| -4)?;
        }
        let host = od.to_host_vec(&stream).map_err(|_| -5)?;
        out.copy_from_slice(&host);
        Ok(())
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
            let rd = DeviceBuffer::from_host(&stream, row).map_err(|_| -3)?;
            let td = DeviceBuffer::from_host(&stream, table).map_err(|_| -3)?;
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
            launch_rc.map_err(|_| -4)?;
        }
        Ok(())
    })
}
