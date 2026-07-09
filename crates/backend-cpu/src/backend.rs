use crate::kernels::{
    attention::attention_step, embedding::embedding_row, matmul, norm::rmsnorm_row,
    sampling::sample, softmax::softmax_rows_inplace, swiglu::silu_gate,
};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{Backend, BackendCaps, KernelDescriptor, KernelHandle};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};

/// The CPU backend.
pub struct CpuBackend {
    caps: BackendCaps,
}

impl CpuBackend {
    /// Detect capabilities and construct.
    #[must_use]
    pub fn new() -> Self {
        let logical = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1) as u32;
        Self {
            caps: BackendCaps {
                simd_f32_lanes: 8,
                has_avx512: cfg!(target_feature = "avx512f"),
                has_avx2: cfg!(target_feature = "avx2"),
                has_neon: cfg!(target_arch = "aarch64"),
                physical_cores: logical,
                logical_threads: logical,
            },
        }
    }

    fn make_handle(op: OpKind) -> KernelHandle {
        KernelHandle(op as u64)
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn id(&self) -> &'static str {
        "cpu"
    }

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn resolve_kernel(
        &self,
        op: OpKind,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<KernelDescriptor> {
        let ok = match op {
            OpKind::MatMul => matches!(
                (input_dtypes.first(), input_dtypes.get(1)),
                (
                    Some(
                        DType::F32
                            | DType::F16
                            | DType::BF16
                            | DType::Q4_0
                            | DType::Q8_0
                            | DType::Q4K
                            | DType::Q6K
                    ),
                    Some(DType::F32)
                )
            ),
            OpKind::Embedding => input_dtypes
                .first()
                .map(|d| {
                    matches!(
                        d,
                        DType::F32
                            | DType::F16
                            | DType::BF16
                            | DType::Q4_0
                            | DType::Q8_0
                            | DType::Q4K
                            | DType::Q6K
                    )
                })
                .unwrap_or(false),
            OpKind::RmsNorm
            | OpKind::Rope
            | OpKind::SiluGate
            | OpKind::Softmax
            | OpKind::Attention
            | OpKind::Add
            | OpKind::Mul
            | OpKind::Reshape
            | OpKind::Cast
            | OpKind::Concat
            | OpKind::Sample
            | OpKind::KvWrite => true,
        };
        if !ok {
            return None;
        }
        Some(KernelDescriptor {
            op,
            input_dtypes: input_dtypes.to_vec(),
            output_dtype,
            handle: Self::make_handle(op),
        })
    }

    fn launch(
        &self,
        handle: KernelHandle,
        attrs: &OpAttrs,
        inputs: &[TensorRef],
        outputs: &mut [TensorMut],
        _stream: StreamHandle,
    ) -> Result<()> {
        let op = decode_handle(handle)?;
        match op {
            OpKind::MatMul => launch_matmul(inputs, outputs),
            OpKind::Embedding => launch_embedding(inputs, outputs),
            OpKind::RmsNorm => launch_rmsnorm(attrs, inputs, outputs),
            OpKind::Rope => launch_rope(attrs, inputs, outputs),
            OpKind::SiluGate => launch_silu_gate(inputs, outputs),
            OpKind::Softmax => launch_softmax(attrs, inputs, outputs),
            OpKind::Attention => launch_attention(attrs, inputs, outputs),
            OpKind::Add => launch_add(inputs, outputs),
            OpKind::Mul => launch_mul(inputs, outputs),
            OpKind::Reshape => launch_reshape(inputs, outputs),
            OpKind::Cast => launch_cast(attrs, inputs, outputs),
            OpKind::Concat => launch_concat(inputs, outputs),
            OpKind::Sample => launch_sample(attrs, inputs, outputs),
            OpKind::KvWrite => launch_kv_write(attrs, inputs, outputs),
        }
    }
}

fn decode_handle(h: KernelHandle) -> Result<OpKind> {
    match h.0 {
        x if x == OpKind::Add as u64 => Ok(OpKind::Add),
        x if x == OpKind::Mul as u64 => Ok(OpKind::Mul),
        x if x == OpKind::MatMul as u64 => Ok(OpKind::MatMul),
        x if x == OpKind::RmsNorm as u64 => Ok(OpKind::RmsNorm),
        x if x == OpKind::Rope as u64 => Ok(OpKind::Rope),
        x if x == OpKind::SiluGate as u64 => Ok(OpKind::SiluGate),
        x if x == OpKind::Softmax as u64 => Ok(OpKind::Softmax),
        x if x == OpKind::Attention as u64 => Ok(OpKind::Attention),
        x if x == OpKind::Embedding as u64 => Ok(OpKind::Embedding),
        x if x == OpKind::Concat as u64 => Ok(OpKind::Concat),
        x if x == OpKind::Reshape as u64 => Ok(OpKind::Reshape),
        x if x == OpKind::Cast as u64 => Ok(OpKind::Cast),
        x if x == OpKind::Sample as u64 => Ok(OpKind::Sample),
        x if x == OpKind::KvWrite as u64 => Ok(OpKind::KvWrite),
        _ => Err(FellmError::other(format!("bad kernel handle {h:?}"))),
    }
}

fn as_f32_slice(t: &TensorRef) -> Result<&[f32]> {
    let d = t.dtype().ok_or_else(|| FellmError::other("bad dtype"))?;
    if d != DType::F32 {
        return Err(FellmError::UnsupportedDType(d));
    }
    // SAFETY: TensorRef is a valid contiguous buffer of `byte_len` bytes.
    let bytes = unsafe { core::slice::from_raw_parts(t.data, t.byte_len as usize) };
    bytemuck::try_cast_slice(bytes).map_err(|e| FellmError::other(format!("cast: {e:?}")))
}

fn as_f32_slice_mut(t: &mut TensorMut) -> Result<&mut [f32]> {
    let d = t.dtype().ok_or_else(|| FellmError::other("bad dtype"))?;
    if d != DType::F32 {
        return Err(FellmError::UnsupportedDType(d));
    }
    // SAFETY: TensorMut is a valid exclusive buffer of `byte_len` bytes.
    let bytes = unsafe { core::slice::from_raw_parts_mut(t.data, t.byte_len as usize) };
    bytemuck::try_cast_slice_mut(bytes).map_err(|e| FellmError::other(format!("cast mut: {e:?}")))
}

fn as_bytes_slice(t: &TensorRef) -> &[u8] {
    // SAFETY: valid by TensorRef contract.
    unsafe { core::slice::from_raw_parts(t.data, t.byte_len as usize) }
}

fn as_u32_slice(t: &TensorRef) -> Result<&[u32]> {
    let bytes = as_bytes_slice(t);
    bytemuck::try_cast_slice(bytes).map_err(|e| FellmError::other(format!("u32 cast: {e:?}")))
}

fn launch_matmul(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // inputs[0] = weight [out_dim, in_dim], inputs[1] = x [in_dim]
    // outputs[0] = y [out_dim]
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("matmul: bad arity"));
    }
    let (y_out, rest) = outputs.split_first_mut().unwrap();
    let _ = rest;
    let w = &inputs[0];
    let x = &inputs[1];
    let w_dtype = w.dtype().ok_or_else(|| FellmError::other("w dtype"))?;
    let out_dim = w.dims_slice()[0] as usize;
    let in_dim = w.dims_slice()[1] as usize;
    let x_slice = as_f32_slice(x)?;
    let y_slice = as_f32_slice_mut(y_out)?;
    match w_dtype {
        DType::F32 => {
            let ws = as_f32_slice(w)?;
            matmul::matvec_f32(ws, x_slice, y_slice, out_dim, in_dim);
        }
        DType::Q4_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            let wb = as_bytes_slice(w);
            matmul::matvec_quant(wb, w_dtype, x_slice, y_slice, out_dim, in_dim)?;
        }
        other => return Err(FellmError::UnsupportedDType(other)),
    }
    Ok(())
}

fn launch_embedding(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("embedding: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let w = &inputs[0];
    let tok = &inputs[1];
    let w_dtype = w.dtype().ok_or_else(|| FellmError::other("w dtype"))?;
    let vocab = w.dims_slice()[0] as usize;
    let dim = w.dims_slice()[1] as usize;
    let ids = as_u32_slice(tok)?;
    let tok_id = ids[0];
    let y_slice = as_f32_slice_mut(y_out)?;
    let wb = as_bytes_slice(w);
    embedding_row(wb, w_dtype, vocab, dim, tok_id, y_slice)
}

fn launch_rmsnorm(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("rmsnorm: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let w = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    rmsnorm_row(x, w, attrs.eps, y);
    Ok(())
}

fn launch_rope(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("rope: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x_in = as_f32_slice(&inputs[0])?;
    let inv_freqs = as_f32_slice(&inputs[1])?;
    let x_out = as_f32_slice_mut(y_out)?;
    x_out.copy_from_slice(x_in);
    crate::kernels::rope::rope_inplace_with_freqs(
        x_out,
        attrs.n_heads as usize,
        attrs.head_dim as usize,
        attrs.rope_dim as usize,
        attrs.position,
        inv_freqs,
    );
    Ok(())
}

fn launch_silu_gate(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("silu_gate: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let gate = as_f32_slice(&inputs[0])?;
    let up = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    silu_gate(gate, up, y);
    Ok(())
}

fn launch_softmax(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("softmax: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let y = as_f32_slice_mut(y_out)?;
    y.copy_from_slice(x);
    // Interpret the last-dim as row length; higher dims collapsed.
    let dims = inputs[0].dims_slice();
    let last = *dims.last().unwrap_or(&(y.len() as u64)) as usize;
    let n_rows = y.len() / last.max(1);
    let causal = if attrs.past_len > 0 {
        Some((attrs.past_len as usize + 1).min(last))
    } else {
        None
    };
    softmax_rows_inplace(y, n_rows, last, causal);
    Ok(())
}

fn launch_attention(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    // inputs: [q, k_cache, v_cache]
    if inputs.len() < 3 || outputs.is_empty() {
        return Err(FellmError::other("attention: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let q = as_f32_slice(&inputs[0])?;
    let k_full = as_f32_slice(&inputs[1])?;
    let v_full = as_f32_slice(&inputs[2])?;
    let out = as_f32_slice_mut(y_out)?;
    let n_heads = attrs.n_heads as usize;
    let n_kv = attrs.n_kv_heads.max(1) as usize;
    let head_dim = attrs.head_dim as usize;
    let past = attrs.past_len as usize;
    let scale = if attrs.scale > 0.0 {
        attrs.scale
    } else {
        1.0 / (head_dim as f32).sqrt()
    };
    let seq = past + 1;
    let kv_elems = seq * n_kv * head_dim;
    if k_full.len() < kv_elems || v_full.len() < kv_elems {
        return Err(FellmError::other(format!(
            "attention: kv buffer too small (need {kv_elems}, k={}, v={})",
            k_full.len(),
            v_full.len()
        )));
    }
    let k = &k_full[..kv_elems];
    let v = &v_full[..kv_elems];
    attention_step(q, k, v, out, n_heads, n_kv, head_dim, past, scale);
    Ok(())
}

fn launch_add(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("add: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let a = as_f32_slice(&inputs[0])?;
    let b = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    for i in 0..y.len() {
        y[i] = a[i] + b[i];
    }
    Ok(())
}

fn launch_mul(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("mul: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let a = as_f32_slice(&inputs[0])?;
    let b = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    for i in 0..y.len() {
        y[i] = a[i] * b[i];
    }
    Ok(())
}

fn launch_reshape(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("reshape: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let src = as_bytes_slice(&inputs[0]);
    // SAFETY: y_out.byte_len is the valid target length.
    let dst = unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
    if dst.len() != src.len() {
        return Err(FellmError::other(format!(
            "reshape: byte length mismatch (src={}, dst={})",
            src.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(src);
    Ok(())
}

fn launch_cast(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("cast: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let src_dtype = inputs[0]
        .dtype()
        .ok_or_else(|| FellmError::other("cast src dtype"))?;
    let dst_dtype = attrs
        .cast_dtype()
        .ok_or_else(|| FellmError::other("cast: dst dtype unset"))?;

    // Phase 1: support f32<->f16, f32<->bf16, and quantized -> f32 (via dequant).
    match (src_dtype, dst_dtype) {
        (DType::F32, DType::F32) => {
            let src = as_f32_slice(&inputs[0])?;
            let dst = as_f32_slice_mut(y_out)?;
            dst.copy_from_slice(src);
        }
        (DType::F16, DType::F32) => {
            let bytes = as_bytes_slice(&inputs[0]);
            let src: &[half::f16] = bytemuck::cast_slice(bytes);
            let dst = as_f32_slice_mut(y_out)?;
            for i in 0..dst.len() {
                dst[i] = src[i].to_f32();
            }
        }
        (DType::BF16, DType::F32) => {
            let bytes = as_bytes_slice(&inputs[0]);
            let src: &[u16] = bytemuck::cast_slice(bytes);
            let dst = as_f32_slice_mut(y_out)?;
            for i in 0..dst.len() {
                dst[i] = f32::from_bits((u32::from(src[i])) << 16);
            }
        }
        (DType::F32, DType::F16) => {
            let src = as_f32_slice(&inputs[0])?;
            // SAFETY: y_out.byte_len valid.
            let dst_bytes =
                unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
            let dst: &mut [half::f16] = bytemuck::cast_slice_mut(dst_bytes);
            for i in 0..src.len() {
                dst[i] = half::f16::from_f32(src[i]);
            }
        }
        (DType::F32, DType::BF16) => {
            let src = as_f32_slice(&inputs[0])?;
            let dst_bytes =
                unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
            let dst: &mut [u16] = bytemuck::cast_slice_mut(dst_bytes);
            for i in 0..src.len() {
                dst[i] = (src[i].to_bits() >> 16) as u16;
            }
        }
        (q, DType::F32) if q.is_quantized() => {
            let dst = as_f32_slice_mut(y_out)?;
            let bytes = as_bytes_slice(&inputs[0]);
            crate::dequant::dequantize_row(q, bytes, dst, dst.len())?;
        }
        (a, b) => {
            return Err(FellmError::other(format!("cast: unsupported {a} -> {b}")));
        }
    }
    Ok(())
}

fn launch_concat(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // Concat along last dim, contiguous f32.
    if outputs.is_empty() {
        return Err(FellmError::other("concat: no outputs"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let dst = as_f32_slice_mut(y_out)?;
    let mut offset = 0usize;
    for inp in inputs {
        let s = as_f32_slice(inp)?;
        dst[offset..offset + s.len()].copy_from_slice(s);
        offset += s.len();
    }
    if offset != dst.len() {
        return Err(FellmError::other(format!(
            "concat: length mismatch (wrote {offset}, dst {})",
            dst.len()
        )));
    }
    Ok(())
}

fn launch_sample(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("sample: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    // We need a mutable f32 copy of logits to modify in place.
    let logits_in = as_f32_slice(&inputs[0])?;
    let mut work = logits_in.to_vec();
    let tok = sample(
        &mut work,
        attrs.temperature,
        attrs.top_k,
        attrs.top_p,
        attrs.seed,
    );
    // Write out as u32.
    // SAFETY: y_out.byte_len is valid and at least 4 bytes.
    let dst_bytes = unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
    if dst_bytes.len() < 4 {
        return Err(FellmError::other("sample: output too small"));
    }
    dst_bytes[..4].copy_from_slice(&tok.to_le_bytes());
    Ok(())
}

fn launch_kv_write(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // inputs[0] = row [dim] f32
    // outputs[0] = kv_buf [max_seq, dim] f32 (aliased mutable storage)
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("kv_write: bad arity"));
    }
    let (buf_out, _) = outputs.split_first_mut().unwrap();
    let dims = buf_out.dims_slice();
    if dims.len() != 2 {
        return Err(FellmError::other("kv_write: kv_buf must be 2D"));
    }
    let max_seq = dims[0] as usize;
    let dim = dims[1] as usize;
    let row = as_f32_slice(&inputs[0])?;
    if row.len() != dim {
        return Err(FellmError::other(format!(
            "kv_write: row len {} != dim {dim}",
            row.len()
        )));
    }
    let pos = attrs.position as usize;
    if pos >= max_seq {
        return Err(FellmError::other(format!(
            "kv_write: position {pos} >= max_seq {max_seq}"
        )));
    }
    let dst = as_f32_slice_mut(buf_out)?;
    dst[pos * dim..(pos + 1) * dim].copy_from_slice(row);
    Ok(())
}
