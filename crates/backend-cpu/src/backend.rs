use crate::cpu_profile::CpuHardwareProfile;
use crate::kernels::{
    attention::{attention_step, attention_step_paged},
    embedding::{embedding_row, weighted_embedding},
    matmul,
    norm::{rmsnorm_groups, rmsnorm_row},
    sampling::sample,
    simd_f32::PulpDispatch,
    softmax::softmax_rows_inplace,
    swiglu::silu_gate,
};
use crossbeam_utils::CachePadded;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi as paged_ctx;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{Backend, BackendCaps, DeviceKind, KernelDescriptor, KernelHandle};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use rayon::ThreadPool;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use thread_local::ThreadLocal;

/// The CPU backend.
pub struct CpuBackend {
    caps: BackendCaps,
    profile: CpuHardwareProfile,
    simd: PulpDispatch,
    /// Rayon pool sized to physical cores (avoids HT L2 thrash in attention).
    pool: ThreadPool,
    /// Cheap backend-wide launch counter, padded to avoid false sharing with
    /// adjacent backend state.
    launches: CachePadded<AtomicU64>,
    /// Per-worker launch counters for diagnosing scheduler imbalance.
    worker_launches: ThreadLocal<AtomicU64>,
}

impl CpuBackend {
    #[must_use]
    pub fn new() -> Self {
        let profile = *CpuHardwareProfile::get();
        let physical = profile.physical_cores.max(1);
        let logical = profile.logical_threads.max(physical);
        let requested_threads =
            std::env::var("FELLM_CPU_THREADS").ok().and_then(|value| {
                match value.trim().to_ascii_lowercase().as_str() {
                    "physical" | "p" => Some(physical),
                    "logical" | "all" => Some(logical),
                    value => value.parse::<usize>().ok().filter(|&n| n > 0),
                }
            });
        let threads = requested_threads.unwrap_or(physical).clamp(1, logical);
        tracing::info!(
            target = "fellm::cpu",
            physical,
            logical,
            threads,
            requested = requested_threads,
            "CPU execution pool configured"
        );
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("fellm-matmul-{i}"))
            .build_global();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("fellm-attn-{i}"))
            .build()
            .unwrap_or_else(|_| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("rayon single-thread pool")
            });
        Self {
            caps: BackendCaps {
                device_kind: DeviceKind::Cpu,
                simd_f32_lanes: profile.simd_f32_lanes,
                has_avx512: profile.has_avx512,
                has_avx2: profile.has_avx2,
                has_neon: profile.has_neon,
                physical_cores: profile.physical_cores as u32,
                logical_threads: profile.logical_threads as u32,
                supports_persistent_device_state: false,
                supports_graph_capture: false,
                supports_async_execution: false,
                supports_read_only_prefix_kv: true,
                supports_grouped_moe: true,
                supports_device_sampling: false,
                supports_bidirectional_attention: true,
                supports_batched_quantized_gemm: true,
                supports_custom_operations: true,
            },
            profile,
            simd: PulpDispatch::new(),
            pool,
            launches: CachePadded::new(AtomicU64::new(0)),
            worker_launches: ThreadLocal::new(),
        }
    }

    fn make_handle(op: OpKind) -> KernelHandle {
        KernelHandle(u64::from(op.raw()))
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn is_supported_matvec_weight_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q6K
    )
}

impl Backend for CpuBackend {
    fn id(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
                            | DType::Q5_0
                            | DType::Q8_0
                            | DType::Q4K
                            | DType::Q6K
                    ),
                    Some(DType::F32)
                )
            ),
            OpKind::GateUpSwiGlu => matches!(
                (input_dtypes.first(), input_dtypes.get(1), input_dtypes.get(2)),
                (Some(gate), Some(up), Some(DType::F32))
                    if is_supported_matvec_weight_dtype(*gate)
                        && is_supported_matvec_weight_dtype(*up)
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
            OpKind::WeightedEmbedding => matches!(
                (input_dtypes.first(), input_dtypes.get(1), output_dtype),
                (Some(weight), Some(DType::F32), DType::F32)
                    if is_supported_matvec_weight_dtype(*weight)
            ),
            OpKind::ShortConv => matches!(
                (
                    input_dtypes.first(),
                    input_dtypes.get(1),
                    input_dtypes.get(2),
                    input_dtypes.get(3)
                ),
                (Some(DType::F32), Some(w0), Some(DType::F32), Some(w1))
                    if is_supported_matvec_weight_dtype(*w0)
                        && is_supported_matvec_weight_dtype(*w1)
            ),
            OpKind::MoE => {
                if input_dtypes.len() >= 7 {
                    let activations_ok = input_dtypes.first() == Some(&DType::F32)
                        && input_dtypes.get(1) == Some(&DType::F32);
                    let weights_ok = input_dtypes[2..7]
                        .iter()
                        .all(|dtype| is_supported_matvec_weight_dtype(*dtype));
                    let bias_ok = input_dtypes.get(7).is_none_or(|dtype| *dtype == DType::F32);
                    return if activations_ok && weights_ok && bias_ok {
                        Some(KernelDescriptor {
                            op,
                            input_dtypes: input_dtypes.to_vec(),
                            output_dtype,
                            handle: Self::make_handle(op),
                        })
                    } else {
                        None
                    };
                }
                let base_ok = matches!(
                    (
                        input_dtypes.first(),
                        input_dtypes.get(1),
                        input_dtypes.get(2),
                        input_dtypes.get(3),
                        input_dtypes.get(4)
                    ),
                    (Some(DType::F32), Some(DType::F32), Some(w0), Some(w1), Some(w2))
                        if is_supported_matvec_weight_dtype(*w0)
                            && is_supported_matvec_weight_dtype(*w1)
                            && is_supported_matvec_weight_dtype(*w2)
                );
                let bias_ok = input_dtypes
                    .get(5)
                    .map(|dtype| *dtype == DType::F32)
                    .unwrap_or(true);
                base_ok && bias_ok
            }
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
            _ => false,
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
        self.launches.fetch_add(1, Ordering::Relaxed);
        self.worker_launches
            .get_or(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        let op = decode_handle(handle)?;
        match op {
            OpKind::MatMul => launch_matmul(attrs, inputs, outputs),
            OpKind::GateUpSwiGlu => launch_gate_up_swiglu(inputs, outputs),
            OpKind::Embedding => launch_embedding(inputs, outputs),
            OpKind::RmsNorm => launch_rmsnorm(attrs, inputs, outputs),
            OpKind::Rope => launch_rope(attrs, inputs, outputs),
            OpKind::SiluGate => launch_silu_gate(inputs, outputs),
            OpKind::Softmax => launch_softmax(attrs, inputs, outputs),
            OpKind::Attention => launch_attention(self, attrs, inputs, outputs),
            OpKind::Add => launch_add(inputs, outputs, self.simd),
            OpKind::Mul => launch_mul(inputs, outputs, self.simd),
            OpKind::Reshape => launch_reshape(inputs, outputs),
            OpKind::Cast => launch_cast(attrs, inputs, outputs),
            OpKind::Concat => launch_concat(inputs, outputs),
            OpKind::Sample => launch_sample(attrs, inputs, outputs),
            OpKind::KvWrite => launch_kv_write(attrs, inputs, outputs),
            OpKind::ShortConv => launch_shortconv(attrs, inputs, outputs),
            OpKind::MoE => launch_moe(attrs, inputs, outputs),
            OpKind::WeightedEmbedding => launch_weighted_embedding(inputs, outputs),
            _ => Err(FellmError::other(
                "custom operation is not implemented by CPU backend",
            )),
        }
    }

    fn begin_step(&self) {
        matmul::begin_q8k_step_cache();
    }

    fn end_step(&self) {
        matmul::end_q8k_step_cache();
    }
}

fn decode_handle(h: KernelHandle) -> Result<OpKind> {
    match h.0 {
        x if x == u64::from(OpKind::Add.raw()) => Ok(OpKind::Add),
        x if x == u64::from(OpKind::Mul.raw()) => Ok(OpKind::Mul),
        x if x == u64::from(OpKind::MatMul.raw()) => Ok(OpKind::MatMul),
        x if x == u64::from(OpKind::RmsNorm.raw()) => Ok(OpKind::RmsNorm),
        x if x == u64::from(OpKind::Rope.raw()) => Ok(OpKind::Rope),
        x if x == u64::from(OpKind::SiluGate.raw()) => Ok(OpKind::SiluGate),
        x if x == u64::from(OpKind::Softmax.raw()) => Ok(OpKind::Softmax),
        x if x == u64::from(OpKind::Attention.raw()) => Ok(OpKind::Attention),
        x if x == u64::from(OpKind::Embedding.raw()) => Ok(OpKind::Embedding),
        x if x == u64::from(OpKind::Concat.raw()) => Ok(OpKind::Concat),
        x if x == u64::from(OpKind::Reshape.raw()) => Ok(OpKind::Reshape),
        x if x == u64::from(OpKind::Cast.raw()) => Ok(OpKind::Cast),
        x if x == u64::from(OpKind::Sample.raw()) => Ok(OpKind::Sample),
        x if x == u64::from(OpKind::KvWrite.raw()) => Ok(OpKind::KvWrite),
        x if x == u64::from(OpKind::ShortConv.raw()) => Ok(OpKind::ShortConv),
        x if x == u64::from(OpKind::MoE.raw()) => Ok(OpKind::MoE),
        x if x == u64::from(OpKind::WeightedEmbedding.raw()) => Ok(OpKind::WeightedEmbedding),
        x if x == u64::from(OpKind::GateUpSwiGlu.raw()) => Ok(OpKind::GateUpSwiGlu),
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

fn launch_matmul(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
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
    // A rank-1 activation is the scalar decode form; higher-rank activations
    // flatten every dimension before the final feature dimension into rows.
    let rows = if x.dims_slice().len() <= 1 {
        1
    } else {
        x.dims_slice()[..x.dims_slice().len() - 1]
            .iter()
            .product::<u64>() as usize
    };
    if rows == 0 || x_slice.len() != rows * in_dim || y_slice.len() != rows * out_dim {
        return Err(FellmError::other(format!(
            "matmul: batched shape mismatch x_dims={:?} x_len={} w_dims={:?} y_len={} rows={} in_dim={} out_dim={}",
            x.dims_slice(),
            x_slice.len(),
            w.dims_slice(),
            y_slice.len(),
            rows,
            in_dim,
            out_dim
        )));
    }
    match w_dtype {
        DType::F32 => {
            let ws = as_f32_slice(w)?;
            matmul::matmul_f32_batch(ws, x_slice, y_slice, rows, out_dim, in_dim)?;
        }
        DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
            let wb = as_bytes_slice(w);
            if rows > 1 {
                matmul::matmul_quant_batch(wb, w_dtype, x_slice, y_slice, rows, out_dim, in_dim)?;
            } else {
                matmul::matvec_quant(wb, w_dtype, x_slice, y_slice, out_dim, in_dim)?;
            }
        }
        other => return Err(FellmError::UnsupportedDType(other)),
    }
    if let Some(residual) = inputs.get(2) {
        let residual = as_f32_slice(residual)?;
        if residual.len() != y_slice.len() {
            return Err(FellmError::other(
                "matmul residual epilogue: shape mismatch",
            ));
        }
        for (value, skip) in y_slice.iter_mut().zip(residual) {
            *value += *skip;
        }
    }
    if attrs.softcap > 0.0 {
        let cap = attrs.softcap;
        for value in y_slice {
            *value = cap * (*value / cap).tanh();
        }
    }
    Ok(())
}

fn launch_gate_up_swiglu(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() != 3 || outputs.is_empty() {
        return Err(FellmError::other("gate_up_swiglu: bad arity"));
    }
    let x = as_f32_slice(&inputs[2])?;
    let out = as_f32_slice_mut(&mut outputs[0])?;
    let rows = inputs[0].dims_slice()[0] as usize;
    let cols = inputs[0].dims_slice()[1] as usize;
    if inputs[1].dims_slice() != inputs[0].dims_slice() || x.len() != cols || out.len() != rows {
        return Err(FellmError::other("gate_up_swiglu: shape mismatch"));
    }
    let mut gate = vec![0.0f32; rows];
    let project = |weight: &TensorRef, dst: &mut [f32]| -> Result<()> {
        let dtype = weight
            .dtype()
            .ok_or_else(|| FellmError::other("gate_up_swiglu: weight dtype"))?;
        match dtype {
            DType::F32 => matmul::matmul_f32_batch(as_f32_slice(weight)?, x, dst, 1, rows, cols),
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q6K => {
                matmul::matvec_quant(as_bytes_slice(weight), dtype, x, dst, rows, cols)
            }
            other => Err(FellmError::UnsupportedDType(other)),
        }
    };
    project(&inputs[0], &mut gate)?;
    project(&inputs[1], out)?;
    for (value, gate) in out.iter_mut().zip(gate) {
        *value *= gate / (1.0 + (-gate).exp());
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
    let y_slice = as_f32_slice_mut(y_out)?;
    if ids.is_empty() || y_slice.len() != ids.len() * dim {
        return Err(FellmError::other("embedding: batched shape mismatch"));
    }
    let wb = as_bytes_slice(w);
    for (tok_id, row) in ids.iter().copied().zip(y_slice.chunks_exact_mut(dim)) {
        embedding_row(wb, w_dtype, vocab, dim, tok_id, row)?;
    }
    Ok(())
}

fn launch_weighted_embedding(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("weighted embedding: bad arity"));
    }
    let weight = &inputs[0];
    let logits = as_f32_slice(&inputs[1])?;
    let output = as_f32_slice_mut(outputs.first_mut().unwrap())?;
    let weight_dtype = weight
        .dtype()
        .ok_or_else(|| FellmError::other("weighted embedding: weight dtype"))?;
    let dims = weight.dims_slice();
    if dims.len() != 2 {
        return Err(FellmError::other(
            "weighted embedding: weight must be rank 2",
        ));
    }
    let vocab = dims[0] as usize;
    let dim = dims[1] as usize;
    if vocab == 0 || dim == 0 {
        return Err(FellmError::other(format!(
            "weighted embedding: invalid weight shape dims={dims:?}"
        )));
    }
    let logit_dims = inputs[1].dims_slice();
    if logit_dims.len() == 2 && logit_dims[1] as usize != vocab {
        let slots = logit_dims[1] as usize;
        if !slots.is_multiple_of(2) {
            return Err(FellmError::other(
                "weighted embedding: packed top-k input must contain pairs",
            ));
        }
        return crate::kernels::embedding::weighted_embedding_topk(
            as_bytes_slice(weight),
            weight_dtype,
            logits,
            output,
            logit_dims[0] as usize,
            slots / 2,
            vocab,
            dim,
        );
    }
    if !logits.len().is_multiple_of(vocab) {
        return Err(FellmError::other("weighted embedding: invalid dense shape"));
    }
    weighted_embedding(
        as_bytes_slice(weight),
        weight_dtype,
        logits,
        output,
        logits.len() / vocab,
        vocab,
        dim,
    )
}

fn launch_rmsnorm(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("rmsnorm: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let w = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    let head_dim = attrs.head_dim as usize;
    let n_heads = attrs.n_heads as usize;
    if head_dim > 0
        && n_heads > 0
        && w.len() == head_dim
        && x.len().is_multiple_of(n_heads * head_dim)
    {
        for (x_row, y_row) in x
            .chunks_exact(n_heads * head_dim)
            .zip(y.chunks_exact_mut(n_heads * head_dim))
        {
            rmsnorm_groups(x_row, w, attrs.eps, head_dim, y_row);
        }
    } else if !w.is_empty() && x.len().is_multiple_of(w.len()) {
        for (x_row, y_row) in x.chunks_exact(w.len()).zip(y.chunks_exact_mut(w.len())) {
            rmsnorm_row(x_row, w, attrs.eps, y_row);
        }
    } else {
        rmsnorm_row(x, w, attrs.eps, y);
    }
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
    let row_width = attrs.n_heads.max(1) as usize * attrs.head_dim.max(1) as usize;
    if row_width == 0 || x_in.len().is_multiple_of(row_width) {
        for (row, values) in x_out.chunks_exact_mut(row_width).enumerate() {
            crate::kernels::rope::rope_inplace_with_freqs(
                values,
                attrs.n_heads as usize,
                attrs.head_dim as usize,
                attrs.rope_dim as usize,
                attrs.position.saturating_add(row as u32),
                inv_freqs,
            );
        }
    }
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
    backend: &CpuBackend,
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if inputs.len() < 3 || outputs.is_empty() {
        return Err(FellmError::other("attention: bad arity"));
    }
    if attrs.attention_mode == 1 && inputs.len() >= 5 {
        return launch_canvas_attention(backend, attrs, inputs, outputs);
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let q = as_f32_slice(&inputs[0])?;
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

    let use_paged = attrs.block_size > 0 && paged_ctx::has_paged_context();

    if use_paged {
        let layer = attrs.layer_ord as usize;
        // Gather from paged arena on this thread, then run tiled attention
        // inside the Rayon pool (contiguous buffers are Send).
        attention_step_paged(
            q,
            out,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            layer,
            &backend.profile,
            backend.simd,
        );
        // Re-run the contiguous kernel under the pool for parallel heads.
        // attention_step_paged already calls attention_step which uses the pool.
        return Ok(());
    }

    let k_full = as_f32_slice(&inputs[1])?;
    let v_full = as_f32_slice(&inputs[2])?;
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
    backend.pool.install(|| {
        attention_step(
            q,
            k,
            v,
            out,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            &backend.profile,
            backend.simd,
        );
    });
    Ok(())
}

fn launch_canvas_attention(
    backend: &CpuBackend,
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let q = as_f32_slice(&inputs[0])?;
    let k_canvas = as_f32_slice(&inputs[3])?;
    let v_canvas = as_f32_slice(&inputs[4])?;
    let out = as_f32_slice_mut(
        outputs
            .first_mut()
            .ok_or_else(|| FellmError::other("canvas attention: no output"))?,
    )?;
    let n_heads = attrs.n_heads.max(1) as usize;
    let n_kv = attrs.n_kv_heads.max(1) as usize;
    let head_dim = attrs.head_dim.max(1) as usize;
    let row_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let rows = attrs.query_len.max(1) as usize;
    let prefix_len = attrs.past_len as usize;
    if q.len() != rows * row_width
        || out.len() != rows * row_width
        || k_canvas.len() < rows * kv_width
        || v_canvas.len() < rows * kv_width
    {
        return Err(FellmError::other("canvas attention: shape mismatch"));
    }
    let heads_per_kv = n_heads / n_kv;
    let window = attrs.attention_window as usize;
    let prefix_start = if window == 0 {
        0
    } else {
        prefix_len.saturating_sub(window)
    };
    let prefix_ctx = if attrs.block_size > 0 {
        paged_ctx::snapshot_paged_context()
    } else {
        None
    };
    let prefix_k = as_f32_slice(&inputs[1]).ok();
    let prefix_v = as_f32_slice(&inputs[2]).ok();
    backend.pool.install(|| {
        q.par_chunks_exact(row_width)
            .zip(out.par_chunks_exact_mut(row_width))
            .for_each(|(q_row, out_row)| {
                for h in 0..n_heads {
                    let kv_h = h / heads_per_kv;
                    let q_head = &q_row[h * head_dim..(h + 1) * head_dim];
                    let mut scores =
                        Vec::with_capacity(prefix_len.saturating_sub(prefix_start) + rows);
                    for t in prefix_start..prefix_len {
                        let mut score = 0.0f32;
                        if let Some(ref ctx) = prefix_ctx {
                            // SAFETY: the active paged context remains valid for the graph step.
                            let k_row = unsafe { ctx.k_row(attrs.layer_ord as usize, t) };
                            let k_head = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
                            for i in 0..head_dim {
                                score += q_head[i] * k_head[i].to_f32();
                            }
                        } else if let Some(prefix_k) = prefix_k {
                            let k_row = &prefix_k[t * kv_width + kv_h * head_dim
                                ..t * kv_width + (kv_h + 1) * head_dim];
                            for i in 0..head_dim {
                                score += q_head[i] * k_row[i];
                            }
                        }
                        scores.push(
                            score
                                * if attrs.scale > 0.0 {
                                    attrs.scale
                                } else {
                                    1.0 / (head_dim as f32).sqrt()
                                },
                        );
                    }
                    for canvas_row in 0..rows {
                        let k_row = &k_canvas[canvas_row * kv_width + kv_h * head_dim
                            ..canvas_row * kv_width + (kv_h + 1) * head_dim];
                        let mut score = 0.0f32;
                        for i in 0..head_dim {
                            score += q_head[i] * k_row[i];
                        }
                        scores.push(
                            score
                                * if attrs.scale > 0.0 {
                                    attrs.scale
                                } else {
                                    1.0 / (head_dim as f32).sqrt()
                                },
                        );
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denom = scores.iter().map(|s| (*s - max).exp()).sum::<f32>();
                    let out_head = &mut out_row[h * head_dim..(h + 1) * head_dim];
                    out_head.fill(0.0);
                    if denom > 0.0 {
                        for (index, score) in scores.iter().copied().enumerate() {
                            let weight = (score - max).exp() / denom;
                            if index < prefix_len.saturating_sub(prefix_start) {
                                let t = prefix_start + index;
                                if let Some(ref ctx) = prefix_ctx {
                                    // SAFETY: the active paged context remains valid for the graph step.
                                    let v_row = unsafe { ctx.v_row(attrs.layer_ord as usize, t) };
                                    let v_head = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
                                    for i in 0..head_dim {
                                        out_head[i] += weight * v_head[i].to_f32();
                                    }
                                } else if let Some(prefix_v) = prefix_v {
                                    let v_row = &prefix_v[t * kv_width + kv_h * head_dim
                                        ..t * kv_width + (kv_h + 1) * head_dim];
                                    for i in 0..head_dim {
                                        out_head[i] += weight * v_row[i];
                                    }
                                }
                            } else {
                                let canvas_row = index - prefix_len.saturating_sub(prefix_start);
                                let v_row = &v_canvas[canvas_row * kv_width + kv_h * head_dim
                                    ..canvas_row * kv_width + (kv_h + 1) * head_dim];
                                for i in 0..head_dim {
                                    out_head[i] += weight * v_row[i];
                                }
                            }
                        }
                    }
                }
            });
    });
    Ok(())
}

fn launch_add(inputs: &[TensorRef], outputs: &mut [TensorMut], simd: PulpDispatch) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("add: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let a = as_f32_slice(&inputs[0])?;
    let b = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    crate::kernels::simd_f32::add_f32(a, b, y, simd);
    Ok(())
}

fn launch_mul(inputs: &[TensorRef], outputs: &mut [TensorMut], simd: PulpDispatch) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("mul: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let a = as_f32_slice(&inputs[0])?;
    let b = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    crate::kernels::simd_f32::mul_f32(a, b, y, simd);
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
    // outputs[0] = kv_buf [max_seq, dim] f32 (aliased mutable storage) — unused when paged
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("kv_write: bad arity"));
    }
    let row = as_f32_slice(&inputs[0])?;
    let pos = attrs.position as usize;
    let use_paged = attrs.block_size > 0 && paged_ctx::has_paged_context();

    if use_paged {
        let layer = attrs.layer_ord as usize;
        let is_v = attrs.kv_slot != 0;
        return paged_ctx::with_paged_context_mut(|ctx| {
            let ctx = ctx.ok_or_else(|| FellmError::other("kv_write: missing paged ctx"))?;
            if row.len() > ctx.tokens_stride {
                return Err(FellmError::other(format!(
                    "kv_write: row len {} > tokens_stride {}",
                    row.len(),
                    ctx.tokens_stride
                )));
            }
            // SAFETY: runtime uniquely owns arena for this step.
            let dst = unsafe { ctx.row_mut(layer, pos, is_v) };
            for (d, &s) in dst.iter_mut().zip(row.iter()) {
                *d = half::f16::from_f32(s);
            }
            Ok(())
        });
    }

    let (buf_out, _) = outputs.split_first_mut().unwrap();
    let dims = buf_out.dims_slice();
    if dims.len() != 2 {
        return Err(FellmError::other("kv_write: kv_buf must be 2D"));
    }
    let max_seq = dims[0] as usize;
    let dim = dims[1] as usize;
    if row.len() != dim {
        return Err(FellmError::other(format!(
            "kv_write: row len {} != dim {dim}",
            row.len()
        )));
    }
    if pos >= max_seq {
        return Err(FellmError::other(format!(
            "kv_write: position {pos} >= max_seq {max_seq}"
        )));
    }
    let dst = as_f32_slice_mut(buf_out)?;
    dst[pos * dim..(pos + 1) * dim].copy_from_slice(row);
    Ok(())
}

fn launch_shortconv(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    // inputs: [x, in_proj, conv, out_proj]
    // outputs: [y, conv_state]
    if inputs.len() < 4 || outputs.len() < 2 {
        return Err(FellmError::other("shortconv: bad arity"));
    }
    let (y_out, rest) = outputs.split_first_mut().unwrap();
    let state_out = rest
        .first_mut()
        .ok_or_else(|| FellmError::other("shortconv: missing state output"))?;

    let x = as_f32_slice(&inputs[0])?;
    let in_proj = &inputs[1];
    let conv = as_f32_slice(&inputs[2])?;
    let out_proj = &inputs[3];
    let y = as_f32_slice_mut(y_out)?;
    let state = as_f32_slice_mut(state_out)?;

    let n_embd = if attrs.n_embd > 0 {
        attrs.n_embd as usize
    } else {
        x.len()
    };
    let conv_dims = inputs[2].dims_slice();
    let l_cache = if attrs.shortconv_l_cache > 0 {
        attrs.shortconv_l_cache as usize
    } else {
        *conv_dims
            .get(1)
            .ok_or_else(|| FellmError::other("shortconv: conv weight must be 2D"))? as usize
    };

    let in_proj_dtype = in_proj
        .dtype()
        .ok_or_else(|| FellmError::other("shortconv: in_proj dtype"))?;
    let out_proj_dtype = out_proj
        .dtype()
        .ok_or_else(|| FellmError::other("shortconv: out_proj dtype"))?;

    crate::kernels::shortconv::shortconv_decode(
        x,
        as_bytes_slice(in_proj),
        in_proj_dtype,
        conv,
        as_bytes_slice(out_proj),
        out_proj_dtype,
        state,
        y,
        n_embd,
        l_cache,
    )
}

fn launch_moe(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // inputs: [x, gate_inp, gate_exps, up_exps, down_exps, optional bias]
    // outputs: [y]
    if inputs.len() < 5 || outputs.is_empty() {
        return Err(FellmError::other("moe: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let gate_inp = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;

    if inputs.len() >= 7 {
        let packed_dims = inputs[2].dims_slice();
        let down_dims = inputs[3].dims_slice();
        let shared_dims = inputs[4].dims_slice();
        if packed_dims.len() != 3 || down_dims.len() != 3 || shared_dims.len() != 2 {
            return Err(FellmError::other(
                "moe gemma: invalid packed/shared dimensions",
            ));
        }
        let n_experts = attrs.n_experts.max(1) as usize;
        let n_expert_used = attrs.n_expert_used.max(1) as usize;
        let n_embd = attrs.n_embd.max(1) as usize;
        let n_ff = packed_dims[1] as usize / 2;
        let shared_ff = shared_dims[0] as usize;
        let bias = if inputs.len() > 7 {
            Some(as_f32_slice(&inputs[7])?)
        } else {
            None
        };
        let x_dims = inputs[0].dims_slice();
        let tokens = if x_dims.len() <= 1 {
            1
        } else {
            x_dims[..x_dims.len() - 1].iter().product::<u64>() as usize
        };
        if tokens > 1 {
            return crate::kernels::moe::moe_decode_gemma_batch(
                x,
                gate_inp,
                as_bytes_slice(&inputs[2]),
                inputs[2]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: packed dtype"))?,
                as_bytes_slice(&inputs[3]),
                inputs[3]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: down dtype"))?,
                as_bytes_slice(&inputs[4]),
                inputs[4]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: shared gate dtype"))?,
                as_bytes_slice(&inputs[5]),
                inputs[5]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: shared up dtype"))?,
                as_bytes_slice(&inputs[6]),
                inputs[6]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: shared down dtype"))?,
                bias,
                y,
                tokens,
                n_experts,
                n_expert_used,
                n_ff,
                shared_ff,
                n_embd,
                attrs.expert_gating_func,
                attrs.routed_scaling_factor,
                attrs.norm_topk_prob != 0,
            );
        }
        return crate::kernels::moe::moe_decode_gemma(
            x,
            gate_inp,
            as_bytes_slice(&inputs[2]),
            inputs[2]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: packed dtype"))?,
            as_bytes_slice(&inputs[3]),
            inputs[3]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: down dtype"))?,
            as_bytes_slice(&inputs[4]),
            inputs[4]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: shared gate dtype"))?,
            as_bytes_slice(&inputs[5]),
            inputs[5]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: shared up dtype"))?,
            as_bytes_slice(&inputs[6]),
            inputs[6]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: shared down dtype"))?,
            bias,
            y,
            n_experts,
            n_expert_used,
            n_ff,
            shared_ff,
            n_embd,
            attrs.expert_gating_func,
            attrs.routed_scaling_factor,
            attrs.norm_topk_prob != 0,
        );
    }

    let gate_dims = inputs[1].dims_slice();
    let expert_dims = inputs[2].dims_slice();
    let down_dims = inputs[4].dims_slice();
    if gate_dims.len() != 2 || expert_dims.len() != 3 || down_dims.len() != 3 {
        return Err(FellmError::other(
            "moe: expected 2D router and 3D expert weights",
        ));
    }

    let n_experts = if attrs.n_experts > 0 {
        attrs.n_experts as usize
    } else {
        gate_dims[0] as usize
    };
    let n_embd = if attrs.n_embd > 0 {
        attrs.n_embd as usize
    } else {
        gate_dims[1] as usize
    };
    let n_ff = expert_dims[1] as usize;
    let n_expert_used = if attrs.n_expert_used > 0 {
        attrs.n_expert_used as usize
    } else {
        1
    };

    if expert_dims[0] as usize != n_experts
        || expert_dims[2] as usize != n_embd
        || down_dims[0] as usize != n_experts
        || down_dims[1] as usize != n_embd
        || down_dims[2] as usize != n_ff
    {
        return Err(FellmError::other("moe: expert dimensions mismatch"));
    }

    let gate_exps_dtype = inputs[2]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: gate expert dtype"))?;
    let up_exps_dtype = inputs[3]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: up expert dtype"))?;
    let down_exps_dtype = inputs[4]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: down expert dtype"))?;
    let bias = if inputs.len() > 5 {
        Some(as_f32_slice(&inputs[5])?)
    } else {
        None
    };

    crate::kernels::moe::moe_decode(
        x,
        gate_inp,
        as_bytes_slice(&inputs[2]),
        gate_exps_dtype,
        as_bytes_slice(&inputs[3]),
        up_exps_dtype,
        as_bytes_slice(&inputs[4]),
        down_exps_dtype,
        bias,
        y,
        n_experts,
        n_expert_used,
        n_ff,
        n_embd,
        attrs.expert_gating_func,
        attrs.routed_scaling_factor,
        attrs.norm_topk_prob != 0,
    )
}
