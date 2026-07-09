//! Generic one-token step graph from a [`ModelSpec`].

use crate::probe::{FfnKind, MixKind, ModelSpec};
use crate::rope::compute_rope_inv_freqs;
use fellm_core::dtype::DType;
use fellm_core::error::Result;
use fellm_core::shape::Shape;
use fellm_gguf::GgufFile;
use fellm_graph::{Graph, GraphBuilder, NodeId};
use fellm_plugin_abi::op::{OpAttrs, OpKind};

/// Build the one-token forward graph for any probed model.
pub fn build_step_graph(gguf: &GgufFile, spec: &ModelSpec) -> Result<Graph> {
    let mut gb = GraphBuilder::new();
    let d_model = spec.d_model;
    let vocab = spec.vocab_size;

    let tok_embd = gb.constant("token_embd", gguf.tensor("token_embd.weight")?);
    let output_w_name = if spec.tied_embeddings {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    let output_w = gb.constant("output_w", gguf.tensor(output_w_name)?);
    let final_norm_w = if gguf.has_tensor("output_norm.weight") {
        gb.constant("output_norm_w", gguf.tensor("output_norm.weight")?)
    } else {
        gb.constant("output_norm_w", gguf.tensor("token_embd_norm.weight")?)
    };

    let inv_freqs = gb.constant(
        "rope_inv_freqs",
        make_f32_tensor(&compute_rope_inv_freqs(spec)),
    );
    let token_id = gb.input("token_id", DType::U32, Shape::new(&[1])?);

    let mut x = gb.op(
        OpKind::Embedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[tok_embd, token_id],
        "tok_embed",
    );

    for layer in &spec.layers {
        x = build_layer(&mut gb, gguf, spec, layer, x, inv_freqs)?;
    }

    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x, final_norm_w],
        "final_norm",
    );
    let logits = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[vocab as u64])?,
        &[output_w, x_norm],
        "lm_head",
    );
    gb.mark_output("logits", logits);
    gb.build()
}

/// Node ids whose attrs must be patched each decode step.
#[derive(Debug, Default, Clone)]
pub struct StepBindings {
    /// RoPE ops.
    pub rope: Vec<NodeId>,
    /// KV write ops.
    pub kv_write: Vec<NodeId>,
    /// Attention ops.
    pub attention: Vec<NodeId>,
}

/// Collect position / past_len binding targets from a built graph.
#[must_use]
pub fn collect_step_bindings(graph: &Graph) -> StepBindings {
    let mut nodes = StepBindings::default();
    for (id, node) in graph.iter_nodes() {
        match node.op {
            Some(OpKind::Rope) => nodes.rope.push(id),
            Some(OpKind::KvWrite) => nodes.kv_write.push(id),
            Some(OpKind::Attention) => nodes.attention.push(id),
            _ => {}
        }
    }
    nodes
}

fn build_layer(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_in: NodeId,
    inv_freqs: NodeId,
) -> Result<NodeId> {
    let i = layer.index;
    let d_model = spec.d_model;
    let attn_norm_w = gb.constant(
        format!("blk.{i}.attn_norm"),
        gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?,
    );
    let ffn_norm_w = gb.constant(
        format!("blk.{i}.ffn_norm"),
        gguf.tensor(&format!("blk.{i}.ffn_norm.weight"))?,
    );
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_in, attn_norm_w],
        format!("blk.{i}.attn_norm_op"),
    );

    let mix = match layer.mix {
        MixKind::ShortConv => build_shortconv(gb, gguf, spec, layer, x_norm)?,
        MixKind::Attention { .. } => build_attention(gb, gguf, spec, layer, x_norm, inv_freqs)?,
    };

    let x_after_mix = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_in, mix],
        format!("blk.{i}.residual1"),
    );

    let ffn_x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_after_mix, ffn_norm_w],
        format!("blk.{i}.ffn_norm_op"),
    );

    let ffn_out = match layer.ffn {
        FfnKind::MoE => build_moe(gb, gguf, spec, layer, ffn_x_norm)?,
        FfnKind::Dense => build_dense_ffn(gb, gguf, spec, layer, ffn_x_norm)?,
    };

    Ok(gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_after_mix, ffn_out],
        format!("blk.{i}.residual2"),
    ))
}

fn build_shortconv(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_norm: NodeId,
) -> Result<NodeId> {
    let i = layer.index;
    let d_model = spec.d_model;
    let conv_ord = layer.conv_ordinal.expect("ShortConv ordinal");
    let in_proj = gb.constant(
        format!("blk.{i}.shortconv.in_proj"),
        gguf.tensor(&format!("blk.{i}.shortconv.in_proj.weight"))?,
    );
    let conv = gb.constant(
        format!("blk.{i}.shortconv.conv"),
        gguf.tensor(&format!("blk.{i}.shortconv.conv.weight"))?,
    );
    let out_proj = gb.constant(
        format!("blk.{i}.shortconv.out_proj"),
        gguf.tensor(&format!("blk.{i}.shortconv.out_proj.weight"))?,
    );
    let conv_in = gb.input(
        format!("conv_in_{conv_ord}"),
        DType::F32,
        Shape::new(&[(spec.shortconv_l_cache.saturating_sub(1) * d_model) as u64])?,
    );
    let attrs = OpAttrs {
        shortconv_l_cache: spec.shortconv_l_cache as u32,
        n_embd: d_model as u32,
        ..Default::default()
    };
    Ok(gb.op(
        OpKind::ShortConv,
        attrs,
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_norm, in_proj, conv, out_proj, conv_in],
        format!("blk.{i}.shortconv"),
    ))
}

fn build_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_norm: NodeId,
    inv_freqs: NodeId,
) -> Result<NodeId> {
    let MixKind::Attention {
        n_kv_heads: n_kv,
        qk_norm,
    } = layer.mix
    else {
        unreachable!("attention mix");
    };
    let i = layer.index;
    let d_model = spec.d_model;
    let head_dim = spec.head_dim;
    let n_heads = spec.n_heads;
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let attn_ord = layer.attn_ordinal.expect("attention ordinal");

    let wq = gb.constant(
        format!("blk.{i}.attn_q"),
        gguf.tensor(&format!("blk.{i}.attn_q.weight"))?,
    );
    let wk = gb.constant(
        format!("blk.{i}.attn_k"),
        gguf.tensor(&format!("blk.{i}.attn_k.weight"))?,
    );
    let wv = gb.constant(
        format!("blk.{i}.attn_v"),
        gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
    );
    let wo = gb.constant(
        format!("blk.{i}.attn_o"),
        gguf.tensor(&format!("blk.{i}.attn_output.weight"))?,
    );

    let q = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[wq, x_norm],
        format!("blk.{i}.q_proj"),
    );
    let k = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[wk, x_norm],
        format!("blk.{i}.k_proj"),
    );
    let v = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[wv, x_norm],
        format!("blk.{i}.v_proj"),
    );

    let (q_for_rope, k_for_rope) = if qk_norm {
        let q_norm_w = gb.constant(
            format!("blk.{i}.attn_q_norm"),
            gguf.tensor(&format!("blk.{i}.attn_q_norm.weight"))?,
        );
        let k_norm_w = gb.constant(
            format!("blk.{i}.attn_k_norm"),
            gguf.tensor(&format!("blk.{i}.attn_k_norm.weight"))?,
        );
        let qn = gb.op(
            OpKind::RmsNorm,
            qk_norm_attrs(spec, n_heads),
            DType::F32,
            Shape::new(&[q_stride as u64])?,
            &[q, q_norm_w],
            format!("blk.{i}.q_norm"),
        );
        let kn = gb.op(
            OpKind::RmsNorm,
            qk_norm_attrs(spec, n_kv),
            DType::F32,
            Shape::new(&[kv_stride as u64])?,
            &[k, k_norm_w],
            format!("blk.{i}.k_norm"),
        );
        (qn, kn)
    } else {
        (q, k)
    };

    let q_rot = gb.op(
        OpKind::Rope,
        rope_attrs(spec, n_heads),
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_for_rope, inv_freqs],
        format!("blk.{i}.q_rope"),
    );
    let k_rot = gb.op(
        OpKind::Rope,
        rope_attrs(spec, n_kv),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[k_for_rope, inv_freqs],
        format!("blk.{i}.k_rope"),
    );

    let k_in = gb.input(
        format!("k_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, kv_stride as u64])?,
    );
    let v_in = gb.input(
        format!("v_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, kv_stride as u64])?,
    );
    let k_cache_updated = gb.op_in_place(
        OpKind::KvWrite,
        OpAttrs {
            layer_ord: attn_ord as u32,
            kv_slot: 0,
            block_size: 16,
            ..OpAttrs::default()
        },
        DType::F32,
        Shape::new(&[spec.context_length as u64, kv_stride as u64])?,
        &[k_rot, k_in],
        1,
        format!("blk.{i}.k_write"),
    );
    let v_cache_updated = gb.op_in_place(
        OpKind::KvWrite,
        OpAttrs {
            layer_ord: attn_ord as u32,
            kv_slot: 1,
            block_size: 16,
            ..OpAttrs::default()
        },
        DType::F32,
        Shape::new(&[spec.context_length as u64, kv_stride as u64])?,
        &[v, v_in],
        1,
        format!("blk.{i}.v_write"),
    );
    let attn_out = gb.op(
        OpKind::Attention,
        OpAttrs {
            layer_ord: attn_ord as u32,
            ..attention_attrs(spec, n_kv)
        },
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_rot, k_cache_updated, v_cache_updated],
        format!("blk.{i}.attn"),
    );

    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[wo, attn_out],
        format!("blk.{i}.o_proj"),
    ))
}

fn build_dense_ffn(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
) -> Result<NodeId> {
    let i = layer.index;
    let d_model = spec.d_model;
    let d_ff = spec.dense_ffn_dim;
    let w_gate = gb.constant(
        format!("blk.{i}.ffn_gate"),
        gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
    );
    let w_up = gb.constant(
        format!("blk.{i}.ffn_up"),
        gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
    );
    let w_down = gb.constant(
        format!("blk.{i}.ffn_down"),
        gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?,
    );
    let gate = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_gate, x],
        format!("blk.{i}.ffn_gate_proj"),
    );
    let up = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_up, x],
        format!("blk.{i}.ffn_up_proj"),
    );
    let gated = gb.op(
        OpKind::SiluGate,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[gate, up],
        format!("blk.{i}.swiglu"),
    );
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[w_down, gated],
        format!("blk.{i}.ffn_down_proj"),
    ))
}

fn build_moe(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
) -> Result<NodeId> {
    let i = layer.index;
    let gate_inp = gb.constant(
        format!("blk.{i}.ffn_gate_inp"),
        gguf.tensor(&format!("blk.{i}.ffn_gate_inp.weight"))?,
    );
    let gate_exps = gb.constant(
        format!("blk.{i}.ffn_gate_exps"),
        gguf.tensor(&format!("blk.{i}.ffn_gate_exps.weight"))?,
    );
    let up_exps = gb.constant(
        format!("blk.{i}.ffn_up_exps"),
        gguf.tensor(&format!("blk.{i}.ffn_up_exps.weight"))?,
    );
    let down_exps = gb.constant(
        format!("blk.{i}.ffn_down_exps"),
        gguf.tensor(&format!("blk.{i}.ffn_down_exps.weight"))?,
    );
    let mut inputs = vec![x, gate_inp, gate_exps, up_exps, down_exps];
    if spec.use_expert_bias && gguf.has_tensor(&format!("blk.{i}.exp_probs_b.bias")) {
        inputs.push(gb.constant(
            format!("blk.{i}.exp_probs_b"),
            gguf.tensor(&format!("blk.{i}.exp_probs_b.bias"))?,
        ));
    }
    Ok(gb.op(
        OpKind::MoE,
        moe_attrs(spec),
        DType::F32,
        Shape::new(&[spec.d_model as u64])?,
        &inputs,
        format!("blk.{i}.moe"),
    ))
}

fn norm_attrs(spec: &ModelSpec) -> OpAttrs {
    OpAttrs {
        eps: spec.norm_eps,
        ..Default::default()
    }
}

fn qk_norm_attrs(spec: &ModelSpec, n_heads: usize) -> OpAttrs {
    OpAttrs {
        eps: spec.norm_eps,
        n_heads: n_heads as u32,
        head_dim: spec.head_dim as u32,
        ..Default::default()
    }
}

fn rope_attrs(spec: &ModelSpec, n_heads: usize) -> OpAttrs {
    OpAttrs {
        n_heads: n_heads as u32,
        head_dim: spec.head_dim as u32,
        rope_dim: spec.rope_dim as u32,
        position: 0,
        rope_base: spec.rope_base,
        ..Default::default()
    }
}

fn attention_attrs(spec: &ModelSpec, n_kv: usize) -> OpAttrs {
    OpAttrs {
        n_heads: spec.n_heads as u32,
        n_kv_heads: n_kv as u32,
        head_dim: spec.head_dim as u32,
        past_len: 0,
        scale: 1.0 / (spec.head_dim as f32).sqrt(),
        block_size: 16,
        ..Default::default()
    }
}

fn moe_attrs(spec: &ModelSpec) -> OpAttrs {
    OpAttrs {
        n_experts: spec.n_experts as u32,
        n_expert_used: spec.n_expert_used as u32,
        expert_gating_func: spec.expert_gating_func,
        routed_scaling_factor: spec.routed_scaling_factor,
        norm_topk_prob: u32::from(spec.norm_topk_prob),
        n_embd: spec.d_model as u32,
        ..Default::default()
    }
}

fn make_f32_tensor(data: &[f32]) -> fellm_core::tensor::Tensor {
    use fellm_core::shape::{Layout, Shape as FShape};
    use fellm_core::storage::{AlignedBuffer, Storage};
    use std::sync::Arc;

    let bytes_len = data.len() * 4;
    let mut buf = AlignedBuffer::new_zeroed(bytes_len, 64);
    let dst: &mut [f32] = bytemuck::cast_slice_mut(buf.as_mut_slice());
    dst.copy_from_slice(data);
    let shape = FShape::new(&[data.len() as u64]).expect("valid shape");
    let layout = Layout::contiguous(DType::F32, shape);
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    fellm_core::tensor::Tensor::from_storage(layout, storage)
}
