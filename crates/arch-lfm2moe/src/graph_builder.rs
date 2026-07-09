use crate::config::Lfm2MoeConfig;
use fellm_core::dtype::DType;
use fellm_core::error::Result;
use fellm_core::shape::Shape;
use fellm_gguf::GgufFile;
use fellm_graph::{Graph, GraphBuilder, NodeId};
use fellm_plugin_abi::op::{OpAttrs, OpKind};

/// Build the one-token LFM2 MoE graph.
pub fn build(gguf: &GgufFile, cfg: &Lfm2MoeConfig) -> Result<Graph> {
    let mut gb = GraphBuilder::new();
    let d_model = cfg.d_model;
    let vocab = cfg.vocab_size;

    let tok_embd = gb.constant("token_embd", gguf.tensor("token_embd.weight")?);
    let output_w_name = if cfg.tied_embeddings {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    let output_w = gb.constant("output_w", gguf.tensor(output_w_name)?);
    // LFM2 stores the final RMSNorm as `token_embd_norm` (llama.cpp
    // OUTPUT_NORM_LFM2). It is applied only after all layers — never on the
    // raw embedding (HF: `embedding_norm` at end of Lfm2Model.forward).
    let final_norm_w = if gguf.has_tensor("output_norm.weight") {
        gb.constant("output_norm_w", gguf.tensor("output_norm.weight")?)
    } else {
        gb.constant("output_norm_w", gguf.tensor("token_embd_norm.weight")?)
    };

    let inv_freqs = gb.constant(
        "rope_inv_freqs",
        make_f32_tensor(&cfg.compute_rope_inv_freqs()),
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

    for layer in 0..cfg.n_layers {
        x = build_layer(&mut gb, gguf, cfg, layer, x, inv_freqs)?;
    }

    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(cfg),
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

fn build_layer(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    cfg: &Lfm2MoeConfig,
    layer: usize,
    x_in: NodeId,
    inv_freqs: NodeId,
) -> Result<NodeId> {
    let d_model = cfg.d_model;
    let attn_norm_w = gb.constant(
        format!("blk.{layer}.attn_norm"),
        gguf.tensor(&format!("blk.{layer}.attn_norm.weight"))?,
    );
    let ffn_norm_w = gb.constant(
        format!("blk.{layer}.ffn_norm"),
        gguf.tensor(&format!("blk.{layer}.ffn_norm.weight"))?,
    );
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(cfg),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_in, attn_norm_w],
        format!("blk.{layer}.attn_norm_op"),
    );

    let mix = if cfg.is_recurrent(layer) {
        build_shortconv(gb, gguf, cfg, layer, x_norm)?
    } else {
        build_attention(gb, gguf, cfg, layer, x_norm, inv_freqs)?
    };

    let x_after_mix = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_in, mix],
        format!("blk.{layer}.residual1"),
    );

    let ffn_x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(cfg),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_after_mix, ffn_norm_w],
        format!("blk.{layer}.ffn_norm_op"),
    );

    let ffn_out = if cfg.is_moe(layer) {
        build_moe(gb, gguf, cfg, layer, ffn_x_norm)?
    } else {
        build_dense_ffn(gb, gguf, cfg, layer, ffn_x_norm)?
    };

    let x_out = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_after_mix, ffn_out],
        format!("blk.{layer}.residual2"),
    );
    Ok(x_out)
}

fn build_shortconv(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    cfg: &Lfm2MoeConfig,
    layer: usize,
    x_norm: NodeId,
) -> Result<NodeId> {
    let d_model = cfg.d_model;
    let conv_ord = cfg.conv_ordinal(layer).expect("recurrent layer");
    let in_proj = gb.constant(
        format!("blk.{layer}.shortconv.in_proj"),
        gguf.tensor(&format!("blk.{layer}.shortconv.in_proj.weight"))?,
    );
    let conv = gb.constant(
        format!("blk.{layer}.shortconv.conv"),
        gguf.tensor(&format!("blk.{layer}.shortconv.conv.weight"))?,
    );
    let out_proj = gb.constant(
        format!("blk.{layer}.shortconv.out_proj"),
        gguf.tensor(&format!("blk.{layer}.shortconv.out_proj.weight"))?,
    );
    let conv_in = gb.input(
        format!("conv_in_{conv_ord}"),
        DType::F32,
        Shape::new(&[(cfg.shortconv_l_cache.saturating_sub(1) * d_model) as u64])?,
    );
    let attrs = OpAttrs {
        shortconv_l_cache: cfg.shortconv_l_cache as u32,
        n_embd: d_model as u32,
        ..Default::default()
    };
    Ok(gb.op(
        OpKind::ShortConv,
        attrs,
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_norm, in_proj, conv, out_proj, conv_in],
        format!("blk.{layer}.shortconv"),
    ))
}

fn build_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    cfg: &Lfm2MoeConfig,
    layer: usize,
    x_norm: NodeId,
    inv_freqs: NodeId,
) -> Result<NodeId> {
    let d_model = cfg.d_model;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.layer_kv_heads[layer];
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let attn_ord = cfg.attn_ordinal(layer).expect("attention layer");

    let wq = gb.constant(
        format!("blk.{layer}.attn_q"),
        gguf.tensor(&format!("blk.{layer}.attn_q.weight"))?,
    );
    let wk = gb.constant(
        format!("blk.{layer}.attn_k"),
        gguf.tensor(&format!("blk.{layer}.attn_k.weight"))?,
    );
    let wv = gb.constant(
        format!("blk.{layer}.attn_v"),
        gguf.tensor(&format!("blk.{layer}.attn_v.weight"))?,
    );
    let wo = gb.constant(
        format!("blk.{layer}.attn_o"),
        gguf.tensor(&format!("blk.{layer}.attn_output.weight"))?,
    );
    let q_norm_w = gb.constant(
        format!("blk.{layer}.attn_q_norm"),
        gguf.tensor(&format!("blk.{layer}.attn_q_norm.weight"))?,
    );
    let k_norm_w = gb.constant(
        format!("blk.{layer}.attn_k_norm"),
        gguf.tensor(&format!("blk.{layer}.attn_k_norm.weight"))?,
    );

    let q = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[wq, x_norm],
        format!("blk.{layer}.q_proj"),
    );
    let k = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[wk, x_norm],
        format!("blk.{layer}.k_proj"),
    );
    let v = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[wv, x_norm],
        format!("blk.{layer}.v_proj"),
    );

    let q_norm = gb.op(
        OpKind::RmsNorm,
        qk_norm_attrs(cfg, n_heads),
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q, q_norm_w],
        format!("blk.{layer}.q_norm"),
    );
    let k_norm = gb.op(
        OpKind::RmsNorm,
        qk_norm_attrs(cfg, n_kv),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[k, k_norm_w],
        format!("blk.{layer}.k_norm"),
    );

    let q_rot = gb.op(
        OpKind::Rope,
        rope_attrs(cfg, n_heads),
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_norm, inv_freqs],
        format!("blk.{layer}.q_rope"),
    );
    let k_rot = gb.op(
        OpKind::Rope,
        rope_attrs(cfg, n_kv),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[k_norm, inv_freqs],
        format!("blk.{layer}.k_rope"),
    );

    let k_in = gb.input(
        format!("k_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[cfg.context_length as u64, kv_stride as u64])?,
    );
    let v_in = gb.input(
        format!("v_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[cfg.context_length as u64, kv_stride as u64])?,
    );
    let k_cache_updated = gb.op_in_place(
        OpKind::KvWrite,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[cfg.context_length as u64, kv_stride as u64])?,
        &[k_rot, k_in],
        1,
        format!("blk.{layer}.k_write"),
    );
    let v_cache_updated = gb.op_in_place(
        OpKind::KvWrite,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[cfg.context_length as u64, kv_stride as u64])?,
        &[v, v_in],
        1,
        format!("blk.{layer}.v_write"),
    );
    let attn_out = gb.op(
        OpKind::Attention,
        attention_attrs(cfg, n_kv),
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_rot, k_cache_updated, v_cache_updated],
        format!("blk.{layer}.attn"),
    );

    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[wo, attn_out],
        format!("blk.{layer}.o_proj"),
    ))
}

fn build_dense_ffn(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    cfg: &Lfm2MoeConfig,
    layer: usize,
    x: NodeId,
) -> Result<NodeId> {
    let d_model = cfg.d_model;
    let d_ff = cfg.dense_ffn_dim;
    let w_gate = gb.constant(
        format!("blk.{layer}.ffn_gate"),
        gguf.tensor(&format!("blk.{layer}.ffn_gate.weight"))?,
    );
    let w_up = gb.constant(
        format!("blk.{layer}.ffn_up"),
        gguf.tensor(&format!("blk.{layer}.ffn_up.weight"))?,
    );
    let w_down = gb.constant(
        format!("blk.{layer}.ffn_down"),
        gguf.tensor(&format!("blk.{layer}.ffn_down.weight"))?,
    );
    let gate = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_gate, x],
        format!("blk.{layer}.ffn_gate_proj"),
    );
    let up = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_up, x],
        format!("blk.{layer}.ffn_up_proj"),
    );
    let gated = gb.op(
        OpKind::SiluGate,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[gate, up],
        format!("blk.{layer}.swiglu"),
    );
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[w_down, gated],
        format!("blk.{layer}.ffn_down_proj"),
    ))
}

fn build_moe(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    cfg: &Lfm2MoeConfig,
    layer: usize,
    x: NodeId,
) -> Result<NodeId> {
    let gate_inp = gb.constant(
        format!("blk.{layer}.ffn_gate_inp"),
        gguf.tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))?,
    );
    let gate_exps = gb.constant(
        format!("blk.{layer}.ffn_gate_exps"),
        gguf.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
    );
    let up_exps = gb.constant(
        format!("blk.{layer}.ffn_up_exps"),
        gguf.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
    );
    let down_exps = gb.constant(
        format!("blk.{layer}.ffn_down_exps"),
        gguf.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    );
    let mut inputs = vec![x, gate_inp, gate_exps, up_exps, down_exps];
    if cfg.use_expert_bias && gguf.has_tensor(&format!("blk.{layer}.exp_probs_b.bias")) {
        inputs.push(gb.constant(
            format!("blk.{layer}.exp_probs_b"),
            gguf.tensor(&format!("blk.{layer}.exp_probs_b.bias"))?,
        ));
    }
    Ok(gb.op(
        OpKind::MoE,
        moe_attrs(cfg),
        DType::F32,
        Shape::new(&[cfg.d_model as u64])?,
        &inputs,
        format!("blk.{layer}.moe"),
    ))
}

fn norm_attrs(cfg: &Lfm2MoeConfig) -> OpAttrs {
    OpAttrs {
        eps: cfg.norm_eps,
        ..Default::default()
    }
}

fn qk_norm_attrs(cfg: &Lfm2MoeConfig, n_heads: usize) -> OpAttrs {
    OpAttrs {
        eps: cfg.norm_eps,
        n_heads: n_heads as u32,
        head_dim: cfg.head_dim as u32,
        ..Default::default()
    }
}

fn rope_attrs(cfg: &Lfm2MoeConfig, n_heads: usize) -> OpAttrs {
    OpAttrs {
        n_heads: n_heads as u32,
        head_dim: cfg.head_dim as u32,
        rope_dim: cfg.head_dim as u32,
        position: 0,
        rope_base: cfg.rope_base,
        ..Default::default()
    }
}

fn attention_attrs(cfg: &Lfm2MoeConfig, n_kv: usize) -> OpAttrs {
    OpAttrs {
        n_heads: cfg.n_heads as u32,
        n_kv_heads: n_kv as u32,
        head_dim: cfg.head_dim as u32,
        past_len: 0,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        ..Default::default()
    }
}

fn moe_attrs(cfg: &Lfm2MoeConfig) -> OpAttrs {
    OpAttrs {
        n_experts: cfg.n_experts as u32,
        n_expert_used: cfg.n_expert_used as u32,
        expert_gating_func: cfg.expert_gating_func,
        routed_scaling_factor: cfg.routed_scaling_factor,
        norm_topk_prob: u32::from(cfg.norm_topk_prob),
        n_embd: cfg.d_model as u32,
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
