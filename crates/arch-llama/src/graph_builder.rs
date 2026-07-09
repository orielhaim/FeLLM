//! Build a per-step Llama forward graph.
//!
//! Signature:
//!   inputs:
//!     token_id: [1] u32
//!     position: [1] u32
//!     past_len: [1] u32
//!     k_in_{layer}: [max_seq, n_kv_heads * head_dim] f32
//!     v_in_{layer}: [max_seq, n_kv_heads * head_dim] f32
//!   outputs:
//!     logits: [vocab] f32
//!     k_out_{layer}: [n_kv_heads * head_dim] f32   (row to append at `position`)
//!     v_out_{layer}: [n_kv_heads * head_dim] f32

use crate::config::LlamaConfig;
use fellm_core::dtype::DType;
use fellm_core::error::Result;
use fellm_core::shape::Shape;
use fellm_gguf::GgufFile;
use fellm_graph::{Graph, GraphBuilder, NodeId};
use fellm_plugin_abi::op::{OpAttrs, OpKind};

/// Build the graph.
pub fn build(gguf: &GgufFile, cfg: &LlamaConfig, position: usize) -> Result<Graph> {
    let mut gb = GraphBuilder::new();

    let d_model = cfg.d_model;
    let head_dim = cfg.head_dim();
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let d_ff = cfg.d_ff;
    let vocab = cfg.vocab_size;
    let kv_stride = n_kv * head_dim;
    let q_stride = n_heads * head_dim;

    // Constants
    let tok_embd = gb.constant(
        "token_embd",
        gguf.tensor("token_embd.weight")?,
    );

    let output_w_name = if cfg.tied_embeddings {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    let output_w = gb.constant("output_w", gguf.tensor(output_w_name)?);
    let output_norm_w = gb.constant("output_norm_w", gguf.tensor("output_norm.weight")?);

    // Inputs
    let token_id = gb.input("token_id", DType::U32, Shape::new(&[1])?);
    let _position_in = gb.input("position", DType::U32, Shape::new(&[1])?);
    let _past_len_in = gb.input("past_len", DType::U32, Shape::new(&[1])?);

    // Embedding lookup.
    let mut x = gb.op(
        OpKind::Embedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[tok_embd, token_id],
        "tok_embed",
    );

    // Per-layer.
    for layer in 0..cfg.n_layers {
        x = build_layer(&mut gb, gguf, cfg, layer, position, x, kv_stride, q_stride)?;
    }

    // Final norm.
    let norm_attrs = OpAttrs {
        eps: cfg.norm_eps,
        ..Default::default()
    };
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs,
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x, output_norm_w],
        "final_norm",
    );

    // LM head.
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

#[allow(clippy::too_many_arguments)]
fn build_layer(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    cfg: &LlamaConfig,
    layer: usize,
    position: usize,
    x_in: NodeId,
    kv_stride: usize,
    q_stride: usize,
) -> Result<NodeId> {
    let d_model = cfg.d_model;
    let head_dim = cfg.head_dim();
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let d_ff = cfg.d_ff;

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
    let attn_norm_w = gb.constant(
        format!("blk.{layer}.attn_norm"),
        gguf.tensor(&format!("blk.{layer}.attn_norm.weight"))?,
    );
    let ffn_norm_w = gb.constant(
        format!("blk.{layer}.ffn_norm"),
        gguf.tensor(&format!("blk.{layer}.ffn_norm.weight"))?,
    );
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

    // Pre-attn norm.
    let norm_attrs = OpAttrs {
        eps: cfg.norm_eps,
        ..Default::default()
    };
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs,
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_in, attn_norm_w],
        format!("blk.{layer}.attn_norm_op"),
    );

    // Q/K/V projections.
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

    // Apply RoPE to Q and K.
    let rope_attrs_q = OpAttrs {
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        rope_dim: cfg.rope_dim as u32,
        position: position as u32,
        rope_base: cfg.rope_base,
        ..Default::default()
    };
    let rope_attrs_k = OpAttrs {
        n_heads: n_kv as u32,
        head_dim: head_dim as u32,
        rope_dim: cfg.rope_dim as u32,
        position: position as u32,
        rope_base: cfg.rope_base,
        ..Default::default()
    };
    let q_rot = gb.op(
        OpKind::Rope,
        rope_attrs_q,
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q],
        format!("blk.{layer}.q_rope"),
    );
    let k_rot = gb.op(
        OpKind::Rope,
        rope_attrs_k,
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[k],
        format!("blk.{layer}.k_rope"),
    );

    // Expose k_out_{layer} / v_out_{layer} as graph outputs.
    gb.mark_output(format!("k_out_{layer}"), k_rot);
    gb.mark_output(format!("v_out_{layer}"), v);

    // Attention. We take the KV-cache inputs (with the current row already
    // conceptually included by the runtime writing before executing the graph
    // at this position — but for correctness we take the KV cache view and
    // let the CPU kernel process past_len+1 tokens after the runtime has
    // written the row. Phase 1 wires this via a pre-step write.
    let k_in = gb.input(
        format!("k_in_{layer}"),
        DType::F32,
        Shape::new(&[cfg.context_length as u64, kv_stride as u64])?,
    );
    let v_in = gb.input(
        format!("v_in_{layer}"),
        DType::F32,
        Shape::new(&[cfg.context_length as u64, kv_stride as u64])?,
    );

    let attn_attrs = OpAttrs {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv as u32,
        head_dim: head_dim as u32,
        past_len: position as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        ..Default::default()
    };
    let attn_out = gb.op(
        OpKind::Attention,
        attn_attrs,
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_rot, k_in, v_in],
        format!("blk.{layer}.attn"),
    );

    // O projection.
    let o = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[wo, attn_out],
        format!("blk.{layer}.o_proj"),
    );

    // Residual.
    let x_after_attn = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_in, o],
        format!("blk.{layer}.residual1"),
    );

    // FFN norm.
    let ffn_x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs,
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_after_attn, ffn_norm_w],
        format!("blk.{layer}.ffn_norm_op"),
    );

    // Gate and Up projections.
    let gate = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_gate, ffn_x_norm],
        format!("blk.{layer}.ffn_gate_proj"),
    );
    let up = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_up, ffn_x_norm],
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

    // Down projection.
    let down = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[w_down, gated],
        format!("blk.{layer}.ffn_down_proj"),
    );

    // Residual 2.
    let x_out = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_after_attn, down],
        format!("blk.{layer}.residual2"),
    );

    Ok(x_out)
}
