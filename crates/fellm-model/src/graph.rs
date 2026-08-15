//! Generic one-token step graph from a [`ModelSpec`].

use crate::probe::{FfnKind, MixKind, ModelSpec};
use crate::rope::{compute_rope_inv_freqs, compute_rope_inv_freqs_with_base};
use fellm_core::dtype::DType;
use fellm_core::error::Result;
use fellm_core::shape::Shape;
use fellm_gguf::GgufFile;
use fellm_graph::{Graph, GraphBuilder, NodeId};
use fellm_plugin_abi::TargetFeature;
use fellm_plugin_abi::op::{OpAttrs, OpKind};

/// Build the one-token forward graph for any probed model.
pub fn build_step_graph(gguf: &GgufFile, spec: &ModelSpec) -> Result<Graph> {
    build_step_graph_with_features(gguf, spec, &[])
}

/// Build a one-token graph that retains only explicitly requested target features.
pub fn build_step_graph_with_features(
    gguf: &GgufFile,
    spec: &ModelSpec,
    requested_features: &[TargetFeature],
) -> Result<Graph> {
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
    let inv_freqs_compress = if spec.compress_rope_base > 0.0 {
        gb.constant(
            "rope_inv_freqs_compress",
            make_f32_tensor(&compute_rope_inv_freqs_with_base(
                spec,
                spec.compress_rope_base,
            )),
        )
    } else {
        inv_freqs
    };
    let token_id = gb.input("token_id", DType::U32, Shape::new(&[1])?);

    let mut x = gb.op(
        OpKind::Embedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[tok_embd, token_id],
        "tok_embed",
    );
    if spec.hc_mult > 1 {
        // llama.cpp repeats the token embedding across every hyper-connection stream.
        let streams = vec![x; spec.hc_mult];
        x = gb.op(
            OpKind::Concat,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[(d_model * spec.hc_mult) as u64])?,
            &streams,
            "hc_embed",
        );
    }
    if requested_features.contains(&TargetFeature::EmbeddingOutput) {
        gb.mark_output("feature.embedding", x);
    }

    for layer in &spec.layers {
        x = build_layer(
            &mut gb,
            gguf,
            spec,
            layer,
            x,
            inv_freqs,
            inv_freqs_compress,
            token_id,
            &format!("blk.{}", layer.index),
        )?;
        if requested_features.contains(&TargetFeature::LayerHiddenState(layer.index as u32)) {
            gb.mark_output(format!("feature.layer.{}", layer.index), x);
        }
    }
    if spec.hc_mult > 1 && gguf.has_tensor("output_hc_fn.weight") {
        let fn_w = gb.constant("output_hc_fn", gguf.tensor("output_hc_fn.weight")?);
        let scale = gb.constant("output_hc_scale", gguf.tensor("output_hc_scale.weight")?);
        let base = gb.constant("output_hc_base", gguf.tensor("output_hc_base.weight")?);
        x = gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 2),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x, fn_w, scale, base],
            "hc_head",
        );
    }
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x, final_norm_w],
        "final_norm",
    );
    if requested_features.contains(&TargetFeature::FinalHiddenState) {
        gb.mark_output("feature.final", x_norm);
    }
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

/// Build one native MTP module invocation. The graph consumes the target
/// trunk's retained final hidden state plus the previously sampled token. It
/// owns an independent one-layer KV stream and reuses the checkpoint's shared
/// token embedding and LM head without copying either tensor.
pub fn build_mtp_step_graph(gguf: &GgufFile, spec: &ModelSpec, stage: usize) -> Result<Graph> {
    if stage >= spec.n_mtp_layers {
        return Err(fellm_core::error::FellmError::other(format!(
            "MTP stage {stage} is out of range for {} modules",
            spec.n_mtp_layers
        )));
    }
    let i = spec.n_layers + stage;
    let mut gb = GraphBuilder::new();
    let d_model = spec.d_model;
    let tok_embd_tensor = gguf.tensor("token_embd.weight")?;
    let tok_embd = gb.constant("token_embd", tok_embd_tensor.clone());
    let token_id = gb.input("token_id", DType::U32, Shape::new(&[1])?);
    let target_hidden = gb.input("target_hidden", DType::F32, Shape::new(&[d_model as u64])?);
    let embedding = gb.op(
        OpKind::Embedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[tok_embd, token_id],
        format!("mtp.{stage}.embedding"),
    );
    let enorm = gb.constant(
        format!("mtp.{stage}.enorm"),
        gguf.tensor(&format!("blk.{i}.nextn.enorm.weight"))?,
    );
    let hnorm = gb.constant(
        format!("mtp.{stage}.hnorm"),
        gguf.tensor(&format!("blk.{i}.nextn.hnorm.weight"))?,
    );
    let embedding = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[embedding, enorm],
        format!("mtp.{stage}.normalized_embedding"),
    );
    let hidden = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[target_hidden, hnorm],
        format!("mtp.{stage}.normalized_target_hidden"),
    );
    let fused = gb.op(
        OpKind::Concat,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[(2 * d_model) as u64])?,
        &[embedding, hidden],
        format!("mtp.{stage}.embedding_hidden_concat"),
    );
    let eh_proj = gb.constant(
        format!("mtp.{stage}.eh_proj"),
        gguf.tensor(&format!("blk.{i}.nextn.eh_proj.weight"))?,
    );
    let fused = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[eh_proj, fused],
        format!("mtp.{stage}.eh_projection"),
    );
    let inv_freqs = gb.constant(
        "rope_inv_freqs",
        make_f32_tensor(&compute_rope_inv_freqs(spec)),
    );
    let layer = crate::probe::LayerSpec {
        index: i,
        attn_ordinal: Some(0),
        conv_ordinal: None,
        mix: MixKind::Attention {
            n_kv_heads: spec.n_kv_heads,
            qk_norm: gguf.has_tensor(&format!("blk.{i}.attn_q_norm.weight")),
            head_dim: spec.head_dim,
            rope_dim: spec.rope_dim,
            is_sliding: false,
            value_reuses_key: !gguf.has_tensor(&format!("blk.{i}.attn_v.weight")),
        },
        ffn: FfnKind::Dense,
        compress_ratio: 0,
    };
    let hidden = build_layer(
        &mut gb,
        gguf,
        spec,
        &layer,
        fused,
        inv_freqs,
        inv_freqs,
        token_id,
        &format!("blk.{i}"),
    )?;
    let head_norm_name = format!("blk.{i}.nextn.shared_head_norm.weight");
    let head_norm = gb.constant(
        format!("mtp.{stage}.head_norm"),
        gguf.tensor(if gguf.has_tensor(&head_norm_name) {
            &head_norm_name
        } else {
            "output_norm.weight"
        })?,
    );
    let normalized = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[hidden, head_norm],
        format!("mtp.{stage}.shared_head_norm"),
    );
    gb.mark_output("mtp_hidden", normalized);
    let private_head_name = format!("blk.{i}.nextn.shared_head_head.weight");
    let head = gb.constant(
        format!("mtp.{stage}.head"),
        if gguf.has_tensor(&private_head_name) {
            gguf.tensor(&private_head_name)?
        } else if spec.tied_embeddings {
            tok_embd_tensor
        } else {
            gguf.tensor("output.weight")?
        },
    );
    let logits = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[spec.vocab_size as u64])?,
        &[head, normalized],
        format!("mtp.{stage}.shared_head"),
    );
    gb.mark_output("logits", logits);
    gb.build()
}

/// Build one physical autoregressive batch. Each row may belong to a different
/// sequence; the runtime supplies row-specific positions and block tables in
/// the paged-KV context.
pub fn build_batch_step_graph(gguf: &GgufFile, spec: &ModelSpec, rows: usize) -> Result<Graph> {
    build_batch_step_graph_with_features(gguf, spec, rows, &[])
}

/// Build a physical batch retaining only explicitly requested target features.
pub fn build_batch_step_graph_with_features(
    gguf: &GgufFile,
    spec: &ModelSpec,
    rows: usize,
    requested_features: &[TargetFeature],
) -> Result<Graph> {
    if rows == 0 {
        return Err(fellm_core::error::FellmError::other(
            "batch must contain rows",
        ));
    }
    let mut gb = GraphBuilder::new();
    let d_model = spec.d_model;
    let vocab = spec.vocab_size;
    let token_embd = gb.constant("token_embd", gguf.tensor("token_embd.weight")?);
    let output_w = gb.constant(
        "output_w",
        gguf.tensor(if spec.tied_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        })?,
    );
    let final_norm_w = gb.constant(
        "output_norm_w",
        gguf.tensor(if gguf.has_tensor("output_norm.weight") {
            "output_norm.weight"
        } else {
            "token_embd_norm.weight"
        })?,
    );
    let inv_freqs = gb.constant(
        "rope_inv_freqs",
        make_f32_tensor(&compute_rope_inv_freqs(spec)),
    );
    let token_ids = gb.input("token_id", DType::U32, Shape::new(&[rows as u64])?);
    let mut x = gb.op(
        OpKind::Embedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[token_embd, token_ids],
        "batch_tok_embed",
    );
    if spec.hc_mult > 1 {
        let streams = vec![x; spec.hc_mult];
        x = gb.op(
            OpKind::Concat,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[rows as u64, (d_model * spec.hc_mult) as u64])?,
            &streams,
            "batch_hc_embed",
        );
    }
    if requested_features.contains(&TargetFeature::EmbeddingOutput) {
        gb.mark_output("feature.embedding", x);
    }
    for layer in &spec.layers {
        x = build_batch_layer(&mut gb, gguf, spec, layer, x, inv_freqs, rows)?;
        if requested_features.contains(&TargetFeature::LayerHiddenState(layer.index as u32)) {
            gb.mark_output(format!("feature.layer.{}", layer.index), x);
        }
    }
    if spec.hc_mult > 1 && gguf.has_tensor("output_hc_fn.weight") {
        let fn_w = gb.constant("output_hc_fn", gguf.tensor("output_hc_fn.weight")?);
        let scale = gb.constant("output_hc_scale", gguf.tensor("output_hc_scale.weight")?);
        let base = gb.constant("output_hc_base", gguf.tensor("output_hc_base.weight")?);
        x = gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 2),
            DType::F32,
            Shape::new(&[rows as u64, d_model as u64])?,
            &[x, fn_w, scale, base],
            "batch_hc_head",
        );
    }
    let normalized = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x, final_norm_w],
        "batch_final_norm",
    );
    if requested_features.contains(&TargetFeature::FinalHiddenState) {
        gb.mark_output("feature.final", normalized);
    }
    let logits = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, vocab as u64])?,
        &[output_w, normalized],
        "batch_lm_head",
    );
    gb.mark_output("logits", logits);
    gb.build()
}

fn build_batch_layer(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    inv_freqs: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    if spec.hc_mult > 1 || gguf.has_tensor(&format!("blk.{i}.attn_q_a.weight")) {
        return build_batch_hc_layer(gb, gguf, spec, layer, x, inv_freqs, rows);
    }
    let shape = Shape::new(&[rows as u64, spec.d_model as u64])?;
    let attn_norm = gb.constant(
        format!("blk.{i}.attn_norm"),
        gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?,
    );
    let ffn_norm_name = if gguf.has_tensor(&format!("blk.{i}.ffn_norm.weight")) {
        format!("blk.{i}.ffn_norm.weight")
    } else {
        format!("blk.{i}.post_attention_norm.weight")
    };
    let ffn_norm = gb.constant(format!("blk.{i}.ffn_norm"), gguf.tensor(&ffn_norm_name)?);
    let normalized = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        shape.clone(),
        &[x, attn_norm],
        format!("blk.{i}.batch_attn_norm"),
    );
    let mix = match layer.mix {
        MixKind::Attention { .. } => build_batch_autoregressive_attention(
            gb, gguf, spec, layer, normalized, inv_freqs, rows,
        )?,
        MixKind::ShortConv => build_batch_shortconv(gb, gguf, spec, layer, normalized, rows)?,
        MixKind::GatedDeltaNet => build_gated_delta_net(gb, gguf, spec, layer, normalized, rows)?,
    };
    let after_attention = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        shape.clone(),
        &[x, mix],
        format!("blk.{i}.batch_residual1"),
    );
    let normalized = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        shape.clone(),
        &[after_attention, ffn_norm],
        format!("blk.{i}.batch_ffn_norm"),
    );
    let ffn = match layer.ffn {
        FfnKind::Dense => build_dense_ffn_batch(gb, gguf, spec, layer, normalized, rows)?,
        FfnKind::MoE => build_moe_batch(gb, gguf, spec, layer, normalized, rows)?,
    };
    Ok(gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        shape,
        &[after_attention, ffn],
        format!("blk.{i}.batch_residual2"),
    ))
}

fn build_batch_hc_layer(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_in: NodeId,
    inv_freqs: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    let prefix = format!("blk.{i}");
    let d_model = spec.d_model;
    let hc = spec.hc_mult.max(1);
    let narrow = Shape::new(&[rows as u64, d_model as u64])?;
    let wide = Shape::new(&[rows as u64, (d_model * hc) as u64])?;
    let hc_attn = if hc > 1 {
        Some((
            gb.constant(
                format!("{prefix}.hc_attn_fn"),
                gguf.tensor(&format!("{prefix}.hc_attn_fn.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_attn_scale"),
                gguf.tensor(&format!("{prefix}.hc_attn_scale.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_attn_base"),
                gguf.tensor(&format!("{prefix}.hc_attn_base.weight"))?,
            ),
        ))
    } else {
        None
    };
    let x_for_attn = if let Some((fn_w, scale, base)) = hc_attn {
        gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 0),
            DType::F32,
            narrow.clone(),
            &[x_in, fn_w, scale, base],
            format!("{prefix}.batch_hc_attn_pre"),
        )
    } else {
        x_in
    };
    let attn_norm_w = gb.constant(
        format!("{prefix}.attn_norm"),
        gguf.tensor(&format!("{prefix}.attn_norm.weight"))?,
    );
    let ffn_norm_name = if gguf.has_tensor(&format!("{prefix}.ffn_norm.weight")) {
        format!("{prefix}.ffn_norm.weight")
    } else {
        format!("{prefix}.post_attention_norm.weight")
    };
    let ffn_norm_w = gb.constant(format!("{prefix}.ffn_norm"), gguf.tensor(&ffn_norm_name)?);
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        narrow.clone(),
        &[x_for_attn, attn_norm_w],
        format!("{prefix}.batch_attn_norm"),
    );
    let mix = if gguf.has_tensor(&format!("{prefix}.attn_q_a.weight")) {
        build_batch_mla_attention(gb, gguf, spec, layer, x_norm, inv_freqs, rows)?
    } else {
        build_batch_autoregressive_attention(gb, gguf, spec, layer, x_norm, inv_freqs, rows)?
    };
    let x_after_mix = if let Some((fn_w, scale, base)) = hc_attn {
        gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 1),
            DType::F32,
            wide.clone(),
            &[mix, fn_w, scale, base, x_in],
            format!("{prefix}.batch_hc_attn_post"),
        )
    } else {
        gb.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            narrow.clone(),
            &[x_in, mix],
            format!("{prefix}.batch_residual1"),
        )
    };
    let hc_ffn = if hc > 1 {
        Some((
            gb.constant(
                format!("{prefix}.hc_ffn_fn"),
                gguf.tensor(&format!("{prefix}.hc_ffn_fn.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_ffn_scale"),
                gguf.tensor(&format!("{prefix}.hc_ffn_scale.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_ffn_base"),
                gguf.tensor(&format!("{prefix}.hc_ffn_base.weight"))?,
            ),
        ))
    } else {
        None
    };
    let x_for_ffn = if let Some((fn_w, scale, base)) = hc_ffn {
        gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 0),
            DType::F32,
            narrow.clone(),
            &[x_after_mix, fn_w, scale, base],
            format!("{prefix}.batch_hc_ffn_pre"),
        )
    } else {
        x_after_mix
    };
    let ffn_x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        narrow.clone(),
        &[x_for_ffn, ffn_norm_w],
        format!("{prefix}.batch_ffn_norm"),
    );
    let ffn_out = match layer.ffn {
        FfnKind::MoE => build_moe_batch(gb, gguf, spec, layer, ffn_x_norm, rows)?,
        FfnKind::Dense => build_dense_ffn_batch(gb, gguf, spec, layer, ffn_x_norm, rows)?,
    };
    if let Some((fn_w, scale, base)) = hc_ffn {
        Ok(gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 1),
            DType::F32,
            wide,
            &[ffn_out, fn_w, scale, base, x_after_mix],
            format!("{prefix}.batch_hc_ffn_post"),
        ))
    } else {
        Ok(gb.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            narrow,
            &[x_after_mix, ffn_out],
            format!("{prefix}.batch_residual2"),
        ))
    }
}

fn build_batch_mla_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_norm: NodeId,
    inv_freqs: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let prefix = format!("blk.{}", layer.index);
    let MixKind::Attention {
        head_dim,
        rope_dim,
        is_sliding,
        ..
    } = layer.mix
    else {
        unreachable!("mla");
    };
    let attn_ord = layer.attn_ordinal.expect("attention ordinal");
    let d_model = spec.d_model;
    let n_heads = spec.n_heads;
    let q_lora = gguf
        .tensor(&format!("{prefix}.attn_q_a.weight"))?
        .shape()
        .dims()[0] as usize;
    let o_groups = spec.output_group_count.max(1);
    let o_lora = spec.output_lora_rank.max(1);
    let k_in = gb.input(
        format!("k_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, head_dim as u64])?,
    );
    let v_in = gb.input(
        format!("v_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, head_dim as u64])?,
    );
    let q_a = gb.constant(
        format!("{prefix}.attn_q_a"),
        gguf.tensor(&format!("{prefix}.attn_q_a.weight"))?,
    );
    let q_a_norm = gb.constant(
        format!("{prefix}.attn_q_a_norm"),
        gguf.tensor(&format!("{prefix}.attn_q_a_norm.weight"))?,
    );
    let q_b = gb.constant(
        format!("{prefix}.attn_q_b"),
        gguf.tensor(&format!("{prefix}.attn_q_b.weight"))?,
    );
    let kv = gb.constant(
        format!("{prefix}.attn_kv"),
        gguf.tensor(&format!("{prefix}.attn_kv.weight"))?,
    );
    let kv_norm = gb.constant(
        format!("{prefix}.attn_kv_a_norm"),
        gguf.tensor(&format!("{prefix}.attn_kv_a_norm.weight"))?,
    );
    let wo_a = gb.constant(
        format!("{prefix}.attn_output_a"),
        gguf.tensor(&format!("{prefix}.attn_output_a.weight"))?,
    );
    let wo_b = gb.constant(
        format!("{prefix}.attn_output_b"),
        gguf.tensor(&format!("{prefix}.attn_output_b.weight"))?,
    );
    let sinks = gb.constant(
        format!("{prefix}.attn_sinks"),
        gguf.tensor(&format!("{prefix}.attn_sinks.weight"))?,
    );
    let mut inputs = vec![
        x_norm, q_a, q_a_norm, q_b, kv, kv_norm, wo_a, wo_b, sinks, inv_freqs, k_in, v_in,
    ];
    if gguf.has_tensor(&format!("{prefix}.attn_compressor_kv.weight")) {
        for name in [
            "attn_compressor_kv",
            "attn_compressor_gate",
            "attn_compressor_ape",
            "attn_compressor_norm",
        ] {
            let tensor_name = format!("{prefix}.{name}.weight");
            if gguf.has_tensor(&tensor_name) {
                inputs.push(gb.constant(format!("{prefix}.{name}"), gguf.tensor(&tensor_name)?));
            }
        }
    }
    Ok(gb.op(
        OpKind::MlaAttention,
        OpAttrs {
            n_embd: d_model as u32,
            n_heads: n_heads as u32,
            n_kv_heads: spec.indexer_n_head as u32,
            query_len: spec.indexer_head_dim as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            rope_pairing: rope_pairing_code(spec),
            gdn_inner_size: q_lora as u32,
            gdn_state_size: o_groups as u32,
            shortconv_l_cache: o_lora as u32,
            attention_window: if is_sliding {
                spec.sliding_window as u32
            } else {
                0
            },
            eps: spec.norm_eps,
            layer_ord: attn_ord as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            block_size: layer.compress_ratio,
            kv_len: spec.indexer_top_k as u32,
            ..Default::default()
        },
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &inputs,
        format!("{prefix}.batch_mla"),
    ))
}

fn build_batch_shortconv(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    let n = spec.d_model;
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
    let state_elements = spec.shortconv_l_cache.saturating_sub(1) * n;
    let state = gb.input(
        format!("conv_in_{conv_ord}"),
        DType::F32,
        Shape::new(&[rows as u64, state_elements as u64])?,
    );
    Ok(gb.op(
        OpKind::ShortConv,
        OpAttrs {
            shortconv_l_cache: spec.shortconv_l_cache as u32,
            n_embd: n as u32,
            ..Default::default()
        },
        DType::F32,
        Shape::new(&[rows as u64, n as u64])?,
        &[x, in_proj, conv, out_proj, state],
        format!("blk.{i}.batch_shortconv"),
    ))
}

fn build_gated_delta_net(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    let recurrent_ord = layer.conv_ordinal.expect("Gated DeltaNet ordinal");
    let constants = [
        ("qkv", "attn_qkv.weight"),
        ("gate", "attn_gate.weight"),
        ("beta", "ssm_beta.weight"),
        ("alpha", "ssm_alpha.weight"),
        ("dt", "ssm_dt.bias"),
        ("a", "ssm_a"),
        ("conv", "ssm_conv1d.weight"),
        ("norm", "ssm_norm.weight"),
        ("out", "ssm_out.weight"),
    ]
    .map(|(label, suffix)| {
        Ok(gb.constant(
            format!("blk.{i}.gdn.{label}"),
            gguf.tensor(&format!("blk.{i}.{suffix}"))?,
        ))
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let mixed_width = spec.gdn_inner_size + 2 * spec.gdn_key_heads * spec.gdn_state_size;
    let conv_elements = spec.gdn_conv_kernel.saturating_sub(1) * mixed_width;
    let ssm_elements = spec.gdn_value_heads * spec.gdn_state_size * spec.gdn_state_size;
    let conv_shape = if rows == 1 {
        Shape::new(&[conv_elements as u64])?
    } else {
        Shape::new(&[rows as u64, conv_elements as u64])?
    };
    let ssm_shape = if rows == 1 {
        Shape::new(&[ssm_elements as u64])?
    } else {
        Shape::new(&[rows as u64, ssm_elements as u64])?
    };
    let conv_state = gb.input(format!("conv_in_{recurrent_ord}"), DType::F32, conv_shape);
    let ssm_state = gb.input(format!("ssm_in_{recurrent_ord}"), DType::F32, ssm_shape);
    let mut inputs = vec![x];
    inputs.extend(constants);
    inputs.extend([conv_state, ssm_state]);
    let output_shape = if rows == 1 {
        Shape::new(&[spec.d_model as u64])?
    } else {
        Shape::new(&[rows as u64, spec.d_model as u64])?
    };
    Ok(gb.op(
        OpKind::GatedDeltaNet,
        OpAttrs {
            eps: spec.norm_eps,
            n_embd: spec.d_model as u32,
            n_heads: spec.gdn_value_heads as u32,
            n_kv_heads: spec.gdn_key_heads as u32,
            gdn_inner_size: spec.gdn_inner_size as u32,
            gdn_state_size: spec.gdn_state_size as u32,
            gdn_conv_kernel: spec.gdn_conv_kernel as u32,
            ..Default::default()
        },
        DType::F32,
        output_shape,
        &inputs,
        format!("blk.{i}.gated_delta_net"),
    ))
}

fn build_batch_autoregressive_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    inv_freqs: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let MixKind::Attention {
        n_kv_heads,
        qk_norm,
        head_dim,
        rope_dim,
        value_reuses_key,
        ..
    } = layer.mix
    else {
        return Err(fellm_core::error::FellmError::other(
            "batch layer is not attention",
        ));
    };
    let i = layer.index;
    let n_heads = spec.n_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let attn_ord = layer.attn_ordinal.expect("attention ordinal");
    let q_weight = gguf.tensor(&format!("blk.{i}.attn_q.weight"))?;
    let q_rows = q_weight.shape().dims()[0] as usize;
    let gated_attention = q_rows == q_width.saturating_mul(2);
    if q_rows != q_width && !gated_attention {
        return Err(fellm_core::error::FellmError::other(format!(
            "blk.{i}.attn_q has {q_rows} rows; expected {q_width} or {} for gated attention",
            q_width * 2
        )));
    }
    let wq = gb.constant(format!("blk.{i}.attn_q"), q_weight);
    let wk = gb.constant(
        format!("blk.{i}.attn_k"),
        gguf.tensor(&format!("blk.{i}.attn_k.weight"))?,
    );
    let wv = if value_reuses_key {
        None
    } else {
        Some(gb.constant(
            format!("blk.{i}.attn_v"),
            gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
        ))
    };
    let wo = gb.constant(
        format!("blk.{i}.attn_o"),
        gguf.tensor(&format!("blk.{i}.attn_output.weight"))?,
    );
    let projection = |gb: &mut GraphBuilder, weight, width, label: String| {
        gb.op(
            OpKind::MatMul,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[rows as u64, width as u64]).expect("batch projection shape"),
            &[weight, x],
            label,
        )
    };
    let q_full = projection(
        gb,
        wq,
        if gated_attention {
            q_width * 2
        } else {
            q_width
        },
        format!("blk.{i}.batch_q_full"),
    );
    let select_lane = |gb: &mut GraphBuilder, lane, label: String| {
        gb.op(
            OpKind::InterleavedHeadSelect,
            OpAttrs {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                kv_slot: lane,
                ..Default::default()
            },
            DType::F32,
            Shape::new(&[rows as u64, q_width as u64]).expect("query lane shape"),
            &[q_full],
            label,
        )
    };
    let q = if gated_attention {
        select_lane(gb, 0, format!("blk.{i}.batch_q"))
    } else {
        q_full
    };
    let q_gate = gated_attention.then(|| select_lane(gb, 1, format!("blk.{i}.batch_q_gate")));
    let k = projection(gb, wk, kv_width, format!("blk.{i}.batch_k"));
    let v = if value_reuses_key {
        k
    } else {
        projection(
            gb,
            wv.expect("v weight"),
            kv_width,
            format!("blk.{i}.batch_v"),
        )
    };
    let normalize =
        |gb: &mut GraphBuilder, input, heads, weight_name: &str, label: String| -> Result<NodeId> {
            if !qk_norm {
                return Ok(input);
            }
            let weight = gb.constant(label.clone() + "_weight", gguf.tensor(weight_name)?);
            Ok(gb.op(
                OpKind::RmsNorm,
                OpAttrs {
                    eps: spec.norm_eps,
                    n_heads: heads as u32,
                    head_dim: head_dim as u32,
                    ..Default::default()
                },
                DType::F32,
                Shape::new(&[rows as u64, (heads * head_dim) as u64])?,
                &[input, weight],
                label,
            ))
        };
    let q = normalize(
        gb,
        q,
        n_heads,
        &format!("blk.{i}.attn_q_norm.weight"),
        format!("blk.{i}.batch_q_norm"),
    )?;
    let k = normalize(
        gb,
        k,
        n_kv_heads,
        &format!("blk.{i}.attn_k_norm.weight"),
        format!("blk.{i}.batch_k_norm"),
    )?;
    let rope =
        |gb: &mut GraphBuilder, input, heads, custom_op_id, label: String| -> Result<NodeId> {
            Ok(gb.op(
                OpKind::Rope,
                OpAttrs {
                    n_heads: heads as u32,
                    head_dim: head_dim as u32,
                    rope_dim: rope_dim as u32,
                    rope_pairing: u32::from(matches!(
                        spec.rope_pairing,
                        crate::probe::RopePairing::SplitHalf
                    )),
                    rope_base: spec.rope_base,
                    layer_ord: attn_ord as u32,
                    custom_op_id,
                    ..Default::default()
                },
                DType::F32,
                Shape::new(&[rows as u64, (heads * head_dim) as u64])?,
                &[input, inv_freqs],
                label,
            ))
        };
    let q = rope(gb, q, n_heads, 0, format!("blk.{i}.batch_q_rope"))?;
    let k = rope(gb, k, n_kv_heads, 1, format!("blk.{i}.batch_k_rope"))?;
    let cache_shape = Shape::new(&[spec.context_length as u64, kv_width as u64])?;
    let k_in = gb.input(format!("k_in_{attn_ord}"), DType::F32, cache_shape.clone());
    let v_in = gb.input(format!("v_in_{attn_ord}"), DType::F32, cache_shape.clone());
    let write = |gb: &mut GraphBuilder, input, cache, slot, label: String| {
        gb.op_in_place(
            OpKind::KvWrite,
            OpAttrs {
                layer_ord: attn_ord as u32,
                kv_slot: slot,
                block_size: 16,
                ..Default::default()
            },
            DType::F32,
            cache_shape.clone(),
            &[input, cache],
            1,
            label,
        )
    };
    let k_cache = write(gb, k, k_in, 0, format!("blk.{i}.batch_k_write"));
    let v_cache = write(gb, v, v_in, 1, format!("blk.{i}.batch_v_write"));
    let attention = gb.op(
        OpKind::Attention,
        OpAttrs {
            layer_ord: attn_ord as u32,
            query_len: rows as u32,
            ..attention_attrs(spec, n_kv_heads, head_dim)
        },
        DType::F32,
        Shape::new(&[rows as u64, q_width as u64])?,
        &[q, k_cache, v_cache],
        format!("blk.{i}.batch_attention"),
    );
    let attention = if let Some(gate) = q_gate {
        gb.op(
            OpKind::SigmoidGate,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[rows as u64, q_width as u64])?,
            &[attention, gate],
            format!("blk.{i}.batch_attention_gate"),
        )
    } else {
        attention
    };
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, spec.d_model as u64])?,
        &[wo, attention],
        format!("blk.{i}.batch_o"),
    ))
}

/// Build DiffusionGemma's reusable full-canvas denoising graph.
///
/// The graph has one embedding/transformer pass for the complete canvas and
/// returns one logit row per canvas position.  Prompt KV inputs are read-only;
/// there are deliberately no `KvWrite` nodes in this graph.
pub fn build_diffusion_canvas_graph(gguf: &GgufFile, spec: &ModelSpec) -> Result<Graph> {
    if !spec.is_diffusion {
        return Err(fellm_core::error::FellmError::other(
            "canvas graph requires diffusion-gemma",
        ));
    }
    let mut gb = GraphBuilder::new();
    let rows = spec.canvas_length;
    let d_model = spec.d_model;
    let vocab = spec.vocab_size;
    let tok_embd = gb.constant("token_embd", gguf.tensor("token_embd.weight")?);
    let output_w = gb.constant("output_w", gguf.tensor("token_embd.weight")?);
    let final_norm_w = gb.constant("output_norm_w", gguf.tensor("output_norm.weight")?);
    let self_cond_pre_norm = gb.constant(
        "self_cond_pre_norm",
        gguf.tensor("self_cond_pre_norm.weight")?,
    );
    let self_cond_gate = gb.constant("self_cond_gate", gguf.tensor("self_cond_gate.weight")?);
    let self_cond_up = gb.constant("self_cond_up", gguf.tensor("self_cond_up.weight")?);
    let self_cond_down = gb.constant("self_cond_down", gguf.tensor("self_cond_down.weight")?);
    let inv_freqs = gb.constant(
        "rope_inv_freqs",
        make_f32_tensor(&compute_rope_inv_freqs(spec)),
    );
    let canvas_ids = gb.input("canvas_tokens", DType::U32, Shape::new(&[rows as u64])?);
    let self_cond_slots = crate::diffusion_self_conditioning_slots(vocab);
    let self_cond_logits = gb.input(
        "self_conditioning_logits",
        DType::F32,
        Shape::new(&[rows as u64, self_cond_slots as u64])?,
    );
    let embed_scale = gb.constant(
        "diffusion_embed_scale",
        make_f32_matrix_tensor(
            &vec![(d_model as f32).sqrt(); rows.saturating_mul(d_model)],
            rows,
            d_model,
        ),
    );
    let canvas_embed = gb.op(
        OpKind::Embedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[tok_embd, canvas_ids],
        "canvas_embed",
    );
    let mut x = gb.op(
        OpKind::Mul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[canvas_embed, embed_scale],
        "canvas_embed_scaled",
    );
    let self_cond_embed = gb.op(
        OpKind::WeightedEmbedding,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[tok_embd, self_cond_logits],
        "self_conditioning_embedding",
    );
    let self_cond_embed = gb.op(
        OpKind::Mul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[self_cond_embed, embed_scale],
        "self_conditioning_embedding_scaled",
    );
    let self_cond_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[self_cond_embed, self_cond_pre_norm],
        "self_conditioning_norm",
    );
    let self_cond_gate_out = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, spec.dense_ffn_dim as u64])?,
        &[self_cond_gate, self_cond_norm],
        "self_conditioning_gate",
    );
    let self_cond_up_out = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, spec.dense_ffn_dim as u64])?,
        &[self_cond_up, self_cond_norm],
        "self_conditioning_up",
    );
    let self_cond_hidden = gb.op(
        OpKind::SiluGate,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, spec.dense_ffn_dim as u64])?,
        &[self_cond_gate_out, self_cond_up_out],
        "self_conditioning_swiglu",
    );
    let self_cond_out = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[self_cond_down, self_cond_hidden],
        "self_conditioning_out",
    );
    let combined = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x, self_cond_out],
        "canvas_with_self_conditioning",
    );
    let post_norm_w = gb.constant(
        "self_cond_post_norm_ones",
        make_f32_tensor(&vec![1.0; d_model]),
    );
    x = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[combined, post_norm_w],
        "canvas_self_conditioned_norm",
    );
    for layer in &spec.layers {
        x = build_canvas_layer(&mut gb, gguf, spec, layer, x, inv_freqs, rows)?;
    }
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x, final_norm_w],
        "canvas_final_norm",
    );
    let logits = gb.op(
        OpKind::MatMul,
        OpAttrs {
            softcap: spec.final_logit_softcapping,
            ..Default::default()
        },
        DType::F32,
        Shape::new(&[rows as u64, vocab as u64])?,
        &[output_w, x_norm],
        "canvas_lm_head",
    );
    gb.mark_output("logits", logits);
    gb.build()
}

fn build_canvas_layer(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_in: NodeId,
    inv_freqs: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    let d_model = spec.d_model;
    let attn_norm_w = gb.constant(
        format!("blk.{i}.attn_norm"),
        gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?,
    );
    let ffn_norm_name = if gguf.has_tensor(&format!("blk.{i}.ffn_norm.weight")) {
        format!("blk.{i}.ffn_norm.weight")
    } else {
        format!("blk.{i}.post_attention_norm.weight")
    };
    let ffn_norm_w = gb.constant(format!("blk.{i}.ffn_norm"), gguf.tensor(&ffn_norm_name)?);
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x_in, attn_norm_w],
        format!("blk.{i}.canvas_attn_norm"),
    );
    let mix = build_canvas_attention(gb, gguf, spec, layer, x_norm, inv_freqs, rows)?;
    let x_after_mix = gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x_in, mix],
        format!("blk.{i}.canvas_residual1"),
    );
    let ffn_x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x_after_mix, ffn_norm_w],
        format!("blk.{i}.canvas_ffn_norm"),
    );
    let ffn_out = match layer.ffn {
        FfnKind::MoE => build_moe_batch(gb, gguf, spec, layer, ffn_x_norm, rows)?,
        FfnKind::Dense => build_dense_ffn_batch(gb, gguf, spec, layer, ffn_x_norm, rows)?,
    };
    Ok(gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[x_after_mix, ffn_out],
        format!("blk.{i}.canvas_residual2"),
    ))
}

fn build_canvas_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_norm: NodeId,
    inv_freqs: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let MixKind::Attention {
        n_kv_heads: n_kv,
        qk_norm,
        head_dim,
        rope_dim,
        is_sliding,
        value_reuses_key,
    } = layer.mix
    else {
        return Err(fellm_core::error::FellmError::other(
            "DiffusionGemma canvas graph encountered recurrent layer",
        ));
    };
    let i = layer.index;
    let d_model = spec.d_model;
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
    let wv = if value_reuses_key {
        None
    } else {
        Some(gb.constant(
            format!("blk.{i}.attn_v"),
            gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
        ))
    };
    let wo = gb.constant(
        format!("blk.{i}.attn_o"),
        gguf.tensor(&format!("blk.{i}.attn_output.weight"))?,
    );
    let q = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, q_stride as u64])?,
        &[wq, x_norm],
        format!("blk.{i}.canvas_q_proj"),
    );
    let k = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, kv_stride as u64])?,
        &[wk, x_norm],
        format!("blk.{i}.canvas_k_proj"),
    );
    let v = if value_reuses_key {
        k
    } else {
        gb.op(
            OpKind::MatMul,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[rows as u64, kv_stride as u64])?,
            &[wv.expect("value projection"), x_norm],
            format!("blk.{i}.canvas_v_proj"),
        )
    };
    let (q, k) = if qk_norm {
        let q_w = gb.constant(
            format!("blk.{i}.attn_q_norm"),
            gguf.tensor(&format!("blk.{i}.attn_q_norm.weight"))?,
        );
        let k_w = gb.constant(
            format!("blk.{i}.attn_k_norm"),
            gguf.tensor(&format!("blk.{i}.attn_k_norm.weight"))?,
        );
        let q = gb.op(
            OpKind::RmsNorm,
            OpAttrs {
                eps: spec.norm_eps,
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                ..Default::default()
            },
            DType::F32,
            Shape::new(&[rows as u64, q_stride as u64])?,
            &[q, q_w],
            format!("blk.{i}.canvas_q_norm"),
        );
        let k = gb.op(
            OpKind::RmsNorm,
            OpAttrs {
                eps: spec.norm_eps,
                n_heads: n_kv as u32,
                head_dim: head_dim as u32,
                ..Default::default()
            },
            DType::F32,
            Shape::new(&[rows as u64, kv_stride as u64])?,
            &[k, k_w],
            format!("blk.{i}.canvas_k_norm"),
        );
        (q, k)
    } else {
        (q, k)
    };
    let rope =
        |gb: &mut GraphBuilder, input: NodeId, heads: usize, label: String| -> Result<NodeId> {
            Ok(gb.op(
                OpKind::Rope,
                OpAttrs {
                    n_heads: heads as u32,
                    head_dim: head_dim as u32,
                    rope_dim: rope_dim as u32,
                    rope_pairing: u32::from(matches!(
                        spec.rope_pairing,
                        crate::probe::RopePairing::SplitHalf
                    )),
                    rope_base: spec.rope_base,
                    ..Default::default()
                },
                DType::F32,
                Shape::new(&[rows as u64, (heads * head_dim) as u64])?,
                &[input, inv_freqs],
                label,
            ))
        };
    let q = rope(gb, q, n_heads, format!("blk.{i}.canvas_q_rope"))?;
    let k = rope(gb, k, n_kv, format!("blk.{i}.canvas_k_rope"))?;
    let prefix_k = gb.input(
        format!("k_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, kv_stride as u64])?,
    );
    let prefix_v = gb.input(
        format!("v_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, kv_stride as u64])?,
    );
    let attn = gb.op(
        OpKind::Attention,
        OpAttrs {
            n_heads: n_heads as u32,
            n_kv_heads: n_kv as u32,
            head_dim: head_dim as u32,
            layer_ord: attn_ord as u32,
            attention_mode: 1,
            attention_window: if is_sliding {
                spec.sliding_window as u32
            } else {
                0
            },
            query_len: rows as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            block_size: 16,
            ..Default::default()
        },
        DType::F32,
        Shape::new(&[rows as u64, q_stride as u64])?,
        &[q, prefix_k, prefix_v, k, v],
        format!("blk.{i}.canvas_attention"),
    );
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_model as u64])?,
        &[wo, attn],
        format!("blk.{i}.canvas_o_proj"),
    ))
}

fn build_dense_ffn_batch(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    let d_ff = spec.dense_ffn_dim;
    let gate_w = gb.constant(
        format!("blk.{i}.ffn_gate"),
        gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
    );
    let up_w = gb.constant(
        format!("blk.{i}.ffn_up"),
        gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
    );
    let down_w = gb.constant(
        format!("blk.{i}.ffn_down"),
        gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?,
    );
    let gate = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_ff as u64])?,
        &[gate_w, x],
        format!("blk.{i}.canvas_ffn_gate"),
    );
    let up = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_ff as u64])?,
        &[up_w, x],
        format!("blk.{i}.canvas_ffn_up"),
    );
    let gated = gb.op(
        OpKind::SiluGate,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, d_ff as u64])?,
        &[gate, up],
        format!("blk.{i}.canvas_swiglu"),
    );
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, spec.d_model as u64])?,
        &[down_w, gated],
        format!("blk.{i}.canvas_ffn_down"),
    ))
}

fn build_moe_batch(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    rows: usize,
) -> Result<NodeId> {
    let i = layer.index;
    let gate_inp = gb.constant(
        format!("blk.{i}.ffn_gate_inp"),
        gguf.tensor(&format!("blk.{i}.ffn_gate_inp.weight"))?,
    );
    let mut inputs = vec![x, gate_inp];
    if gguf.has_tensor(&format!("blk.{i}.ffn_gate_up_exps.weight")) {
        inputs.extend([
            gb.constant(
                format!("blk.{i}.ffn_gate_up_exps"),
                gguf.tensor(&format!("blk.{i}.ffn_gate_up_exps.weight"))?,
            ),
            gb.constant(
                format!("blk.{i}.ffn_down_exps"),
                gguf.tensor(&format!("blk.{i}.ffn_down_exps.weight"))?,
            ),
            gb.constant(
                format!("blk.{i}.ffn_gate"),
                gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
            ),
            gb.constant(
                format!("blk.{i}.ffn_up"),
                gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
            ),
            gb.constant(
                format!("blk.{i}.ffn_down"),
                gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?,
            ),
        ]);
    } else {
        for name in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            inputs.push(gb.constant(
                format!("blk.{i}.{name}"),
                gguf.tensor(&format!("blk.{i}.{name}.weight"))?,
            ));
        }
        if gguf.has_tensor(&format!("blk.{i}.ffn_gate_shexp.weight")) {
            for name in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"] {
                inputs.push(gb.constant(
                    format!("blk.{i}.{name}"),
                    gguf.tensor(&format!("blk.{i}.{name}.weight"))?,
                ));
            }
        }
    }
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
        Shape::new(&[rows as u64, spec.d_model as u64])?,
        &inputs,
        format!("blk.{i}.canvas_moe"),
    ))
}

/// Node ids whose attrs must be patched each decode step.
#[derive(Debug, Default, Clone)]
pub struct StepBindings {
    /// `RoPE` ops.
    pub rope: Vec<NodeId>,
    /// KV write ops.
    pub kv_write: Vec<NodeId>,
    /// Attention ops.
    pub attention: Vec<NodeId>,
}

/// Collect position / `past_len` binding targets from a built graph.
#[must_use]
pub fn collect_step_bindings(graph: &Graph) -> StepBindings {
    let mut nodes = StepBindings::default();
    for (id, node) in graph.iter_nodes() {
        match node.op {
            Some(OpKind::Rope) => nodes.rope.push(id),
            Some(OpKind::KvWrite) => nodes.kv_write.push(id),
            Some(OpKind::Attention) | Some(OpKind::MlaAttention) => nodes.attention.push(id),
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
    inv_freqs_compress: NodeId,
    token_id: NodeId,
    prefix: &str,
) -> Result<NodeId> {
    let d_model = spec.d_model;
    let hc = spec.hc_mult;
    let hc_attn = if hc > 1 {
        Some((
            gb.constant(
                format!("{prefix}.hc_attn_fn"),
                gguf.tensor(&format!("{prefix}.hc_attn_fn.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_attn_scale"),
                gguf.tensor(&format!("{prefix}.hc_attn_scale.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_attn_base"),
                gguf.tensor(&format!("{prefix}.hc_attn_base.weight"))?,
            ),
        ))
    } else {
        None
    };
    let x_for_attn = if let Some((fn_w, scale, base)) = hc_attn {
        gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 0),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x_in, fn_w, scale, base],
            format!("{prefix}.hc_attn_pre"),
        )
    } else {
        x_in
    };
    let attn_norm_w = gb.constant(
        format!("{prefix}.attn_norm"),
        gguf.tensor(&format!("{prefix}.attn_norm.weight"))?,
    );
    let ffn_norm_name = if gguf.has_tensor(&format!("{prefix}.ffn_norm.weight")) {
        format!("{prefix}.ffn_norm.weight")
    } else {
        format!("{prefix}.post_attention_norm.weight")
    };
    let ffn_norm_w = gb.constant(format!("{prefix}.ffn_norm"), gguf.tensor(&ffn_norm_name)?);
    let x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_for_attn, attn_norm_w],
        format!("{prefix}.attn_norm_op"),
    );

    let mix_is_fused = matches!(layer.mix, MixKind::Attention { .. });
    let mix = match layer.mix {
        MixKind::ShortConv => build_shortconv(gb, gguf, spec, layer, x_norm)?,
        MixKind::GatedDeltaNet => build_gated_delta_net(gb, gguf, spec, layer, x_norm, 1)?,
        MixKind::Attention { .. } => {
            build_attention(
                gb,
                gguf,
                spec,
                layer,
                x_norm,
                if layer.compress_ratio != 0 {
                    inv_freqs_compress
                } else {
                    inv_freqs
                },
                x_for_attn,
                prefix,
            )?
        }
    };

    let x_after_mix = if let Some((fn_w, scale, base)) = hc_attn {
        gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 1),
            DType::F32,
            Shape::new(&[(d_model * hc) as u64])?,
            &[mix, fn_w, scale, base, x_in],
            format!("{prefix}.hc_attn_post"),
        )
    } else if mix_is_fused {
        mix
    } else {
        gb.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x_in, mix],
            format!("{prefix}.residual1"),
        )
    };

    let hc_ffn = if hc > 1 {
        Some((
            gb.constant(
                format!("{prefix}.hc_ffn_fn"),
                gguf.tensor(&format!("{prefix}.hc_ffn_fn.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_ffn_scale"),
                gguf.tensor(&format!("{prefix}.hc_ffn_scale.weight"))?,
            ),
            gb.constant(
                format!("{prefix}.hc_ffn_base"),
                gguf.tensor(&format!("{prefix}.hc_ffn_base.weight"))?,
            ),
        ))
    } else {
        None
    };
    let x_for_ffn = if let Some((fn_w, scale, base)) = hc_ffn {
        gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 0),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x_after_mix, fn_w, scale, base],
            format!("{prefix}.hc_ffn_pre"),
        )
    } else {
        x_after_mix
    };

    let ffn_x_norm = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x_for_ffn, ffn_norm_w],
        format!("{prefix}.ffn_norm_op"),
    );

    let ffn_out = match layer.ffn {
        FfnKind::MoE => build_moe(gb, gguf, spec, layer, ffn_x_norm, token_id, prefix)?,
        FfnKind::Dense => build_dense_ffn(gb, gguf, spec, layer, ffn_x_norm, x_for_ffn)?,
    };
    if let Some((fn_w, scale, base)) = hc_ffn {
        Ok(gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 1),
            DType::F32,
            Shape::new(&[(d_model * hc) as u64])?,
            &[ffn_out, fn_w, scale, base, x_after_mix],
            format!("{prefix}.hc_ffn_post"),
        ))
    } else if matches!(layer.ffn, FfnKind::Dense) {
        Ok(ffn_out)
    } else {
        Ok(gb.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x_after_mix, ffn_out],
            format!("{prefix}.residual2"),
        ))
    }
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

fn rope_pairing_code(spec: &ModelSpec) -> u32 {
    match spec.rope_pairing {
        crate::probe::RopePairing::Adjacent => 0,
        crate::probe::RopePairing::SplitHalf => 1,
        crate::probe::RopePairing::TailAdjacent => 2,
    }
}

fn hc_attrs(spec: &ModelSpec, slot: u32) -> OpAttrs {
    OpAttrs {
        n_embd: spec.d_model as u32,
        gdn_state_size: spec.hc_mult.max(1) as u32,
        kv_slot: slot,
        gdn_conv_kernel: spec.hc_sinkhorn_iters.max(1),
        eps: spec.hc_eps.max(spec.norm_eps),
        ..Default::default()
    }
}

fn build_mla_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_norm: NodeId,
    inv_freqs: NodeId,
    _residual: NodeId,
    prefix: &str,
) -> Result<NodeId> {
    let MixKind::Attention {
        head_dim,
        rope_dim,
        is_sliding,
        ..
    } = layer.mix
    else {
        unreachable!("mla");
    };
    let attn_ord = layer.attn_ordinal.expect("attention ordinal");
    let d_model = spec.d_model;
    let n_heads = spec.n_heads;
    let q_lora = gguf
        .tensor(&format!("{prefix}.attn_q_a.weight"))?
        .shape()
        .dims()[0] as usize;
    let o_groups = spec.output_group_count.max(1);
    let o_lora = spec.output_lora_rank.max(1);
    let k_in = gb.input(
        format!("k_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, head_dim as u64])?,
    );
    let v_in = gb.input(
        format!("v_in_{attn_ord}"),
        DType::F32,
        Shape::new(&[spec.context_length as u64, head_dim as u64])?,
    );
    let q_a = gb.constant(
        format!("{prefix}.attn_q_a"),
        gguf.tensor(&format!("{prefix}.attn_q_a.weight"))?,
    );
    let q_a_norm = gb.constant(
        format!("{prefix}.attn_q_a_norm"),
        gguf.tensor(&format!("{prefix}.attn_q_a_norm.weight"))?,
    );
    let q_b = gb.constant(
        format!("{prefix}.attn_q_b"),
        gguf.tensor(&format!("{prefix}.attn_q_b.weight"))?,
    );
    let kv = gb.constant(
        format!("{prefix}.attn_kv"),
        gguf.tensor(&format!("{prefix}.attn_kv.weight"))?,
    );
    let kv_norm = gb.constant(
        format!("{prefix}.attn_kv_a_norm"),
        gguf.tensor(&format!("{prefix}.attn_kv_a_norm.weight"))?,
    );
    let wo_a = gb.constant(
        format!("{prefix}.attn_output_a"),
        gguf.tensor(&format!("{prefix}.attn_output_a.weight"))?,
    );
    let wo_b = gb.constant(
        format!("{prefix}.attn_output_b"),
        gguf.tensor(&format!("{prefix}.attn_output_b.weight"))?,
    );
    let sinks = gb.constant(
        format!("{prefix}.attn_sinks"),
        gguf.tensor(&format!("{prefix}.attn_sinks.weight"))?,
    );
    let mut inputs = vec![
        x_norm, q_a, q_a_norm, q_b, kv, kv_norm, wo_a, wo_b, sinks, inv_freqs, k_in, v_in,
    ];
    if gguf.has_tensor(&format!("{prefix}.attn_compressor_kv.weight")) {
        inputs.push(gb.constant(
            format!("{prefix}.attn_compressor_kv"),
            gguf.tensor(&format!("{prefix}.attn_compressor_kv.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.attn_compressor_gate"),
            gguf.tensor(&format!("{prefix}.attn_compressor_gate.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.attn_compressor_ape"),
            gguf.tensor(&format!("{prefix}.attn_compressor_ape.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.attn_compressor_norm"),
            gguf.tensor(&format!("{prefix}.attn_compressor_norm.weight"))?,
        ));
    }
    if gguf.has_tensor(&format!("{prefix}.indexer.attn_q_b.weight")) {
        for name in [
            "indexer.attn_q_b",
            "indexer.proj",
            "indexer_compressor_kv",
            "indexer_compressor_gate",
            "indexer_compressor_ape",
            "indexer_compressor_norm",
        ] {
            let tensor_name = format!("{prefix}.{name}.weight");
            if !gguf.has_tensor(&tensor_name) {
                return Err(fellm_core::error::FellmError::other(format!(
                    "missing {tensor_name}"
                )));
            }
            inputs.push(gb.constant(format!("{prefix}.{name}"), gguf.tensor(&tensor_name)?));
        }
    }
    Ok(gb.op(
        OpKind::MlaAttention,
        OpAttrs {
            n_embd: d_model as u32,
            n_heads: n_heads as u32,
            n_kv_heads: spec.indexer_n_head as u32,
            query_len: spec.indexer_head_dim as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            rope_pairing: rope_pairing_code(spec),
            gdn_inner_size: q_lora as u32,
            gdn_state_size: o_groups as u32,
            shortconv_l_cache: o_lora as u32,
            attention_window: if is_sliding {
                spec.sliding_window as u32
            } else {
                0
            },
            eps: spec.norm_eps,
            layer_ord: attn_ord as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            block_size: layer.compress_ratio,
            kv_len: spec.indexer_top_k as u32,
            ..Default::default()
        },
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &inputs,
        format!("{prefix}.mla"),
    ))
}

fn build_attention(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x_norm: NodeId,
    inv_freqs: NodeId,
    residual: NodeId,
    prefix: &str,
) -> Result<NodeId> {
    let MixKind::Attention {
        n_kv_heads: n_kv,
        qk_norm,
        head_dim,
        rope_dim,
        value_reuses_key,
        ..
    } = layer.mix
    else {
        unreachable!("attention mix");
    };
    let i = layer.index;
    let d_model = spec.d_model;
    let n_heads = spec.n_heads;
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let attn_ord = layer.attn_ordinal.expect("attention ordinal");

    if gguf.has_tensor(&format!("{prefix}.attn_q_a.weight")) {
        return build_mla_attention(gb, gguf, spec, layer, x_norm, inv_freqs, residual, prefix);
    }

    let q_weight = gguf.tensor(&format!("blk.{i}.attn_q.weight"))?;
    let q_rows = q_weight.shape().dims()[0] as usize;
    let gated_attention = q_rows == q_stride.saturating_mul(2);
    if q_rows != q_stride && !gated_attention {
        return Err(fellm_core::error::FellmError::other(format!(
            "blk.{i}.attn_q has {q_rows} rows; expected {q_stride} or {} for gated attention",
            q_stride * 2
        )));
    }
    let wq = gb.constant(format!("blk.{i}.attn_q"), q_weight);
    let wk = gb.constant(
        format!("blk.{i}.attn_k"),
        gguf.tensor(&format!("blk.{i}.attn_k.weight"))?,
    );
    let wv = if value_reuses_key {
        None
    } else {
        Some(gb.constant(
            format!("blk.{i}.attn_v"),
            gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
        ))
    };
    let wo = gb.constant(
        format!("blk.{i}.attn_o"),
        gguf.tensor(&format!("blk.{i}.attn_output.weight"))?,
    );

    let q_full = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[(if gated_attention {
            q_stride * 2
        } else {
            q_stride
        }) as u64])?,
        &[wq, x_norm],
        format!("blk.{i}.q_proj"),
    );
    let select_lane = |gb: &mut GraphBuilder, lane, label: String| {
        gb.op(
            OpKind::InterleavedHeadSelect,
            OpAttrs {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                kv_slot: lane,
                ..Default::default()
            },
            DType::F32,
            Shape::new(&[q_stride as u64]).expect("query shape"),
            &[q_full],
            label,
        )
    };
    let q = if gated_attention {
        select_lane(gb, 0, format!("blk.{i}.q_proj"))
    } else {
        q_full
    };
    let q_gate = gated_attention.then(|| select_lane(gb, 1, format!("blk.{i}.q_gate_proj")));
    let k = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[kv_stride as u64])?,
        &[wk, x_norm],
        format!("blk.{i}.k_proj"),
    );
    let v = if value_reuses_key {
        k
    } else {
        gb.op(
            OpKind::MatMul,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[kv_stride as u64])?,
            &[wv.expect("value projection is present"), x_norm],
            format!("blk.{i}.v_proj"),
        )
    };

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
            OpAttrs {
                eps: spec.norm_eps,
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                ..Default::default()
            },
            DType::F32,
            Shape::new(&[q_stride as u64])?,
            &[q, q_norm_w],
            format!("blk.{i}.q_norm"),
        );
        let kn = gb.op(
            OpKind::RmsNorm,
            OpAttrs {
                eps: spec.norm_eps,
                n_heads: n_kv as u32,
                head_dim: head_dim as u32,
                ..Default::default()
            },
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
        OpAttrs {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            rope_pairing: u32::from(matches!(
                spec.rope_pairing,
                crate::probe::RopePairing::SplitHalf
            )),
            position: 0,
            rope_base: spec.rope_base,
            ..Default::default()
        },
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_for_rope, inv_freqs],
        format!("blk.{i}.q_rope"),
    );
    let k_rot = gb.op(
        OpKind::Rope,
        OpAttrs {
            n_heads: n_kv as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            rope_pairing: u32::from(matches!(
                spec.rope_pairing,
                crate::probe::RopePairing::SplitHalf
            )),
            position: 0,
            rope_base: spec.rope_base,
            // Snapshot pre-RoPE K for sequence-state policies (TriAttention).
            layer_ord: attn_ord as u32,
            custom_op_id: 1, // 1 = store pre-RoPE key before rotation
            ..Default::default()
        },
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
            ..attention_attrs(spec, n_kv, head_dim)
        },
        DType::F32,
        Shape::new(&[q_stride as u64])?,
        &[q_rot, k_cache_updated, v_cache_updated],
        format!("blk.{i}.attn"),
    );
    let attn_out = if let Some(gate) = q_gate {
        gb.op(
            OpKind::SigmoidGate,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[q_stride as u64])?,
            &[attn_out, gate],
            format!("blk.{i}.attention_gate"),
        )
    } else {
        attn_out
    };

    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[wo, attn_out, residual],
        format!("blk.{i}.o_proj_residual"),
    ))
}

fn build_dense_ffn(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    residual: NodeId,
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
    let gated = gb.op(
        OpKind::GateUpSwiGlu,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_ff as u64])?,
        &[w_gate, w_up, x],
        format!("blk.{i}.gate_up_swiglu"),
    );
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[w_down, gated, residual],
        format!("blk.{i}.ffn_down_residual"),
    ))
}

fn build_moe(
    gb: &mut GraphBuilder,
    gguf: &GgufFile,
    spec: &ModelSpec,
    layer: &crate::probe::LayerSpec,
    x: NodeId,
    token_id: NodeId,
    prefix: &str,
) -> Result<NodeId> {
    let gate_inp = gb.constant(
        format!("{prefix}.ffn_gate_inp"),
        gguf.tensor(&format!("{prefix}.ffn_gate_inp.weight"))?,
    );
    let mut inputs = vec![x, gate_inp];

    if gguf.has_tensor(&format!("{prefix}.ffn_gate_up_exps.weight")) {
        inputs.push(gb.constant(
            format!("{prefix}.ffn_gate_up_exps"),
            gguf.tensor(&format!("{prefix}.ffn_gate_up_exps.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.ffn_down_exps"),
            gguf.tensor(&format!("{prefix}.ffn_down_exps.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.ffn_gate"),
            gguf.tensor(&format!("{prefix}.ffn_gate.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.ffn_up"),
            gguf.tensor(&format!("{prefix}.ffn_up.weight"))?,
        ));
        inputs.push(gb.constant(
            format!("{prefix}.ffn_down"),
            gguf.tensor(&format!("{prefix}.ffn_down.weight"))?,
        ));
    } else {
        for name in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            inputs.push(gb.constant(
                format!("{prefix}.{name}"),
                gguf.tensor(&format!("{prefix}.{name}.weight"))?,
            ));
        }
        if gguf.has_tensor(&format!("{prefix}.ffn_gate_shexp.weight")) {
            for name in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"] {
                inputs.push(gb.constant(
                    format!("{prefix}.{name}"),
                    gguf.tensor(&format!("{prefix}.{name}.weight"))?,
                ));
            }
        }
    }
    let hash_map = gguf.has_tensor(&format!("{prefix}.ffn_gate_tid2eid.weight"));
    if spec.use_expert_bias && !hash_map && gguf.has_tensor(&format!("{prefix}.exp_probs_b.bias")) {
        inputs.push(gb.constant(
            format!("{prefix}.exp_probs_b"),
            gguf.tensor(&format!("{prefix}.exp_probs_b.bias"))?,
        ));
    }
    if hash_map {
        inputs.push(gb.constant(
            format!("{prefix}.ffn_gate_tid2eid"),
            gguf.tensor(&format!("{prefix}.ffn_gate_tid2eid.weight"))?,
        ));
        inputs.push(token_id);
    }
    Ok(gb.op(
        OpKind::MoE,
        moe_attrs(spec),
        DType::F32,
        Shape::new(&[spec.d_model as u64])?,
        &inputs,
        format!("{prefix}.moe"),
    ))
}

fn norm_attrs(spec: &ModelSpec) -> OpAttrs {
    OpAttrs {
        eps: spec.norm_eps,
        ..Default::default()
    }
}

fn attention_attrs(spec: &ModelSpec, n_kv: usize, head_dim: usize) -> OpAttrs {
    OpAttrs {
        n_heads: spec.n_heads as u32,
        n_kv_heads: n_kv as u32,
        head_dim: head_dim as u32,
        past_len: 0,
        scale: 1.0 / (head_dim as f32).sqrt(),
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

fn make_f32_matrix_tensor(data: &[f32], rows: usize, cols: usize) -> fellm_core::tensor::Tensor {
    use fellm_core::shape::{Layout, Shape as FShape};
    use fellm_core::storage::{AlignedBuffer, Storage};
    use std::sync::Arc;

    assert_eq!(data.len(), rows.saturating_mul(cols));
    let mut buf = AlignedBuffer::new_zeroed(data.len() * 4, 64);
    let dst: &mut [f32] = bytemuck::cast_slice_mut(buf.as_mut_slice());
    dst.copy_from_slice(data);
    let shape = FShape::new(&[rows as u64, cols as u64]).expect("valid matrix shape");
    let layout = Layout::contiguous(DType::F32, shape);
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    fellm_core::tensor::Tensor::from_storage(layout, storage)
}

/// One-token DeepSeek DSpark draft graph using `mtp.*` stages from `support`
/// and the target checkpoint's embedding / LM head.
pub fn build_dspark_proposal_graph(
    support: &GgufFile,
    target: &GgufFile,
    spec: &ModelSpec,
    stages: usize,
) -> Result<Graph> {
    let template = spec
        .layers
        .iter()
        .rev()
        .find(|layer| matches!(layer.ffn, crate::probe::FfnKind::MoE))
        .cloned()
        .or_else(|| spec.layers.first().cloned())
        .ok_or_else(|| fellm_core::error::FellmError::other("DSpark target has no layers"))?;
    let d_model = spec.d_model;
    let vocab = spec.vocab_size;
    let mut gb = GraphBuilder::new();
    let tok_embd = gb.constant("token_embd", target.tensor("token_embd.weight")?);
    let output_w = gb.constant(
        "output_w",
        target.tensor(if spec.tied_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        })?,
    );
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
        "dspark_tok_embed",
    );
    if support.has_tensor("mtp.0.main_proj.weight") {
        let feature_width = d_model.saturating_mul(3);
        let features = gb.input(
            "target_features",
            DType::F32,
            Shape::new(&[feature_width as u64])?,
        );
        let main_proj = gb.constant(
            "mtp.0.main_proj",
            support.tensor("mtp.0.main_proj.weight")?,
        );
        let mut projected = gb.op(
            OpKind::MatMul,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[main_proj, features],
            "dspark_main_proj",
        );
        if support.has_tensor("mtp.0.main_norm.weight") {
            let main_norm = gb.constant(
                "mtp.0.main_norm",
                support.tensor("mtp.0.main_norm.weight")?,
            );
            projected = gb.op(
                OpKind::RmsNorm,
                norm_attrs(spec),
                DType::F32,
                Shape::new(&[d_model as u64])?,
                &[projected, main_norm],
                "dspark_main_norm",
            );
        }
        x = gb.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x, projected],
            "dspark_fused_embed",
        );
    }
    if spec.hc_mult > 1 {
        let streams = vec![x; spec.hc_mult];
        x = gb.op(
            OpKind::Concat,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[(d_model * spec.hc_mult) as u64])?,
            &streams,
            "dspark_hc_embed",
        );
    }
    for stage in 0..stages {
        let mut layer = template.clone();
        layer.index = stage;
        layer.attn_ordinal = Some(stage);
        layer.compress_ratio = 0;
        x = build_layer(
            &mut gb,
            support,
            spec,
            &layer,
            x,
            inv_freqs,
            inv_freqs,
            token_id,
            &format!("mtp.{stage}"),
        )?;
    }
    if spec.hc_mult > 1 && support.has_tensor("mtp.2.hc_head_fn.weight") {
        let fn_w = gb.constant("hc_head_fn", support.tensor("mtp.2.hc_head_fn.weight")?);
        let scale = gb.constant(
            "hc_head_scale",
            support.tensor("mtp.2.hc_head_scale.weight")?,
        );
        let base = gb.constant("hc_head_base", support.tensor("mtp.2.hc_head_base.weight")?);
        x = gb.op(
            OpKind::HyperConnection,
            hc_attrs(spec, 2),
            DType::F32,
            Shape::new(&[d_model as u64])?,
            &[x, fn_w, scale, base],
            "dspark_hc_head",
        );
    }
    let last_norm = if support.has_tensor("mtp.2.norm.weight") {
        support.tensor("mtp.2.norm.weight")?
    } else {
        target.tensor("output_norm.weight")?
    };
    let norm_w = gb.constant("dspark_norm", last_norm);
    let normalized = gb.op(
        OpKind::RmsNorm,
        norm_attrs(spec),
        DType::F32,
        Shape::new(&[d_model as u64])?,
        &[x, norm_w],
        "dspark_final_norm",
    );
    gb.mark_output("draft_hidden", normalized);
    let logits = gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[vocab as u64])?,
        &[output_w, normalized],
        "dspark_lm_head",
    );
    gb.mark_output("logits", logits);
    gb.build()
}
