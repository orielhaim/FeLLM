//! Probe GGUF metadata + tensor names into a model recipe.
//!
//! `general.architecture` is used only as a metadata key prefix. Layer topology
//! is inferred from which `blk.N.*` tensors exist.

use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;
use fellm_gguf::meta::MetaValue;

/// Kind of RoPE frequency scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    /// No scaling.
    None,
    /// Linear scaling.
    Linear,
    /// Llama-3 piecewise interpolation.
    Llama3,
}

/// Mix (token-mixing) block for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixKind {
    /// Multi-head / GQA attention.
    Attention {
        /// KV heads for this layer.
        n_kv_heads: usize,
        /// Whether `attn_q_norm` / `attn_k_norm` tensors exist.
        qk_norm: bool,
    },
    /// Short convolution recurrent mix.
    ShortConv,
}

/// Feed-forward block for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    /// Dense SwiGLU FFN.
    Dense,
    /// MoE routed experts.
    MoE,
}

/// One transformer / hybrid block.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Block index in the GGUF (`blk.{i}`).
    pub index: usize,
    /// Ordinal among attention layers only (`k_in_{ord}`), if attention.
    pub attn_ordinal: Option<usize>,
    /// Ordinal among ShortConv layers only (`conv_in_{ord}`), if ShortConv.
    pub conv_ordinal: Option<usize>,
    /// Mix block kind.
    pub mix: MixKind,
    /// FFN block kind.
    pub ffn: FfnKind,
}

/// Architecture-agnostic model recipe derived from a GGUF file.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// GGUF `general.architecture` string (metadata prefix only).
    pub arch_id: String,
    /// Number of blocks.
    pub n_layers: usize,
    /// Hidden size.
    pub d_model: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Default KV heads (uniform attention models).
    pub n_kv_heads: usize,
    /// Per-head dim (`d_model / n_heads`).
    pub head_dim: usize,
    /// Dense FFN hidden dim.
    pub dense_ffn_dim: usize,
    /// MoE expert FFN dim.
    pub expert_ffn_dim: usize,
    /// Expert count (0 if unused).
    pub n_experts: usize,
    /// Experts used per token.
    pub n_expert_used: usize,
    /// MoE gating: 1 = softmax, 2 = sigmoid.
    pub expert_gating_func: u32,
    /// Renormalize MoE top-k probs.
    pub norm_topk_prob: bool,
    /// MoE route scale.
    pub routed_scaling_factor: f32,
    /// Expert bias tensor present on MoE layers.
    pub use_expert_bias: bool,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base.
    pub rope_base: f32,
    /// RoPE rotation dim.
    pub rope_dim: usize,
    /// RoPE scaling.
    pub rope_scaling_type: RopeScalingType,
    /// RoPE scale factor.
    pub rope_scaling_factor: f32,
    /// Original ctx for Llama3 scaling.
    pub rope_original_ctx: u32,
    /// Llama3 low-freq factor.
    pub rope_low_freq_factor: f32,
    /// Llama3 high-freq factor.
    pub rope_high_freq_factor: f32,
    /// Trained / GGUF max context length.
    pub context_length: usize,
    /// Vocab size.
    pub vocab_size: usize,
    /// Tied embeddings (no `output.weight`).
    pub tied_embeddings: bool,
    /// ShortConv window (`l_cache`); 0 if unused.
    pub shortconv_l_cache: usize,
    /// Per-layer recipe.
    pub layers: Vec<LayerSpec>,
}

impl ModelSpec {
    /// Probe a GGUF file into a [`ModelSpec`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch_id = gguf.metadata.arch()?.to_string();
        let p = arch_id.as_str();
        let m = &gguf.metadata;

        let n_layers = m.get_u32(&format!("{p}.block_count"))? as usize;
        let d_model = m.get_u32(&format!("{p}.embedding_length"))? as usize;
        let n_heads = m.get_u32(&format!("{p}.attention.head_count"))? as usize;
        if n_heads == 0 {
            return Err(FellmError::other("attention.head_count is 0"));
        }
        let head_dim = d_model / n_heads;

        let kv_meta = read_kv_heads(m, p, n_layers, n_heads)?;
        let n_kv_heads = kv_meta.iter().copied().find(|&n| n > 0).unwrap_or(n_heads);

        let dense_ffn_dim = m.get_u32(&format!("{p}.feed_forward_length")).unwrap_or(0) as usize;
        let n_experts = m.get_u32(&format!("{p}.expert_count")).unwrap_or(0) as usize;
        let n_expert_used = m.get_u32(&format!("{p}.expert_used_count")).unwrap_or(0) as usize;
        let expert_ffn_dim = m
            .get_u32(&format!("{p}.expert_feed_forward_length"))
            .unwrap_or(0) as usize;
        let expert_gating_func = parse_gating_func(
            m.get(&format!("{p}.expert_gating_func")),
            m.get_u32(&format!("{p}.expert_gating_func")).ok(),
        );
        let leading_dense = m
            .get_u32(&format!("{p}.leading_dense_block_count"))
            .unwrap_or(0) as usize;
        let norm_topk_prob = get_boolish(m.get(&format!("{p}.norm_topk_prob"))).unwrap_or(true);
        let routed_scaling_factor = m
            .get_f32(&format!("{p}.routed_scaling_factor"))
            .unwrap_or(1.0);
        let use_expert_bias =
            (0..n_layers).any(|i| gguf.has_tensor(&format!("blk.{i}.exp_probs_b.bias")));

        let norm_eps = m
            .get_f32(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_base = m.get_f32(&format!("{p}.rope.freq_base")).unwrap_or(10000.0);
        let rope_dim = m
            .get_u32(&format!("{p}.rope.dimension_count"))
            .map(|x| x as usize)
            .unwrap_or(head_dim);
        let context_length = m.get_u32(&format!("{p}.context_length")).unwrap_or(4096) as usize;
        let vocab_size = m
            .get_u32(&format!("{p}.vocab_size"))
            .map(|x| x as usize)
            .unwrap_or_else(|_| {
                m.get_string_array("tokenizer.ggml.tokens")
                    .map(<[String]>::len)
                    .unwrap_or(0)
            });
        let tied_embeddings = !gguf.has_tensor("output.weight");
        let shortconv_l_cache = m.get_u32(&format!("{p}.shortconv.l_cache")).unwrap_or(0) as usize;

        let (rope_scaling_type, rope_scaling_factor, rope_original_ctx, rope_low, rope_high) =
            read_rope_scaling(m, p);

        let mut layers = Vec::with_capacity(n_layers);
        let mut attn_ord = 0usize;
        let mut conv_ord = 0usize;

        for i in 0..n_layers {
            let has_attn = gguf.has_tensor(&format!("blk.{i}.attn_q.weight"));
            let has_shortconv = gguf.has_tensor(&format!("blk.{i}.shortconv.in_proj.weight"));
            let has_dense = gguf.has_tensor(&format!("blk.{i}.ffn_gate.weight"));
            let has_moe = gguf.has_tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                || gguf.has_tensor(&format!("blk.{i}.ffn_gate_exps.weight"));

            let mix = if has_shortconv && !has_attn {
                MixKind::ShortConv
            } else if has_attn {
                let n_kv = kv_meta.get(i).copied().unwrap_or(n_kv_heads).max(1);
                let qk_norm = gguf.has_tensor(&format!("blk.{i}.attn_q_norm.weight"));
                MixKind::Attention {
                    n_kv_heads: n_kv,
                    qk_norm,
                }
            } else if kv_meta.get(i).copied() == Some(0) && shortconv_l_cache > 0 {
                // Metadata says recurrent but tensors missing — still error clearly.
                return Err(FellmError::other(format!(
                    "layer {i}: metadata marks recurrent but no shortconv tensors found"
                )));
            } else {
                return Err(FellmError::other(format!(
                    "layer {i}: unknown mix block (no attn_q or shortconv tensors)"
                )));
            };

            // Prefer tensors; fall back to leading_dense_block_count for MoE models.
            let ffn = if has_moe {
                FfnKind::MoE
            } else if has_dense {
                FfnKind::Dense
            } else if n_experts > 0 && i >= leading_dense {
                FfnKind::MoE
            } else if dense_ffn_dim > 0 {
                FfnKind::Dense
            } else {
                return Err(FellmError::other(format!(
                    "layer {i}: unknown FFN block (no ffn_gate or MoE tensors)"
                )));
            };

            let (attn_ordinal, conv_ordinal) = match mix {
                MixKind::Attention { .. } => {
                    let o = Some(attn_ord);
                    attn_ord += 1;
                    (o, None)
                }
                MixKind::ShortConv => {
                    let o = Some(conv_ord);
                    conv_ord += 1;
                    (None, o)
                }
            };

            layers.push(LayerSpec {
                index: i,
                attn_ordinal,
                conv_ordinal,
                mix,
                ffn,
            });
        }

        if layers.iter().any(|l| matches!(l.mix, MixKind::ShortConv)) && shortconv_l_cache == 0 {
            return Err(FellmError::other(format!(
                "ShortConv layers present but {p}.shortconv.l_cache is missing/0"
            )));
        }

        Ok(Self {
            arch_id,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            dense_ffn_dim: if dense_ffn_dim > 0 { dense_ffn_dim } else { 0 },
            expert_ffn_dim,
            n_experts,
            n_expert_used,
            expert_gating_func,
            norm_topk_prob,
            routed_scaling_factor,
            use_expert_bias,
            norm_eps,
            rope_base,
            rope_dim,
            rope_scaling_type,
            rope_scaling_factor,
            rope_original_ctx,
            rope_low_freq_factor: rope_low,
            rope_high_freq_factor: rope_high,
            context_length,
            vocab_size,
            tied_embeddings,
            shortconv_l_cache,
            layers,
        })
    }

    /// Number of attention layers.
    #[must_use]
    pub fn n_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.mix, MixKind::Attention { .. }))
            .count()
    }

    /// Number of ShortConv layers.
    #[must_use]
    pub fn n_conv_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.mix, MixKind::ShortConv))
            .count()
    }

    /// Whether any layer uses ShortConv (hybrid state needed).
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        self.n_conv_layers() > 0
    }

    /// Per-layer KV head counts (`0` = ShortConv) for hybrid state allocation.
    #[must_use]
    pub fn layer_kv_heads_for_state(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|l| match l.mix {
                MixKind::Attention { n_kv_heads, .. } => n_kv_heads,
                MixKind::ShortConv => 0,
            })
            .collect()
    }
}

fn read_kv_heads(
    m: &fellm_gguf::meta::MetaMap,
    p: &str,
    n_layers: usize,
    n_heads: usize,
) -> Result<Vec<usize>> {
    let key = format!("{p}.attention.head_count_kv");
    if let Ok(arr) = m.get_i32_array(&key) {
        let v: Vec<usize> = arr.iter().map(|&x| x.max(0) as usize).collect();
        if v.len() == n_layers {
            return Ok(v);
        }
        if v.len() == 1 {
            return Ok(vec![v[0]; n_layers]);
        }
    }
    if let Ok(n) = m.get_u32(&key) {
        return Ok(vec![n as usize; n_layers]);
    }
    Ok(vec![n_heads; n_layers])
}

fn read_rope_scaling(
    m: &fellm_gguf::meta::MetaMap,
    p: &str,
) -> (RopeScalingType, f32, u32, f32, f32) {
    let scaling_type_str = m
        .get_string(&format!("{p}.rope.scaling.type"))
        .ok()
        .map(str::to_string);
    let factor = m
        .get_f32(&format!("{p}.rope.scaling.factor"))
        .or_else(|_| m.get_f32(&format!("{p}.rope.scale_linear")))
        .unwrap_or(1.0);
    let original_ctx = m
        .get_u32(&format!("{p}.rope.scaling.original_context_length"))
        .unwrap_or(0);
    let low = m
        .get_f32(&format!("{p}.rope.scaling.low_freq_factor"))
        .unwrap_or(1.0);
    let high = m
        .get_f32(&format!("{p}.rope.scaling.high_freq_factor"))
        .unwrap_or(4.0);

    let kind = match scaling_type_str.as_deref() {
        Some("llama3") => RopeScalingType::Llama3,
        Some("linear") => RopeScalingType::Linear,
        Some("none") | None => {
            if original_ctx > 0 && factor > 1.0 {
                RopeScalingType::Llama3
            } else {
                RopeScalingType::None
            }
        }
        _ => RopeScalingType::None,
    };
    (kind, factor, original_ctx, low, high)
}

fn get_boolish(v: Option<&MetaValue>) -> Option<bool> {
    match v {
        Some(MetaValue::Bool(x)) => Some(*x),
        Some(MetaValue::U32(x)) => Some(*x != 0),
        Some(MetaValue::U8(x)) => Some(*x != 0),
        Some(MetaValue::I32(x)) => Some(*x != 0),
        _ => None,
    }
}

fn parse_gating_func(v: Option<&MetaValue>, numeric: Option<u32>) -> u32 {
    if let Some(n) = numeric {
        return n;
    }
    match v {
        Some(MetaValue::String(s)) if s.eq_ignore_ascii_case("sigmoid") => 2,
        Some(MetaValue::String(s)) if s.eq_ignore_ascii_case("softmax") => 1,
        _ => 1,
    }
}
