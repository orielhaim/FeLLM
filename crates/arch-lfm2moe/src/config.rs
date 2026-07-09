//! LFM2 MoE config extracted from GGUF metadata.

use fellm_core::error::Result;
use fellm_gguf::GgufFile;
use fellm_gguf::meta::MetaValue;

/// Config values needed to build an LFM2 MoE one-token graph.
#[derive(Debug, Clone)]
pub struct Lfm2MoeConfig {
    /// Number of blocks.
    pub n_layers: usize,
    /// Model hidden dim.
    pub d_model: usize,
    /// Trained context length.
    pub context_length: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Per-layer KV heads. `0` marks recurrent ShortConv layers.
    pub layer_kv_heads: Vec<usize>,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// Expert count.
    pub n_experts: usize,
    /// Experts selected per token.
    pub n_expert_used: usize,
    /// MoE expert hidden dim.
    pub expert_ffn_dim: usize,
    /// MoE gating function as backend enum.
    pub expert_gating_func: u32,
    /// Dense FFN hidden dim.
    pub dense_ffn_dim: usize,
    /// Number of leading dense FFN blocks.
    pub leading_dense_block_count: usize,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// ShortConv window length.
    pub shortconv_l_cache: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// True if `output.weight` is tied to `token_embd.weight`.
    pub tied_embeddings: bool,
    /// Whether selected MoE probabilities are renormalized.
    pub norm_topk_prob: bool,
    /// Multiplier applied to selected MoE routes.
    pub routed_scaling_factor: f32,
    /// Whether expert bias tensor exists.
    pub use_expert_bias: bool,
}

impl Lfm2MoeConfig {
    /// Extract from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let m = &gguf.metadata;
        let n_layers = m.get_u32("lfm2moe.block_count")? as usize;
        let d_model = m.get_u32("lfm2moe.embedding_length")? as usize;
        let context_length = m.get_u32("lfm2moe.context_length")? as usize;
        let vocab_size = m.get_u32("lfm2moe.vocab_size").unwrap_or_else(|_| {
            m.get_string_array("tokenizer.ggml.tokens")
                .map(<[String]>::len)
                .unwrap_or(0) as u32
        }) as usize;
        let n_heads = m.get_u32("lfm2moe.attention.head_count")? as usize;
        let layer_kv_heads = m
            .get_i32_array("lfm2moe.attention.head_count_kv")?
            .iter()
            .map(|&x| x.max(0) as usize)
            .collect::<Vec<_>>();
        let norm_eps = m
            .get_f32("lfm2moe.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let n_experts = m.get_u32("lfm2moe.expert_count")? as usize;
        let n_expert_used = m.get_u32("lfm2moe.expert_used_count")? as usize;
        let expert_ffn_dim = m.get_u32("lfm2moe.expert_feed_forward_length")? as usize;
        let expert_gating_func = parse_gating_func(
            m.get("lfm2moe.expert_gating_func"),
            m.get_u32("lfm2moe.expert_gating_func").ok(),
        );
        let dense_ffn_dim = m.get_u32("lfm2moe.feed_forward_length")? as usize;
        let leading_dense_block_count =
            m.get_u32("lfm2moe.leading_dense_block_count").unwrap_or(0) as usize;
        let rope_base = m.get_f32("lfm2moe.rope.freq_base").unwrap_or(10000.0);
        let shortconv_l_cache = m.get_u32("lfm2moe.shortconv.l_cache")? as usize;
        let head_dim = d_model / n_heads;
        let tied_embeddings = !gguf.has_tensor("output.weight");
        let norm_topk_prob = get_boolish(m.get("lfm2moe.norm_topk_prob")).unwrap_or(true);
        let routed_scaling_factor = m.get_f32("lfm2moe.routed_scaling_factor").unwrap_or(1.0);
        let use_expert_bias = gguf.has_tensor("blk.2.exp_probs_b.bias")
            || (0..n_layers).any(|i| gguf.has_tensor(&format!("blk.{i}.exp_probs_b.bias")));

        Ok(Self {
            n_layers,
            d_model,
            context_length,
            vocab_size,
            n_heads,
            layer_kv_heads,
            norm_eps,
            n_experts,
            n_expert_used,
            expert_ffn_dim,
            expert_gating_func,
            dense_ffn_dim,
            leading_dense_block_count,
            rope_base,
            shortconv_l_cache,
            head_dim,
            tied_embeddings,
            norm_topk_prob,
            routed_scaling_factor,
            use_expert_bias,
        })
    }

    /// Whether a block is recurrent ShortConv.
    #[must_use]
    pub fn is_recurrent(&self, layer: usize) -> bool {
        self.layer_kv_heads[layer] == 0
    }

    /// Whether a block uses MoE FFN.
    #[must_use]
    pub fn is_moe(&self, layer: usize) -> bool {
        layer >= self.leading_dense_block_count
    }

    /// Number of attention layers.
    #[must_use]
    pub fn n_attn_layers(&self) -> usize {
        self.layer_kv_heads.iter().filter(|&&n| n > 0).count()
    }

    /// Number of recurrent ShortConv layers.
    #[must_use]
    pub fn n_conv_layers(&self) -> usize {
        self.layer_kv_heads.iter().filter(|&&n| n == 0).count()
    }

    /// Attention ordinal for a block, if it is attention.
    #[must_use]
    pub fn attn_ordinal(&self, layer: usize) -> Option<usize> {
        (!self.is_recurrent(layer)).then(|| {
            self.layer_kv_heads[..layer]
                .iter()
                .filter(|&&n| n > 0)
                .count()
        })
    }

    /// ShortConv ordinal for a block, if it is recurrent.
    #[must_use]
    pub fn conv_ordinal(&self, layer: usize) -> Option<usize> {
        self.is_recurrent(layer).then(|| {
            self.layer_kv_heads[..layer]
                .iter()
                .filter(|&&n| n == 0)
                .count()
        })
    }

    /// Precompute inverse frequencies for RoPE.
    #[must_use]
    pub fn compute_rope_inv_freqs(&self) -> Vec<f32> {
        let half = self.head_dim / 2;
        let mut out = Vec::with_capacity(half);
        for i in 0..half {
            let exp = -((2 * i) as f32) / self.head_dim as f32;
            out.push(self.rope_base.powf(exp));
        }
        out
    }
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
