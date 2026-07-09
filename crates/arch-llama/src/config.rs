//! Llama config extracted from GGUF metadata.

use fellm_core::error::Result;
use fellm_gguf::GgufFile;

/// Config values needed to build a Llama forward graph.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Model hidden dim.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of KV heads (== n_heads for pre-GQA models).
    pub n_kv_heads: usize,
    /// Feed-forward hidden dim.
    pub d_ff: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// RoPE rotation dimension (equal to head_dim in most llama configs).
    pub rope_dim: usize,
    /// Trained context length.
    pub context_length: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// True if `output.weight` is tied to `token_embd.weight`.
    pub tied_embeddings: bool,
}

impl LlamaConfig {
    /// Head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Extract from a GGUF metadata map.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let m = &gguf.metadata;
        let n_layers = m.get_u32("llama.block_count")? as usize;
        let d_model = m.get_u32("llama.embedding_length")? as usize;
        let n_heads = m.get_u32("llama.attention.head_count")? as usize;
        let n_kv_heads = m
            .get_u32("llama.attention.head_count_kv")
            .unwrap_or(n_heads as u32) as usize;
        let d_ff = m.get_u32("llama.feed_forward_length")? as usize;
        let norm_eps = m
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let rope_base = m.get_f32("llama.rope.freq_base").unwrap_or(10000.0);
        let head_dim = d_model / n_heads;
        let rope_dim = m
            .get_u32("llama.rope.dimension_count")
            .map(|x| x as usize)
            .unwrap_or(head_dim);
        let context_length = m.get_u32("llama.context_length").unwrap_or(4096) as usize;
        let vocab_size = m
            .get_string_array("tokenizer.ggml.tokens")
            .map(<[String]>::len)
            .unwrap_or(0);

        // Detect tied embeddings by checking whether `output.weight` exists.
        let tied_embeddings = !gguf.has_tensor("output.weight");

        Ok(Self {
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            d_ff,
            norm_eps,
            rope_base,
            rope_dim,
            context_length,
            vocab_size,
            tied_embeddings,
        })
    }
}
