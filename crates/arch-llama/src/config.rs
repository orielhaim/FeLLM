//! Llama config extracted from GGUF metadata.

use fellm_core::error::Result;
use fellm_gguf::GgufFile;

/// Kind of RoPE frequency scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    /// No scaling.
    None,
    /// Linear scaling (Llama 2 style).
    Linear,
    /// Llama 3 style scaling (piecewise interpolation).
    Llama3,
}

/// Config values needed to build a Llama forward graph.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Model hidden dim.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of KV heads.
    pub n_kv_heads: usize,
    /// Feed-forward hidden dim.
    pub d_ff: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// RoPE rotation dimension.
    pub rope_dim: usize,
    /// Trained context length.
    pub context_length: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// True if `output.weight` is tied to `token_embd.weight`.
    pub tied_embeddings: bool,
    /// RoPE scaling type.
    pub rope_scaling_type: RopeScalingType,
    /// RoPE scaling factor (32.0 for Llama 3.2 1B).
    pub rope_scaling_factor: f32,
    /// Original context length before scaling (8192 for Llama 3.2).
    pub rope_original_ctx: u32,
    /// Low-freq factor for llama3 scaling.
    pub rope_low_freq_factor: f32,
    /// High-freq factor for llama3 scaling.
    pub rope_high_freq_factor: f32,
}

impl LlamaConfig {
    /// Per-head dimension (`d_model / n_heads`).
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

        let tied_embeddings = !gguf.has_tensor("output.weight");

        let scaling_type_str = m
            .get_string("llama.rope.scaling.type")
            .ok()
            .map(str::to_string);
        let rope_scaling_factor = m
            .get_f32("llama.rope.scaling.factor")
            .or_else(|_| m.get_f32("llama.rope.scale_linear"))
            .unwrap_or(1.0);
        let rope_original_ctx = m
            .get_u32("llama.rope.scaling.original_context_length")
            .unwrap_or(0);
        let rope_low_freq_factor = m
            .get_f32("llama.rope.scaling.low_freq_factor")
            .unwrap_or(1.0);
        let rope_high_freq_factor = m
            .get_f32("llama.rope.scaling.high_freq_factor")
            .unwrap_or(4.0);

        let rope_scaling_type = match scaling_type_str.as_deref() {
            Some("llama3") => RopeScalingType::Llama3,
            Some("linear") => RopeScalingType::Linear,
            Some("none") | None => {
                if rope_original_ctx > 0 && rope_scaling_factor > 1.0 {
                    // Heuristic: Llama 3.x GGUFs sometimes omit the type string.
                    RopeScalingType::Llama3
                } else {
                    RopeScalingType::None
                }
            }
            _ => RopeScalingType::None,
        };

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
            rope_scaling_type,
            rope_scaling_factor,
            rope_original_ctx,
            rope_low_freq_factor,
            rope_high_freq_factor,
        })
    }

    /// Precompute inverse frequencies for RoPE, applying any scaling.
    ///
    /// Returns a vector of length `rope_dim / 2` containing `inv_freq[i]` such
    /// that the rotation angle for position `p`, pair index `i` is
    /// `p * inv_freq[i]`.
    #[must_use]
    pub fn compute_rope_inv_freqs(&self) -> Vec<f32> {
        let half = self.rope_dim / 2;
        let mut out = Vec::with_capacity(half);
        for i in 0..half {
            let exp = -((2 * i) as f32) / self.rope_dim as f32;
            let base_freq = self.rope_base.powf(exp);
            let scaled = match self.rope_scaling_type {
                RopeScalingType::None => base_freq,
                RopeScalingType::Linear => base_freq / self.rope_scaling_factor,
                RopeScalingType::Llama3 => llama3_scale_freq(
                    base_freq,
                    self.rope_scaling_factor,
                    self.rope_low_freq_factor,
                    self.rope_high_freq_factor,
                    self.rope_original_ctx.max(1) as f32,
                ),
            };
            out.push(scaled);
        }
        out
    }
}

fn llama3_scale_freq(
    freq: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    old_context_len: f32,
) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let wavelen = two_pi / freq;
    let low_wavelen = old_context_len / low_freq_factor;
    let high_wavelen = old_context_len / high_freq_factor;
    if wavelen < high_wavelen {
        // High frequency: no change.
        freq
    } else if wavelen > low_wavelen {
        // Low frequency: divide by factor.
        freq / factor
    } else {
        // Smooth interpolation.
        let smooth =
            (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        (1.0 - smooth) * (freq / factor) + smooth * freq
    }
}
