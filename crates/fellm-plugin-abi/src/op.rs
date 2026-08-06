//! Op kind + attributes.

use fellm_core::dtype::DType;

/// Stable numeric operation identity.
///
/// Built-ins occupy the low range. Plugins use [`Self::custom`] to derive a
/// namespaced numeric id at graph-build time. The compiled graph and backend
/// registry carry only this integer, so the execution loop never performs a
/// string lookup. This is intentionally a transparent value rather than a
/// closed enum: adding an operation does not require editing the engine.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpKind(pub u32);

#[allow(non_upper_case_globals)]
impl OpKind {
    /// Element-wise add.
    pub const Add: Self = Self(0);
    /// Element-wise multiply.
    pub const Mul: Self = Self(1);
    /// Matmul: `C = A @ B^T` (weights row-major).
    pub const MatMul: Self = Self(2);
    /// RMSNorm with a learned weight vector.
    pub const RmsNorm: Self = Self(3);
    /// Rotary position embedding applied in-place to Q or K.
    pub const Rope: Self = Self(4);
    /// SiLU/Swish gate: `out = silu(gate) * up`.
    pub const SiluGate: Self = Self(5);
    /// Softmax (last dim), numerically stable, with optional causal mask.
    pub const Softmax: Self = Self(6);
    /// Full-attention over cached KV: takes Q, K-cache-view, V-cache-view.
    pub const Attention: Self = Self(7);
    /// Embedding lookup: gather rows from an embedding matrix by token id.
    pub const Embedding: Self = Self(8);
    /// Concatenate along the last dim (mostly for KV cache append).
    pub const Concat: Self = Self(9);
    /// Reshape (no-op if strides permit).
    pub const Reshape: Self = Self(10);
    /// Convert dtype.
    pub const Cast: Self = Self(11);
    /// Sample the next token from a logit vector.
    pub const Sample: Self = Self(12);
    /// Write a K/V row into the cache at `position`, in-place on the cache buffer.
    pub const KvWrite: Self = Self(13);
    /// Short convolution block used by LFM2-style decode.
    pub const ShortConv: Self = Self(14);
    /// Mixture-of-Experts feed-forward block.
    pub const MoE: Self = Self(15);
    /// Softmax-weighted embedding projection used by diffusion self-conditioning.
    pub const WeightedEmbedding: Self = Self(16);

    const CUSTOM_TAG: u32 = 0x8000_0000;

    /// Derive a stable namespaced plugin operation id using FNV-1a.
    #[must_use]
    pub fn custom(namespace: &str, name: &str) -> Self {
        let mut hash = 0x811c_9dc5u32;
        for byte in namespace
            .as_bytes()
            .iter()
            .chain(std::iter::once(&b':'))
            .chain(name.as_bytes())
        {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        Self(hash | Self::CUSTOM_TAG)
    }

    /// Raw C-ABI discriminant / plugin registry key.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Whether this is a plugin-defined operation.
    #[must_use]
    pub const fn is_custom(self) -> bool {
        self.0 & Self::CUSTOM_TAG != 0
    }
}

impl OpKind {
    /// Reconstruct from the C-ABI `u32` discriminant.
    #[must_use]
    pub fn from_u32(v: u32) -> Option<Self> {
        if v <= Self::WeightedEmbedding.raw() || v & Self::CUSTOM_TAG != 0 {
            Some(Self(v))
        } else {
            None
        }
    }

    /// A stable string name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::MatMul => "matmul",
            Self::RmsNorm => "rmsnorm",
            Self::Rope => "rope",
            Self::SiluGate => "silu_gate",
            Self::Softmax => "softmax",
            Self::Attention => "attention",
            Self::Embedding => "embedding",
            Self::Concat => "concat",
            Self::Reshape => "reshape",
            Self::Cast => "cast",
            Self::Sample => "sample",
            Self::KvWrite => "kv_write",
            Self::ShortConv => "shortconv",
            Self::MoE => "moe",
            Self::WeightedEmbedding => "weighted_embedding",
            _ if self.is_custom() => "custom",
            _ => "unknown",
        }
    }
}

/// Attributes attached to an op node (small, `Copy`-friendly).
///
/// `#[repr(C)]` so the layout is stable across the dynamic plugin boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OpAttrs {
    /// RMSNorm epsilon.
    pub eps: f32,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// RoPE dimension (number of head dims that get rotated).
    pub rope_dim: u32,
    /// Number of attention heads.
    pub n_heads: u32,
    /// Number of KV heads (GQA).
    pub n_kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Current position (for RoPE / causal masking).
    pub position: u32,
    /// Cached-sequence length (for attention: how many past tokens exist).
    pub past_len: u32,
    /// Softmax scale.
    pub scale: f32,
    /// Cast: destination dtype (as `u32` from `DType`).
    pub cast_to: u32,
    /// Sampling: temperature.
    pub temperature: f32,
    /// Sampling: top-k (0 = disabled).
    pub top_k: u32,
    /// Sampling: top-p (>= 1.0 = disabled).
    pub top_p: f32,
    /// Sampling: RNG seed.
    pub seed: u64,
    /// MoE: number of experts.
    pub n_experts: u32,
    /// MoE: number of experts selected per token.
    pub n_expert_used: u32,
    /// MoE: gating function (1 = softmax, 2 = sigmoid).
    pub expert_gating_func: u32,
    /// MoE: multiplier applied to selected expert weights.
    pub routed_scaling_factor: f32,
    /// MoE: whether selected top-k probabilities are renormalized.
    pub norm_topk_prob: u32,
    /// ShortConv: convolution window length.
    pub shortconv_l_cache: u32,
    /// Shared model embedding dimension.
    pub n_embd: u32,
    /// Paged KV: block size in tokens (`0` = contiguous legacy layout).
    pub block_size: u32,
    /// Paged KvWrite: `0` = K row, `1` = V row.
    pub kv_slot: u32,
    /// Attention / KvWrite layer ordinal (paged path).
    pub layer_ord: u32,
    /// Attention mode: `0` causal one-step, `1` bidirectional canvas.
    pub attention_mode: u32,
    /// Sliding-window size (`0` = unrestricted).
    pub attention_window: u32,
    /// Query row count for batched attention.
    pub query_len: u32,
    /// Explicit KV row count for batched attention.
    pub kv_len: u32,
    /// Prefix row count visible to a canvas query.
    pub prefix_len: u32,
    /// Final-logit soft cap (`0` = disabled).
    pub softcap: f32,
    /// Namespaced custom operation id for extension dispatch.
    pub custom_op_id: u32,
}

impl OpAttrs {
    /// Helper: destination dtype for a Cast op.
    #[must_use]
    pub fn cast_dtype(&self) -> Option<DType> {
        DType::from_ggml_code(self.cast_to).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::OpKind;

    #[test]
    fn custom_ops_are_namespaced_numeric_ids() {
        let a = OpKind::custom("example", "fused_bias");
        let b = OpKind::custom("other", "fused_bias");
        assert!(a.is_custom());
        assert_ne!(a, b);
        assert_eq!(OpKind::from_u32(a.raw()), Some(a));
        assert_eq!(a.name(), "custom");
        assert!(OpKind::from_u32(17).is_none());
    }
}
