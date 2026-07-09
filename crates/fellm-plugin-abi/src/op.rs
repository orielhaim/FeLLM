//! Op kind + attributes.

use fellm_core::dtype::DType;

/// The set of operations FeLLM knows about.
///
/// The core dispatches by matching `(OpKind, input dtypes)` in the backend's
/// kernel registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Element-wise add.
    Add,
    /// Element-wise multiply.
    Mul,
    /// Matmul: `C = A @ B^T` (weights row-major).
    MatMul,
    /// RMSNorm with a learned weight vector.
    RmsNorm,
    /// Rotary position embedding applied in-place to Q or K.
    Rope,
    /// SiLU/Swish gate: `out = silu(gate) * up`.
    SiluGate,
    /// Softmax (last dim), numerically stable, with optional causal mask.
    Softmax,
    /// Full-attention over cached KV: takes Q, K-cache-view, V-cache-view.
    Attention,
    /// Embedding lookup: gather rows from an embedding matrix by token id.
    Embedding,
    /// Concatenate along the last dim (mostly for KV cache append).
    Concat,
    /// Reshape (no-op if strides permit).
    Reshape,
    /// Convert dtype.
    Cast,
    /// Sample the next token from a logit vector.
    Sample,
    /// Write a K/V row into the cache at `position`, in-place on the cache buffer.
    KvWrite,
    /// Short convolution block used by LFM2-style decode.
    ShortConv,
    /// Mixture-of-Experts feed-forward block.
    MoE,
}

impl OpKind {
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
        }
    }
}

/// Attributes attached to an op node (small, `Copy`-friendly).
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
}

impl OpAttrs {
    /// Helper: destination dtype for a Cast op.
    #[must_use]
    pub fn cast_dtype(&self) -> Option<DType> {
        DType::from_ggml_code(self.cast_to).ok()
    }
}
