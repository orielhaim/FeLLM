//! Op kind + attributes.

use fellm_core::dtype::DType;

/// The set of operations FeLLM knows about.
///
/// The core dispatches by matching `(OpKind, input dtypes)` in the backend's
/// kernel registry.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Element-wise add.
    Add = 0,
    /// Element-wise multiply.
    Mul = 1,
    /// Matmul: `C = A @ B^T` (weights row-major).
    MatMul = 2,
    /// RMSNorm with a learned weight vector.
    RmsNorm = 3,
    /// Rotary position embedding applied in-place to Q or K.
    Rope = 4,
    /// SiLU/Swish gate: `out = silu(gate) * up`.
    SiluGate = 5,
    /// Softmax (last dim), numerically stable, with optional causal mask.
    Softmax = 6,
    /// Full-attention over cached KV: takes Q, K-cache-view, V-cache-view.
    Attention = 7,
    /// Embedding lookup: gather rows from an embedding matrix by token id.
    Embedding = 8,
    /// Concatenate along the last dim (mostly for KV cache append).
    Concat = 9,
    /// Reshape (no-op if strides permit).
    Reshape = 10,
    /// Convert dtype.
    Cast = 11,
    /// Sample the next token from a logit vector.
    Sample = 12,
    /// Write a K/V row into the cache at `position`, in-place on the cache buffer.
    KvWrite = 13,
    /// Short convolution block used by LFM2-style decode.
    ShortConv = 14,
    /// Mixture-of-Experts feed-forward block.
    MoE = 15,
}

impl OpKind {
    /// Reconstruct from the C-ABI `u32` discriminant.
    #[must_use]
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Add,
            1 => Self::Mul,
            2 => Self::MatMul,
            3 => Self::RmsNorm,
            4 => Self::Rope,
            5 => Self::SiluGate,
            6 => Self::Softmax,
            7 => Self::Attention,
            8 => Self::Embedding,
            9 => Self::Concat,
            10 => Self::Reshape,
            11 => Self::Cast,
            12 => Self::Sample,
            13 => Self::KvWrite,
            14 => Self::ShortConv,
            15 => Self::MoE,
            _ => return None,
        })
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
}

impl OpAttrs {
    /// Helper: destination dtype for a Cast op.
    #[must_use]
    pub fn cast_dtype(&self) -> Option<DType> {
        DType::from_ggml_code(self.cast_to).ok()
    }
}
