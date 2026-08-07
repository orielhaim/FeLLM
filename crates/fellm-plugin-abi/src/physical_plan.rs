//! Backend physical-plan contract.
//!
//! Semantic graphs cross this boundary once, during model compilation.  The
//! steady-state executor uses stable allocation ids and device addresses; it
//! never rediscovers device storage through a host pointer.

use fellm_core::dtype::DType;

/// Stable tensor identifier within one compiled physical plan.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanTensorId(pub u32);

/// Stable allocation identifier owned by a backend plan.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId(pub u32);

/// Raw accelerator address. It is meaningful only to the owning backend.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

/// Tensor storage lifetime selected during backend lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    /// Immutable, model-lifetime storage (normally packed weights).
    Model,
    /// Request-lifetime mutable state such as KV or recurrent state.
    Request,
    /// Reusable storage in the static execution arena.
    Transient,
    /// Small device-owned control or result storage.
    Control,
}

/// Backend-neutral tensor descriptor supplied to a plan compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanTensorDesc {
    /// Stable tensor id.
    pub id: PlanTensorId,
    /// Element type.
    pub dtype: DType,
    /// Row-major dimensions.
    pub shape: Vec<u64>,
    /// Byte alignment required by candidate kernels.
    pub alignment: usize,
    /// Storage lifetime.
    pub storage: StorageClass,
    /// First macro-operation that reads or writes this value.
    pub first_use: u32,
    /// Last macro-operation that reads this value.
    pub last_use: u32,
}

impl PlanTensorDesc {
    /// Checked storage size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> Option<usize> {
        let elements = self
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))?;
        self.dtype.byte_size(elements).into()
    }
}

/// A direct descriptor used by prepared kernels at run time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceTensor {
    /// Base device address including [`Self::offset_bytes`].
    pub ptr: DevicePtr,
    /// Owning allocation.
    pub allocation: AllocationId,
    /// Element type.
    pub dtype: DType,
    /// Dimensions.
    pub shape: Vec<u64>,
    /// Byte strides.
    pub strides: Vec<u64>,
    /// Offset from the allocation base.
    pub offset_bytes: usize,
}

/// Stable arena assignment produced by liveness-based memory planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSlot {
    /// Tensor assigned to this slot.
    pub tensor: PlanTensorId,
    /// Byte offset from the arena base.
    pub offset: usize,
    /// Reserved bytes.
    pub size: usize,
    /// Required alignment.
    pub alignment: usize,
    /// First operation using the slot.
    pub first_use: u32,
    /// Last operation using the slot.
    pub last_use: u32,
}

/// Device-resident parameters updated once per generation step.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceStepParams {
    /// Current token id.
    pub token_id: u32,
    /// Absolute token position.
    pub position: u32,
    /// Visible sequence length including the current token.
    pub sequence_length: u32,
    /// Active request batch.
    pub active_batch: u32,
    /// Physical KV block receiving the current row.
    pub kv_write_block: u32,
    /// Token slot inside that block.
    pub kv_write_slot: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Nucleus probability.
    pub top_p: f32,
    /// Top-k candidate count.
    pub top_k: u32,
    /// Number of recent token ids used by repetition adjustment.
    pub recent_count: u32,
    /// Deterministic RNG seed/state.
    pub seed: u64,
}

/// Stable numeric macro-operation identity.
///
/// Built-ins occupy the low range. Plugins derive namespaced ids via
/// [`Self::custom`]. Semantic attention requirements use names such as
/// [`Self::PagedAttentionDecode`] — never product/algorithm brands.
/// Prepared plans store this integer plus a `kernel_variant` handle; steady-
/// state execution never string-dispatches.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroOpKind(pub u32);

#[allow(non_upper_case_globals)]
impl MacroOpKind {
    /// Embedding lookup into the static activation arena.
    pub const Embedding: Self = Self(0);
    /// RMS normalization fused with Q8_1 activation preparation.
    pub const RmsNormQuantizeQ8_1: Self = Self(1);
    /// Packed Q/K/V quantized projection.
    pub const QkvMmvq: Self = Self(2);
    /// RoPE plus direct device KV commit.
    pub const RopeKvCommit: Self = Self(3);
    /// Paged attention decode (single-query). Provider selects the kernel.
    pub const PagedAttentionDecode: Self = Self(4);
    /// Contiguous (non-paged) attention decode.
    pub const ContiguousAttentionDecode: Self = Self(5);
    /// Paged attention prefill (multi-query).
    pub const PagedAttentionPrefill: Self = Self(6);
    /// Contiguous attention prefill.
    pub const ContiguousAttentionPrefill: Self = Self(7);
    /// Output projection with residual epilogue.
    pub const OutputProjectionResidual: Self = Self(8);
    /// Packed gate/up projection with SwiGLU epilogue.
    pub const GateUpMmvqSwiglu: Self = Self(9);
    /// Down projection with residual epilogue.
    pub const DownProjectionResidual: Self = Self(10);
    /// Device LM-head projection and sampling pipeline.
    pub const LmHeadSample: Self = Self(11);
    /// GPU routing, grouping, expert execution and scatter-add.
    pub const GroupedMoe: Self = Self(12);

    const CUSTOM_TAG: u32 = 0x8000_0000;

    /// Derive a stable namespaced plugin macro-op id (FNV-1a).
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

    /// Raw discriminant.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Plugin-defined macro-op.
    #[must_use]
    pub const fn is_custom(self) -> bool {
        self.0 & Self::CUSTOM_TAG != 0
    }

    /// Stable diagnostic name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::RmsNormQuantizeQ8_1 => "rmsnorm_quantize_q8_1",
            Self::QkvMmvq => "qkv_mmvq",
            Self::RopeKvCommit => "rope_kv_commit",
            Self::PagedAttentionDecode => "paged_attention_decode",
            Self::ContiguousAttentionDecode => "contiguous_attention_decode",
            Self::PagedAttentionPrefill => "paged_attention_prefill",
            Self::ContiguousAttentionPrefill => "contiguous_attention_prefill",
            Self::OutputProjectionResidual => "output_projection_residual",
            Self::GateUpMmvqSwiglu => "gate_up_mmvq_swiglu",
            Self::DownProjectionResidual => "down_projection_residual",
            Self::LmHeadSample => "lm_head_sample",
            Self::GroupedMoe => "grouped_moe",
            _ if self.is_custom() => "custom",
            _ => "unknown",
        }
    }
}

/// One fully resolved macro-operation. Kernel variant selection is complete.
#[derive(Debug, Clone)]
pub struct PreparedMacroOp {
    /// Macro-operation semantic kind (built-in or plugin custom).
    pub kind: MacroOpKind,
    /// Input tensors.
    pub inputs: Vec<PlanTensorId>,
    /// Output tensors.
    pub outputs: Vec<PlanTensorId>,
    /// Backend-owned prepared kernel variant id (direct dispatch; not a name).
    pub kernel_variant: u64,
    /// Prepared attention / policy provider id when applicable (`0` = none).
    pub provider_id: u64,
}

/// Device physical plan shared by prefill and decode plan implementations.
#[derive(Debug, Clone, Default)]
pub struct PhysicalPlan {
    /// Stable arena assignments.
    pub arena: Vec<ArenaSlot>,
    /// Total transient arena size.
    pub arena_bytes: usize,
    /// Prepared operations in launch order.
    pub operations: Vec<PreparedMacroOp>,
}
