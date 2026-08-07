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

/// Macro-operation understood by device plan compilers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroOpKind {
    /// Embedding lookup into the static activation arena.
    Embedding,
    /// RMS normalization fused with Q8_1 activation preparation.
    RmsNormQuantizeQ8_1,
    /// Packed Q/K/V quantized projection.
    QkvMmvq,
    /// RoPE plus direct device KV commit.
    RopeKvCommit,
    /// Tiled paged single-query attention.
    PagedFlashDecode,
    /// Output projection with residual epilogue.
    OutputProjectionResidual,
    /// Packed gate/up projection with SwiGLU epilogue.
    GateUpMmvqSwiglu,
    /// Down projection with residual epilogue.
    DownProjectionResidual,
    /// Device LM-head projection and sampling pipeline.
    LmHeadSample,
    /// GPU routing, grouping, expert execution and scatter-add.
    GroupedMoe,
}

/// One fully resolved macro-operation. Kernel variant selection is complete.
#[derive(Debug, Clone)]
pub struct PreparedMacroOp {
    /// Macro-operation semantic kind.
    pub kind: MacroOpKind,
    /// Input tensors.
    pub inputs: Vec<PlanTensorId>,
    /// Output tensors.
    pub outputs: Vec<PlanTensorId>,
    /// Backend-owned prepared kernel variant id.
    pub kernel_variant: u64,
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
