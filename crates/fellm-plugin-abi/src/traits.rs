//! The Backend / Kernel / Architecture traits.
//!
//! In Phase 1 these are pure Rust traits (not `#[sabi_trait]`) because we
//! statically link. The types are already shaped so that switching to
//! `abi_stable::sabi_trait` in Phase 2 is mechanical.

use crate::op::OpKind;
use crate::{StreamHandle, TensorMut, TensorRef};
use fellm_core::dtype::DType;
use fellm_core::error::Result;
use std::any::Any;

/// Metadata and tensor inventory exposed to an architecture provider.
#[derive(Debug, Clone, Default)]
pub struct ModelSource {
    /// GGUF architecture identifier.
    pub architecture_id: String,
    /// Metadata represented as stable string values for the plugin boundary.
    pub metadata: Vec<(String, String)>,
    /// Tensor names and layouts represented as stable strings.
    pub tensors: Vec<(String, String)>,
}

/// Architecture configuration returned by probing a source.
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    /// Architecture identifier.
    pub architecture_id: String,
    /// Architecture-owned configuration payload.
    pub data: String,
}

/// Backend capabilities visible during graph compilation.
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    /// Diagnostic backend id. Architecture decisions must use `caps`, not this string.
    pub backend_id: String,
    /// Underlying backend capabilities.
    pub caps: BackendCaps,
}

/// Hardware family without naming a concrete backend implementation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DeviceKind {
    /// General-purpose host processor.
    #[default]
    Cpu,
    /// Discrete or integrated accelerator.
    Gpu,
}

/// Stable graph identifier within a compiled model program.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(pub u32);

/// A graph descriptor owned by the architecture provider.
#[derive(Debug, Clone)]
pub struct GraphSpec {
    /// Stable graph id.
    pub id: GraphId,
    /// Human-readable name.
    pub name: String,
}

/// Compiled architecture program.  Concrete graph plans remain runtime-owned;
/// this descriptor keeps the plugin plane independent from `fellm-graph`.
#[derive(Debug, Clone, Default)]
pub struct ModelProgram {
    /// Architecture id.
    pub architecture_id: String,
    /// Graphs required by this program.
    pub graphs: Vec<GraphSpec>,
}

/// Generation request passed to a provider.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Prompt token ids.
    pub prompt: Vec<u32>,
    /// Maximum emitted tokens.
    pub max_tokens: u32,
    /// Deterministic seed.
    pub seed: u64,
}

/// Named input binding requested by a generation driver.
#[derive(Debug, Clone)]
pub struct InputBinding {
    /// Binding name.
    pub name: String,
    /// Integer values for token/id bindings.
    pub values: Vec<u32>,
    /// F32 values for activation/state bindings.
    pub float_values: Vec<f32>,
}

/// State binding requested by a generation driver.
#[derive(Debug, Clone)]
pub struct StateBinding {
    /// Binding name.
    pub name: String,
}

/// Graph output delivered to a driver.
#[derive(Debug, Clone)]
pub struct GraphOutput {
    /// Output name.
    pub name: String,
    /// Row-major f32 values.
    pub values: Vec<f32>,
    /// Number of rows.
    pub rows: usize,
    /// Values per row.
    pub cols: usize,
}

/// Driver event delivered by the core scheduler.
#[derive(Debug, Clone)]
pub enum DriverEvent {
    /// Start a new request.
    Started,
    /// A requested graph completed.
    GraphCompleted {
        graph: GraphId,
        outputs: Vec<GraphOutput>,
    },
    /// A cache commit completed.
    CacheCommitted { token_count: usize },
    /// The request was cancelled.
    Cancelled,
}

/// Bindings for a graph invocation.
#[derive(Debug, Clone)]
pub struct InputBindings {
    /// Input values.
    pub inputs: Vec<InputBinding>,
}

/// Runtime state bindings for a graph invocation.
#[derive(Debug, Clone)]
pub struct StateBindings {
    /// State names.
    pub states: Vec<StateBinding>,
}

/// A cache transition requested by a driver.
#[derive(Debug, Clone)]
pub struct CacheCommit {
    /// Tokens to append causally.
    pub token_ids: Vec<u32>,
}

/// A token batch emitted by a driver.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    /// Emitted token ids.
    pub token_ids: Vec<u32>,
    /// Complete finalized block that must be appended to the causal cache
    /// before the driver starts its next diffusion block.  Keeping this in
    /// the action makes cache ownership explicit at the host boundary while
    /// allowing the visible batch to be capped by `max_tokens`.
    pub commit_token_ids: Vec<u32>,
}

/// Next action returned by a generation driver.
#[derive(Debug, Clone)]
pub enum DriverAction {
    /// Invoke one precompiled graph.
    InvokeGraph {
        graph: GraphId,
        inputs: InputBindings,
        state: StateBindings,
    },
    /// Commit completed tokens into the persistent causal cache.
    CommitCache(CacheCommit),
    /// Emit one or more finalized tokens.
    Emit(TokenBatch),
    /// Finish the request.
    Done,
}

/// Architecture-specific generation state machine.
pub trait GenerationDriver: Send {
    /// Advance the driver after a scheduler event.
    fn next_action(&mut self, event: DriverEvent) -> Result<DriverAction>;
}

/// Architecture provider contract.  Providers own probing, graph selection,
/// architecture state, and generation sequencing; the host still owns
/// backend kernels, memory, graph execution, and scheduling.
pub trait ArchitectureProvider: Send + Sync {
    /// Stable architecture identifier.
    fn architecture_id(&self) -> &str;
    /// Validate source metadata/tensors and return architecture config.
    fn probe(&self, source: &ModelSource) -> Result<ArchitectureConfig>;
    /// Compile the provider's multi-graph program.
    fn compile(
        &self,
        source: &ModelSource,
        config: &ArchitectureConfig,
        backend: &BackendCapabilities,
    ) -> Result<ModelProgram>;
    /// Create a generation state machine for a compiled program.
    fn create_generation_driver(
        &self,
        program: &ModelProgram,
        request: GenerationRequest,
    ) -> Result<Box<dyn GenerationDriver>>;
}

/// Opaque handle to a resolved kernel implementation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelHandle(pub u64);

/// Backend capabilities discovered at startup.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCaps {
    /// Broad hardware family.
    pub device_kind: DeviceKind,
    /// Widest SIMD vector width in f32 lanes.
    pub simd_f32_lanes: u32,
    /// True if AVX-512 available.
    pub has_avx512: bool,
    /// True if AVX2/FMA available.
    pub has_avx2: bool,
    /// True if NEON available.
    pub has_neon: bool,
    /// Physical cores.
    pub physical_cores: u32,
    /// Logical threads.
    pub logical_threads: u32,
    /// Persistent model/request tensors can remain device-resident.
    pub supports_persistent_device_state: bool,
    /// Backend can capture stable execution graphs.
    pub supports_graph_capture: bool,
    /// Launches and transfers can be enqueued without a global synchronization.
    pub supports_async_execution: bool,
    /// Attention can read immutable prefix KV while writing separate causal state.
    pub supports_read_only_prefix_kv: bool,
    /// Routed experts can execute as grouped device work.
    pub supports_grouped_moe: bool,
    /// Sampling and stopping reductions can remain on the device.
    pub supports_device_sampling: bool,
    /// Bidirectional attention is available.
    pub supports_bidirectional_attention: bool,
    /// Quantized matrix multiplication supports multiple input rows.
    pub supports_batched_quantized_gemm: bool,
    /// Namespaced semantic custom operations can be resolved at compile time.
    pub supports_custom_operations: bool,
    /// GPU compute capability major (`0` = unknown / CPU).
    pub compute_major: u32,
    /// GPU compute capability minor.
    pub compute_minor: u32,
    /// Shared memory per SM in bytes (`0` if unknown).
    pub smem_per_sm: u32,
    /// Hardware supports Ampere/Ada-class tensor-core style paths.
    pub has_ampere_ada_features: bool,
    /// Hardware supports Hopper-class async pipeline / TMA / WGMMA-class paths.
    pub has_hopper_features: bool,
    /// Hardware supports Blackwell-class features (reserved).
    pub has_blackwell_features: bool,
}

/// Current allocatable memory reported by a backend's physical device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceMemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl BackendCaps {
    /// Convert hardware flags into a [`crate::capability::FeatureSet`] for
    /// provider negotiation. Prefer this over stringly-typed GPU product names.
    #[must_use]
    pub fn feature_set(&self) -> crate::capability::FeatureSet {
        use crate::capability::{FeatureId, FeatureSet};
        let mut f = FeatureSet::new();
        if self.supports_bidirectional_attention {
            f.insert(FeatureId::ATTN_BIDIRECTIONAL);
        }
        f.insert(FeatureId::ATTN_CAUSAL);
        f.insert(FeatureId::ATTN_MHA);
        f.insert(FeatureId::ATTN_MQA);
        f.insert(FeatureId::ATTN_GQA);
        f.insert(FeatureId::ATTN_CONTIGUOUS_KV);
        f.insert(FeatureId::ATTN_PAGED_KV);
        f.insert(FeatureId::ATTN_PREFILL);
        f.insert(FeatureId::ATTN_DECODE);
        f.insert(FeatureId::ATTN_BATCHED_DECODE);
        f.insert(FeatureId::ATTN_FP16);
        f.insert(FeatureId::ATTN_BF16);
        f.insert(FeatureId::ATTN_SLIDING_WINDOW);
        f.insert(FeatureId::ATTN_INDIRECT_POSITIONS);
        f.insert(FeatureId::ATTN_PER_HEAD_KV_VIEWS);
        // Runtime sequence-state contracts available to all backends.
        f.insert(FeatureId::KV_LOGICAL_POSITIONS);
        f.insert(FeatureId::KV_PREFIX_PRIVATE_SPLIT);
        f.insert(FeatureId::KV_MUTABLE_REMAP);
        f.insert(FeatureId::KV_PHYSICAL_RECLAIM);
        if self.has_ampere_ada_features {
            f.insert(FeatureId::HW_AMPERE_ADA);
        }
        if self.has_hopper_features {
            f.insert(FeatureId::HW_HOPPER);
            f.insert(FeatureId::HW_ASYNC_PIPELINE);
            f.insert(FeatureId::HW_TMA_CLASS);
            f.insert(FeatureId::HW_WGMMA_CLASS);
        }
        if self.has_blackwell_features {
            f.insert(FeatureId::HW_BLACKWELL);
        }
        f
    }
}

/// A resolved kernel launch descriptor.
#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    /// The op.
    pub op: OpKind,
    /// Input dtypes.
    pub input_dtypes: Vec<DType>,
    /// Output dtype.
    pub output_dtype: DType,
    /// Opaque handle for the backend.
    pub handle: KernelHandle,
}

/// A compute backend.
pub trait Backend: Send + Sync + 'static {
    /// Stable id, e.g. `"cpu"`.
    fn id(&self) -> &'static str;

    /// Capabilities.
    fn capabilities(&self) -> BackendCaps;

    /// Live memory capacity for the device that owns model and KV allocations.
    fn memory_info(&self) -> Option<DeviceMemoryInfo> {
        None
    }

    /// Resolve a kernel.
    fn resolve_kernel(
        &self,
        op: OpKind,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<KernelDescriptor>;

    /// Launch a resolved kernel.
    fn launch(
        &self,
        handle: KernelHandle,
        attrs: &crate::op::OpAttrs,
        inputs: &[TensorRef],
        outputs: &mut [TensorMut],
        stream: StreamHandle,
    ) -> Result<()>;

    /// Schedule immutable weights before their execution group reaches the critical path.
    /// Backends without tiered residency may ignore this hint.
    fn prefetch_weight_group(
        &self,
        _group_id: u64,
        _weights: &[TensorRef],
        _required: bool,
    ) -> Result<()> {
        Ok(())
    }

    /// Mark the beginning of one forward step.
    ///
    /// Backends may use this boundary to reuse immutable per-step activation
    /// transforms. The default is a no-op for backends that do not need it.
    fn begin_step(&self) {}

    /// Mark the end of one forward step.
    ///
    /// The default is a no-op; CPU implementations use it to close any
    /// per-step scratch-cache lifetime.
    fn end_step(&self) {}

    /// Sample directly from a device-resident logit tensor when supported.
    /// Returns `None` when the requested sampling policy needs the host path.
    fn sample_device(
        &self,
        _logits: TensorRef,
        _attrs: &crate::op::OpAttrs,
    ) -> Result<Option<u32>> {
        Ok(None)
    }

    /// Make a device-resident tensor's host storage coherent on an explicit
    /// fallback/debug boundary.
    fn materialize(&self, _tensor: TensorRef, _host: TensorMut) -> Result<()> {
        Ok(())
    }

    /// Downcast to a concrete backend (CUDA graph / VRAM hooks).
    fn as_any(&self) -> &dyn Any;

    /// Synchronize the default compute stream (no-op on CPU).
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// A model architecture.
///
/// Phase 1 builds graphs via `fellm-model::ModelSpec` (GGUF probe + tensor
/// presence). This trait remains as the future plugin ABI surface for
/// architectures that cannot be inferred automatically.
///
/// Implementors are responsible for reading a GGUF file, extracting their
/// hyperparameters, and constructing a `fellm_graph::Graph` that can be
/// executed by a `Backend`. To avoid a circular dependency between
/// `fellm-plugin-abi` and `fellm-graph`, the graph type is exposed as a
/// generic associated type: each architecture parameterizes over its own
/// graph and config representation. This is fine for the static-linking case
/// in Phase 1 and can be widened to a boxed dynamic type when needed.
pub trait Architecture: Send + Sync + 'static {
    /// The config type this architecture extracts from GGUF metadata.
    type Config;

    /// The graph type this architecture produces.
    type Graph;

    /// Stable architecture id (matches `general.architecture` in GGUF).
    fn id(&self) -> &'static str;

    /// Extract a config from raw GGUF metadata (as key-value pairs).
    ///
    /// The concrete GGUF file type isn't referenced here to keep this crate
    /// free of a `fellm-gguf` dependency. Day-to-day loading uses
    /// `fellm_model::ModelSpec::from_gguf`; this trait method is a
    /// placeholder for Phase 2 dynamic architecture plugins.
    fn config_from_metadata(&self, metadata_json: &str) -> Result<Self::Config>;

    /// Build a per-step forward graph.
    fn build_step_graph(&self, config: &Self::Config, position: usize) -> Result<Self::Graph>;
}
