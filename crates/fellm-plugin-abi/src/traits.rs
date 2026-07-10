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

/// Opaque handle to a resolved kernel implementation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelHandle(pub u64);

/// Backend capabilities discovered at startup.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCaps {
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
