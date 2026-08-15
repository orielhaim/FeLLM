//! Host-side architecture-plugin boundary.
//!
//! The runtime owns scheduling, tensor storage, graph execution, and the CPU
//! backend.  An architecture plugin owns model-family probing, graph
//! construction, and generation semantics.  Keeping the preparation object
//! deliberately small means the engine never needs a DiffusionGemma branch
//! or a model-specific tensor name.

use fellm_core::error::Result;
use fellm_gguf::GgufFile;
use fellm_graph::Graph;
use fellm_model::ModelSpec;
use fellm_plugin_abi::{
    Backend, BackendCapabilities, GenerationDriver, GenerationRequest, ModelProgram, ModelSource,
};
use std::sync::Arc;

/// Generation strategy selected by an architecture plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureGenerationMode {
    /// Conventional causal token-at-a-time generation.
    Autoregressive,
    /// Fixed-canvas denoising followed by causal block commits.
    BlockDiffusion,
}

/// Prepared architecture data handed from a plugin to the generic engine.
pub struct ArchitecturePreparation {
    /// Stable architecture program descriptor.
    pub program: ModelProgram,
    /// Generic generation strategy selected by the plugin.
    pub generation_mode: ArchitectureGenerationMode,
    /// Optional graph used by the selected strategy.  The causal graph remains
    /// the runtime's generic GGUF graph; architecture plugins add only graphs
    /// that are genuinely family-specific.
    pub canvas_graph: Option<Graph>,
}

/// Runtime-facing architecture plugin.
///
/// This is the in-process implementation boundary.  A dynamic plugin uses the
/// equivalent C registration record from `fellm-plugin-abi`; the host can
/// adapt that record to this trait without putting architecture code in the
/// engine.
pub trait ArchitecturePlugin: Send + Sync {
    /// Stable architecture id claimed by this plugin.
    fn architecture_id(&self) -> &str;

    /// Probe and prepare a model. `None` means this plugin does not claim it.
    fn prepare(
        &self,
        gguf: &GgufFile,
        spec: &ModelSpec,
        backend: &dyn Backend,
    ) -> Result<Option<ArchitecturePreparation>>;

    /// Create the architecture-owned generation state machine.
    fn create_generation_driver(
        &self,
        program: &ModelProgram,
        request: GenerationRequest,
    ) -> Result<Box<dyn GenerationDriver>>;
}

/// Generic GGUF source conversion for architecture plugins.
#[must_use]
pub fn source_from_gguf(gguf: &GgufFile) -> ModelSource {
    ModelSource {
        architecture_id: gguf.metadata.arch().unwrap_or_default().to_owned(),
        metadata: gguf
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), format!("{value:?}")))
            .collect(),
        tensors: gguf
            .tensors()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    format!("{} {:?}", tensor.dtype, tensor.shape),
                )
            })
            .collect(),
    }
}

/// Shared plugin handle used by builders and host registries.
pub type ArchitecturePluginHandle = Arc<dyn ArchitecturePlugin>;

/// Graphs and requirements produced by a model-native speculator plugin.
pub struct SpeculatorPreparation {
    pub compatibility: fellm_plugin_abi::SpeculatorCompatibility,
    pub graphs: Vec<Graph>,
}

/// Preparation boundary for speculators whose weights live in the target
/// checkpoint. Execution, KV, sampling, and verification remain runtime-owned.
pub trait ModelSpeculatorPlugin: Send + Sync {
    fn id(&self) -> &'static str;
    fn prepare(&self, gguf: &GgufFile, spec: &ModelSpec) -> Result<Option<SpeculatorPreparation>>;
}

pub type ModelSpeculatorPluginHandle = Arc<dyn ModelSpeculatorPlugin>;

/// Build backend capabilities for a plugin preparation call.
#[must_use]
pub fn backend_capabilities(backend: &dyn Backend) -> BackendCapabilities {
    BackendCapabilities {
        backend_id: backend.id().into(),
        caps: backend.capabilities(),
    }
}
