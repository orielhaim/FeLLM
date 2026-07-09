//! LFM2 MoE architecture support.

#![deny(missing_docs)]

/// LFM2 MoE config extraction.
pub mod config;
/// Per-step forward graph construction.
pub mod graph_builder;

pub use config::Lfm2MoeConfig;

use fellm_core::error::Result;
use fellm_gguf::GgufFile;
use fellm_graph::Graph;

/// The LFM2 MoE architecture implementation.
pub struct Lfm2MoeArch;

impl Lfm2MoeArch {
    /// New instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract config from GGUF metadata.
    pub fn config_from_gguf(&self, gguf: &GgufFile) -> Result<Lfm2MoeConfig> {
        Lfm2MoeConfig::from_gguf(gguf)
    }

    /// Build a graph that executes one token forward.
    pub fn build_graph(&self, gguf: &GgufFile, config: &Lfm2MoeConfig) -> Result<Graph> {
        graph_builder::build(gguf, config)
    }

    /// Collect nodes whose attrs must be patched each decode step.
    #[must_use]
    pub fn collect_step_nodes(graph: &Graph) -> StepNodes {
        let mut nodes = StepNodes::default();
        for (id, node) in graph.iter_nodes() {
            match node.op {
                Some(fellm_plugin_abi::op::OpKind::Rope) => nodes.rope.push(id),
                Some(fellm_plugin_abi::op::OpKind::KvWrite) => nodes.kv_write.push(id),
                Some(fellm_plugin_abi::op::OpKind::Attention) => nodes.attention.push(id),
                _ => {}
            }
        }
        nodes
    }
}

/// Node ids that carry position / past_len attrs.
#[derive(Debug, Default, Clone)]
pub struct StepNodes {
    /// RoPE ops.
    pub rope: Vec<fellm_graph::NodeId>,
    /// KV write ops.
    pub kv_write: Vec<fellm_graph::NodeId>,
    /// Attention ops.
    pub attention: Vec<fellm_graph::NodeId>,
}

impl Default for Lfm2MoeArch {
    fn default() -> Self {
        Self::new()
    }
}
