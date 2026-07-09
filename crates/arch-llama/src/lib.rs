//! Llama-family architecture as an implementation of the FeLLM plugin ABI.
//!
//! Reads `general.architecture = "llama"` from GGUF and produces a per-step
//! graph that executes one token.
//!
//! Weight tensor naming follows the GGUF convention:
//!   token_embd.weight
//!   blk.{i}.attn_norm.weight
//!   blk.{i}.attn_q.weight
//!   blk.{i}.attn_k.weight
//!   blk.{i}.attn_v.weight
//!   blk.{i}.attn_output.weight
//!   blk.{i}.ffn_norm.weight
//!   blk.{i}.ffn_gate.weight
//!   blk.{i}.ffn_up.weight
//!   blk.{i}.ffn_down.weight
//!   output_norm.weight
//!   output.weight

#![deny(missing_docs)]

/// Llama hyperparameter extraction from GGUF metadata.
pub mod config;
/// Per-step forward graph construction.
pub mod graph_builder;
/// Llama-family assistant tool-call response parsing.
pub mod tools;

pub use config::LlamaConfig;
pub use tools::parse_assistant_output;

use fellm_core::error::Result;
use fellm_gguf::GgufFile;
use fellm_graph::Graph;

/// The Llama architecture implementation.
pub struct LlamaArch;

impl LlamaArch {
    /// New instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract config from GGUF metadata.
    pub fn config_from_gguf(&self, gguf: &GgufFile) -> Result<LlamaConfig> {
        LlamaConfig::from_gguf(gguf)
    }

    /// Build a graph that executes one token forward.
    ///
    /// Position-dependent attrs are placeholders; the runtime patches them
    /// each step.
    pub fn build_graph(&self, gguf: &GgufFile, config: &LlamaConfig) -> Result<Graph> {
        graph_builder::build(gguf, config)
    }

    /// Collect nodes whose attrs must be patched each decode step.
    #[must_use]
    pub fn collect_position_nodes(graph: &Graph) -> PositionNodes {
        let mut nodes = PositionNodes::default();
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
pub struct PositionNodes {
    /// RoPE ops.
    pub rope: Vec<fellm_graph::NodeId>,
    /// KV write ops.
    pub kv_write: Vec<fellm_graph::NodeId>,
    /// Attention ops.
    pub attention: Vec<fellm_graph::NodeId>,
}

impl Default for LlamaArch {
    fn default() -> Self {
        Self::new()
    }
}
