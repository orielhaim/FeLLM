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

pub mod config;
pub mod graph_builder;

pub use config::LlamaConfig;

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

    /// Build a graph that executes one token forward at `position`.
    pub fn build_step_graph(
        &self,
        gguf: &GgufFile,
        config: &LlamaConfig,
        position: usize,
    ) -> Result<Graph> {
        graph_builder::build(gguf, config, position)
    }
}

impl Default for LlamaArch {
    fn default() -> Self {
        Self::new()
    }
}
