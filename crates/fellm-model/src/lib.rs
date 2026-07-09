//! GGUF-driven model recipe: probe tensors/metadata, build a petgraph step graph.
//!
//! The engine never names model families. Topology comes from which `blk.N.*`
//! tensors exist under the GGUF architecture metadata prefix.

#![deny(missing_docs)]

/// One-token step graph construction.
pub mod graph;
/// Probe GGUF into [`ModelSpec`].
pub mod probe;
/// RoPE inv-freq helpers.
pub mod rope;
/// Tool-call response parsing heuristics.
pub mod tools;

pub use graph::{StepBindings, build_step_graph, collect_step_bindings};
pub use probe::{FfnKind, LayerSpec, MixKind, ModelSpec, RopeScalingType};
pub use tools::parse_assistant_output;
