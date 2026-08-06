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

pub use graph::{
    StepBindings, build_diffusion_canvas_graph, build_step_graph, collect_step_bindings,
};
pub use probe::{FfnKind, LayerSpec, MixKind, ModelSpec, RopeScalingType};
pub use tools::parse_assistant_output;

/// Default number of vocabulary candidates retained for DiffusionGemma
/// self-conditioning.  A compact `(token_id, logit)` pair is stored for each
/// candidate instead of materializing a dense `[canvas, vocab]` matrix.
pub const DEFAULT_DIFFUSION_SELF_COND_TOP_K: usize = 256;

/// Number of F32 values required by the self-conditioning graph input.
///
/// `0` is an explicit correctness/debug fallback to the dense vocabulary
/// representation.  The normal path uses two values per retained candidate:
/// token id encoded exactly as F32, followed by its logit.
#[must_use]
pub fn diffusion_self_conditioning_slots(vocab: usize) -> usize {
    let top_k = std::env::var("FELLM_DIFFUSION_SELF_COND_TOP_K")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_DIFFUSION_SELF_COND_TOP_K);
    if top_k == 0 {
        vocab
    } else {
        top_k.min(vocab).saturating_mul(2)
    }
}
