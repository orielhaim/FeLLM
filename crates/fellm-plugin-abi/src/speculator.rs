//! Generic proposal-provider contract for speculative decoding.
//!
//! Verification, sampling, KV transactions, and scheduling intentionally do
//! not appear here: they are runtime responsibilities.

use crate::capability::ProviderDescriptor;
use crate::{DeviceKind, TensorRef};
use fellm_core::error::Result;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetFeature {
    EmbeddingOutput,
    LayerHiddenState(u32),
    FinalHiddenState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureTap {
    pub feature: TargetFeature,
    /// Keep the tensor in its execution placement when possible.
    pub preserve_placement: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VocabularyMapping {
    Identity {
        vocabulary_size: u32,
    },
    /// Draft id -> target id. A target token absent here cannot be proposed.
    DraftToTarget(Vec<u32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalTopology {
    Linear,
    Tree,
    DirectedAcyclicGraph,
}

#[derive(Debug, Clone)]
pub struct SpeculatorCompatibility {
    pub target_architectures: Vec<String>,
    pub maximum_proposal_length: u32,
    pub topology: ProposalTopology,
    pub vocabulary: VocabularyMapping,
    pub required_features: Vec<FeatureTap>,
    pub required_tensor_patterns: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ProposalScores {
    /// Unprocessed vocabulary logits. Core applies the target sampling policy.
    Logits(Vec<f32>),
    /// A normalized distribution, for algorithms whose checkpoint produces
    /// probabilities directly. Core still applies grammar and token masks.
    Probabilities(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct ProposalNode {
    pub parent: Option<u32>,
    pub scores: ProposalScores,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct SpeculatorProposal {
    pub nodes: Vec<ProposalNode>,
}

/// A borrowed target activation retained by the target execution plan.
///
/// `tensor` keeps the backend's ordinary logical identity and host address, so
/// a colocated speculator backend can consume the resident allocation without
/// a D2H copy. A heterogeneous speculator may explicitly materialize/transfer
/// it through the normal backend and Memory Fabric paths.
#[derive(Debug, Clone, Copy)]
pub struct CapturedTargetFeature<'a> {
    pub feature: TargetFeature,
    pub tensor: TensorRef,
    pub device: DeviceKind,
    _lifetime: PhantomData<&'a [u8]>,
}

impl<'a> CapturedTargetFeature<'a> {
    /// Bind a live execution-plan tensor to the duration of one proposal call.
    #[must_use]
    pub fn new(feature: TargetFeature, tensor: TensorRef, device: DeviceKind) -> Self {
        Self {
            feature,
            tensor,
            device,
            _lifetime: PhantomData,
        }
    }
}

pub struct SpeculatorContext<'a> {
    pub prefix_tokens: &'a [u32],
    /// Tokens sampled by core earlier in this proposal round. Sequential heads
    /// may condition on this slice; block-parallel heads may ignore it.
    pub proposed_tokens: &'a [u32],
    pub maximum_length: u32,
    pub captured_features: &'a [CapturedTargetFeature<'a>],
}

/// Core-owned result delivered after target verification. Plugins use it to
/// commit accepted private state and discard rejected provisional state.
pub struct SpeculatorRoundOutcome<'a> {
    pub accepted: usize,
    pub emitted: &'a [u32],
    pub terminal: bool,
}

/// One target token that has become irrevocably committed, together with the
/// target features produced while processing it. Stateful drafters use this to
/// keep private KV/recurrent state aligned with the target prefix.
pub struct CommittedTargetToken<'a> {
    pub token: u32,
    pub position: u32,
    pub captured_features: &'a [CapturedTargetFeature<'a>],
}

/// Request-owned proposer state. Factories/providers may be shared across
/// threads, but an active proposal transaction is deliberately local to one
/// request and need not make arena-backed graph state `Send` or `Sync`.
pub trait Speculator {
    fn descriptor(&self) -> &ProviderDescriptor;
    fn compatibility(&self) -> &SpeculatorCompatibility;
    fn validate_target(&self, architecture: &str, vocabulary_size: u32) -> Result<()>;
    /// Observe a target token only after core has committed it. Stateless
    /// speculators can ignore this; MTP-style drafters use it for KV catch-up.
    fn observe_committed(&mut self, _observation: &CommittedTargetToken<'_>) -> Result<()> {
        Ok(())
    }
    /// Start a provisional proposal transaction.
    fn begin_round(&mut self, context: &SpeculatorContext<'_>) -> Result<()>;
    /// Produce the next linear/tree frontier. Core owns token sampling and may
    /// call this repeatedly with an extended `proposed_tokens` prefix.
    fn propose_next(&mut self, context: &SpeculatorContext<'_>) -> Result<SpeculatorProposal>;
    /// Atomically reconcile private state with the target verifier's result.
    fn finish_round(&mut self, outcome: &SpeculatorRoundOutcome<'_>) -> Result<()>;
    /// Discard all provisional state after cancellation or target failure.
    fn abort_round(&mut self);
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_taps_express_eagle_three_style_fusion_without_algorithm_branches() {
        let taps = [1, 12, 31].map(|layer| FeatureTap {
            feature: TargetFeature::LayerHiddenState(layer),
            preserve_placement: true,
        });
        assert_eq!(taps.len(), 3);
        assert!(taps.iter().all(|tap| tap.preserve_placement));
    }

    #[test]
    fn proposal_nodes_already_support_non_linear_parentage() {
        let proposal = SpeculatorProposal {
            nodes: vec![
                ProposalNode {
                    parent: None,
                    scores: ProposalScores::Probabilities(vec![0.0, 1.0]),
                    confidence: Some(0.9),
                },
                ProposalNode {
                    parent: Some(0),
                    scores: ProposalScores::Probabilities(vec![1.0, 0.0]),
                    confidence: None,
                },
            ],
        };
        assert_eq!(proposal.nodes[1].parent, Some(0));
    }
}
