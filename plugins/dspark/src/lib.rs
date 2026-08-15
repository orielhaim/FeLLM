//! DSpark block-parallel speculator with a sequential Markov head.
//!
//! The architecture-specific parallel backbone is deliberately separate from
//! the paper's generic Markov and confidence heads. Target verification,
//! sampling, and scheduling remain runtime responsibilities.

mod checkpoint;
mod deepseek;
mod native;

pub use checkpoint::DsparkCheckpoint;
pub use deepseek::DeepseekDsparkBackbone;
pub use native::NativeDsparkBackbone;

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::{
    CapabilityKind, FeatureTap, ProposalNode, ProposalScores, ProposalTopology, ProviderDescriptor,
    ProviderVersion, Speculator, SpeculatorCompatibility, SpeculatorContext, SpeculatorProposal,
    SpeculatorRoundOutcome, TargetFeature, VocabularyMapping,
};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

/// Architecture and feature contract serialized by released DeepSpec
/// checkpoints. Weight loading is kept separate so the same interpretation
/// can be backed by safetensors today and a future native quantized container.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct DsparkCheckpointConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub block_size: u32,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_target_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<u32>,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub enable_confidence_head: bool,
    pub confidence_head_with_markov: bool,
}

impl DsparkCheckpointConfig {
    pub fn from_directory(directory: &Path) -> Result<Self> {
        let path = directory.join("config.json");
        let bytes = std::fs::read(&path).map_err(|error| {
            FellmError::other(format!(
                "failed to read DSpark config {}: {error}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|error| {
            FellmError::other(format!("invalid DSpark config {}: {error}", path.display()))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.architectures.len() != 1
            || !matches!(
                self.architectures[0].as_str(),
                "Qwen3DSparkModel" | "Gemma4DSparkModel"
            )
        {
            return Err(FellmError::other(
                "unsupported DSpark checkpoint architecture",
            ));
        }
        if self.block_size == 0
            || self.hidden_size == 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.markov_rank == 0
        {
            return Err(FellmError::other(
                "invalid zero-sized DSpark checkpoint field",
            ));
        }
        if self.markov_head_type != "vanilla" {
            return Err(FellmError::other(format!(
                "released DSpark execution currently requires vanilla Markov head, got {}",
                self.markov_head_type
            )));
        }
        if !self.enable_confidence_head || !self.confidence_head_with_markov {
            return Err(FellmError::other(
                "released DSpark execution requires Markov-conditioned confidence",
            ));
        }
        if self.target_layer_ids.is_empty()
            || self
                .target_layer_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .target_layer_ids
                .iter()
                .any(|&layer| layer as usize >= self.num_target_layers)
        {
            return Err(FellmError::other(
                "DSpark target layer ids must be strictly increasing and in range",
            ));
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(FellmError::other(
                "invalid DSpark grouped-query attention shape",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn required_features(&self) -> Vec<FeatureTap> {
        self.target_layer_ids
            .iter()
            .copied()
            .map(|layer| FeatureTap {
                feature: TargetFeature::LayerHiddenState(layer),
                preserve_placement: true,
            })
            .collect()
    }
}

/// One block-parallel backbone evaluation. Rows are ordered by proposal position.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkBackboneOutput {
    pub base_logits: Vec<Vec<f32>>,
    pub hidden_states: Vec<Vec<f32>>,
}

/// Architecture/checkpoint-specific DFlash-style parallel backbone.
pub trait ParallelBackbone {
    fn draft_block(&mut self, context: &SpeculatorContext<'_>) -> Result<DsparkBackboneOutput>;
    fn observe_committed(
        &mut self,
        _observation: &fellm_plugin_abi::CommittedTargetToken<'_>,
    ) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self);
}

/// Released DSpark vanilla-Markov and conditional-acceptance heads.
#[derive(Debug, Clone)]
pub struct MarkovHeads {
    vocabulary_size: usize,
    rank: usize,
    hidden_size: usize,
    markov_embeddings: Vec<f32>,
    markov_output: Vec<f32>,
    confidence_weight: Vec<f32>,
    confidence_bias: f32,
    confidence_temperature: f32,
}

impl MarkovHeads {
    pub fn from_checkpoint(checkpoint: &DsparkCheckpoint) -> Result<Self> {
        let config = &checkpoint.config;
        let confidence_weight = checkpoint.tensor_f32("confidence_head.proj.weight")?;
        let confidence_bias = checkpoint
            .tensor_f32("confidence_head.proj.bias")?
            .into_iter()
            .next()
            .ok_or_else(|| FellmError::other("empty DSpark confidence bias"))?;
        Self::new(
            config.vocab_size,
            config.markov_rank,
            config.hidden_size,
            checkpoint.tensor_f32("markov_head.markov_w1.weight")?,
            checkpoint.tensor_f32("markov_head.markov_w2.weight")?,
            confidence_weight,
            confidence_bias,
            1.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocabulary_size: usize,
        rank: usize,
        hidden_size: usize,
        markov_embeddings: Vec<f32>,
        markov_output: Vec<f32>,
        confidence_weight: Vec<f32>,
        confidence_bias: f32,
        confidence_temperature: f32,
    ) -> Result<Self> {
        if vocabulary_size == 0 || rank == 0 || hidden_size == 0 {
            return Err(FellmError::other("DSpark head dimensions must be non-zero"));
        }
        if markov_embeddings.len() != vocabulary_size * rank {
            return Err(FellmError::other("DSpark markov embedding shape mismatch"));
        }
        if markov_output.len() != vocabulary_size * rank {
            return Err(FellmError::other("DSpark markov output shape mismatch"));
        }
        if confidence_weight.len() != hidden_size + rank {
            return Err(FellmError::other("DSpark confidence head shape mismatch"));
        }
        if !confidence_temperature.is_finite() || confidence_temperature <= 0.0 {
            return Err(FellmError::other(
                "DSpark confidence calibration temperature must be positive",
            ));
        }
        Ok(Self {
            vocabulary_size,
            rank,
            hidden_size,
            markov_embeddings,
            markov_output,
            confidence_weight,
            confidence_bias,
            confidence_temperature,
        })
    }

    fn correct_logits(&self, previous_token: u32, base: &[f32]) -> Result<Vec<f32>> {
        let token = usize::try_from(previous_token)
            .map_err(|_| FellmError::other("DSpark token id does not fit usize"))?;
        if token >= self.vocabulary_size || base.len() != self.vocabulary_size {
            return Err(FellmError::other("DSpark vocabulary mismatch"));
        }
        let embedding = &self.markov_embeddings[token * self.rank..(token + 1) * self.rank];
        Ok(base
            .iter()
            .enumerate()
            .map(|(vocabulary_token, &logit)| {
                let output_row = &self.markov_output
                    [vocabulary_token * self.rank..(vocabulary_token + 1) * self.rank];
                logit
                    + output_row
                        .iter()
                        .zip(embedding)
                        .map(|(&weight, &value)| weight * value)
                        .sum::<f32>()
            })
            .collect())
    }

    fn confidence(&self, previous_token: u32, hidden: &[f32]) -> Result<f32> {
        let token = usize::try_from(previous_token)
            .map_err(|_| FellmError::other("DSpark token id does not fit usize"))?;
        if token >= self.vocabulary_size || hidden.len() != self.hidden_size {
            return Err(FellmError::other("DSpark confidence input shape mismatch"));
        }
        let embedding = &self.markov_embeddings[token * self.rank..(token + 1) * self.rank];
        let raw = self.confidence_bias
            + self.confidence_weight[..self.hidden_size]
                .iter()
                .zip(hidden)
                .map(|(&weight, &value)| weight * value)
                .sum::<f32>()
            + self.confidence_weight[self.hidden_size..]
                .iter()
                .zip(embedding)
                .map(|(&weight, &value)| weight * value)
                .sum::<f32>();
        let calibrated = raw / self.confidence_temperature;
        Ok(if calibrated >= 0.0 {
            1.0 / (1.0 + (-calibrated).exp())
        } else {
            let exponential = calibrated.exp();
            exponential / (1.0 + exponential)
        })
    }
}

pub struct DsparkSpeculator<B> {
    descriptor: ProviderDescriptor,
    compatibility: SpeculatorCompatibility,
    backbone: B,
    heads: MarkovHeads,
    provisional: Option<DsparkBackboneOutput>,
}

impl<B: ParallelBackbone> DsparkSpeculator<B> {
    pub fn new(
        architecture: impl Into<String>,
        maximum_proposal_length: u32,
        required_features: Vec<FeatureTap>,
        backbone: B,
        heads: MarkovHeads,
    ) -> Result<Self> {
        if maximum_proposal_length == 0 {
            return Err(FellmError::other("DSpark proposal length must be non-zero"));
        }
        let vocabulary_size = u32::try_from(heads.vocabulary_size)
            .map_err(|_| FellmError::other("DSpark vocabulary exceeds u32"))?;
        let compatibility = SpeculatorCompatibility {
            target_architectures: vec![architecture.into()],
            maximum_proposal_length,
            topology: ProposalTopology::Linear,
            vocabulary: VocabularyMapping::Identity { vocabulary_size },
            required_features,
            required_tensor_patterns: Vec::new(),
        };
        let descriptor = ProviderDescriptor::new(
            "speculator.dspark_markov",
            CapabilityKind::Speculator,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "DSpark parallel draft with sequential Markov and confidence heads",
        )
        .with_priority(80)
        .with_meta("head", "markov")
        .with_meta("confidence", "calibrated_conditional_acceptance");
        Ok(Self {
            descriptor,
            compatibility,
            backbone,
            heads,
            provisional: None,
        })
    }
}

impl DsparkSpeculator<NativeDsparkBackbone> {
    pub fn from_checkpoint(
        checkpoint: Arc<DsparkCheckpoint>,
        backend: fellm_runtime::BackendSelect,
        maximum_sequence_length: usize,
    ) -> Result<Self> {
        let heads = MarkovHeads::from_checkpoint(&checkpoint)?;
        let backbone =
            NativeDsparkBackbone::new(checkpoint.clone(), backend, maximum_sequence_length)?;
        Self::new(
            checkpoint.config.model_type.clone(),
            checkpoint.config.block_size,
            checkpoint.config.required_features(),
            backbone,
            heads,
        )
    }
}

impl DsparkSpeculator<crate::DeepseekDsparkBackbone> {
    pub fn from_support_gguf(
        support_path: impl AsRef<std::path::Path>,
        target: &fellm_gguf::GgufFile,
        spec: &fellm_model::ModelSpec,
        backend: fellm_runtime::BackendSelect,
        maximum_sequence_length: usize,
    ) -> Result<Self> {
        let (backbone, heads, features, block_size) = crate::DeepseekDsparkBackbone::open(
            support_path,
            target,
            spec,
            backend,
            maximum_sequence_length,
        )?;
        Self::new(spec.arch_id.clone(), block_size, features, backbone, heads)
    }
}

impl<B: ParallelBackbone> Speculator for DsparkSpeculator<B> {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    fn compatibility(&self) -> &SpeculatorCompatibility {
        &self.compatibility
    }

    fn validate_target(&self, architecture: &str, vocabulary_size: u32) -> Result<()> {
        let architecture_matches = self
            .compatibility
            .target_architectures
            .iter()
            .any(|candidate| {
                candidate == architecture
                    || architecture.starts_with(candidate)
                    || candidate.starts_with(architecture)
            });
        if !architecture_matches || vocabulary_size as usize != self.heads.vocabulary_size {
            return Err(FellmError::other(
                "target is incompatible with DSpark checkpoint",
            ));
        }
        Ok(())
    }

    fn observe_committed(
        &mut self,
        observation: &fellm_plugin_abi::CommittedTargetToken<'_>,
    ) -> Result<()> {
        if self.provisional.is_some() {
            return Err(FellmError::other(
                "cannot append DSpark context during an active proposal transaction",
            ));
        }
        self.backbone.observe_committed(observation)
    }

    fn begin_round(&mut self, context: &SpeculatorContext<'_>) -> Result<()> {
        if self.provisional.is_some() {
            return Err(FellmError::other(
                "DSpark proposal transaction is already active",
            ));
        }
        let block = self.backbone.draft_block(context)?;
        let expected = usize::try_from(context.maximum_length)
            .unwrap_or(usize::MAX)
            .min(self.compatibility.maximum_proposal_length as usize);
        if block.base_logits.len() < expected || block.hidden_states.len() < expected {
            return Err(FellmError::other(
                "DSpark backbone returned a short proposal block",
            ));
        }
        self.provisional = Some(block);
        Ok(())
    }

    fn propose_next(&mut self, context: &SpeculatorContext<'_>) -> Result<SpeculatorProposal> {
        let index = context.proposed_tokens.len();
        let block = self
            .provisional
            .as_ref()
            .ok_or_else(|| FellmError::other("DSpark proposal transaction is not active"))?;
        let maximum = usize::try_from(context.maximum_length)
            .unwrap_or(usize::MAX)
            .min(self.compatibility.maximum_proposal_length as usize);
        if index >= maximum {
            return Ok(SpeculatorProposal::default());
        }
        let previous_token = context
            .proposed_tokens
            .last()
            .or_else(|| context.prefix_tokens.last())
            .copied()
            .ok_or_else(|| FellmError::other("DSpark requires a preceding token"))?;
        let logits = self
            .heads
            .correct_logits(previous_token, &block.base_logits[index])?;
        let confidence = self
            .heads
            .confidence(previous_token, &block.hidden_states[index])?;
        Ok(SpeculatorProposal {
            nodes: vec![ProposalNode {
                parent: index.checked_sub(1).map(|parent| parent as u32),
                scores: ProposalScores::Logits(logits),
                confidence: Some(confidence),
            }],
        })
    }

    fn finish_round(&mut self, _outcome: &SpeculatorRoundOutcome<'_>) -> Result<()> {
        if self.provisional.take().is_none() {
            return Err(FellmError::other(
                "DSpark proposal transaction is not active",
            ));
        }
        Ok(())
    }

    fn abort_round(&mut self) {
        self.provisional = None;
    }

    fn reset(&mut self) {
        self.provisional = None;
        self.backbone.reset();
    }
}

/// The feature request used by released DSpark checkpoints unless their model
/// configuration specifies a different fusion set.
#[must_use]
pub fn final_hidden_feature() -> FeatureTap {
    FeatureTap {
        feature: TargetFeature::FinalHiddenState,
        preserve_placement: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_qwen_config_maps_selected_target_layers_to_feature_taps() {
        let config: DsparkCheckpointConfig = serde_json::from_str(
            r#"{
                "architectures":["Qwen3DSparkModel"], "model_type":"qwen3", "block_size":7,
                "hidden_size":2560, "vocab_size":151936,
                "num_hidden_layers":5, "num_target_layers":36,
                "num_attention_heads":32, "num_key_value_heads":8,
                "intermediate_size":9728, "head_dim":128,
                "mask_token_id":151669,
                "target_layer_ids":[1,9,17,25,33],
                "markov_rank":256, "markov_head_type":"vanilla",
                "enable_confidence_head":true,
                "confidence_head_with_markov":true
            }"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.block_size, 7);
        assert_eq!(
            config
                .required_features()
                .iter()
                .map(|tap| tap.feature)
                .collect::<Vec<_>>(),
            [1, 9, 17, 25, 33]
                .map(TargetFeature::LayerHiddenState)
                .to_vec()
        );
    }

    struct MockBackbone;

    impl ParallelBackbone for MockBackbone {
        fn draft_block(
            &mut self,
            _context: &SpeculatorContext<'_>,
        ) -> Result<DsparkBackboneOutput> {
            Ok(DsparkBackboneOutput {
                base_logits: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                hidden_states: vec![vec![0.5], vec![1.0]],
            })
        }

        fn reset(&mut self) {}
    }

    fn speculator() -> DsparkSpeculator<MockBackbone> {
        let heads = MarkovHeads::new(
            2,
            1,
            1,
            vec![2.0, 3.0],
            vec![10.0, 20.0],
            vec![2.0, 1.0],
            -1.0,
            2.0,
        )
        .unwrap();
        DsparkSpeculator::new("test", 2, vec![final_hidden_feature()], MockBackbone, heads).unwrap()
    }

    #[test]
    fn markov_head_uses_the_preceding_sampled_token_and_calibrates_confidence() {
        let mut speculator = speculator();
        let initial = SpeculatorContext {
            prefix_tokens: &[0],
            proposed_tokens: &[],
            maximum_length: 2,
            captured_features: &[],
        };
        speculator.begin_round(&initial).unwrap();
        let first = speculator.propose_next(&initial).unwrap();
        let ProposalScores::Logits(first_logits) = &first.nodes[0].scores else {
            panic!("expected logits")
        };
        assert_eq!(first_logits, &[21.0, 42.0]);
        let expected = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!((first.nodes[0].confidence.unwrap() - expected).abs() < 1e-6);

        let sequential = SpeculatorContext {
            prefix_tokens: &[0],
            proposed_tokens: &[1],
            maximum_length: 2,
            captured_features: &[],
        };
        let second = speculator.propose_next(&sequential).unwrap();
        let ProposalScores::Logits(second_logits) = &second.nodes[0].scores else {
            panic!("expected logits")
        };
        assert_eq!(second_logits, &[33.0, 64.0]);
        assert_eq!(second.nodes[0].parent, Some(0));
    }

    #[test]
    fn proposal_state_is_transactional() {
        let mut speculator = speculator();
        let context = SpeculatorContext {
            prefix_tokens: &[0],
            proposed_tokens: &[],
            maximum_length: 2,
            captured_features: &[],
        };
        speculator.begin_round(&context).unwrap();
        assert!(speculator.begin_round(&context).is_err());
        speculator.abort_round();
        speculator.begin_round(&context).unwrap();
        speculator
            .finish_round(&SpeculatorRoundOutcome {
                accepted: 0,
                emitted: &[],
                terminal: false,
            })
            .unwrap();
        assert!(speculator.propose_next(&context).is_err());
    }
}
