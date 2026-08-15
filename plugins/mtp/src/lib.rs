//! Native multi-token-prediction speculator preparation.

use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_gguf::GgufFile;
use fellm_graph::ExecutionPlan;
use fellm_graph::graph::OpValue;
use fellm_model::{ModelSpec, build_mtp_step_graph, collect_step_bindings};
use fellm_plugin_abi::{
    CapabilityKind, FeatureTap, ProposalNode, ProposalScores, ProposalTopology, ProviderDescriptor,
    ProviderVersion, Speculator, SpeculatorCompatibility, SpeculatorContext, SpeculatorProposal,
    SpeculatorRoundOutcome, TargetFeature, VocabularyMapping,
};
use fellm_runtime::architecture::{ModelSpeculatorPlugin, SpeculatorPreparation};
use fellm_runtime::compiled::{CompiledStep, MutableBinding};
use fellm_runtime::{BackendSelect, DummyKvBuffers};
use std::collections::HashMap;
use std::sync::Arc;

/// Detects appended `nextn` modules and constructs one reusable graph per stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtpPlugin;

/// How physical MTP modules map to logical proposal depths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDepthLayout {
    /// One checkpoint block is recurrently invoked for every logical depth.
    SharedRecurrent,
    /// Each logical depth uses its corresponding physical checkpoint block.
    Staged,
}

struct MtpStage {
    graph: fellm_graph::Graph,
    bindings: fellm_model::StepBindings,
    step: CompiledStep,
    _kv: DummyKvBuffers,
    length: usize,
}

/// Executable native-MTP speculator using ordinary FeLLM graphs and backend
/// kernels. Target verification and token sampling remain runtime-owned.
pub struct MtpSpeculator {
    descriptor: ProviderDescriptor,
    compatibility: SpeculatorCompatibility,
    backend: Box<dyn fellm_plugin_abi::Backend>,
    stages: Vec<MtpStage>,
    layout: MtpDepthLayout,
    d_model: usize,
    pending_target_hidden: Vec<f32>,
    target_hidden: Option<Vec<f32>>,
    recurrent_hidden: Option<Vec<f32>>,
    round_start_lengths: Vec<usize>,
}

impl MtpSpeculator {
    pub fn from_model(
        gguf: &GgufFile,
        spec: &ModelSpec,
        backend: BackendSelect,
        max_sequence_length: usize,
        maximum_proposal_length: u32,
    ) -> Result<Self> {
        let preparation = MtpPlugin
            .prepare(gguf, spec)?
            .ok_or_else(|| FellmError::other("model does not contain native MTP modules"))?;
        if maximum_proposal_length == 0 {
            return Err(FellmError::other("MTP proposal length must be non-zero"));
        }
        let backend = backend.resolve()?;
        let required_weight_group = preparation
            .graphs
            .iter()
            .flat_map(|graph| {
                graph.iter_nodes().map(|(id, _)| {
                    graph
                        .inputs_slice(id)
                        .iter()
                        .filter_map(|&input| match &graph.node(input).value {
                            OpValue::Constant(tensor) => Some(tensor.as_bytes().len() as u64),
                            _ => None,
                        })
                        .sum::<u64>()
                })
            })
            .max()
            .unwrap_or(0);
        backend.ensure_weight_group_capacity_bytes(required_weight_group)?;
        let mut stages = Vec::with_capacity(preparation.graphs.len());
        for graph in preparation.graphs {
            let plan = ExecutionPlan::from_graph(&graph)?;
            let kv = DummyKvBuffers::new(
                1,
                max_sequence_length.max(maximum_proposal_length as usize),
                spec.n_kv_heads,
                spec.head_dim,
            )?;
            let shape = Shape::new(&[kv.max_seq as u64, (spec.n_kv_heads * spec.head_dim) as u64])?;
            let mutable_inputs = HashMap::from([
                (
                    "k_in_0".to_owned(),
                    MutableBinding {
                        dtype: DType::F32,
                        shape: shape.clone(),
                        buffer: kv.k_buffer(0),
                    },
                ),
                (
                    "v_in_0".to_owned(),
                    MutableBinding {
                        dtype: DType::F32,
                        shape,
                        buffer: kv.v_buffer(0),
                    },
                ),
            ]);
            let step = CompiledStep::compile(&graph, &plan, backend.as_ref(), &mutable_inputs)?;
            stages.push(MtpStage {
                bindings: collect_step_bindings(&graph),
                graph,
                step,
                _kv: kv,
                length: 0,
            });
        }
        let layout = if stages.len() == 1 {
            MtpDepthLayout::SharedRecurrent
        } else {
            MtpDepthLayout::Staged
        };
        let mut compatibility = preparation.compatibility;
        compatibility.maximum_proposal_length = maximum_proposal_length;
        let descriptor = ProviderDescriptor::new(
            "speculator.mtp",
            CapabilityKind::Speculator,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "native shared-component multi-token prediction",
        )
        .with_priority(100)
        .with_meta(
            "depth_layout",
            match layout {
                MtpDepthLayout::SharedRecurrent => "shared_recurrent",
                MtpDepthLayout::Staged => "staged",
            },
        );
        Ok(Self {
            descriptor,
            compatibility,
            backend,
            stages,
            layout,
            d_model: spec.d_model,
            pending_target_hidden: vec![0.0; spec.d_model],
            target_hidden: None,
            recurrent_hidden: None,
            round_start_lengths: Vec::new(),
        })
    }

    fn stage_index(&self, depth: usize) -> Option<usize> {
        match self.layout {
            MtpDepthLayout::SharedRecurrent => Some(0),
            MtpDepthLayout::Staged => (depth < self.stages.len()).then_some(depth),
        }
    }

    fn execute_stage(
        &mut self,
        stage_index: usize,
        token: u32,
        hidden: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let stage = &mut self.stages[stage_index];
        let position = stage.length as u32;
        for &id in &stage.bindings.rope {
            let mut attrs = stage.graph.node(id).attrs;
            attrs.position = position;
            stage.step.set_attrs(id, attrs);
        }
        for &id in &stage.bindings.kv_write {
            let mut attrs = stage.graph.node(id).attrs;
            attrs.position = position;
            // MTP owns an isolated contiguous DummyKvBuffers cache. A non-zero
            // block size would incorrectly route CUDA through the target's
            // paged KV snapshot.
            attrs.block_size = 0;
            stage.step.set_attrs(id, attrs);
        }
        for &id in &stage.bindings.attention {
            let mut attrs = stage.graph.node(id).attrs;
            attrs.past_len = position;
            attrs.query_len = 1;
            attrs.kv_len = position + 1;
            attrs.block_size = 0;
            stage.step.set_attrs(id, attrs);
        }
        stage.step.bind_input("token_id", scalar_u32_tensor(token));
        stage
            .step
            .bind_input("target_hidden", f32_vector_tensor(hidden)?);
        self.backend.begin_step();
        let logits = stage.step.run(self.backend.as_ref(), true);
        self.backend.end_step();
        let logits = logits?;
        let next_hidden = stage
            .step
            .materialize_named_output(self.backend.as_ref(), "mtp_hidden")?;
        stage.length += 1;
        Ok((
            logits.as_slice::<f32>()?.to_vec(),
            next_hidden.as_slice::<f32>()?.to_vec(),
        ))
    }
}

impl Speculator for MtpSpeculator {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    fn compatibility(&self) -> &SpeculatorCompatibility {
        &self.compatibility
    }

    fn validate_target(&self, architecture: &str, vocabulary_size: u32) -> Result<()> {
        if !self
            .compatibility
            .target_architectures
            .iter()
            .any(|candidate| candidate == architecture)
        {
            return Err(FellmError::other(
                "target architecture is incompatible with MTP",
            ));
        }
        match self.compatibility.vocabulary {
            VocabularyMapping::Identity {
                vocabulary_size: expected,
            } if expected == vocabulary_size => Ok(()),
            _ => Err(FellmError::other(
                "target vocabulary is incompatible with MTP",
            )),
        }
    }

    fn observe_committed(
        &mut self,
        observation: &fellm_plugin_abi::CommittedTargetToken<'_>,
    ) -> Result<()> {
        if self.target_hidden.is_some() {
            return Err(FellmError::other(
                "cannot catch up MTP state during an active proposal transaction",
            ));
        }
        let feature = observation
            .captured_features
            .iter()
            .find(|feature| feature.feature == TargetFeature::FinalHiddenState)
            .ok_or_else(|| FellmError::other("MTP catch-up requires final hidden state"))?;
        if feature.tensor.dtype() != Some(DType::F32)
            || feature.tensor.byte_len as usize != self.d_model * 4
        {
            return Err(FellmError::other("MTP catch-up feature shape mismatch"));
        }
        // SAFETY: the observation keeps the captured tensor alive for this call.
        let bytes = unsafe { feature.tensor.as_bytes() };
        let current: &[f32] = bytemuck::try_cast_slice(bytes)
            .map_err(|_| FellmError::other("MTP catch-up feature is not aligned f32"))?;
        if self
            .stages
            .iter()
            .any(|stage| stage.length != observation.position as usize)
        {
            return Err(FellmError::other(
                "MTP draft KV position is not aligned with committed target prefix",
            ));
        }
        let conditioning = self.pending_target_hidden.clone();
        for stage in 0..self.stages.len() {
            let _ = self.execute_stage(stage, observation.token, &conditioning)?;
        }
        self.pending_target_hidden.copy_from_slice(current);
        Ok(())
    }

    fn begin_round(&mut self, context: &SpeculatorContext<'_>) -> Result<()> {
        if self.target_hidden.is_some() {
            return Err(FellmError::other(
                "MTP proposal transaction is already active",
            ));
        }
        let feature = context
            .captured_features
            .iter()
            .find(|feature| feature.feature == TargetFeature::FinalHiddenState)
            .ok_or_else(|| FellmError::other("MTP requires the target final hidden state"))?;
        if feature.tensor.dtype() != Some(DType::F32)
            || feature.tensor.byte_len as usize != self.d_model * 4
        {
            return Err(FellmError::other(
                "MTP target hidden-state shape or dtype mismatch",
            ));
        }
        // SAFETY: CapturedTargetFeature guarantees the view remains live for
        // this call; copy it into request-owned recurrent proposal state.
        let bytes = unsafe { feature.tensor.as_bytes() };
        let hidden: &[f32] = bytemuck::try_cast_slice(bytes)
            .map_err(|_| FellmError::other("MTP target hidden state is not aligned f32"))?;
        self.round_start_lengths = self.stages.iter().map(|stage| stage.length).collect();
        self.target_hidden = Some(hidden.to_vec());
        self.recurrent_hidden = None;
        Ok(())
    }

    fn propose_next(&mut self, context: &SpeculatorContext<'_>) -> Result<SpeculatorProposal> {
        let depth = context.proposed_tokens.len();
        let maximum = context
            .maximum_length
            .min(self.compatibility.maximum_proposal_length) as usize;
        if depth >= maximum || self.stage_index(depth).is_none() {
            return Ok(SpeculatorProposal::default());
        }
        let token = context
            .proposed_tokens
            .last()
            .or_else(|| context.prefix_tokens.last())
            .copied()
            .ok_or_else(|| FellmError::other("MTP requires an aligned input token"))?;
        let hidden = self
            .recurrent_hidden
            .as_ref()
            .or(self.target_hidden.as_ref())
            .ok_or_else(|| FellmError::other("MTP proposal transaction is not active"))?
            .clone();
        let stage = self
            .stage_index(depth)
            .ok_or_else(|| FellmError::other("MTP logical depth exceeds staged modules"))?;
        let (logits, next_hidden) = self.execute_stage(stage, token, &hidden)?;
        self.recurrent_hidden = Some(next_hidden);
        Ok(SpeculatorProposal {
            nodes: vec![ProposalNode {
                parent: depth.checked_sub(1).map(|parent| parent as u32),
                scores: ProposalScores::Logits(logits),
                confidence: None,
            }],
        })
    }

    fn finish_round(&mut self, _outcome: &SpeculatorRoundOutcome<'_>) -> Result<()> {
        if self.target_hidden.take().is_none() {
            return Err(FellmError::other("MTP proposal transaction is not active"));
        }
        self.recurrent_hidden = None;
        for (stage, &length) in self.stages.iter_mut().zip(&self.round_start_lengths) {
            stage.length = length;
        }
        self.round_start_lengths.clear();
        Ok(())
    }

    fn abort_round(&mut self) {
        self.target_hidden = None;
        self.recurrent_hidden = None;
        for (stage, &length) in self.stages.iter_mut().zip(&self.round_start_lengths) {
            stage.length = length;
        }
        self.round_start_lengths.clear();
    }

    fn reset(&mut self) {
        self.target_hidden = None;
        self.recurrent_hidden = None;
        self.round_start_lengths.clear();
        for stage in &mut self.stages {
            stage.length = 0;
        }
        self.pending_target_hidden.fill(0.0);
    }
}

fn scalar_u32_tensor(value: u32) -> Tensor {
    let mut buffer = AlignedBuffer::new_zeroed(4, 4);
    buffer.as_mut_slice().copy_from_slice(&value.to_le_bytes());
    Tensor::from_storage(
        Layout::contiguous(DType::U32, Shape::new(&[1]).expect("valid scalar shape")),
        Arc::new(Storage::Owned(Arc::new(buffer))),
    )
}

fn f32_vector_tensor(values: &[f32]) -> Result<Tensor> {
    let mut buffer = AlignedBuffer::new_zeroed(values.len() * 4, 64);
    buffer
        .as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(values));
    Ok(Tensor::from_storage(
        Layout::contiguous(DType::F32, Shape::new(&[values.len() as u64])?),
        Arc::new(Storage::Owned(Arc::new(buffer))),
    ))
}

impl ModelSpeculatorPlugin for MtpPlugin {
    fn id(&self) -> &'static str {
        "mtp"
    }

    fn prepare(&self, gguf: &GgufFile, spec: &ModelSpec) -> Result<Option<SpeculatorPreparation>> {
        if spec.n_mtp_layers == 0 {
            return Ok(None);
        }
        if spec.vocab_size == 0 {
            return Err(FellmError::other(
                "MTP requires a non-empty target vocabulary",
            ));
        }
        let mut required_tensor_patterns = Vec::with_capacity(spec.n_mtp_layers * 3);
        let mut graphs = Vec::with_capacity(spec.n_mtp_layers);
        for (stage, layer) in spec.mtp_layer_indices().enumerate() {
            for suffix in [
                "nextn.eh_proj.weight",
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
            ] {
                let name = format!("blk.{layer}.{suffix}");
                if !gguf.has_tensor(&name) {
                    return Err(FellmError::other(format!(
                        "MTP stage {stage} is missing required tensor {name}"
                    )));
                }
                required_tensor_patterns.push(name);
            }
            graphs.push(build_mtp_step_graph(gguf, spec, stage)?);
        }
        Ok(Some(SpeculatorPreparation {
            compatibility: SpeculatorCompatibility {
                target_architectures: vec![spec.arch_id.clone()],
                // Physical module count and logical recurrent rollout depth are
                // distinct. The executable instance supplies its configured K.
                maximum_proposal_length: spec.n_mtp_layers.max(1) as u32,
                topology: ProposalTopology::Linear,
                vocabulary: VocabularyMapping::Identity {
                    vocabulary_size: spec.vocab_size as u32,
                },
                required_features: vec![FeatureTap {
                    feature: TargetFeature::FinalHiddenState,
                    preserve_placement: true,
                }],
                required_tensor_patterns,
            },
            graphs,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_plugin_id() {
        assert_eq!(MtpPlugin.id(), "mtp");
    }
}
