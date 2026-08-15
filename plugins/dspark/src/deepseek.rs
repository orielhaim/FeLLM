use crate::{DsparkBackboneOutput, MarkovHeads, ParallelBackbone};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_gguf::GgufFile;
use fellm_graph::ExecutionPlan;
use fellm_model::{ModelSpec, build_dspark_proposal_graph, collect_step_bindings};
use fellm_plugin_abi::{CommittedTargetToken, FeatureTap, SpeculatorContext, TargetFeature};
use fellm_runtime::compiled::{CompiledStep, MutableBinding};
use fellm_runtime::{BackendSelect, DummyKvBuffers};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Fused DeepSeek V4 DSpark drafter backed by a support GGUF + target weights.
pub struct DeepseekDsparkBackbone {
    support: Arc<GgufFile>,
    target_layers: Vec<u32>,
    noise_token_id: u32,
    hidden_size: usize,
    hc_mult: usize,
    backend: Box<dyn fellm_plugin_abi::Backend>,
    graph: fellm_graph::Graph,
    step: CompiledStep,
    rope: Vec<fellm_graph::graph::NodeId>,
    attention: Vec<fellm_graph::graph::NodeId>,
    _kv: DummyKvBuffers,
    fused: Vec<f32>,
    context_length: usize,
}

impl DeepseekDsparkBackbone {
    pub fn open(
        support_path: impl AsRef<Path>,
        target: &GgufFile,
        spec: &ModelSpec,
        backend: BackendSelect,
        capacity: usize,
    ) -> Result<(Self, MarkovHeads, Vec<FeatureTap>, u32)> {
        let support = Arc::new(GgufFile::open(support_path.as_ref())?);
        let meta = &support.metadata;
        let stages = meta.get_u32("dspark.n_layers").unwrap_or(3) as usize;
        let block_size = meta.get_u32("dspark.block_size").unwrap_or(5);
        let noise_token_id = meta.get_u32("dspark.noise_token_id").unwrap_or(128799);
        let markov_rank = meta.get_u32("dspark.markov_rank").unwrap_or(256) as usize;
        let target_layers = meta
            .get_u32_array("dspark.target_layer_ids")
            .map(|values| values.to_vec())
            .unwrap_or_else(|_| vec![40, 41, 42]);
        let features = target_layers
            .iter()
            .copied()
            .map(|layer| FeatureTap {
                feature: TargetFeature::LayerHiddenState(layer),
                preserve_placement: true,
            })
            .collect();
        let heads = markov_heads_from_support(&support, spec.vocab_size, markov_rank, spec.d_model)?;
        let backend = backend.resolve()?;
        backend.ensure_weight_group_capacity_bytes(32 * 1024 * 1024)?;
        let kv = DummyKvBuffers::new(stages, capacity, 1, spec.head_dim)?;
        let mut mutable = HashMap::new();
        let shape = Shape::new(&[kv.max_seq as u64, kv.tokens_stride as u64])?;
        for layer in 0..stages {
            mutable.insert(
                format!("k_in_{layer}"),
                MutableBinding {
                    dtype: DType::F32,
                    shape: shape.clone(),
                    buffer: kv.k_buffer(layer),
                },
            );
            mutable.insert(
                format!("v_in_{layer}"),
                MutableBinding {
                    dtype: DType::F32,
                    shape: shape.clone(),
                    buffer: kv.v_buffer(layer),
                },
            );
        }
        let mut draft_spec = spec.clone();
        draft_spec.context_length = capacity.max(1).min(spec.context_length.max(1));
        let graph = build_dspark_proposal_graph(&support, target, &draft_spec, stages)?;
        let plan = ExecutionPlan::from_graph(&graph)?;
        let bindings = collect_step_bindings(&graph);
        let step = CompiledStep::compile(&graph, &plan, backend.as_ref(), &mutable)?;
        install_cpu_storage(backend.as_ref(), &graph)?;
        let fused_len = spec.d_model * target_layers.len();
        Ok((
            Self {
                support,
                target_layers,
                noise_token_id,
                hidden_size: spec.d_model,
                hc_mult: spec.hc_mult.max(1),
                backend,
                graph,
                step,
                rope: bindings.rope,
                attention: bindings.attention,
                _kv: kv,
                fused: vec![0.0; fused_len],
                context_length: 0,
            },
            heads,
            features,
            block_size,
        ))
    }

    fn pool_layer(&self, values: &[f32]) -> Result<Vec<f32>> {
        if values.len() == self.hidden_size {
            return Ok(values.to_vec());
        }
        if values.len() != self.hidden_size * self.hc_mult {
            return Err(FellmError::other(format!(
                "DSpark target feature width {} does not match hidden {} or hc {}",
                values.len(),
                self.hidden_size,
                self.hc_mult
            )));
        }
        let mut pooled = vec![0.0f32; self.hidden_size];
        for stream in 0..self.hc_mult {
            let start = stream * self.hidden_size;
            for (dst, &src) in pooled.iter_mut().zip(&values[start..start + self.hidden_size]) {
                *dst += src;
            }
        }
        let scale = 1.0 / self.hc_mult as f32;
        for value in &mut pooled {
            *value *= scale;
        }
        Ok(pooled)
    }

    fn run_token(&mut self, token: u32, position: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        for &node in &self.rope {
            let mut attrs = self.graph.node(node).attrs;
            attrs.position = position;
            self.step.set_attrs(node, attrs);
        }
        for &node in &self.attention {
            let mut attrs = self.graph.node(node).attrs;
            attrs.past_len = position;
            attrs.position = position;
            self.step.set_attrs(node, attrs);
        }
        self.step.bind_input("token_id", u32_tensor(&[token])?);
        self.step.bind_input(
            "target_features",
            f32_tensor(&self.fused, &[self.fused.len() as u64])?,
        );
        self.backend.begin_step();
        let logits = self.step.run(self.backend.as_ref(), true);
        self.backend.end_step();
        let logits = logits?;
        let hidden = self
            .step
            .materialize_named_output(self.backend.as_ref(), "draft_hidden")?;
        Ok((logits.as_slice::<f32>()?.to_vec(), hidden.as_slice::<f32>()?.to_vec()))
    }
}

impl ParallelBackbone for DeepseekDsparkBackbone {
    fn observe_committed(&mut self, observation: &CommittedTargetToken<'_>) -> Result<()> {
        let mut fused = Vec::with_capacity(self.hidden_size * self.target_layers.len());
        for &layer in &self.target_layers {
            let capture = observation
                .captured_features
                .iter()
                .find(|capture| capture.feature == TargetFeature::LayerHiddenState(layer))
                .ok_or_else(|| {
                    FellmError::other(format!("DSpark missing target layer {layer}"))
                })?;
            let values: &[f32] = bytemuck::try_cast_slice(unsafe { capture.tensor.as_bytes() })
                .map_err(|_| FellmError::other("unaligned DSpark target feature"))?;
            fused.extend(self.pool_layer(values)?);
        }
        self.fused = fused;
        self.run_token(observation.token, observation.position)?;
        self.context_length = observation.position as usize + 1;
        Ok(())
    }

    fn draft_block(&mut self, context: &SpeculatorContext<'_>) -> Result<DsparkBackboneOutput> {
        let rows = context.maximum_length.max(1) as usize;
        if context.prefix_tokens.is_empty() {
            return Err(FellmError::other("DSpark requires a preceding target token"));
        }
        let mut ids = vec![self.noise_token_id; rows];
        let mut base_logits = Vec::with_capacity(rows);
        let mut hidden_states = Vec::with_capacity(rows);
        let start = self.context_length as u32;
        for (offset, token) in ids.into_iter().enumerate() {
            let (logits, hidden) = self.run_token(token, start + offset as u32)?;
            base_logits.push(logits);
            hidden_states.push(hidden);
        }
        let _ = &self.support;
        Ok(DsparkBackboneOutput {
            base_logits,
            hidden_states,
        })
    }

    fn reset(&mut self) {
        self.context_length = 0;
        self.fused.fill(0.0);
    }
}

fn markov_heads_from_support(
    support: &GgufFile,
    vocab: usize,
    rank: usize,
    hidden: usize,
) -> Result<MarkovHeads> {
    let w1 = dequant_tensor(&support.tensor("mtp.2.markov_head.markov_w1.weight")?)?;
    let w2 = dequant_tensor(&support.tensor("mtp.2.markov_head.markov_w2.weight")?)?;
    let conf = dequant_tensor(&support.tensor("mtp.2.confidence_head.proj.weight")?)?;
    MarkovHeads::new(vocab, rank, hidden, w1, w2, conf, 0.0, 1.0)
}

fn dequant_tensor(tensor: &Tensor) -> Result<Vec<f32>> {
    let n = tensor.shape().num_elements();
    let mut out = vec![0.0f32; n];
    let bytes_storage;
    let bytes: &[u8] = match tensor.storage().as_ref() {
        Storage::FileExtent { path, offset, len } => {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(path.as_ref())?;
            file.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len];
            file.read_exact(&mut buf)?;
            bytes_storage = Some(buf);
            bytes_storage.as_deref().expect("filled")
        }
        _ => {
            bytes_storage = None;
            tensor.as_bytes()
        }
    };
    backend_cpu::dequant::dequantize_row(tensor.dtype(), bytes, &mut out, n)?;
    Ok(out)
}

fn install_cpu_storage(backend: &dyn fellm_plugin_abi::Backend, graph: &fellm_graph::Graph) -> Result<()> {
    let Some(cpu) = backend.as_any().downcast_ref::<backend_cpu::CpuBackend>() else {
        return Ok(());
    };
    let mut weights = Vec::new();
    for (_, node) in graph.iter_nodes() {
        let fellm_graph::graph::OpValue::Constant(tensor) = &node.value else {
            continue;
        };
        let Some((path, offset, len)) = tensor.file_extent() else {
            continue;
        };
        weights.push(fellm_memory::WeightDescriptor {
            id: fellm_memory::WeightId(tensor.logical_id()),
            name: node.label.clone(),
            home: fellm_memory::StorageExtent {
                provider: "file".into(),
                path: path.clone(),
                offset,
                len: len as u64,
                alignment: 4096,
            },
            byte_len: len as u64,
            replicas: Vec::new(),
        });
    }
    if weights.is_empty() {
        return Ok(());
    }
    let streamed = weights
        .iter()
        .filter(|weight| !fellm_memory::is_moe_expert_bank(&weight.name))
        .cloned()
        .collect::<Vec<_>>();
    let groups = streamed
        .iter()
        .enumerate()
        .map(|(index, weight)| fellm_memory::ExecutionGroup {
            id: index as u32,
            weights: vec![weight.id],
            byte_len: weight.byte_len,
            first_op: 0,
            last_op: 0,
            reuse_count: 1,
            cpu_compute_time: None,
        })
        .collect::<Vec<_>>();
    let objects = fellm_memory::StorageObjectIndex::from_execution_groups(
        &streamed,
        &groups,
        0,
        u64::MAX,
    )?;
    cpu.configure_weight_storage(
        &weights,
        &objects,
        fellm_memory::StorageProviderKind::Buffered,
        24,
        0,
        false,
    )
}

fn u32_tensor(ids: &[u32]) -> Result<Tensor> {
    let mut buf = AlignedBuffer::new_zeroed(ids.len() * 4, 64);
    buf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(ids));
    Ok(Tensor::from_storage(
        Layout::contiguous(DType::U32, Shape::new(&[ids.len() as u64])?),
        Arc::new(Storage::Owned(Arc::new(buf))),
    ))
}

fn f32_tensor(values: &[f32], dims: &[u64]) -> Result<Tensor> {
    let mut buf = AlignedBuffer::new_zeroed(values.len() * 4, 64);
    buf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(values));
    Ok(Tensor::from_storage(
        Layout::contiguous(DType::F32, Shape::new(dims)?),
        Arc::new(Storage::Owned(Arc::new(buf))),
    ))
}
