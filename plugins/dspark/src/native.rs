use crate::{DsparkBackboneOutput, DsparkCheckpoint, ParallelBackbone};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_graph::{ExecutionPlan, Graph, GraphBuilder, NodeId};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::{CommittedTargetToken, SpeculatorContext, TargetFeature};
use fellm_runtime::compiled::{CompiledStep, MutableBinding};
use fellm_runtime::{BackendSelect, DummyKvBuffers};
use std::collections::HashMap;
use std::sync::Arc;

struct ContextGraph {
    graph: Graph,
    step: CompiledStep,
    rope: Vec<NodeId>,
    writes: Vec<NodeId>,
}

struct ProposalGraph {
    graph: Graph,
    step: CompiledStep,
    rope: Vec<NodeId>,
    attention: Vec<NodeId>,
}

/// Request-local Qwen3 DSpark executor backed by ordinary FeLLM graphs.
pub struct NativeDsparkBackbone {
    checkpoint: Arc<DsparkCheckpoint>,
    backend: Box<dyn fellm_plugin_abi::Backend>,
    context: ContextGraph,
    proposal: ProposalGraph,
    _kv: DummyKvBuffers,
    context_length: usize,
    capacity: usize,
}

impl NativeDsparkBackbone {
    pub fn new(
        checkpoint: Arc<DsparkCheckpoint>,
        backend: BackendSelect,
        capacity: usize,
    ) -> Result<Self> {
        if checkpoint.config.architectures[0] != "Qwen3DSparkModel" {
            return Err(FellmError::other(
                "native DSpark currently supports released Qwen3 checkpoints",
            ));
        }
        let backend = backend.resolve()?;
        const PREFERRED_WEIGHT_GROUP_BYTES: u64 = 16 * 1024 * 1024;
        backend.ensure_weight_group_capacity_bytes(PREFERRED_WEIGHT_GROUP_BYTES)?;
        let weight_group_capacity = backend
            .weight_group_capacity_bytes()
            .and_then(|bytes| usize::try_from(bytes).ok());
        let c = &checkpoint.config;
        let kv = DummyKvBuffers::new(
            c.num_hidden_layers,
            capacity,
            c.num_key_value_heads,
            c.head_dim,
        )?;
        let mutable = mutable_kv_bindings(&kv)?;
        let (context_graph, context_rope, writes) =
            build_context_graph(&checkpoint, capacity, weight_group_capacity)?;
        let plan = ExecutionPlan::from_graph(&context_graph)?;
        let context_step =
            CompiledStep::compile(&context_graph, &plan, backend.as_ref(), &mutable)?;
        let (proposal_graph, proposal_rope, attention) =
            build_proposal_graph(&checkpoint, capacity, weight_group_capacity)?;
        let plan = ExecutionPlan::from_graph(&proposal_graph)?;
        let proposal_step =
            CompiledStep::compile(&proposal_graph, &plan, backend.as_ref(), &mutable)?;
        Ok(Self {
            checkpoint,
            backend,
            context: ContextGraph {
                graph: context_graph,
                step: context_step,
                rope: context_rope,
                writes,
            },
            proposal: ProposalGraph {
                graph: proposal_graph,
                step: proposal_step,
                rope: proposal_rope,
                attention,
            },
            _kv: kv,
            context_length: 0,
            capacity,
        })
    }

    fn append_feature_row(&mut self, observation: &CommittedTargetToken<'_>) -> Result<()> {
        if observation.position as usize != self.context_length {
            return Err(FellmError::other(format!(
                "DSpark context position {} does not match private KV length {}",
                observation.position, self.context_length
            )));
        }
        if self.context_length >= self.capacity {
            return Err(FellmError::other("DSpark private KV capacity exceeded"));
        }
        let c = &self.checkpoint.config;
        let mut fused = Vec::with_capacity(c.target_layer_ids.len() * c.hidden_size);
        for &layer in &c.target_layer_ids {
            let capture = observation
                .captured_features
                .iter()
                .find(|capture| capture.feature == TargetFeature::LayerHiddenState(layer))
                .ok_or_else(|| {
                    FellmError::other(format!(
                        "DSpark observation is missing target layer {layer}"
                    ))
                })?;
            if capture.tensor.dtype() != Some(DType::F32)
                || capture.tensor.byte_len as usize != c.hidden_size * 4
            {
                return Err(FellmError::other(format!(
                    "DSpark target layer {layer} feature shape mismatch"
                )));
            }
            // SAFETY: the observation owns the feature view for this call.
            let values: &[f32] = bytemuck::try_cast_slice(unsafe { capture.tensor.as_bytes() })
                .map_err(|_| FellmError::other("unaligned DSpark target feature"))?;
            fused.extend_from_slice(values);
        }
        let position = self.context_length as u32;
        patch_position(
            &self.context.graph,
            &mut self.context.step,
            &self.context.rope,
            position,
        );
        patch_position(
            &self.context.graph,
            &mut self.context.step,
            &self.context.writes,
            position,
        );
        self.context.step.bind_input(
            "target_features",
            f32_tensor(&fused, &[fused.len() as u64])?,
        );
        self.backend.begin_step();
        let result = self.context.step.run(self.backend.as_ref(), true);
        self.backend.end_step();
        result?;
        self.context_length += 1;
        Ok(())
    }
}

impl ParallelBackbone for NativeDsparkBackbone {
    fn draft_block(&mut self, context: &SpeculatorContext<'_>) -> Result<DsparkBackboneOutput> {
        let c = &self.checkpoint.config;
        let rows = c.block_size as usize;
        if self.context_length + rows > self.capacity {
            return Err(FellmError::other(
                "DSpark proposal exceeds private KV capacity",
            ));
        }
        let mut ids = vec![c.mask_token_id; rows];
        ids[0] = context
            .prefix_tokens
            .last()
            .copied()
            .ok_or_else(|| FellmError::other("DSpark requires a preceding target token"))?;
        patch_position(
            &self.proposal.graph,
            &mut self.proposal.step,
            &self.proposal.rope,
            self.context_length as u32,
        );
        for &node in &self.proposal.attention {
            let mut attrs = self.proposal.graph.node(node).attrs;
            attrs.past_len = self.context_length as u32;
            self.proposal.step.set_attrs(node, attrs);
        }
        let mut embeddings = Vec::with_capacity(rows * c.hidden_size);
        for &token in &ids {
            embeddings.extend(
                self.checkpoint
                    .tensor_row_f32("embed_tokens.weight", token as usize)?,
            );
        }
        self.proposal.step.bind_input(
            "noise_embeddings",
            f32_tensor(&embeddings, &[rows as u64, c.hidden_size as u64])?,
        );
        self.backend.begin_step();
        let logits = self.proposal.step.run(self.backend.as_ref(), true);
        self.backend.end_step();
        let logits = logits?;
        let hidden = self
            .proposal
            .step
            .materialize_named_output(self.backend.as_ref(), "draft_hidden")?;
        Ok(DsparkBackboneOutput {
            base_logits: logits
                .as_slice::<f32>()?
                .chunks_exact(c.vocab_size)
                .map(<[f32]>::to_vec)
                .collect(),
            hidden_states: hidden
                .as_slice::<f32>()?
                .chunks_exact(c.hidden_size)
                .map(<[f32]>::to_vec)
                .collect(),
        })
    }

    fn observe_committed(&mut self, observation: &CommittedTargetToken<'_>) -> Result<()> {
        self.append_feature_row(observation)
    }

    fn reset(&mut self) {
        self.context_length = 0;
    }
}

fn patch_position(graph: &Graph, step: &mut CompiledStep, nodes: &[NodeId], position: u32) {
    for &node in nodes {
        let mut attrs = graph.node(node).attrs;
        attrs.position = position;
        step.set_attrs(node, attrs);
    }
}

fn mutable_kv_bindings(kv: &DummyKvBuffers) -> Result<HashMap<String, MutableBinding>> {
    let shape = Shape::new(&[kv.max_seq as u64, kv.tokens_stride as u64])?;
    let mut result = HashMap::with_capacity(kv.n_layers * 2);
    for layer in 0..kv.n_layers {
        result.insert(
            format!("k_in_{layer}"),
            MutableBinding {
                dtype: DType::F32,
                shape: shape.clone(),
                buffer: kv.k_buffer(layer),
            },
        );
        result.insert(
            format!("v_in_{layer}"),
            MutableBinding {
                dtype: DType::F32,
                shape: shape.clone(),
                buffer: kv.v_buffer(layer),
            },
        );
    }
    Ok(result)
}

fn build_context_graph(
    checkpoint: &DsparkCheckpoint,
    capacity: usize,
    weight_group_capacity: Option<usize>,
) -> Result<(Graph, Vec<NodeId>, Vec<NodeId>)> {
    let c = &checkpoint.config;
    let h = c.hidden_size;
    let kv_width = c.num_key_value_heads * c.head_dim;
    let mut gb = GraphBuilder::new();
    let features = gb.input(
        "target_features",
        DType::F32,
        Shape::new(&[(c.target_layer_ids.len() * h) as u64])?,
    );
    let projected = chunked_projection(
        &mut gb,
        checkpoint,
        "fc.weight",
        features,
        1,
        h,
        "context_fc".into(),
        weight_group_capacity,
    )?;
    let norm = gb.constant("hidden_norm", checkpoint.tensor("hidden_norm.weight")?);
    let target = gb.op(
        OpKind::RmsNorm,
        norm_attrs(1, h),
        DType::F32,
        Shape::new(&[h as u64])?,
        &[projected, norm],
        "context_hidden_norm",
    );
    let inv_freq = gb.constant("inv_freq", inv_freq_tensor(c.head_dim, 1_000_000.0)?);
    let cache_shape = Shape::new(&[capacity as u64, kv_width as u64])?;
    let mut rope = Vec::new();
    let mut writes = Vec::new();
    for layer in 0..c.num_hidden_layers {
        let base = format!("layers.{layer}.self_attn");
        let wk = gb.constant(
            "context_k_weight",
            checkpoint.tensor(&format!("{base}.k_proj.weight"))?,
        );
        let wv = gb.constant(
            "context_v_weight",
            checkpoint.tensor(&format!("{base}.v_proj.weight"))?,
        );
        let k = matmul(
            &mut gb,
            wk,
            target,
            &[kv_width as u64],
            format!("context.{layer}.k"),
        )?;
        let v = matmul(
            &mut gb,
            wv,
            target,
            &[kv_width as u64],
            format!("context.{layer}.v"),
        )?;
        let kn = gb.constant(
            "context_k_norm",
            checkpoint.tensor(&format!("{base}.k_norm.weight"))?,
        );
        let k = gb.op(
            OpKind::RmsNorm,
            norm_attrs(c.num_key_value_heads, c.head_dim),
            DType::F32,
            Shape::new(&[kv_width as u64])?,
            &[k, kn],
            format!("context.{layer}.k_norm"),
        );
        let k = gb.op(
            OpKind::Rope,
            rope_attrs(c.num_key_value_heads, c.head_dim),
            DType::F32,
            Shape::new(&[kv_width as u64])?,
            &[k, inv_freq],
            format!("context.{layer}.k_rope"),
        );
        rope.push(k);
        let kc = gb.input(format!("k_in_{layer}"), DType::F32, cache_shape.clone());
        let vc = gb.input(format!("v_in_{layer}"), DType::F32, cache_shape.clone());
        let kw = gb.op_in_place(
            OpKind::KvWrite,
            OpAttrs {
                layer_ord: layer as u32,
                kv_slot: 0,
                ..Default::default()
            },
            DType::F32,
            cache_shape.clone(),
            &[k, kc],
            1,
            format!("context.{layer}.k_write"),
        );
        let vw = gb.op_in_place(
            OpKind::KvWrite,
            OpAttrs {
                layer_ord: layer as u32,
                kv_slot: 1,
                ..Default::default()
            },
            DType::F32,
            cache_shape.clone(),
            &[v, vc],
            1,
            format!("context.{layer}.v_write"),
        );
        writes.extend([kw, vw]);
    }
    gb.mark_output("logits", target);
    Ok((gb.build()?, rope, writes))
}

fn build_proposal_graph(
    checkpoint: &DsparkCheckpoint,
    capacity: usize,
    weight_group_capacity: Option<usize>,
) -> Result<(Graph, Vec<NodeId>, Vec<NodeId>)> {
    let c = &checkpoint.config;
    let rows = c.block_size as usize;
    let h = c.hidden_size;
    let q_width = c.num_attention_heads * c.head_dim;
    let kv_width = c.num_key_value_heads * c.head_dim;
    let ff = c.intermediate_size;
    let mut gb = GraphBuilder::new();
    let mut x = gb.input(
        "noise_embeddings",
        DType::F32,
        Shape::new(&[rows as u64, h as u64])?,
    );
    let inv = gb.constant("inv_freq", inv_freq_tensor(c.head_dim, 1_000_000.0)?);
    let cache_shape = Shape::new(&[capacity as u64, kv_width as u64])?;
    let mut rope_nodes = Vec::new();
    let mut attention_nodes = Vec::new();
    for layer in 0..c.num_hidden_layers {
        let base = format!("layers.{layer}");
        let norm = gb.constant(
            "input_norm",
            checkpoint.tensor(&format!("{base}.input_layernorm.weight"))?,
        );
        let n = gb.op(
            OpKind::RmsNorm,
            norm_attrs(1, h),
            DType::F32,
            Shape::new(&[rows as u64, h as u64])?,
            &[x, norm],
            format!("draft.{layer}.attn_norm"),
        );
        let attn = format!("{base}.self_attn");
        let q = projection(
            &mut gb,
            checkpoint,
            &format!("{attn}.q_proj.weight"),
            n,
            rows,
            q_width,
            format!("draft.{layer}.q"),
            weight_group_capacity,
        )?;
        let k = projection(
            &mut gb,
            checkpoint,
            &format!("{attn}.k_proj.weight"),
            n,
            rows,
            kv_width,
            format!("draft.{layer}.k"),
            weight_group_capacity,
        )?;
        let v = projection(
            &mut gb,
            checkpoint,
            &format!("{attn}.v_proj.weight"),
            n,
            rows,
            kv_width,
            format!("draft.{layer}.v"),
            weight_group_capacity,
        )?;
        let qn = gb.constant(
            "q_norm",
            checkpoint.tensor(&format!("{attn}.q_norm.weight"))?,
        );
        let kn = gb.constant(
            "k_norm",
            checkpoint.tensor(&format!("{attn}.k_norm.weight"))?,
        );
        let q = gb.op(
            OpKind::RmsNorm,
            norm_attrs(c.num_attention_heads, c.head_dim),
            DType::F32,
            Shape::new(&[rows as u64, q_width as u64])?,
            &[q, qn],
            format!("draft.{layer}.q_norm"),
        );
        let k = gb.op(
            OpKind::RmsNorm,
            norm_attrs(c.num_key_value_heads, c.head_dim),
            DType::F32,
            Shape::new(&[rows as u64, kv_width as u64])?,
            &[k, kn],
            format!("draft.{layer}.k_norm"),
        );
        let q = gb.op(
            OpKind::Rope,
            rope_attrs(c.num_attention_heads, c.head_dim),
            DType::F32,
            Shape::new(&[rows as u64, q_width as u64])?,
            &[q, inv],
            format!("draft.{layer}.q_rope"),
        );
        let k = gb.op(
            OpKind::Rope,
            rope_attrs(c.num_key_value_heads, c.head_dim),
            DType::F32,
            Shape::new(&[rows as u64, kv_width as u64])?,
            &[k, inv],
            format!("draft.{layer}.k_rope"),
        );
        rope_nodes.extend([q, k]);
        let pk = gb.input(format!("k_in_{layer}"), DType::F32, cache_shape.clone());
        let pv = gb.input(format!("v_in_{layer}"), DType::F32, cache_shape.clone());
        let mixed = gb.op(
            OpKind::Attention,
            OpAttrs {
                n_heads: c.num_attention_heads as u32,
                n_kv_heads: c.num_key_value_heads as u32,
                head_dim: c.head_dim as u32,
                layer_ord: layer as u32,
                attention_mode: 1,
                query_len: rows as u32,
                scale: 1.0 / (c.head_dim as f32).sqrt(),
                ..Default::default()
            },
            DType::F32,
            Shape::new(&[rows as u64, q_width as u64])?,
            &[q, pk, pv, k, v],
            format!("draft.{layer}.attention"),
        );
        attention_nodes.push(mixed);
        let mixed = projection(
            &mut gb,
            checkpoint,
            &format!("{attn}.o_proj.weight"),
            mixed,
            rows,
            h,
            format!("draft.{layer}.o"),
            weight_group_capacity,
        )?;
        let residual = add(
            &mut gb,
            x,
            mixed,
            rows,
            h,
            format!("draft.{layer}.residual1"),
        )?;
        let post = gb.constant(
            "post_norm",
            checkpoint.tensor(&format!("{base}.post_attention_layernorm.weight"))?,
        );
        let ffn_in = gb.op(
            OpKind::RmsNorm,
            norm_attrs(1, h),
            DType::F32,
            Shape::new(&[rows as u64, h as u64])?,
            &[residual, post],
            format!("draft.{layer}.ffn_norm"),
        );
        let gate = projection(
            &mut gb,
            checkpoint,
            &format!("{base}.mlp.gate_proj.weight"),
            ffn_in,
            rows,
            ff,
            format!("draft.{layer}.gate"),
            weight_group_capacity,
        )?;
        let up = projection(
            &mut gb,
            checkpoint,
            &format!("{base}.mlp.up_proj.weight"),
            ffn_in,
            rows,
            ff,
            format!("draft.{layer}.up"),
            weight_group_capacity,
        )?;
        let activated = gb.op(
            OpKind::SiluGate,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[rows as u64, ff as u64])?,
            &[gate, up],
            format!("draft.{layer}.silu"),
        );
        let down = projection(
            &mut gb,
            checkpoint,
            &format!("{base}.mlp.down_proj.weight"),
            activated,
            rows,
            h,
            format!("draft.{layer}.down"),
            weight_group_capacity,
        )?;
        x = add(
            &mut gb,
            residual,
            down,
            rows,
            h,
            format!("draft.{layer}.residual2"),
        )?;
    }
    let norm = gb.constant("final_norm_weight", checkpoint.tensor("norm.weight")?);
    let hidden = gb.op(
        OpKind::RmsNorm,
        norm_attrs(1, h),
        DType::F32,
        Shape::new(&[rows as u64, h as u64])?,
        &[x, norm],
        "final_norm",
    );
    let logits = chunked_projection(
        &mut gb,
        checkpoint,
        "lm_head.weight",
        hidden,
        rows,
        c.vocab_size,
        "lm_head".into(),
        weight_group_capacity,
    )?;
    gb.mark_output("draft_hidden", hidden);
    gb.mark_output("logits", logits);
    Ok((gb.build()?, rope_nodes, attention_nodes))
}

fn projection(
    gb: &mut GraphBuilder,
    checkpoint: &DsparkCheckpoint,
    name: &str,
    input: NodeId,
    rows: usize,
    width: usize,
    label: String,
    weight_group_capacity: Option<usize>,
) -> Result<NodeId> {
    chunked_projection(
        gb,
        checkpoint,
        name,
        input,
        rows,
        width,
        label,
        weight_group_capacity,
    )
}

fn chunked_projection(
    gb: &mut GraphBuilder,
    checkpoint: &DsparkCheckpoint,
    name: &str,
    input: NodeId,
    rows: usize,
    width: usize,
    label: String,
    weight_group_capacity: Option<usize>,
) -> Result<NodeId> {
    const DEFAULT_WEIGHT_CHUNK_BYTES: usize = 16 * 1024 * 1024;
    let max_weight_chunk_bytes = weight_group_capacity
        .unwrap_or(DEFAULT_WEIGHT_CHUNK_BYTES)
        .min(DEFAULT_WEIGHT_CHUNK_BYTES)
        .max(1);
    let tensor = checkpoint.tensor(name)?;
    let input_width = tensor.shape().dims()[1] as usize;
    let bytes_per_row = tensor.dtype().byte_size(input_width);
    let chunk_rows = (max_weight_chunk_bytes / bytes_per_row).max(1);
    let mut chunks = Vec::with_capacity(width.div_ceil(chunk_rows));
    for start in (0..width).step_by(chunk_rows) {
        let count = (width - start).min(chunk_rows);
        let weight = gb.constant(
            format!("{label}.weight.{start}"),
            tensor.rows(start, count)?,
        );
        chunks.push((
            matmul(
                gb,
                weight,
                input,
                &[rows as u64, count as u64],
                format!("{label}.{start}"),
            )?,
            count,
        ));
    }
    let (mut output, mut accumulated) = chunks[0];
    for (index, &(chunk, count)) in chunks.iter().enumerate().skip(1) {
        accumulated += count;
        output = gb.op(
            OpKind::Concat,
            OpAttrs::default(),
            DType::F32,
            Shape::new(&[rows as u64, accumulated as u64])?,
            &[output, chunk],
            format!("{label}.concat.{index}"),
        );
    }
    Ok(output)
}

fn matmul(
    gb: &mut GraphBuilder,
    weight: NodeId,
    input: NodeId,
    shape: &[u64],
    label: String,
) -> Result<NodeId> {
    Ok(gb.op(
        OpKind::MatMul,
        OpAttrs::default(),
        DType::F32,
        Shape::new(shape)?,
        &[weight, input],
        label,
    ))
}

fn add(
    gb: &mut GraphBuilder,
    left: NodeId,
    right: NodeId,
    rows: usize,
    width: usize,
    label: String,
) -> Result<NodeId> {
    Ok(gb.op(
        OpKind::Add,
        OpAttrs::default(),
        DType::F32,
        Shape::new(&[rows as u64, width as u64])?,
        &[left, right],
        label,
    ))
}

fn norm_attrs(heads: usize, head_dim: usize) -> OpAttrs {
    OpAttrs {
        eps: 1e-6,
        n_heads: heads as u32,
        head_dim: head_dim as u32,
        ..Default::default()
    }
}

fn rope_attrs(heads: usize, head_dim: usize) -> OpAttrs {
    OpAttrs {
        n_heads: heads as u32,
        head_dim: head_dim as u32,
        rope_dim: head_dim as u32,
        rope_pairing: 1,
        rope_base: 1_000_000.0,
        ..Default::default()
    }
}

fn inv_freq_tensor(head_dim: usize, base: f32) -> Result<Tensor> {
    let values = (0..head_dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    f32_tensor(&values, &[values.len() as u64])
}

fn f32_tensor(values: &[f32], shape: &[u64]) -> Result<Tensor> {
    let mut buffer = AlignedBuffer::new_zeroed(values.len() * 4, 64);
    buffer
        .as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(values));
    Ok(Tensor::from_storage(
        Layout::contiguous(DType::F32, Shape::new(shape)?),
        Arc::new(Storage::Owned(Arc::new(buffer))),
    ))
}
