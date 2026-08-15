//! Lowering from the backend-neutral tensor DAG to CUDA decode macro-operations.
//!
//! This module performs compile-time work only. The resulting plan contains no
//! host tensor identity and is suitable for stable CUDA graph replay.

use fellm_core::error::{FellmError, Result};
use fellm_graph::Graph;
use fellm_graph::graph::OpValue;
use fellm_graph::plan::ExecutionPlan;
#[cfg(feature = "backend-cuda")]
use fellm_plugin_abi::PhysicalPlan;
use fellm_plugin_abi::{MacroOpKind, PlanTensorDesc, PlanTensorId, PreparedMacroOp, StorageClass};
use std::collections::HashMap;

/// Backend-neutral input to the CUDA plan compiler.
#[derive(Debug)]
pub struct LoweredDecodeGraph {
    /// Tensor lifetimes and storage classes.
    pub tensors: Vec<PlanTensorDesc>,
    /// Recognized transformer macro-operations.
    pub operations: Vec<PreparedMacroOp>,
}

/// Lower a semantic autoregressive graph into tensor lifetimes and macro-ops.
pub fn lower_decode_graph(graph: &Graph, plan: &ExecutionPlan) -> Result<LoweredDecodeGraph> {
    let index: HashMap<_, _> = plan
        .order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut last_use: Vec<u32> = (0..plan.order.len()).map(|i| i as u32).collect();
    for (consumer, &id) in plan.order.iter().enumerate() {
        for input in graph.inputs_slice(id) {
            let producer = *index
                .get(input)
                .ok_or_else(|| FellmError::other("decode input is absent from execution plan"))?;
            last_use[producer] = last_use[producer].max(consumer as u32);
        }
    }

    let mut tensors = Vec::new();
    for (i, &id) in plan.order.iter().enumerate() {
        let node = graph.node(id);
        let (dtype, shape, storage) = match &node.value {
            OpValue::Runtime { dtype, shape } => {
                let storage = if node.op == Some(fellm_plugin_abi::OpKind::KvWrite) {
                    StorageClass::Request
                } else {
                    StorageClass::Transient
                };
                (*dtype, shape, storage)
            }
            OpValue::Constant(tensor) => (tensor.dtype(), tensor.shape(), StorageClass::Model),
            OpValue::Input { dtype, shape, .. } => (*dtype, shape, StorageClass::Control),
            OpValue::Output { .. } => continue,
        };
        tensors.push(PlanTensorDesc {
            id: PlanTensorId(i as u32),
            dtype,
            shape: shape.dims().to_vec(),
            alignment: required_alignment(node.label.as_str()),
            storage,
            first_use: i as u32,
            last_use: last_use[i],
        });
    }

    let operations = recognize_macro_ops(graph, plan, &index)?;
    Ok(LoweredDecodeGraph {
        tensors,
        operations,
    })
}

fn required_alignment(label: &str) -> usize {
    if label.ends_with("q_proj")
        || label.ends_with("k_proj")
        || label.ends_with("v_proj")
        || label.contains("ffn_")
    {
        128
    } else {
        64
    }
}

fn recognize_macro_ops(
    graph: &Graph,
    plan: &ExecutionPlan,
    index: &HashMap<fellm_graph::graph::NodeId, usize>,
) -> Result<Vec<PreparedMacroOp>> {
    let by_label: HashMap<&str, fellm_graph::graph::NodeId> = graph
        .iter_nodes()
        .map(|(id, node)| (node.label.as_str(), id))
        .collect();
    let tensor_id = |id: fellm_graph::graph::NodeId| {
        index
            .get(&id)
            .copied()
            .map(|i| PlanTensorId(i as u32))
            .ok_or_else(|| FellmError::other("macro tensor absent from execution plan"))
    };
    let label_id = |label: &str| {
        by_label
            .get(label)
            .copied()
            .ok_or_else(|| FellmError::other(format!("required CUDA macro node {label} is absent")))
    };
    let mut operations = Vec::new();
    for &id in &plan.order {
        let node = graph.node(id);
        let label = node.label.as_str();
        // Macro recognition is opportunistic. Architectures may legitimately
        // reuse these semantic suffixes for fused/unary operations; those must
        // remain ordinary graph nodes instead of being indexed as dense GEMMs.
        if (label.ends_with(".q_proj") && graph.inputs_slice(id).len() < 2)
            || (label.ends_with(".residual2") && graph.inputs_slice(id).len() < 2)
        {
            continue;
        }
        let (kind, input_nodes, output_nodes) = if label == "tok_embed" {
            (MacroOpKind::Embedding, graph.inputs_of(id), vec![id])
        } else if label.ends_with(".attn_norm_op") || label.ends_with(".ffn_norm_op") {
            (
                MacroOpKind::RmsNormQuantizeQ8_1,
                graph.inputs_of(id),
                vec![id],
            )
        } else if let Some(prefix) = label.strip_suffix(".q_proj") {
            let k = label_id(&format!("{prefix}.k_proj"))?;
            let v = label_id(&format!("{prefix}.v_proj"))?;
            let q_inputs = graph.inputs_slice(id);
            let k_inputs = graph.inputs_slice(k);
            let v_inputs = graph.inputs_slice(v);
            if k_inputs.is_empty() || v_inputs.is_empty() {
                continue;
            }
            (
                MacroOpKind::QkvMmvq,
                vec![q_inputs[1], q_inputs[0], k_inputs[0], v_inputs[0]],
                vec![id, k, v],
            )
        } else if let Some(prefix) = label.strip_suffix(".k_write") {
            let q_rope = label_id(&format!("{prefix}.q_rope"))?;
            let v_write = label_id(&format!("{prefix}.v_write"))?;
            let mut inputs = graph.inputs_of(q_rope);
            inputs.extend_from_slice(graph.inputs_slice(id));
            inputs.extend_from_slice(graph.inputs_slice(v_write));
            (MacroOpKind::RopeKvCommit, inputs, vec![q_rope, id, v_write])
        } else if label.ends_with(".attn") {
            (
                MacroOpKind::PagedAttentionDecode,
                graph.inputs_of(id),
                vec![id],
            )
        } else if label.ends_with(".o_proj_residual") {
            (
                MacroOpKind::OutputProjectionResidual,
                graph.inputs_of(id),
                vec![id],
            )
        } else if label.ends_with(".gate_up_swiglu") {
            (MacroOpKind::GateUpMmvqSwiglu, graph.inputs_of(id), vec![id])
        } else if label.ends_with(".ffn_down_residual") {
            (
                MacroOpKind::DownProjectionResidual,
                graph.inputs_of(id),
                vec![id],
            )
        } else if label.ends_with(".residual2") {
            let residual_inputs = graph.inputs_slice(id);
            let ffn = residual_inputs[1];
            if graph.node(ffn).label.ends_with(".ffn_down_proj") {
                let mut inputs = graph.inputs_of(ffn);
                inputs.push(residual_inputs[0]);
                (MacroOpKind::DownProjectionResidual, inputs, vec![id])
            } else if graph.node(ffn).label.contains(".moe") {
                let mut inputs = graph.inputs_of(ffn);
                inputs.push(residual_inputs[0]);
                (MacroOpKind::GroupedMoe, inputs, vec![id])
            } else {
                continue;
            }
        } else if label == "lm_head" {
            (MacroOpKind::LmHeadSample, graph.inputs_of(id), vec![id])
        } else {
            continue;
        };
        let inputs = input_nodes
            .into_iter()
            .map(tensor_id)
            .collect::<Result<Vec<_>>>()?;
        let outputs = output_nodes
            .into_iter()
            .map(tensor_id)
            .collect::<Result<Vec<_>>>()?;
        operations.push(PreparedMacroOp {
            kind,
            inputs,
            outputs,
            // Zero means variant selection remains for the CUDA compiler/autotuner.
            kernel_variant: 0,
            // Filled at provider prepare time for attention ops.
            provider_id: 0,
        });
    }
    Ok(operations)
}

#[cfg(test)]
fn macro_kind(label: &str) -> Option<MacroOpKind> {
    if label == "tok_embed" {
        Some(MacroOpKind::Embedding)
    } else if label.ends_with(".attn_norm_op") || label.ends_with(".ffn_norm_op") {
        Some(MacroOpKind::RmsNormQuantizeQ8_1)
    } else if label.ends_with(".q_proj") {
        // The CUDA compiler consumes adjacent q/k/v projections as one packed op.
        Some(MacroOpKind::QkvMmvq)
    } else if label.ends_with(".k_write") {
        Some(MacroOpKind::RopeKvCommit)
    } else if label.ends_with(".attn") {
        Some(MacroOpKind::PagedAttentionDecode)
    } else if label.ends_with(".o_proj") {
        Some(MacroOpKind::OutputProjectionResidual)
    } else if label.ends_with(".ffn_gate_proj") {
        Some(MacroOpKind::GateUpMmvqSwiglu)
    } else if label.ends_with(".ffn_down_proj") {
        Some(MacroOpKind::DownProjectionResidual)
    } else if label.contains(".moe") {
        Some(MacroOpKind::GroupedMoe)
    } else if label == "lm_head" {
        Some(MacroOpKind::LmHeadSample)
    } else {
        None
    }
}

/// Attach CUDA-owned arena assignments to a lowered semantic plan.
#[cfg(feature = "backend-cuda")]
pub fn compile_cuda_layout(lowered: &LoweredDecodeGraph) -> Result<PhysicalPlan> {
    let mut physical = backend_cuda::plan_static_arena(&lowered.tensors)?;
    physical.operations.clone_from(&lowered.operations);
    Ok(physical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_dense_decode_macro_boundaries() {
        assert_eq!(macro_kind("tok_embed"), Some(MacroOpKind::Embedding));
        assert_eq!(
            macro_kind("blk.3.attn_norm_op"),
            Some(MacroOpKind::RmsNormQuantizeQ8_1)
        );
        assert_eq!(macro_kind("blk.3.q_proj"), Some(MacroOpKind::QkvMmvq));
        assert_eq!(
            macro_kind("blk.3.ffn_gate_proj"),
            Some(MacroOpKind::GateUpMmvqSwiglu)
        );
        assert_eq!(macro_kind("blk.3.k_proj"), None);
    }
}
