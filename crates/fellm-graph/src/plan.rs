//! Execution plan: topologically sorted graph.

use crate::graph::{Graph, NodeId, OpValue};
use fellm_core::error::{FellmError, Result};
use petgraph::algo::toposort;
use std::collections::HashMap;

/// A frozen, topologically-sorted execution order.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Node ids in execution order.
    pub order: Vec<NodeId>,
    /// Reusable host/device-independent activation layout.
    pub memory: MemoryPlan,
}

/// One runtime tensor's placement in the reusable activation arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorAllocation {
    /// Byte offset from the start of the arena.
    pub offset: usize,
    /// Tensor storage size in bytes.
    pub size: usize,
    /// First operation index at which the tensor exists.
    pub first: usize,
    /// Last operation index that consumes the tensor.
    pub last: usize,
}

/// Logical activation-arena plan shared by physical backends.
#[derive(Debug, Clone, Default)]
pub struct MemoryPlan {
    /// Required arena size in bytes.
    pub arena_bytes: usize,
    /// Placement by graph node. Explicit in-place aliases point at the same range.
    pub allocations: HashMap<NodeId, TensorAllocation>,
}

impl ExecutionPlan {
    /// Compute an execution plan from a graph.
    pub fn from_graph(g: &Graph) -> Result<Self> {
        let order = toposort(&g.inner, None)
            .map_err(|c| FellmError::InvalidGraph(format!("cycle at node {:?}", c.node_id())))?;
        let memory = MemoryPlan::from_order(g, &order)?;
        Ok(Self { order, memory })
    }
}

impl MemoryPlan {
    const ALIGN: usize = 64;

    fn from_order(graph: &Graph, order: &[NodeId]) -> Result<Self> {
        let index: HashMap<NodeId, usize> = order
            .iter()
            .enumerate()
            .map(|(position, &id)| (id, position))
            .collect();
        let mut last_use: HashMap<NodeId, usize> = index.clone();
        for (consumer_pos, &id) in order.iter().enumerate() {
            for &input in graph.inputs_slice(id) {
                last_use
                    .entry(input)
                    .and_modify(|last| *last = (*last).max(consumer_pos));
            }
        }

        let mut allocations: HashMap<NodeId, TensorAllocation> = HashMap::new();
        let mut active: Vec<(usize, usize, usize)> = Vec::new(); // last, offset, size
        let mut free: Vec<(usize, usize)> = Vec::new();
        let mut arena_bytes = 0usize;

        for (position, &id) in order.iter().enumerate() {
            let mut cursor = 0;
            while cursor < active.len() {
                if active[cursor].0 < position {
                    let (_, offset, size) = active.swap_remove(cursor);
                    insert_free_range(&mut free, offset, size);
                } else {
                    cursor += 1;
                }
            }

            let node = graph.node(id);
            let OpValue::Runtime { dtype, shape } = &node.value else {
                continue;
            };
            if let Some(slot) = node.in_place_output_from {
                let source = graph.inputs_slice(id).get(slot as usize).ok_or_else(|| {
                    FellmError::InvalidGraph(format!("invalid in-place slot on {}", node.label))
                })?;
                // Mutable graph inputs (KV/recurrent state) are materialized by
                // the runtime, not the activation arena. Their in-place outputs
                // deliberately have no arena allocation.
                let Some(source_alloc) = allocations.get(source).copied() else {
                    continue;
                };
                let size = dtype.byte_size(shape.num_elements());
                if size > source_alloc.size {
                    return Err(FellmError::InvalidGraph(format!(
                        "in-place output {} needs {size} bytes but source has {}",
                        node.label, source_alloc.size
                    )));
                }
                let allocation = TensorAllocation {
                    offset: source_alloc.offset,
                    size,
                    first: position,
                    last: *last_use.get(&id).unwrap_or(&position),
                };
                allocations.insert(id, allocation);
                if let Some(entry) = active
                    .iter_mut()
                    .find(|(_, offset, _)| *offset == source_alloc.offset)
                {
                    entry.0 = entry.0.max(allocation.last);
                }
                continue;
            }

            let size = align_up(dtype.byte_size(shape.num_elements()), Self::ALIGN);
            let offset = take_best_fit(&mut free, size).unwrap_or_else(|| {
                let offset = align_up(arena_bytes, Self::ALIGN);
                arena_bytes = offset.saturating_add(size);
                offset
            });
            let allocation = TensorAllocation {
                offset,
                size,
                first: position,
                last: *last_use.get(&id).unwrap_or(&position),
            };
            allocations.insert(id, allocation);
            active.push((allocation.last, offset, size));
        }
        Ok(Self {
            arena_bytes,
            allocations,
        })
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment).saturating_mul(alignment)
}

fn take_best_fit(free: &mut Vec<(usize, usize)>, size: usize) -> Option<usize> {
    let (index, &(offset, available)) = free
        .iter()
        .enumerate()
        .filter(|(_, (_, available))| *available >= size)
        .min_by_key(|(_, (_, available))| *available)?;
    free.swap_remove(index);
    if available > size {
        insert_free_range(free, offset + size, available - size);
    }
    Some(offset)
}

fn insert_free_range(free: &mut Vec<(usize, usize)>, offset: usize, size: usize) {
    free.push((offset, size));
    free.sort_unstable_by_key(|range| range.0);
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(free.len());
    for &(start, len) in free.iter() {
        if let Some((prior_start, prior_len)) = merged.last_mut()
            && prior_start.saturating_add(*prior_len) == start
        {
            *prior_len = prior_len.saturating_add(len);
        } else {
            merged.push((start, len));
        }
    }
    *free = merged;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphBuilder;
    use fellm_core::{dtype::DType, shape::Shape};
    use fellm_plugin_abi::op::{OpAttrs, OpKind};

    #[test]
    fn reuses_ranges_only_after_last_consumer() {
        let mut builder = GraphBuilder::new();
        let shape = Shape::new(&[16]).unwrap();
        let input = builder.input("input", DType::F32, shape.clone());
        let first = builder.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            shape.clone(),
            &[input, input],
            "first",
        );
        let second = builder.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            shape.clone(),
            &[first, input],
            "second",
        );
        let third = builder.op(
            OpKind::Add,
            OpAttrs::default(),
            DType::F32,
            shape,
            &[second, input],
            "third",
        );
        builder.mark_output("output", third);
        let graph = builder.build().unwrap();
        let plan = ExecutionPlan::from_graph(&graph).unwrap();
        let first_allocation = plan.memory.allocations[&first];
        let second_allocation = plan.memory.allocations[&second];
        let third_allocation = plan.memory.allocations[&third];
        assert_ne!(first_allocation.offset, second_allocation.offset);
        assert_eq!(first_allocation.offset, third_allocation.offset);
        assert!(plan.memory.arena_bytes < first_allocation.size * 3);
    }
}
