//! Graph node and value types.

use fellm_core::dtype::DType;
use fellm_core::shape::Shape;
use fellm_core::tensor::Tensor;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use petgraph::Directed;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::EdgeRef;

/// Node id in the graph.
pub type NodeId = NodeIndex;

/// The value produced by a node.
#[derive(Debug, Clone)]
pub enum OpValue {
    /// A constant tensor known at load time.
    Constant(Tensor),
    /// A runtime tensor produced by executing this node.
    Runtime {
        /// Element type.
        dtype: DType,
        /// Shape.
        shape: Shape,
    },
    /// A named graph input.
    Input {
        /// Name.
        name: String,
        /// dtype.
        dtype: DType,
        /// shape.
        shape: Shape,
    },
    /// A named graph output.
    Output {
        /// Name.
        name: String,
    },
}

/// A node in the operator DAG.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// The op (None for constants/inputs/outputs).
    pub op: Option<OpKind>,
    /// Attributes.
    pub attrs: OpAttrs,
    /// The value this node produces.
    pub value: OpValue,
    /// Human-readable label.
    pub label: String,
    /// If set, the runtime output reuses the storage of this input slot.
    pub in_place_output_from: Option<u32>,
}

/// Edge metadata.
#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    /// Input slot index on the target node.
    pub input_slot: u32,
}

/// The operator DAG.
pub struct Graph {
    pub(crate) inner: StableGraph<OpNode, EdgeInfo, Directed>,
    pub(crate) inputs: Vec<NodeId>,
    pub(crate) outputs: Vec<NodeId>,
}

impl Graph {
    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Access a node.
    #[must_use]
    pub fn node(&self, id: NodeId) -> &OpNode {
        &self.inner[id]
    }

    /// Mutable access.
    pub fn node_mut(&mut self, id: NodeId) -> &mut OpNode {
        &mut self.inner[id]
    }

    /// Graph inputs.
    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Graph outputs.
    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Incoming edges sorted by input slot.
    #[must_use]
    pub fn inputs_of(&self, id: NodeId) -> Vec<NodeId> {
        let mut edges: Vec<_> = self
            .inner
            .edges_directed(id, petgraph::Direction::Incoming)
            .map(|e| (e.weight().input_slot, e.source()))
            .collect();
        edges.sort_by_key(|(slot, _)| *slot);
        edges.into_iter().map(|(_, s)| s).collect()
    }

    /// Iterate all nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeId, &OpNode)> {
        self.inner.node_indices().map(|i| (i, &self.inner[i]))
    }
}

impl core::fmt::Debug for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.inner.node_count())
            .field("edges", &self.inner.edge_count())
            .field("inputs", &self.inputs.len())
            .field("outputs", &self.outputs.len())
            .finish()
    }
}
