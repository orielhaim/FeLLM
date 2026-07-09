//! Execution plan: topologically sorted graph.

use crate::graph::{Graph, NodeId};
use fellm_core::error::{FellmError, Result};
use petgraph::algo::toposort;

/// A frozen, topologically-sorted execution order.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Node ids in execution order.
    pub order: Vec<NodeId>,
}

impl ExecutionPlan {
    /// Compute an execution plan from a graph.
    pub fn from_graph(g: &Graph) -> Result<Self> {
        let order = toposort(&g.inner, None)
            .map_err(|c| FellmError::InvalidGraph(format!("cycle at node {:?}", c.node_id())))?;
        Ok(Self { order })
    }
}
