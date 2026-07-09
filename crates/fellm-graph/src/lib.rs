//! Operator DAG built on `petgraph::StableGraph`.

#![deny(missing_docs)]

pub mod builder;
pub mod graph;
pub mod plan;

pub use builder::GraphBuilder;
pub use graph::{EdgeInfo, Graph, NodeId, OpNode, OpValue};
pub use plan::ExecutionPlan;
