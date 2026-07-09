//! GraphBuilder — the API architecture crates use to construct a Graph.

use crate::graph::{EdgeInfo, Graph, NodeId, OpNode, OpValue};
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::Shape;
use fellm_core::tensor::Tensor;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use petgraph::Directed;
use petgraph::stable_graph::StableGraph;

/// Builder for a [`Graph`].
pub struct GraphBuilder {
    inner: StableGraph<OpNode, EdgeInfo, Directed>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl GraphBuilder {
    /// New empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: StableGraph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a named graph input.
    pub fn input(&mut self, name: impl Into<String>, dtype: DType, shape: Shape) -> NodeId {
        let name = name.into();
        let node = OpNode {
            op: None,
            attrs: OpAttrs::default(),
            value: OpValue::Input {
                name: name.clone(),
                dtype,
                shape,
            },
            label: format!("input:{name}"),
            in_place_output_from: None,
        };
        let id = self.inner.add_node(node);
        self.inputs.push(id);
        id
    }

    /// Add a constant tensor.
    pub fn constant(&mut self, label: impl Into<String>, t: Tensor) -> NodeId {
        let label = label.into();
        let node = OpNode {
            op: None,
            attrs: OpAttrs::default(),
            value: OpValue::Constant(t),
            label: format!("const:{label}"),
            in_place_output_from: None,
        };
        self.inner.add_node(node)
    }

        /// Add an operation node.
        pub fn op(
          &mut self,
          op: OpKind,
          attrs: OpAttrs,
          out_dtype: DType,
          out_shape: Shape,
          inputs: &[NodeId],
          label: impl Into<String>,
      ) -> NodeId {
          let node = OpNode {
              op: Some(op),
              attrs,
              value: OpValue::Runtime {
                  dtype: out_dtype,
                  shape: out_shape,
              },
              label: label.into(),
              in_place_output_from: None,
          };
          let id = self.inner.add_node(node);
          for (slot, &inp) in inputs.iter().enumerate() {
              self.inner.add_edge(
                  inp,
                  id,
                  EdgeInfo {
                      input_slot: slot as u32,
                  },
              );
          }
          id
      }
  
      /// Add an operation that writes in place into one of its inputs.
      ///
      /// The `alias_input` argument is the slot index (0-based) of the input
      /// whose storage will be used as the op's output.
      pub fn op_in_place(
          &mut self,
          op: OpKind,
          attrs: OpAttrs,
          out_dtype: DType,
          out_shape: Shape,
          inputs: &[NodeId],
          alias_input: u32,
          label: impl Into<String>,
      ) -> NodeId {
          let node = OpNode {
              op: Some(op),
              attrs,
              value: OpValue::Runtime {
                  dtype: out_dtype,
                  shape: out_shape,
              },
              label: label.into(),
              in_place_output_from: Some(alias_input),
          };
          let id = self.inner.add_node(node);
          for (slot, &inp) in inputs.iter().enumerate() {
              self.inner.add_edge(
                  inp,
                  id,
                  EdgeInfo {
                      input_slot: slot as u32,
                  },
              );
          }
          id
      }  

    /// Mark a node as a graph output.
    pub fn mark_output(&mut self, name: impl Into<String>, node: NodeId) {
        let out_node = OpNode {
            op: None,
            attrs: OpAttrs::default(),
            value: OpValue::Output { name: name.into() },
            label: "output".into(),
            in_place_output_from: None,
        };
        let id = self.inner.add_node(out_node);
        self.inner
            .add_edge(node, id, EdgeInfo { input_slot: 0 });
        self.outputs.push(id);
    }

    /// Finalize the builder.
    pub fn build(self) -> Result<Graph> {
        if self.outputs.is_empty() {
            return Err(FellmError::InvalidGraph("graph has no outputs".into()));
        }
        Ok(Graph {
            inner: self.inner,
            inputs: self.inputs,
            outputs: self.outputs,
        })
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
