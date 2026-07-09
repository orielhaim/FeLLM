//! The single-request executor.
//!
//! In Phase 1 the executor takes a fully-built [`fellm_graph::Graph`] plus a
//! [`ExecutionPlan`] and walks it node-by-node, allocating activation buffers
//! on demand and forwarding kernel launches to the backend.

use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::Shape;
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_graph::graph::{Graph, NodeId, OpValue};
use fellm_graph::plan::ExecutionPlan;
use fellm_plugin_abi::op::OpAttrs;
use fellm_plugin_abi::traits::Backend;
use fellm_plugin_abi::{TensorMut, TensorRef};
use std::collections::HashMap;
use std::sync::Arc;

/// A materialized tensor value at a graph node.
enum NodeValue {
    /// A borrowed constant from the graph.
    Constant(Tensor),
    /// A freshly-allocated runtime buffer.
    Runtime {
        dtype: DType,
        shape: Shape,
        buffer: AlignedBuffer,
    },
    /// A user-supplied input.
    Input(Tensor),
}

impl NodeValue {
    fn dtype(&self) -> DType {
        match self {
            Self::Constant(t) => t.dtype(),
            Self::Runtime { dtype, .. } => *dtype,
            Self::Input(t) => t.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            Self::Constant(t) => t.shape().clone(),
            Self::Runtime { shape, .. } => shape.clone(),
            Self::Input(t) => t.shape().clone(),
        }
    }

    fn as_tensor_ref(&self) -> TensorRef {
        match self {
            Self::Constant(t) | Self::Input(t) => {
                let bytes = t.as_bytes();
                let dims = t.shape().dims().to_vec();
                let strides = t.shape().row_major_strides().as_slice().to_vec();
                // SAFETY: bytes.as_ptr() valid for bytes.len() bytes.
                unsafe {
                    TensorRef::from_raw(t.dtype(), &dims, &strides, bytes.as_ptr(), bytes.len())
                }
            }
            Self::Runtime { dtype, shape, buffer } => {
                let dims = shape.dims().to_vec();
                let strides = shape.row_major_strides().as_slice().to_vec();
                let bytes = buffer.as_slice();
                // SAFETY: bytes.as_ptr() valid for bytes.len() bytes.
                unsafe {
                    TensorRef::from_raw(*dtype, &dims, &strides, bytes.as_ptr(), bytes.len())
                }
            }
        }
    }

    fn as_tensor_mut(&mut self) -> TensorMut {
        match self {
            Self::Runtime { dtype, shape, buffer } => {
                let dims = shape.dims().to_vec();
                let strides = shape.row_major_strides().as_slice().to_vec();
                let bytes = buffer.as_mut_slice();
                let len = bytes.len();
                // SAFETY: bytes.as_mut_ptr() exclusive for `len` bytes.
                unsafe {
                    TensorMut::from_raw(*dtype, &dims, &strides, bytes.as_mut_ptr(), len)
                }
            }
            _ => panic!("cannot take mutable view of a non-runtime node value"),
        }
    }
}

/// Run a graph forward.
pub struct GraphExecutor<'a> {
    graph: &'a Graph,
    plan: &'a ExecutionPlan,
    backend: &'a dyn Backend,
    /// Map from graph input name to a user-supplied tensor.
    inputs: HashMap<String, Tensor>,
    /// Attributes overrides applied to specific nodes (for e.g. `position`
    /// which changes every step).
    attr_overrides: HashMap<NodeId, OpAttrs>,
}

impl<'a> GraphExecutor<'a> {
    /// New executor.
    pub fn new(graph: &'a Graph, plan: &'a ExecutionPlan, backend: &'a dyn Backend) -> Self {
        Self {
            graph,
            plan,
            backend,
            inputs: HashMap::new(),
            attr_overrides: HashMap::new(),
        }
    }

    /// Bind an input tensor by name.
    pub fn bind_input(&mut self, name: impl Into<String>, t: Tensor) {
        self.inputs.insert(name.into(), t);
    }

    /// Override attributes on a specific node.
    pub fn set_attrs(&mut self, node: NodeId, attrs: OpAttrs) {
        self.attr_overrides.insert(node, attrs);
    }

    /// Execute the plan.
    ///
    /// Returns a map from output-name -> owned Tensor.
    pub fn run(&self) -> Result<HashMap<String, Tensor>> {
        let mut values: HashMap<NodeId, NodeValue> = HashMap::with_capacity(self.plan.order.len());

        for &id in &self.plan.order {
            let node = self.graph.node(id);
            match &node.value {
                OpValue::Input { name, .. } => {
                    let t = self
                        .inputs
                        .get(name)
                        .ok_or_else(|| FellmError::other(format!("missing input {name}")))?
                        .clone();
                    values.insert(id, NodeValue::Input(t));
                }
                OpValue::Constant(t) => {
                    values.insert(id, NodeValue::Constant(t.clone()));
                }
                OpValue::Output { name: _ } => {
                    // Output is a passthrough — value comes from its single predecessor.
                    // We handle it below when constructing the result map.
                }
                OpValue::Runtime { dtype, shape } => {
                    let bytes = dtype.byte_size(shape.num_elements());
                    let buffer = AlignedBuffer::new_zeroed(bytes, 64);
                    let mut nv = NodeValue::Runtime {
                        dtype: *dtype,
                        shape: shape.clone(),
                        buffer,
                    };
                    let inputs = self.graph.inputs_of(id);
                    // Gather input TensorRefs.
                    let input_refs: Vec<TensorRef> = inputs
                        .iter()
                        .map(|iid| values.get(iid).expect("value").as_tensor_ref())
                        .collect();
                    let op = node
                        .op
                        .ok_or_else(|| FellmError::other("Runtime node without op"))?;
                    let attrs = self
                        .attr_overrides
                        .get(&id)
                        .copied()
                        .unwrap_or(node.attrs);
                    let input_dtypes: Vec<DType> = input_refs
                        .iter()
                        .map(|r| r.dtype().unwrap_or(DType::F32))
                        .collect();
                    let desc = self
                        .backend
                        .resolve_kernel(op, &input_dtypes, *dtype)
                        .ok_or_else(|| FellmError::NoKernel {
                            op: op.name().into(),
                            dtypes: input_dtypes
                                .iter()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        })?;
                    let mut out_mut = nv.as_tensor_mut();
                    let mut outs = [out_mut];
                    self.backend
                        .launch(desc.handle, &attrs, &input_refs, &mut outs, 0)?;
                    let _ = out_mut;
                    values.insert(id, nv);
                }
            }
        }

        // Collect outputs.
        let mut result = HashMap::new();
        for &oid in self.graph.outputs() {
            let out_node = self.graph.node(oid);
            let name = match &out_node.value {
                OpValue::Output { name } => name.clone(),
                _ => continue,
            };
            let preds = self.graph.inputs_of(oid);
            let src = preds.first().copied().ok_or_else(|| {
                FellmError::InvalidGraph("output node has no source".into())
            })?;
            let nv = values.get(&src).ok_or_else(|| {
                FellmError::InvalidGraph("output source not computed".into())
            })?;
            // Materialize as owned Tensor by cloning bytes into a fresh buffer.
            let dtype = nv.dtype();
            let shape = nv.shape();
            let bytes_len = dtype.byte_size(shape.num_elements());
            let mut buf = AlignedBuffer::new_zeroed(bytes_len, 64);
            {
                let dst = buf.as_mut_slice();
                let tref = nv.as_tensor_ref();
                // SAFETY: tref is valid for tref.byte_len bytes.
                let src_bytes =
                    unsafe { core::slice::from_raw_parts(tref.data, tref.byte_len as usize) };
                dst.copy_from_slice(&src_bytes[..dst.len()]);
            }
            let layout = fellm_core::shape::Layout::contiguous(dtype, shape);
            let storage = Arc::new(Storage::Owned(Arc::new(buf)));
            result.insert(name, Tensor::from_storage(layout, storage));
        }
        Ok(result)
    }
}
