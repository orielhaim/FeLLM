//! The single-request executor.

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
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

/// Shared mutable buffer for a graph node's runtime value.
type SharedBuf = Rc<RefCell<AlignedBuffer>>;

/// A materialized tensor value at a graph node.
enum NodeValue {
    /// A borrowed constant from the graph.
    Constant(Tensor),
    /// A freshly-allocated runtime buffer (possibly shared with other nodes
    /// via in-place aliasing).
    Runtime {
        dtype: DType,
        shape: Shape,
        buffer: SharedBuf,
    },
    /// A user-supplied read-only input.
    Input(Tensor),
    /// A user-supplied mutable input (KV cache slot). Storage is shared.
    MutableInput {
        dtype: DType,
        shape: Shape,
        buffer: SharedBuf,
    },
}

impl NodeValue {
    fn dtype(&self) -> DType {
        match self {
            Self::Constant(t) => t.dtype(),
            Self::Runtime { dtype, .. } | Self::MutableInput { dtype, .. } => *dtype,
            Self::Input(t) => t.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            Self::Constant(t) => t.shape().clone(),
            Self::Runtime { shape, .. } | Self::MutableInput { shape, .. } => shape.clone(),
            Self::Input(t) => t.shape().clone(),
        }
    }

    fn shared_buffer(&self) -> Option<SharedBuf> {
        match self {
            Self::Runtime { buffer, .. } | Self::MutableInput { buffer, .. } => {
                Some(buffer.clone())
            }
            _ => None,
        }
    }
}

/// Build a TensorRef for reading.
///
/// # Safety
/// Caller must ensure no concurrent mutable borrow of the same buffer.
fn tensor_ref_from(nv: &NodeValue) -> TensorRef {
    match nv {
        NodeValue::Constant(t) | NodeValue::Input(t) => {
            let bytes = t.as_bytes();
            let dims = t.shape().dims().to_vec();
            let strides = t.shape().row_major_strides().as_slice().to_vec();
            // SAFETY: bytes.as_ptr() valid for bytes.len() bytes for &t's lifetime.
            unsafe { TensorRef::from_raw(t.dtype(), &dims, &strides, bytes.as_ptr(), bytes.len()) }
        }
        NodeValue::Runtime {
            dtype,
            shape,
            buffer,
        }
        | NodeValue::MutableInput {
            dtype,
            shape,
            buffer,
        } => {
            let dims = shape.dims().to_vec();
            let strides = shape.row_major_strides().as_slice().to_vec();
            // We take a read-only borrow through Ref -> raw ptr. The pointer is
            // valid as long as the borrow lives; TensorRef by construction is
            // only used within one op-launch scope, so we drop the Ref right
            // after taking the pointer.
            let borrow = buffer.borrow();
            let ptr = borrow.as_slice().as_ptr();
            let len = borrow.len();
            drop(borrow);
            // SAFETY: no other &mut exists during this op's kernel launch; the
            // executor never launches two ops in parallel per step.
            unsafe { TensorRef::from_raw(*dtype, &dims, &strides, ptr, len) }
        }
    }
}

/// Build a TensorMut for writing.
///
/// # Safety
/// Caller must ensure exclusive access.
fn tensor_mut_from(nv: &NodeValue) -> TensorMut {
    match nv {
        NodeValue::Runtime {
            dtype,
            shape,
            buffer,
        }
        | NodeValue::MutableInput {
            dtype,
            shape,
            buffer,
        } => {
            let dims = shape.dims().to_vec();
            let strides = shape.row_major_strides().as_slice().to_vec();
            let mut borrow = buffer.borrow_mut();
            let ptr = borrow.as_mut_slice().as_mut_ptr();
            let len = borrow.len();
            drop(borrow);
            // SAFETY: the executor is single-threaded per step and only holds
            // one mutable output at a time.
            unsafe { TensorMut::from_raw(*dtype, &dims, &strides, ptr, len) }
        }
        _ => panic!("cannot take mutable view of a non-mutable node value"),
    }
}

/// A mutable input binding.
pub struct MutableBinding {
    /// Element type.
    pub dtype: DType,
    /// Shape.
    pub shape: Shape,
    /// Shared buffer holding this tensor's bytes.
    pub buffer: SharedBuf,
}

/// Run a graph forward.
pub struct GraphExecutor<'a> {
    graph: &'a Graph,
    plan: &'a ExecutionPlan,
    backend: &'a dyn Backend,
    inputs: HashMap<String, Tensor>,
    mutable_inputs: HashMap<String, MutableBinding>,
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
            mutable_inputs: HashMap::new(),
            attr_overrides: HashMap::new(),
        }
    }

    /// Bind a read-only input.
    pub fn bind_input(&mut self, name: impl Into<String>, t: Tensor) {
        self.inputs.insert(name.into(), t);
    }

    /// Bind a mutable input (shared storage the graph can modify in place).
    pub fn bind_mutable(&mut self, name: impl Into<String>, b: MutableBinding) {
        self.mutable_inputs.insert(name.into(), b);
    }

    /// Override attributes on a specific node.
    pub fn set_attrs(&mut self, node: NodeId, attrs: OpAttrs) {
        self.attr_overrides.insert(node, attrs);
    }

    /// Execute the plan.
    pub fn run(&self) -> Result<HashMap<String, Tensor>> {
        let mut values: HashMap<NodeId, NodeValue> = HashMap::with_capacity(self.plan.order.len());

        for &id in &self.plan.order {
            let node = self.graph.node(id);
            match &node.value {
                OpValue::Input { name, dtype, shape } => {
                    // Prefer mutable binding if present, else read-only.
                    if let Some(mb) = self.mutable_inputs.get(name) {
                        if mb.dtype != *dtype {
                            return Err(FellmError::other(format!(
                                "mutable input {name}: dtype mismatch {} vs {}",
                                mb.dtype, dtype
                            )));
                        }
                        values.insert(
                            id,
                            NodeValue::MutableInput {
                                dtype: mb.dtype,
                                shape: mb.shape.clone(),
                                buffer: mb.buffer.clone(),
                            },
                        );
                    } else {
                        let t = self
                            .inputs
                            .get(name)
                            .ok_or_else(|| FellmError::other(format!("missing input {name}")))?
                            .clone();
                        values.insert(id, NodeValue::Input(t));
                    }
                }
                OpValue::Constant(t) => {
                    values.insert(id, NodeValue::Constant(t.clone()));
                }
                OpValue::Output { .. } => {
                    // Handled below.
                }
                OpValue::Runtime { dtype, shape } => {
                    let inputs = self.graph.inputs_of(id);
                    // Determine output buffer: fresh, or aliased to an input.
                    let (out_buffer, out_shape, out_dtype) =
                        if let Some(slot) = node.in_place_output_from {
                            let src_id = inputs.get(slot as usize).copied().ok_or_else(|| {
                                FellmError::other(format!(
                                    "in-place op {} at slot {slot}: no such input",
                                    node.label
                                ))
                            })?;
                            let src = values.get(&src_id).ok_or_else(|| {
                                FellmError::other("in-place source not yet computed")
                            })?;
                            let buf = src.shared_buffer().ok_or_else(|| {
                                FellmError::other("in-place source is not a runtime/mutable value")
                            })?;
                            (buf, src.shape(), src.dtype())
                        } else {
                            let bytes = dtype.byte_size(shape.num_elements());
                            let buf = Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes, 64)));
                            (buf, shape.clone(), *dtype)
                        };

                    // Build input TensorRefs.
                    let mut input_refs: Vec<TensorRef> = Vec::with_capacity(inputs.len());
                    for iid in &inputs {
                        let nv = values.get(iid).ok_or_else(|| {
                            FellmError::other("dep not computed (should not happen)")
                        })?;
                        input_refs.push(tensor_ref_from(nv));
                    }
                    let op = node
                        .op
                        .ok_or_else(|| FellmError::other("Runtime node without op"))?;
                    let attrs = self.attr_overrides.get(&id).copied().unwrap_or(node.attrs);
                    let input_dtypes: Vec<DType> = input_refs
                        .iter()
                        .map(|r| r.dtype().unwrap_or(DType::F32))
                        .collect();
                    let desc = self
                        .backend
                        .resolve_kernel(op, &input_dtypes, out_dtype)
                        .ok_or_else(|| FellmError::NoKernel {
                            op: op.name().into(),
                            dtypes: input_dtypes
                                .iter()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        })?;

                    // Insert the value FIRST so tensor_mut_from can find it,
                    // then take a mutable view.
                    let nv = NodeValue::Runtime {
                        dtype: out_dtype,
                        shape: out_shape,
                        buffer: out_buffer,
                    };
                    values.insert(id, nv);
                    let nv_ref = values.get(&id).unwrap();
                    let mut out_mut = tensor_mut_from(nv_ref);
                    let mut outs = [out_mut];
                    self.backend
                        .launch(desc.handle, &attrs, &input_refs, &mut outs, 0)?;
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
            let src = preds
                .first()
                .copied()
                .ok_or_else(|| FellmError::InvalidGraph("output node has no source".into()))?;
            let nv = values
                .get(&src)
                .ok_or_else(|| FellmError::InvalidGraph("output source not computed".into()))?;
            let dtype = nv.dtype();
            let shape = nv.shape();
            let bytes_len = dtype.byte_size(shape.num_elements());
            let mut buf = AlignedBuffer::new_zeroed(bytes_len, 64);
            {
                let dst = buf.as_mut_slice();
                let tref = tensor_ref_from(nv);
                // SAFETY: tref valid for tref.byte_len bytes at this instant.
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
