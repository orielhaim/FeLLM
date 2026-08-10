use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_graph::graph::{Graph, NodeId, OpValue};
use fellm_graph::plan::ExecutionPlan;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{Backend, KernelHandle};
use fellm_plugin_abi::{TensorMut, TensorRef};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

/// Shared mutable buffer for a runtime / mutable-input slot.
type SharedBuf = Rc<RefCell<AlignedBuffer>>;

/// Mutable graph input backed by stable reusable storage.
pub struct MutableBinding {
    /// Element type.
    pub dtype: DType,
    /// Logical shape.
    pub shape: Shape,
    /// Shared storage modified by in-place graph operations.
    pub buffer: Rc<RefCell<AlignedBuffer>>,
}

/// A per-slot value backing.
enum SlotKind {
    /// Borrowed constant tensor (Arc-shared, cloned once at compile time).
    Constant(Tensor),
    /// Read-only input bound by name each step (e.g. `token_id`).
    Input(String),
    /// Placeholder slot that carries no buffer (output nodes).
    None,
    /// Runtime buffer or mutable input: shared, reused across steps.
    Buffer(SharedBuf),
    /// A byte range in the graph-wide reusable activation arena.
    Arena {
        buffer: SharedBuf,
        offset: usize,
        len: usize,
    },
}

/// One compiled node in plan order.
struct CompiledNode {
    /// Plan indices of this node's inputs, sorted by input slot.
    inputs: Vec<usize>,
    /// The value backing for this node.
    kind: SlotKind,
    /// dtype of the value.
    dtype: DType,
    /// dims of the value.
    dims: Vec<u64>,
    /// row-major strides of the value.
    strides: Vec<u64>,
    /// For runtime ops: the resolved kernel + op + default attrs.
    runtime: Option<RuntimeOp>,
}

/// Precomputed launch info for a runtime node.
struct RuntimeOp {
    op: OpKind,
    handle: KernelHandle,
    attrs: OpAttrs,
    /// Number of inputs passed as read-only refs (`ShortConv` hides its state input).
    input_ref_count: usize,
}

/// A read-only input bound for the current step.
struct BoundInput {
    tensor: Tensor,
}

/// A compiled, reusable step schedule.
pub struct CompiledStep {
    nodes: Vec<CompiledNode>,
    /// Map graph node id -> plan index.
    index_of: HashMap<NodeId, usize>,
    /// Plan index whose buffer feeds the `logits` output.
    logits_src: usize,
    /// First plan index of post-body ops (`final_norm` / `lm_head`).
    /// When `compute_logits` is false, runtime ops at and after this index are skipped.
    body_end: usize,
    /// dtype / shape of the logits output.
    logits_dtype: DType,
    logits_shape: Shape,
    /// Read-only inputs bound this step, keyed by input name.
    inputs: HashMap<String, BoundInput>,
    /// Per-step attribute patches keyed by plan index.
    attr_patches: HashMap<usize, OpAttrs>,
}

impl CompiledStep {
    /// Stable host identities used only once while binding the CUDA static arena.
    /// The hot path uses the corresponding device addresses registered at compile time.
    #[cfg(feature = "backend-cuda")]
    pub(crate) fn arena_bindings(&self) -> Vec<(fellm_plugin_abi::PlanTensorId, *const u8, usize)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| match &node.kind {
                SlotKind::Buffer(buffer) => {
                    let borrow = buffer.borrow();
                    Some((
                        fellm_plugin_abi::PlanTensorId(index as u32),
                        borrow.as_slice().as_ptr(),
                        borrow.len(),
                    ))
                }
                SlotKind::Arena {
                    buffer,
                    offset,
                    len,
                } => {
                    let borrow = buffer.borrow();
                    // SAFETY: the memory plan validates that offset + len is within the arena.
                    let ptr = unsafe { borrow.as_slice().as_ptr().add(*offset) };
                    Some((fellm_plugin_abi::PlanTensorId(index as u32), ptr, *len))
                }
                _ => None,
            })
            .collect()
    }

    /// Compile a step schedule from a fixed graph + plan + backend.
    ///
    /// `mutable_inputs` supplies the stable shared buffers for KV / conv slots;
    /// their storage persists for the model's lifetime so we bind them once.
    pub fn compile(
        graph: &Graph,
        plan: &ExecutionPlan,
        backend: &dyn Backend,
        mutable_inputs: &HashMap<String, MutableBinding>,
    ) -> Result<Self> {
        // Resolve the complete operation surface before allocating or executing.
        // Strict accelerators therefore report every missing dtype combination
        // in one compile error and can never silently cross a host boundary.
        let mut inferred_dtypes: HashMap<NodeId, DType> = HashMap::new();
        let mut missing = BTreeSet::new();
        for &id in &plan.order {
            let node = graph.node(id);
            let mut input_dtypes: Vec<DType> = graph
                .inputs_slice(id)
                .iter()
                .filter_map(|input| inferred_dtypes.get(input).copied())
                .collect();
            // The recurrent ShortConv state is an aliased mutable output, not
            // a read-only kernel input (see the launch path below).
            if node.op == Some(OpKind::ShortConv) {
                input_dtypes.truncate(input_dtypes.len().saturating_sub(1));
            }
            let output_dtype = match &node.value {
                OpValue::Input { dtype, .. } | OpValue::Runtime { dtype, .. } => *dtype,
                OpValue::Constant(t) => t.dtype(),
                OpValue::Output { .. } => input_dtypes.first().copied().unwrap_or(DType::F32),
            };
            let output_dtype = node
                .in_place_output_from
                .and_then(|slot| input_dtypes.get(slot as usize).copied())
                .unwrap_or(output_dtype);
            inferred_dtypes.insert(id, output_dtype);
            if let Some(op) = node.op
                && backend
                    .resolve_kernel(op, &input_dtypes, output_dtype)
                    .is_none()
            {
                missing.insert(format!(
                    "{}({}) -> {}",
                    op.name(),
                    input_dtypes
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    output_dtype
                ));
            }
        }
        if !missing.is_empty() {
            let boundary =
                if backend.capabilities().device_kind == fellm_plugin_abi::DeviceKind::Gpu {
                    "strict CUDA execution would require CPU fallback"
                } else {
                    "backend execution plan is incomplete"
                };
            return Err(FellmError::other(format!(
                "{boundary}; missing kernels:\n  {}",
                missing.into_iter().collect::<Vec<_>>().join("\n  ")
            )));
        }

        // Plan index for each node id.
        let mut index_of: HashMap<NodeId, usize> = HashMap::with_capacity(plan.order.len());
        for (i, &id) in plan.order.iter().enumerate() {
            index_of.insert(id, i);
        }

        let mut nodes: Vec<CompiledNode> = Vec::with_capacity(plan.order.len());
        let activation_arena = Rc::new(RefCell::new(AlignedBuffer::new_zeroed(
            plan.memory.arena_bytes,
            64,
        )));

        for &id in &plan.order {
            let node = graph.node(id);
            let input_ids = graph.inputs_slice(id);
            let inputs: Vec<usize> = input_ids
                .iter()
                .map(|iid| {
                    index_of
                        .get(iid)
                        .copied()
                        .ok_or_else(|| FellmError::other("input not before node in plan"))
                })
                .collect::<Result<_>>()?;

            match &node.value {
                OpValue::Input { name, dtype, .. } => {
                    if let Some(mb) = mutable_inputs.get(name) {
                        if mb.dtype != *dtype {
                            return Err(FellmError::other(format!(
                                "mutable input {name}: dtype mismatch {} vs {}",
                                mb.dtype, dtype
                            )));
                        }
                        let (dims, strides) = dims_strides(&mb.shape);
                        nodes.push(CompiledNode {
                            inputs,
                            kind: SlotKind::Buffer(mb.buffer.clone()),
                            dtype: mb.dtype,
                            dims,
                            strides,
                            runtime: None,
                        });
                    } else {
                        // Read-only input bound by name each step (token_id).
                        nodes.push(CompiledNode {
                            inputs,
                            kind: SlotKind::Input(name.clone()),
                            dtype: *dtype,
                            dims: Vec::new(),
                            strides: Vec::new(),
                            runtime: None,
                        });
                    }
                }
                OpValue::Constant(t) => {
                    let (dims, strides) = dims_strides(t.shape());
                    nodes.push(CompiledNode {
                        inputs,
                        kind: SlotKind::Constant(t.clone()),
                        dtype: t.dtype(),
                        dims,
                        strides,
                        runtime: None,
                    });
                }
                OpValue::Output { .. } => {
                    // Outputs carry no buffer; they reference their source slot.
                    nodes.push(CompiledNode {
                        inputs,
                        kind: SlotKind::None,
                        dtype: DType::F32,
                        dims: Vec::new(),
                        strides: Vec::new(),
                        runtime: None,
                    });
                }
                OpValue::Runtime { dtype, shape } => {
                    let op = node
                        .op
                        .ok_or_else(|| FellmError::other("Runtime node without op"))?;

                    // Output buffer: fresh, or aliased to an input slot (in-place).
                    let (kind, out_dtype, out_shape) = if let Some(slot) = node.in_place_output_from
                    {
                        let src_idx = *inputs.get(slot as usize).ok_or_else(|| {
                            FellmError::other(format!(
                                "in-place op {} slot {slot}: no such input",
                                node.label
                            ))
                        })?;
                        let src = &nodes[src_idx];
                        let buf = match &src.kind {
                            SlotKind::Buffer(b) => b.clone(),
                            SlotKind::Arena { buffer, .. } => buffer.clone(),
                            _ => {
                                return Err(FellmError::other(
                                    "in-place source is not a runtime/mutable value",
                                ));
                            }
                        };
                        let kind = match &src.kind {
                            SlotKind::Arena { offset, len, .. } => SlotKind::Arena {
                                buffer: buf,
                                offset: *offset,
                                len: *len,
                            },
                            _ => SlotKind::Buffer(buf),
                        };
                        (kind, src.dtype, shape_from_dims(&src.dims))
                    } else {
                        let bytes = dtype.byte_size(shape.num_elements());
                        let allocation = plan.memory.allocations.get(&id).ok_or_else(|| {
                            FellmError::other(format!(
                                "missing activation allocation for {}",
                                node.label
                            ))
                        })?;
                        debug_assert!(allocation.size >= bytes);
                        (
                            SlotKind::Arena {
                                buffer: activation_arena.clone(),
                                offset: allocation.offset,
                                len: bytes,
                            },
                            *dtype,
                            shape.clone(),
                        )
                    };

                    let (dims, strides) = dims_strides(&out_shape);

                    // Resolve the kernel using compile-time input dtypes.
                    let input_ref_count = if op == OpKind::ShortConv {
                        inputs.len().saturating_sub(1)
                    } else {
                        inputs.len()
                    };
                    let input_dtypes: Vec<DType> = inputs
                        .iter()
                        .take(input_ref_count)
                        .map(|&i| nodes[i].dtype)
                        .collect();
                    let desc = backend
                        .resolve_kernel(op, &input_dtypes, out_dtype)
                        .ok_or_else(|| FellmError::NoKernel {
                            op: op.name().into(),
                            dtypes: input_dtypes
                                .iter()
                                .map(std::string::ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                        })?;

                    nodes.push(CompiledNode {
                        inputs,
                        kind,
                        dtype: out_dtype,
                        dims,
                        strides,
                        runtime: Some(RuntimeOp {
                            op,
                            handle: desc.handle,
                            attrs: node.attrs,
                            input_ref_count,
                        }),
                    });
                }
            }
        }

        // Resolve the logits output source once.
        let mut logits_src = None;
        let mut logits_dtype = DType::F32;
        let mut logits_shape = Shape::new(&[1])?;
        for &oid in graph.outputs() {
            let out_node = graph.node(oid);
            let name = match &out_node.value {
                OpValue::Output { name } => name.clone(),
                _ => continue,
            };
            if name != "logits" {
                continue;
            }
            let preds = graph.inputs_slice(oid);
            let src = preds
                .first()
                .copied()
                .ok_or_else(|| FellmError::InvalidGraph("logits output has no source".into()))?;
            let src_idx = *index_of
                .get(&src)
                .ok_or_else(|| FellmError::InvalidGraph("logits source not in plan".into()))?;
            logits_src = Some(src_idx);
            logits_dtype = nodes[src_idx].dtype;
            logits_shape = shape_from_dims(&nodes[src_idx].dims);
        }
        let logits_src =
            logits_src.ok_or_else(|| FellmError::other("graph has no logits output"))?;

        // Prefill mid-tokens only need KV/conv updates. Skip from `final_norm`
        // (or `lm_head` if the label is missing) through the end of the plan.
        let body_end = plan
            .order
            .iter()
            .position(|&id| {
                let label = graph.node(id).label.as_str();
                label == "final_norm" || label == "lm_head"
            })
            .unwrap_or(plan.order.len());

        Ok(Self {
            nodes,
            index_of,
            logits_src,
            body_end,
            logits_dtype,
            logits_shape,
            inputs: HashMap::new(),
            attr_patches: HashMap::new(),
        })
    }

    /// Bind a read-only input for the current step.
    pub fn bind_input(&mut self, name: impl Into<String>, t: Tensor) {
        self.inputs.insert(name.into(), BoundInput { tensor: t });
    }

    /// Patch attributes on a specific graph node for the current step.
    pub fn set_attrs(&mut self, node: NodeId, attrs: OpAttrs) {
        if let Some(&idx) = self.index_of.get(&node) {
            self.attr_patches.insert(idx, attrs);
        }
    }

    /// Execute the compiled schedule.
    ///
    /// When `compute_logits` is false, ops from `body_end` onward
    /// (`final_norm` + `lm_head`) are skipped and an empty F32 tensor is returned.
    /// When true, logits are returned by swapping the compiled logits slot buffer
    /// into an owned tensor (no full copy); a fresh zeroed buffer is left in the slot.
    pub fn run(&mut self, backend: &dyn Backend, compute_logits: bool) -> Result<Tensor> {
        let end = if compute_logits {
            self.nodes.len()
        } else {
            self.body_end
        };
        let profile_ops = std::env::var_os("FELLM_PROFILE_OPS").is_some();
        let mut profile: HashMap<String, (u32, u128)> = HashMap::new();

        for i in 0..end {
            let node = &self.nodes[i];
            let Some(rt) = &node.runtime else { continue };

            let mut input_refs: SmallVec<[TensorRef; 6]> = SmallVec::new();
            for &iid in node.inputs.iter().take(rt.input_ref_count) {
                input_refs.push(self.tensor_ref(iid)?);
            }

            let attrs = self.attr_patches.get(&i).copied().unwrap_or(rt.attrs);
            let out_mut = self.tensor_mut(i)?;
            // Profiling must bracket completed GPU work. Merely timing the host
            // launch attributes all queued work to the eventual D2H boundary.
            if profile_ops {
                backend.synchronize()?;
            }
            let started = profile_ops.then(Instant::now);

            if rt.op == OpKind::ShortConv {
                let state_idx = *node
                    .inputs
                    .get(4)
                    .ok_or_else(|| FellmError::other("shortconv node missing state input"))?;
                let state_mut = self.tensor_mut(state_idx)?;
                let mut outs = [out_mut, state_mut];
                backend
                    .launch(rt.handle, &attrs, &input_refs, &mut outs, 0)
                    .map_err(|error| {
                        FellmError::other(format!(
                            "CUDA/runtime launch failed at {} ({}): {error}",
                            rt.op.name(),
                            node.dims
                                .iter()
                                .map(u64::to_string)
                                .collect::<Vec<_>>()
                                .join("x")
                        ))
                    })?;
            } else {
                let mut outs = [out_mut];
                backend
                    .launch(rt.handle, &attrs, &input_refs, &mut outs, 0)
                    .map_err(|error| {
                        FellmError::other(format!(
                            "CUDA/runtime launch failed at {} ({}): {error}",
                            rt.op.name(),
                            node.dims
                                .iter()
                                .map(u64::to_string)
                                .collect::<Vec<_>>()
                                .join("x")
                        ))
                    })?;
            }
            if let Some(started) = started {
                backend.synchronize()?;
                let shape = node
                    .dims
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let key = format!("{}[{}]", rt.op.name(), shape);
                let entry = profile.entry(key).or_default();
                entry.0 += 1;
                entry.1 += started.elapsed().as_nanos();
            }
        }

        if profile_ops {
            let mut profile: Vec<_> = profile.into_iter().collect();
            profile.sort_unstable_by_key(|(_, (_, nanos))| std::cmp::Reverse(*nanos));
            tracing::info!(
                ops = ?profile.iter().map(|(op, (count, nanos))| {
                    format!("{op}:{count}={:.3}ms", *nanos as f64 / 1_000_000.0)
                }).collect::<Vec<_>>(),
                "forward op profile"
            );
        }

        if !compute_logits {
            let layout = Layout::contiguous(DType::F32, Shape::new(&[0])?);
            let storage = Arc::new(Storage::Owned(Arc::new(AlignedBuffer::new_zeroed(0, 64))));
            return Ok(Tensor::from_storage(layout, storage));
        }

        // Copy the logits out of the persistent activation arena. Sampling owns
        // this result while the arena is immediately reusable by another batch.
        let bytes_len = self
            .logits_dtype
            .byte_size(self.logits_shape.num_elements());
        let mut taken = AlignedBuffer::new_zeroed(bytes_len, 64);
        let source = self.tensor_ref(self.logits_src)?;
        // SAFETY: tensor_ref describes a live contiguous logits buffer for this step.
        let bytes = unsafe { core::slice::from_raw_parts(source.data, source.byte_len as usize) };
        taken.as_mut_slice().copy_from_slice(&bytes[..bytes_len]);
        let layout = Layout::contiguous(self.logits_dtype, self.logits_shape.clone());
        let storage = Arc::new(Storage::Owned(Arc::new(taken)));
        Ok(Tensor::from_storage(layout, storage))
    }

    /// Build a read-only tensor view for slot `idx`.
    ///
    /// # Safety
    /// The executor launches one op at a time; no concurrent mutable borrow of
    /// the same buffer exists during a single launch.
    fn tensor_ref(&self, idx: usize) -> Result<TensorRef> {
        let node = &self.nodes[idx];
        match &node.kind {
            SlotKind::Constant(t) => {
                let bytes = t.as_bytes();
                // SAFETY: bytes valid for the tensor's lifetime (Arc-shared).
                Ok(unsafe {
                    TensorRef::from_raw(
                        t.dtype(),
                        &node.dims,
                        &node.strides,
                        bytes.as_ptr(),
                        bytes.len(),
                    )
                })
            }
            SlotKind::Input(name) => {
                let bound = self
                    .inputs
                    .get(name)
                    .ok_or_else(|| FellmError::other(format!("missing input {name}")))?;
                let t = &bound.tensor;
                let bytes = t.as_bytes();
                let dims = t.shape().dims().to_vec();
                let strides = t.shape().row_major_strides().as_slice().to_vec();
                // SAFETY: bytes valid for the tensor's lifetime.
                Ok(unsafe {
                    TensorRef::from_raw(t.dtype(), &dims, &strides, bytes.as_ptr(), bytes.len())
                })
            }
            SlotKind::None => Err(FellmError::other("output slot has no tensor value")),
            SlotKind::Buffer(buf) => {
                let borrow = buf.borrow();
                let ptr = borrow.as_slice().as_ptr();
                let len = borrow.len();
                drop(borrow);
                // SAFETY: single-op-at-a-time; no live &mut to this buffer now.
                Ok(unsafe { TensorRef::from_raw(node.dtype, &node.dims, &node.strides, ptr, len) })
            }
            SlotKind::Arena {
                buffer,
                offset,
                len,
            } => {
                let borrow = buffer.borrow();
                // SAFETY: planned range is within the stable arena allocation.
                let ptr = unsafe { borrow.as_slice().as_ptr().add(*offset) };
                Ok(
                    unsafe {
                        TensorRef::from_raw(node.dtype, &node.dims, &node.strides, ptr, *len)
                    },
                )
            }
        }
    }

    /// Build a writable tensor view for slot `idx`.
    ///
    /// # Safety
    /// Exclusive access is guaranteed by single-op-at-a-time execution.
    fn tensor_mut(&self, idx: usize) -> Result<TensorMut> {
        let node = &self.nodes[idx];
        match &node.kind {
            SlotKind::Buffer(buf) => {
                let mut borrow = buf.borrow_mut();
                let ptr = borrow.as_mut_slice().as_mut_ptr();
                let len = borrow.len();
                drop(borrow);
                // SAFETY: single-op-at-a-time; caller holds only one mutable view.
                Ok(unsafe { TensorMut::from_raw(node.dtype, &node.dims, &node.strides, ptr, len) })
            }
            SlotKind::Arena {
                buffer,
                offset,
                len,
            } => {
                let mut borrow = buffer.borrow_mut();
                // SAFETY: liveness planning prevents overlapping live outputs; launches are serial.
                let ptr = unsafe { borrow.as_mut_slice().as_mut_ptr().add(*offset) };
                Ok(
                    unsafe {
                        TensorMut::from_raw(node.dtype, &node.dims, &node.strides, ptr, *len)
                    },
                )
            }
            _ => Err(FellmError::other(
                "cannot take mutable view of a non-buffer slot",
            )),
        }
    }
}

fn dims_strides(shape: &Shape) -> (Vec<u64>, Vec<u64>) {
    let dims = shape.dims().to_vec();
    let strides = shape.row_major_strides().as_slice().to_vec();
    (dims, strides)
}

fn shape_from_dims(dims: &[u64]) -> Shape {
    Shape::new(dims).unwrap_or_else(|_| Shape::new(&[1]).expect("scalar shape"))
}
