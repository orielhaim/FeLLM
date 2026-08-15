use crate::cpu_profile::CpuHardwareProfile;
use crate::kernels::{
    attention::{attention_step, attention_step_paged},
    embedding::{embedding_row, weighted_embedding},
    matmul,
    norm::{rmsnorm_groups, rmsnorm_row},
    sampling::sample,
    simd_f32::PulpDispatch,
    softmax::softmax_rows_inplace,
    swiglu::{silu_gate, silu_gate_inplace},
};
use crossbeam_utils::CachePadded;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi as paged_ctx;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::{
    Backend, BackendCaps, DeviceKind, DeviceMemoryInfo, KernelDescriptor, KernelHandle,
};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use rayon::ThreadPool;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use thread_local::ThreadLocal;

/// The CPU backend.
pub struct CpuBackend {
    caps: BackendCaps,
    profile: CpuHardwareProfile,
    simd: PulpDispatch,
    /// Rayon pool sized to physical cores (avoids HT L2 thrash in attention).
    pool: ThreadPool,
    /// Cheap backend-wide launch counter, padded to avoid false sharing with
    /// adjacent backend state.
    launches: CachePadded<AtomicU64>,
    /// Per-worker launch counters for diagnosing scheduler imbalance.
    worker_launches: ThreadLocal<AtomicU64>,
    weight_storage: Mutex<Option<CpuWeightStorage>>,
    storage_wait_nanos: AtomicU64,
    mmap_execution_bytes: AtomicU64,
}

struct CpuWeightStorage {
    providers: std::collections::HashMap<std::path::PathBuf, Arc<dyn fellm_memory::TransferProvider>>,
    slot_bytes: usize,
    objects: std::collections::HashMap<u64, fellm_memory::StorageObject>,
    weight_objects: std::collections::HashMap<u64, u64>,
    sparse: std::collections::HashMap<u64, SparseWeight>,
    pending: std::collections::HashMap<u64, std::thread::JoinHandle<Result<CpuRead>>>,
    resident: Vec<Option<CpuRead>>,
    available: Vec<fellm_core::storage::AlignedBuffer>,
    physical_reads: u64,
    physical_bytes: u64,
    expert_slice_reads: u64,
    expert_slice_bytes: u64,
    overlap: bool,
    provider_kind: fellm_memory::StorageProviderKind,
    read_nanos: u64,
}

#[derive(Clone)]
struct SparseWeight {
    path: std::path::PathBuf,
    offset: u64,
    len: u64,
}

struct CpuRead {
    object: u64,
    buffer: fellm_core::storage::AlignedBuffer,
    valid_len: usize,
    read_nanos: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuStorageMetrics {
    pub physical_reads: u64,
    pub physical_bytes: u64,
    pub storage_stall_nanos: u64,
    pub staging_bytes: u64,
    pub mmap_execution_bytes: u64,
    pub buffered_storage_bytes: u64,
    pub direct_storage_bytes: u64,
    pub io_compute_overlap_percent: f64,
    pub expert_slice_reads: u64,
    pub expert_slice_bytes: u64,
    pub resident_dense_bytes: u64,
    pub avg_read_bytes: f64,
}

impl CpuBackend {
    #[must_use]
    pub fn new() -> Self {
        let profile = *CpuHardwareProfile::get();
        let physical = profile.physical_cores.max(1);
        let logical = profile.logical_threads.max(physical);
        let requested_threads =
            std::env::var("FELLM_CPU_THREADS").ok().and_then(|value| {
                match value.trim().to_ascii_lowercase().as_str() {
                    "physical" | "p" => Some(physical),
                    "logical" | "all" => Some(logical),
                    value => value.parse::<usize>().ok().filter(|&n| n > 0),
                }
            });
        let automatic_threads = if cfg!(target_os = "linux")
            && std::fs::read_to_string("/proc/sys/kernel/osrelease")
                .is_ok_and(|release| release.to_ascii_lowercase().contains("microsoft"))
        {
            // WSL currently hides hybrid-core classes and presents every vCPU as an identical
            // physical core. Bandwidth-bound quantized projections regress when all 24 virtual
            // cores contend for guest memory on Arrow Lake; leave one third idle unless the user
            // supplied an explicit topology choice.
            physical.saturating_mul(2).div_ceil(3).max(1)
        } else {
            physical
        };
        let threads = requested_threads
            .unwrap_or(automatic_threads)
            .clamp(1, logical);
        tracing::debug!(
            cpu_threads = threads,
            automatic = requested_threads.is_none(),
            "selected CPU execution pool"
        );
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("fellm-matmul-{i}"))
            .build_global();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("fellm-attn-{i}"))
            .build()
            .unwrap_or_else(|_| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("rayon single-thread pool")
            });
        Self {
            caps: BackendCaps {
                device_kind: DeviceKind::Cpu,
                simd_f32_lanes: profile.simd_f32_lanes,
                has_avx512: profile.has_avx512,
                has_avx2: profile.has_avx2,
                has_neon: profile.has_neon,
                physical_cores: profile.physical_cores as u32,
                logical_threads: profile.logical_threads as u32,
                supports_persistent_device_state: false,
                supports_graph_capture: false,
                supports_async_execution: false,
                supports_read_only_prefix_kv: true,
                supports_grouped_moe: true,
                supports_device_sampling: false,
                supports_bidirectional_attention: true,
                supports_batched_quantized_gemm: true,
                supports_custom_operations: true,
                compute_major: 0,
                compute_minor: 0,
                smem_per_sm: 0,
                has_ampere_ada_features: false,
                has_hopper_features: false,
                has_blackwell_features: false,
            },
            profile,
            simd: PulpDispatch::new(),
            pool,
            launches: CachePadded::new(AtomicU64::new(0)),
            worker_launches: ThreadLocal::new(),
            weight_storage: Mutex::new(None),
            storage_wait_nanos: AtomicU64::new(0),
            mmap_execution_bytes: AtomicU64::new(0),
        }
    }

    pub fn configure_weight_storage(
        &self,
        weights: &[fellm_memory::WeightDescriptor],
        objects: &fellm_memory::StorageObjectIndex,
        provider_kind: fellm_memory::StorageProviderKind,
        buffer_count: usize,
        buffer_bytes: usize,
        overlap: bool,
    ) -> Result<()> {
        let mut providers = std::collections::HashMap::<
            std::path::PathBuf,
            Arc<dyn fellm_memory::TransferProvider>,
        >::new();
        for weight in weights {
            let path = weight.home.path.clone();
            if providers.contains_key(&path) {
                continue;
            }
            let provider: Arc<dyn fellm_memory::TransferProvider> = match provider_kind {
                fellm_memory::StorageProviderKind::Mmap => {
                    let file = std::fs::File::open(&path).map_err(FellmError::Io)?;
                    Arc::new(fellm_memory::MmapProvider::open(&file)?)
                }
                fellm_memory::StorageProviderKind::PageCache
                | fellm_memory::StorageProviderKind::Buffered => {
                    Arc::new(fellm_memory::FileProvider::open(&path)?)
                }
                fellm_memory::StorageProviderKind::Direct => {
                    #[cfg(any(target_os = "linux", windows))]
                    {
                        Arc::new(fellm_memory::DirectFileProvider::open(&path)?)
                    }
                    #[cfg(not(any(target_os = "linux", windows)))]
                    {
                        return Err(FellmError::other("native direct I/O unavailable"));
                    }
                }
                fellm_memory::StorageProviderKind::IoUring => {
                    return Err(FellmError::other("io_uring provider is not operational"));
                }
                fellm_memory::StorageProviderKind::Gds => {
                    return Err(FellmError::other("GDS cannot feed CPU execution"));
                }
            };
            providers.insert(path, provider);
        }
        if providers.is_empty() {
            return Err(FellmError::other("CPU storage weight catalog is empty"));
        }
        let object_map = objects
            .objects()
            .iter()
            .cloned()
            .map(|object| (object.id.0, object))
            .collect::<std::collections::HashMap<_, _>>();
        let weight_objects = objects
            .objects()
            .iter()
            .flat_map(|object| {
                object
                    .members
                    .iter()
                    .map(move |member| (member.weight.0, object.id.0))
            })
            .collect();
        let mut sparse = std::collections::HashMap::new();
        for weight in weights {
            if fellm_memory::is_moe_expert_bank(&weight.name) {
                sparse.insert(
                    weight.id.0,
                    SparseWeight {
                        path: weight.home.path.clone(),
                        offset: weight.home.offset,
                        len: weight.home.len,
                    },
                );
            }
        }
        let maximum = object_map
            .values()
            .map(|object| object.extent.len)
            .max()
            .unwrap_or(1) as usize;
        let slot_bytes = buffer_bytes.max(maximum).next_multiple_of(4096);
        let slots = buffer_count.max(2);
        *self.weight_storage.lock().expect("CPU weight storage lock") = Some(CpuWeightStorage {
            providers,
            slot_bytes,
            objects: object_map,
            weight_objects,
            sparse,
            pending: std::collections::HashMap::new(),
            resident: (0..slots).map(|_| None).collect(),
            available: (0..slots)
                .map(|_| fellm_core::storage::AlignedBuffer::new_zeroed(slot_bytes, 4096))
                .collect(),
            physical_reads: 0,
            physical_bytes: 0,
            expert_slice_reads: 0,
            expert_slice_bytes: 0,
            overlap,
            provider_kind,
            read_nanos: 0,
        });
        tracing::debug!(
            storage_provider = provider_kind.name(),
            storage_queue_depth = slots,
            storage_staging_bytes = slots.saturating_mul(slot_bytes),
            host_weight_cache_bytes = 0,
            "configured bounded CPU storage-native weight ring"
        );
        Ok(())
    }

    #[must_use]
    pub fn storage_metrics(&self) -> CpuStorageMetrics {
        let guard = self.weight_storage.lock().expect("CPU weight storage lock");
        let Some(state) = guard.as_ref() else {
            return CpuStorageMetrics {
                mmap_execution_bytes: self.mmap_execution_bytes.load(Ordering::Relaxed),
                ..CpuStorageMetrics::default()
            };
        };
        CpuStorageMetrics {
            physical_reads: state.physical_reads,
            physical_bytes: state.physical_bytes,
            storage_stall_nanos: self.storage_wait_nanos.load(Ordering::Relaxed),
            staging_bytes: (state.resident.len() * state.slot_bytes) as u64,
            mmap_execution_bytes: self.mmap_execution_bytes.load(Ordering::Relaxed),
            buffered_storage_bytes: if matches!(
                state.provider_kind,
                fellm_memory::StorageProviderKind::PageCache
                    | fellm_memory::StorageProviderKind::Buffered
                    | fellm_memory::StorageProviderKind::Mmap
            ) {
                state.physical_bytes
            } else {
                0
            },
            direct_storage_bytes: if state.provider_kind
                == fellm_memory::StorageProviderKind::Direct
            {
                state.physical_bytes
            } else {
                0
            },
            io_compute_overlap_percent: if state.read_nanos == 0 {
                0.0
            } else {
                100.0
                    * state
                        .read_nanos
                        .saturating_sub(self.storage_wait_nanos.load(Ordering::Relaxed))
                        as f64
                    / state.read_nanos as f64
            },
            expert_slice_reads: state.expert_slice_reads,
            expert_slice_bytes: state.expert_slice_bytes,
            resident_dense_bytes: state
                .resident
                .iter()
                .filter_map(|slot| slot.as_ref().map(|read| read.valid_len as u64))
                .sum(),
            avg_read_bytes: if state.physical_reads == 0 {
                0.0
            } else {
                state.physical_bytes as f64 / state.physical_reads as f64
            },
        }
    }

    fn make_handle(op: OpKind) -> KernelHandle {
        KernelHandle(u64::from(op.raw()))
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn is_supported_matvec_weight_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::BF16
            | DType::Q4_0
            | DType::Q5_0
            | DType::Q8_0
            | DType::F16
            | DType::Q2K
            | DType::Q4K
            | DType::Q5K
            | DType::Q6K
            | DType::IQ2XXS
            | DType::IQ2XS
            | DType::IQ3XXS
            | DType::IQ3S
            | DType::MXFP4
    )
}

impl Backend for CpuBackend {
    fn id(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn memory_info(&self) -> Option<DeviceMemoryInfo> {
        let system = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        Some(DeviceMemoryInfo {
            total_bytes: system.total_memory(),
            available_bytes: system.available_memory(),
        })
    }

    fn resolve_kernel(
        &self,
        op: OpKind,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<KernelDescriptor> {
        let ok = match op {
            OpKind::MatMul => matches!(
                (input_dtypes.first(), input_dtypes.get(1)),
                (
                    Some(
                        DType::F32
                            | DType::F16
                            | DType::BF16
                            | DType::Q4_0
                            | DType::Q5_0
                            | DType::Q8_0
                            | DType::Q4K
                            | DType::Q5K
                            | DType::Q6K
                            | DType::IQ2XS
                            | DType::IQ3XXS
                            | DType::IQ3S
                            | DType::MXFP4
                    ),
                    Some(DType::F32)
                )
            ),
            OpKind::GateUpSwiGlu => matches!(
                (input_dtypes.first(), input_dtypes.get(1), input_dtypes.get(2)),
                (Some(gate), Some(up), Some(DType::F32))
                    if is_supported_matvec_weight_dtype(*gate)
                        && is_supported_matvec_weight_dtype(*up)
            ),
            OpKind::Embedding => input_dtypes
                .first()
                .map(|d| {
                    matches!(
                        d,
                        DType::F32
                            | DType::F16
                            | DType::BF16
                            | DType::Q4_0
                            | DType::Q5_0
                            | DType::Q8_0
                            | DType::Q4K
                            | DType::Q5K
                            | DType::Q6K
                    )
                })
                .unwrap_or(false),
            OpKind::WeightedEmbedding => matches!(
                (input_dtypes.first(), input_dtypes.get(1), output_dtype),
                (Some(weight), Some(DType::F32), DType::F32)
                    if is_supported_matvec_weight_dtype(*weight)
            ),
            OpKind::ShortConv => matches!(
                (
                    input_dtypes.first(),
                    input_dtypes.get(1),
                    input_dtypes.get(2),
                    input_dtypes.get(3)
                ),
                (Some(DType::F32), Some(w0), Some(DType::F32), Some(w1))
                    if is_supported_matvec_weight_dtype(*w0)
                        && is_supported_matvec_weight_dtype(*w1)
            ),
            OpKind::GatedDeltaNet => {
                input_dtypes.len() == 10
                    && input_dtypes[0] == DType::F32
                    && [1, 2, 3, 4, 9]
                        .into_iter()
                        .all(|index| is_supported_matvec_weight_dtype(input_dtypes[index]))
                    && input_dtypes[5..9].iter().all(|dtype| *dtype == DType::F32)
                    && output_dtype == DType::F32
            }
            OpKind::MoE => {
                let activations_ok = input_dtypes.first() == Some(&DType::F32)
                    && input_dtypes.get(1).is_some_and(|dtype| {
                        *dtype == DType::F32 || is_supported_matvec_weight_dtype(*dtype)
                    });
                let rest_ok = input_dtypes.iter().skip(2).all(|dtype| {
                    matches!(*dtype, DType::F32 | DType::I32 | DType::U32)
                        || is_supported_matvec_weight_dtype(*dtype)
                });
                activations_ok && rest_ok && input_dtypes.len() >= 5
            }
            OpKind::MlaAttention | OpKind::HyperConnection => {
                input_dtypes.first() == Some(&DType::F32)
                    && input_dtypes.iter().all(|dtype| {
                        *dtype == DType::F32 || is_supported_matvec_weight_dtype(*dtype)
                    })
                    && output_dtype == DType::F32
            }
            OpKind::RmsNorm
            | OpKind::Rope
            | OpKind::SiluGate
            | OpKind::Softmax
            | OpKind::Attention
            | OpKind::Add
            | OpKind::Mul
            | OpKind::SigmoidGate
            | OpKind::InterleavedHeadSelect
            | OpKind::Reshape
            | OpKind::Cast
            | OpKind::Concat
            | OpKind::Sample
            | OpKind::KvWrite => true,
            _ => false,
        };
        if !ok {
            return None;
        }
        Some(KernelDescriptor {
            op,
            input_dtypes: input_dtypes.to_vec(),
            output_dtype,
            handle: Self::make_handle(op),
        })
    }

    fn launch(
        &self,
        handle: KernelHandle,
        attrs: &OpAttrs,
        inputs: &[TensorRef],
        outputs: &mut [TensorMut],
        _stream: StreamHandle,
    ) -> Result<()> {
        self.launches.fetch_add(1, Ordering::Relaxed);
        self.worker_launches
            .get_or(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        let op = decode_handle(handle)?;
        let mut storage = self.weight_storage.lock().expect("CPU weight storage lock");
        if storage.is_none() {
            let mapped = inputs
                .iter()
                .filter(|input| input.logical_id != 0)
                .fold(0u64, |total, input| total.saturating_add(input.byte_len));
            self.mmap_execution_bytes
                .fetch_add(mapped, Ordering::Relaxed);
        }
        let mut aligned_copies = smallvec::SmallVec::<[fellm_core::storage::AlignedBuffer; 4]>::new();
        let mut prepared = smallvec::SmallVec::<[TensorRef; 8]>::new();
        for input in inputs {
            let mut view = *input;
            if let Some(state) = storage.as_ref()
                && state.sparse.contains_key(&input.logical_id)
            {
                prepared.push(view);
                continue;
            }
            if input.data.is_null() {
                if storage.is_none() {
                    return Err(FellmError::other(format!(
                        "CPU storage-backed weight {} has no storage installed",
                        input.logical_id
                    )));
                }
                if storage
                    .as_ref()
                    .is_none_or(|state| !state.weight_objects.contains_key(&input.logical_id))
                {
                    return Err(FellmError::other(format!(
                        "CPU storage-backed weight {} is not in any published object",
                        input.logical_id
                    )));
                }
            }
            if let Some(state) = storage.as_ref()
                && let Some(&object_id) = state.weight_objects.get(&input.logical_id)
            {
                let object = state
                    .objects
                    .get(&object_id)
                    .ok_or_else(|| FellmError::other("CPU storage object missing"))?;
                let member = object
                    .members
                    .iter()
                    .find(|member| member.weight.0 == input.logical_id)
                    .ok_or_else(|| FellmError::other("CPU storage object member missing"))?;
                let read = state
                    .resident
                    .iter()
                    .find_map(|slot| match slot {
                        Some(read) if read.object == object_id => Some(read),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        FellmError::other(format!(
                            "CPU weight {} was not published before kernel launch",
                            input.logical_id
                        ))
                    })?;
                let src = unsafe { read.buffer.as_slice().as_ptr().add(member.offset as usize) };
                view.byte_len = member.len;
                if (src as usize) % 64 == 0 {
                    view.data = src;
                } else {
                    let mut copy =
                        fellm_core::storage::AlignedBuffer::new_zeroed(member.len as usize, 64);
                    copy.as_mut_slice()[..member.len as usize]
                        .copy_from_slice(unsafe { std::slice::from_raw_parts(src, member.len as usize) });
                    view.data = copy.as_slice().as_ptr();
                    aligned_copies.push(copy);
                }
            }
            prepared.push(view);
        }
        let inputs = prepared.as_slice();
        let result = match op {
            OpKind::MatMul => launch_matmul(attrs, inputs, outputs),
            OpKind::GateUpSwiGlu => launch_gate_up_swiglu(inputs, outputs),
            OpKind::Embedding => launch_embedding(inputs, outputs),
            OpKind::RmsNorm => launch_rmsnorm(attrs, inputs, outputs),
            OpKind::Rope => launch_rope(attrs, inputs, outputs),
            OpKind::SiluGate => launch_silu_gate(inputs, outputs),
            OpKind::Softmax => launch_softmax(attrs, inputs, outputs),
            OpKind::Attention => launch_attention(self, attrs, inputs, outputs),
            OpKind::Add => launch_add(inputs, outputs, self.simd),
            OpKind::Mul => launch_mul(inputs, outputs, self.simd),
            OpKind::SigmoidGate => launch_sigmoid_gate(inputs, outputs),
            OpKind::Reshape => launch_reshape(inputs, outputs),
            OpKind::Cast => launch_cast(attrs, inputs, outputs),
            OpKind::Concat => launch_concat(inputs, outputs),
            OpKind::Sample => launch_sample(attrs, inputs, outputs),
            OpKind::KvWrite => launch_kv_write(attrs, inputs, outputs),
            OpKind::ShortConv => launch_shortconv(attrs, inputs, outputs),
            OpKind::GatedDeltaNet => launch_gated_delta_net(attrs, inputs, outputs),
            OpKind::InterleavedHeadSelect => launch_interleaved_head_select(attrs, inputs, outputs),
            OpKind::MlaAttention => launch_mla_attention(attrs, inputs, outputs),
            OpKind::HyperConnection => launch_hyper_connection(attrs, inputs, outputs),
            OpKind::MoE => launch_moe(attrs, inputs, outputs, storage.as_mut()),
            OpKind::WeightedEmbedding => launch_weighted_embedding(inputs, outputs),
            _ => Err(FellmError::other(
                "custom operation is not implemented by CPU backend",
            )),
        };
        drop(aligned_copies);
        result
    }

    fn prefetch_weight_group(
        &self,
        _group_id: u64,
        weights: &[TensorRef],
        required: bool,
    ) -> Result<()> {
        let mut guard = self.weight_storage.lock().expect("CPU weight storage lock");
        let Some(state) = guard.as_mut() else {
            return Ok(());
        };
        if !required && !state.overlap {
            return Ok(());
        }
        let mut objects = Vec::new();
        for weight in weights {
            if let Some(&object) = state.weight_objects.get(&weight.logical_id)
                && !objects.contains(&object)
            {
                objects.push(object);
            }
        }
        if required && !objects.is_empty() {
            let needed = objects.iter().copied().collect::<std::collections::HashSet<_>>();
            for resident in &mut state.resident {
                if matches!(resident, Some(read) if needed.contains(&read.object)) {
                    continue;
                }
                if let Some(read) = resident.take() {
                    state.available.push(read.buffer);
                }
            }
        }
        for object_id in objects.iter().copied() {
            if state
                .resident
                .iter()
                .any(|slot| matches!(slot, Some(read) if read.object == object_id))
            {
                continue;
            }
            if !state.pending.contains_key(&object_id) {
                let extent = state
                    .objects
                    .get(&object_id)
                    .ok_or_else(|| FellmError::other("CPU prefetch object missing"))?
                    .extent
                    .clone();
                if required && state.available.is_empty() {
                    let stale_id = state.pending.keys().copied().find(|id| {
                        !objects.contains(id)
                    });
                    if let Some(stale_id) = stale_id {
                    let stale = state.pending.remove(&stale_id).expect("pending key exists");
                    let stale = stale.join().map_err(|_| {
                        FellmError::other("CPU speculative storage reader panicked")
                    })??;
                    state.physical_reads = state.physical_reads.saturating_add(1);
                    state.physical_bytes =
                        state.physical_bytes.saturating_add(stale.valid_len as u64);
                    state.read_nanos = state.read_nanos.saturating_add(stale.read_nanos);
                    state.available.push(stale.buffer);
                    }
                }
                if let Some(mut buffer) = state.available.pop() {
                    let provider = state
                        .providers
                        .get(&extent.path)
                        .cloned()
                        .ok_or_else(|| {
                            FellmError::other(format!(
                                "no storage provider for {}",
                                extent.path.display()
                            ))
                        })?;
                    let valid_len = extent.len as usize;
                    let handle = std::thread::Builder::new()
                        .name(format!("fellm-cpu-storage-{}", object_id))
                        .spawn(move || {
                            let started = std::time::Instant::now();
                            provider
                                .read_at(extent.offset, &mut buffer.as_mut_slice()[..valid_len])
                                .map_err(|error| {
                                    FellmError::other(format!(
                                        "storage object {object_id} path={} offset={} len={}: {error}",
                                        extent.path.display(),
                                        extent.offset,
                                        extent.len
                                    ))
                                })?;
                            Ok(CpuRead {
                                object: object_id,
                                buffer,
                                valid_len,
                                read_nanos: started.elapsed().as_nanos().min(u128::from(u64::MAX))
                                    as u64,
                            })
                        })
                        .map_err(|error| {
                            FellmError::other(format!("spawn CPU storage reader: {error}"))
                        })?;
                    state.pending.insert(object_id, handle);
                } else if required {
                    return Err(FellmError::other("CPU storage ring has no demand slot"));
                }
            }
            if required {
                let reader = state
                    .pending
                    .remove(&object_id)
                    .ok_or_else(|| FellmError::other("CPU required prefetch missing"))?;
                let started = std::time::Instant::now();
                let read = reader.join().map_err(|_| {
                    FellmError::other(format!(
                        "CPU storage reader panicked for object {object_id}"
                    ))
                })??;
                self.storage_wait_nanos.fetch_add(
                    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    Ordering::Relaxed,
                );
                state.physical_reads = state.physical_reads.saturating_add(1);
                state.physical_bytes = state.physical_bytes.saturating_add(read.valid_len as u64);
                state.read_nanos = state.read_nanos.saturating_add(read.read_nanos);
                let slot = state
                    .resident
                    .iter()
                    .position(Option::is_none)
                    .ok_or_else(|| {
                        FellmError::other(format!(
                            "CPU execution bundle exceeds {} fixed storage slots",
                            state.resident.len()
                        ))
                    })?;
                state.resident[slot] = Some(read);
            }
        }
        Ok(())
    }

    fn begin_step(&self) {
        matmul::begin_q8k_step_cache();
    }

    fn end_step(&self) {
        matmul::end_q8k_step_cache();
    }
}

fn decode_handle(h: KernelHandle) -> Result<OpKind> {
    let raw = u32::try_from(h.0).map_err(|_| FellmError::other(format!("bad kernel handle {h:?}")))?;
    OpKind::from_u32(raw).ok_or_else(|| FellmError::other(format!("bad kernel handle {h:?}")))
}

fn as_f32_slice(t: &TensorRef) -> Result<&[f32]> {
    let d = t.dtype().ok_or_else(|| FellmError::other("bad dtype"))?;
    if d != DType::F32 {
        return Err(FellmError::UnsupportedDType(d));
    }
    // SAFETY: TensorRef is a valid contiguous buffer of `byte_len` bytes.
    let bytes = unsafe { core::slice::from_raw_parts(t.data, t.byte_len as usize) };
    bytemuck::try_cast_slice(bytes).map_err(|e| FellmError::other(format!("cast: {e:?}")))
}

fn as_f32_slice_mut(t: &mut TensorMut) -> Result<&mut [f32]> {
    let d = t.dtype().ok_or_else(|| FellmError::other("bad dtype"))?;
    if d != DType::F32 {
        return Err(FellmError::UnsupportedDType(d));
    }
    // SAFETY: TensorMut is a valid exclusive buffer of `byte_len` bytes.
    let bytes = unsafe { core::slice::from_raw_parts_mut(t.data, t.byte_len as usize) };
    bytemuck::try_cast_slice_mut(bytes).map_err(|e| FellmError::other(format!("cast mut: {e:?}")))
}

fn as_bytes_slice(t: &TensorRef) -> &[u8] {
    // SAFETY: valid by TensorRef contract.
    unsafe { core::slice::from_raw_parts(t.data, t.byte_len as usize) }
}

fn as_u32_slice(t: &TensorRef) -> Result<&[u32]> {
    let bytes = as_bytes_slice(t);
    bytemuck::try_cast_slice(bytes).map_err(|e| FellmError::other(format!("u32 cast: {e:?}")))
}

fn as_i32_slice(t: &TensorRef) -> Result<&[i32]> {
    let bytes = as_bytes_slice(t);
    bytemuck::try_cast_slice(bytes).map_err(|e| FellmError::other(format!("i32 cast: {e:?}")))
}

/// Optional trailing `(tid2eid, token_id)` pair: a frozen token-id → expert-id table.
fn moe_forced_experts<'a>(
    inputs: &'a [TensorRef],
    n_expert_used: usize,
) -> Result<(usize, Option<Vec<i32>>)> {
    if inputs.len() < 2 {
        return Ok((inputs.len(), None));
    }
    let map = &inputs[inputs.len() - 2];
    let tok = &inputs[inputs.len() - 1];
    if map.dtype() != Some(DType::I32) || tok.dtype() != Some(DType::U32) {
        return Ok((inputs.len(), None));
    }
    let table = as_i32_slice(map)?;
    let ids = as_u32_slice(tok)?;
    let token = *ids.first().ok_or_else(|| FellmError::other("moe: empty token_id"))?;
    let used = n_expert_used.max(1);
    let start = token as usize * used;
    let end = start.saturating_add(used);
    if end > table.len() {
        return Err(FellmError::other(format!(
            "moe: token {token} expert map out of range (table={}, used={used})",
            table.len()
        )));
    }
    Ok((inputs.len() - 2, Some(table[start..end].to_vec())))
}

fn launch_matmul(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // inputs[0] = weight [out_dim, in_dim], inputs[1] = x [in_dim]
    // outputs[0] = y [out_dim]
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("matmul: bad arity"));
    }
    let (y_out, rest) = outputs.split_first_mut().unwrap();
    let _ = rest;
    let w = &inputs[0];
    let x = &inputs[1];
    let w_dtype = w.dtype().ok_or_else(|| FellmError::other("w dtype"))?;
    let out_dim = w.dims_slice()[0] as usize;
    let in_dim = w.dims_slice()[1] as usize;
    let x_slice = as_f32_slice(x)?;
    let y_slice = as_f32_slice_mut(y_out)?;
    // A rank-1 activation is the scalar decode form; higher-rank activations
    // flatten every dimension before the final feature dimension into rows.
    let rows = if x.dims_slice().len() <= 1 {
        1
    } else {
        x.dims_slice()[..x.dims_slice().len() - 1]
            .iter()
            .product::<u64>() as usize
    };
    if rows == 0 || x_slice.len() != rows * in_dim || y_slice.len() != rows * out_dim {
        return Err(FellmError::other(format!(
            "matmul: batched shape mismatch x_dims={:?} x_len={} w_dims={:?} y_len={} rows={} in_dim={} out_dim={}",
            x.dims_slice(),
            x_slice.len(),
            w.dims_slice(),
            y_slice.len(),
            rows,
            in_dim,
            out_dim
        )));
    }
    match w_dtype {
        DType::F32 => {
            let ws = as_f32_slice(w)?;
            matmul::matmul_f32_batch(ws, x_slice, y_slice, rows, out_dim, in_dim)?;
        }
        DType::F16 | DType::BF16 => {
            matmul::matmul_dense16_batch(
                as_bytes_slice(w),
                w_dtype,
                x_slice,
                y_slice,
                rows,
                out_dim,
                in_dim,
            )?;
        }
        DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q5K | DType::Q6K => {
            let wb = as_bytes_slice(w);
            if rows > 1 {
                matmul::matmul_quant_batch(wb, w_dtype, x_slice, y_slice, rows, out_dim, in_dim)?;
            } else {
                matmul::matvec_quant(wb, w_dtype, x_slice, y_slice, out_dim, in_dim)?;
            }
        }
        other => return Err(FellmError::UnsupportedDType(other)),
    }
    if let Some(residual) = inputs.get(2) {
        let residual = as_f32_slice(residual)?;
        if residual.len() != y_slice.len() {
            return Err(FellmError::other(
                "matmul residual epilogue: shape mismatch",
            ));
        }
        static SIMD: std::sync::OnceLock<crate::kernels::simd_f32::PulpDispatch> =
            std::sync::OnceLock::new();
        let simd = *SIMD.get_or_init(crate::kernels::simd_f32::PulpDispatch::new);
        crate::kernels::simd_f32::axpy_f32(y_slice, residual, 1.0, simd);
    }
    if attrs.softcap > 0.0 {
        let cap = attrs.softcap;
        for value in y_slice {
            *value = cap * (*value / cap).tanh();
        }
    }
    Ok(())
}

fn launch_gate_up_swiglu(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() != 3 || outputs.is_empty() {
        return Err(FellmError::other("gate_up_swiglu: bad arity"));
    }
    let x = as_f32_slice(&inputs[2])?;
    let out = as_f32_slice_mut(&mut outputs[0])?;
    let rows = inputs[0].dims_slice()[0] as usize;
    let cols = inputs[0].dims_slice()[1] as usize;
    if inputs[1].dims_slice() != inputs[0].dims_slice() || x.len() != cols || out.len() != rows {
        return Err(FellmError::other("gate_up_swiglu: shape mismatch"));
    }
    let gate_dtype = inputs[0]
        .dtype()
        .ok_or_else(|| FellmError::other("gate_up_swiglu: gate dtype"))?;
    let up_dtype = inputs[1]
        .dtype()
        .ok_or_else(|| FellmError::other("gate_up_swiglu: up dtype"))?;

    thread_local! {
        static GATE_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        static COMBINED_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    if gate_dtype == up_dtype && matches!(gate_dtype, DType::Q4K | DType::Q5K | DType::Q6K) {
        return COMBINED_SCRATCH.with(|combined_cell| {
            let mut combined = combined_cell.borrow_mut();
            if combined.len() < rows * 2 {
                combined.resize(rows * 2, 0.0);
            }
            let mats = [
                matmul::MatDesc {
                    w: as_bytes_slice(&inputs[0]),
                    dtype: gate_dtype,
                    out_dim: rows,
                    in_dim: cols,
                    x_off: 0,
                },
                matmul::MatDesc {
                    w: as_bytes_slice(&inputs[1]),
                    dtype: up_dtype,
                    out_dim: rows,
                    in_dim: cols,
                    x_off: 0,
                },
            ];
            matmul::matvec_quant_multi(x, &mats, &mut combined[..rows * 2])?;
            // combined = [gate | up] → out[i] = silu(gate[i]) * up[i]
            let (gate, up) = combined.split_at(rows);
            silu_gate(gate, up, out);
            Ok(())
        });
    }

    GATE_SCRATCH.with(|cell| {
        let mut gate = cell.borrow_mut();
        if gate.len() < rows {
            gate.resize(rows, 0.0);
        }
        let gate = &mut gate[..rows];
        match gate_dtype {
            DType::F32 => {
                matmul::matmul_f32_batch(as_f32_slice(&inputs[0])?, x, gate, 1, rows, cols)?
            }
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q5K | DType::Q6K => {
                matmul::matvec_quant(as_bytes_slice(&inputs[0]), gate_dtype, x, gate, rows, cols)?
            }
            other => return Err(FellmError::UnsupportedDType(other)),
        }
        match up_dtype {
            DType::F32 => {
                matmul::matmul_f32_batch(as_f32_slice(&inputs[1])?, x, out, 1, rows, cols)?
            }
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Q4K | DType::Q5K | DType::Q6K => {
                matmul::matvec_quant(as_bytes_slice(&inputs[1]), up_dtype, x, out, rows, cols)?
            }
            other => return Err(FellmError::UnsupportedDType(other)),
        }
        silu_gate_inplace(gate, out);
        Ok(())
    })
}

fn launch_embedding(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("embedding: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let w = &inputs[0];
    let tok = &inputs[1];
    let w_dtype = w.dtype().ok_or_else(|| FellmError::other("w dtype"))?;
    let vocab = w.dims_slice()[0] as usize;
    let dim = w.dims_slice()[1] as usize;
    let ids = as_u32_slice(tok)?;
    let y_slice = as_f32_slice_mut(y_out)?;
    if ids.is_empty() || y_slice.len() != ids.len() * dim {
        return Err(FellmError::other("embedding: batched shape mismatch"));
    }
    let wb = as_bytes_slice(w);
    for (tok_id, row) in ids.iter().copied().zip(y_slice.chunks_exact_mut(dim)) {
        embedding_row(wb, w_dtype, vocab, dim, tok_id, row)?;
    }
    Ok(())
}

fn launch_weighted_embedding(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("weighted embedding: bad arity"));
    }
    let weight = &inputs[0];
    let logits = as_f32_slice(&inputs[1])?;
    let output = as_f32_slice_mut(outputs.first_mut().unwrap())?;
    let weight_dtype = weight
        .dtype()
        .ok_or_else(|| FellmError::other("weighted embedding: weight dtype"))?;
    let dims = weight.dims_slice();
    if dims.len() != 2 {
        return Err(FellmError::other(
            "weighted embedding: weight must be rank 2",
        ));
    }
    let vocab = dims[0] as usize;
    let dim = dims[1] as usize;
    if vocab == 0 || dim == 0 {
        return Err(FellmError::other(format!(
            "weighted embedding: invalid weight shape dims={dims:?}"
        )));
    }
    let logit_dims = inputs[1].dims_slice();
    if logit_dims.len() == 2 && logit_dims[1] as usize != vocab {
        let slots = logit_dims[1] as usize;
        if !slots.is_multiple_of(2) {
            return Err(FellmError::other(
                "weighted embedding: packed top-k input must contain pairs",
            ));
        }
        return crate::kernels::embedding::weighted_embedding_topk(
            as_bytes_slice(weight),
            weight_dtype,
            logits,
            output,
            logit_dims[0] as usize,
            slots / 2,
            vocab,
            dim,
        );
    }
    if !logits.len().is_multiple_of(vocab) {
        return Err(FellmError::other("weighted embedding: invalid dense shape"));
    }
    weighted_embedding(
        as_bytes_slice(weight),
        weight_dtype,
        logits,
        output,
        logits.len() / vocab,
        vocab,
        dim,
    )
}

fn launch_rmsnorm(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("rmsnorm: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let w = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    let head_dim = attrs.head_dim as usize;
    let n_heads = attrs.n_heads as usize;
    if head_dim > 0
        && n_heads > 0
        && w.len() == head_dim
        && x.len().is_multiple_of(n_heads * head_dim)
    {
        for (x_row, y_row) in x
            .chunks_exact(n_heads * head_dim)
            .zip(y.chunks_exact_mut(n_heads * head_dim))
        {
            rmsnorm_groups(x_row, w, attrs.eps, head_dim, y_row);
        }
    } else if !w.is_empty() && x.len().is_multiple_of(w.len()) {
        for (x_row, y_row) in x.chunks_exact(w.len()).zip(y.chunks_exact_mut(w.len())) {
            rmsnorm_row(x_row, w, attrs.eps, y_row);
        }
    } else {
        rmsnorm_row(x, w, attrs.eps, y);
    }
    Ok(())
}

fn launch_rope(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("rope: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x_in = as_f32_slice(&inputs[0])?;
    let inv_freqs = as_f32_slice(&inputs[1])?;
    let x_out = as_f32_slice_mut(y_out)?;
    x_out.copy_from_slice(x_in);
    let row_width = attrs.n_heads.max(1) as usize * attrs.head_dim.max(1) as usize;
    let paged = paged_ctx::snapshot_paged_context();
    if row_width == 0 || x_in.len().is_multiple_of(row_width) {
        for (row, values) in x_out.chunks_exact_mut(row_width).enumerate() {
            let position = paged
                .as_ref()
                .and_then(|ctx| ctx.row_rope_positions.get(row).copied())
                .unwrap_or_else(|| attrs.position.saturating_add(row as u32));
            if attrs.custom_op_id == 1 {
                fellm_plugin_abi::pre_rope_write(
                    attrs.layer_ord as usize,
                    position as usize,
                    &x_in[row * row_width..(row + 1) * row_width],
                );
            }
            crate::kernels::rope::rope_inplace_with_freqs_ex(
                values,
                attrs.n_heads as usize,
                attrs.head_dim as usize,
                attrs.rope_dim as usize,
                position,
                inv_freqs,
                attrs.rope_pairing == 1,
                attrs.rope_pairing == 2,
            );
        }
    }
    Ok(())
}

fn launch_silu_gate(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("silu_gate: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let gate = as_f32_slice(&inputs[0])?;
    let up = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    silu_gate(gate, up, y);
    Ok(())
}

fn launch_interleaved_head_select(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(FellmError::other("interleaved_head_select: bad arity"));
    }
    let input = as_f32_slice(&inputs[0])?;
    let output = as_f32_slice_mut(&mut outputs[0])?;
    let heads = attrs.n_heads as usize;
    let width = attrs.head_dim as usize;
    let source_row = heads * width * 2;
    let output_row = heads * width;
    let lane = attrs.kv_slot as usize;
    if heads == 0
        || width == 0
        || lane > 1
        || !input.len().is_multiple_of(source_row)
        || output.len() != input.len() / 2
    {
        return Err(FellmError::other("interleaved_head_select: shape mismatch"));
    }
    for (source, target) in input
        .chunks_exact(source_row)
        .zip(output.chunks_exact_mut(output_row))
    {
        for head in 0..heads {
            let source_start = (head * 2 + lane) * width;
            let target_start = head * width;
            target[target_start..target_start + width]
                .copy_from_slice(&source[source_start..source_start + width]);
        }
    }
    Ok(())
}

fn launch_softmax(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("softmax: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let y = as_f32_slice_mut(y_out)?;
    y.copy_from_slice(x);
    // Interpret the last-dim as row length; higher dims collapsed.
    let dims = inputs[0].dims_slice();
    let last = *dims.last().unwrap_or(&(y.len() as u64)) as usize;
    let n_rows = y.len() / last.max(1);
    let causal = if attrs.past_len > 0 {
        Some((attrs.past_len as usize + 1).min(last))
    } else {
        None
    };
    softmax_rows_inplace(y, n_rows, last, causal);
    Ok(())
}

fn launch_attention(
    backend: &CpuBackend,
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if inputs.len() < 3 || outputs.is_empty() {
        return Err(FellmError::other("attention: bad arity"));
    }
    if attrs.attention_mode == 1 && inputs.len() >= 5 {
        return launch_canvas_attention(backend, attrs, inputs, outputs);
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let q = as_f32_slice(&inputs[0])?;
    let out = as_f32_slice_mut(y_out)?;
    let n_heads = attrs.n_heads as usize;
    let n_kv = attrs.n_kv_heads.max(1) as usize;
    let head_dim = attrs.head_dim as usize;
    let past = attrs.past_len as usize;
    let scale = if attrs.scale > 0.0 {
        attrs.scale
    } else {
        1.0 / (head_dim as f32).sqrt()
    };

    let use_paged = attrs.block_size > 0 && paged_ctx::has_paged_context();
    let path = fellm_plugin_abi::resolve_path(attrs.query_len.max(1), attrs.custom_op_id);
    let use_fa2_host = matches!(
        path,
        fellm_plugin_abi::AttentionKernelPath::HostFa2
            | fellm_plugin_abi::AttentionKernelPath::Fa2Decode
            | fellm_plugin_abi::AttentionKernelPath::Fa2Prefill
            | fellm_plugin_abi::AttentionKernelPath::Auto
    );
    let dispatch = fellm_plugin_abi::attention_dispatch();
    let br = dispatch.q_tile.max(4) as usize;
    let bc = dispatch.kv_tile.max(16) as usize;

    if use_paged {
        let layer = attrs.layer_ord as usize;
        let ctx = paged_ctx::snapshot_paged_context()
            .ok_or_else(|| FellmError::other("attention: missing paged ctx"))?;
        let row_width = n_heads * head_dim;
        if ctx.batch_size() > 1 || q.len() > row_width {
            if q.len() != ctx.batch_size() * row_width || out.len() != q.len() {
                return Err(FellmError::other(
                    "attention: physical batch shape mismatch",
                ));
            }
            for row in 0..ctx.batch_size() {
                let q_row = &q[row * row_width..(row + 1) * row_width];
                let out_row = &mut out[row * row_width..(row + 1) * row_width];
                let seq = ctx.row_lengths[row] as usize;
                fellm_plugin_abi::fa2_style_attention_paged_f32(
                    q_row,
                    out_row,
                    n_heads,
                    n_kv,
                    head_dim,
                    seq,
                    scale,
                    true,
                    attrs.attention_window as usize,
                    br,
                    bc,
                    |t, is_v, values| {
                        let full = unsafe {
                            if is_v {
                                ctx.v_row_for(row, layer, t)
                            } else {
                                ctx.k_row_for(row, layer, t)
                            }
                        };
                        for (dst, &src) in values.iter_mut().zip(full.iter()) {
                            *dst = src.to_f32();
                        }
                    },
                );
            }
            return Ok(());
        }
        if use_fa2_host {
            // Prepared FA2-style path over paged storage (host).
            let seq = past + 1;
            let window = attrs.attention_window as usize;
            let causal = attrs.attention_mode == 0;
            fellm_plugin_abi::fa2_style_attention_paged_f32(
                q,
                out,
                n_heads,
                n_kv,
                head_dim,
                seq,
                scale,
                causal,
                window,
                br,
                bc,
                |t, is_v, row| {
                    let full = unsafe {
                        if is_v {
                            ctx.v_row(layer, t)
                        } else {
                            ctx.k_row(layer, t)
                        }
                    };
                    for (d, &s) in row.iter_mut().zip(full.iter()) {
                        *d = s.to_f32();
                    }
                },
            );
            return Ok(());
        }
        attention_step_paged(
            q,
            out,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            layer,
            &backend.profile,
            backend.simd,
        );
        return Ok(());
    }

    let k_full = as_f32_slice(&inputs[1])?;
    let v_full = as_f32_slice(&inputs[2])?;
    let seq = past + 1;
    let kv_elems = seq * n_kv * head_dim;
    if k_full.len() < kv_elems || v_full.len() < kv_elems {
        return Err(FellmError::other(format!(
            "attention: kv buffer too small (need {kv_elems}, k={}, v={})",
            k_full.len(),
            v_full.len()
        )));
    }
    let k = &k_full[..kv_elems];
    let v = &v_full[..kv_elems];
    if use_fa2_host {
        let window = attrs.attention_window as usize;
        let causal = attrs.attention_mode == 0;
        let q_len = attrs.query_len.max(1) as usize;
        fellm_plugin_abi::fa2_style_attention_f32(
            q, k, v, out, n_heads, n_kv, head_dim, q_len, seq, scale, causal, window, br, bc,
        );
        return Ok(());
    }
    backend.pool.install(|| {
        attention_step(
            q,
            k,
            v,
            out,
            n_heads,
            n_kv,
            head_dim,
            past,
            scale,
            &backend.profile,
            backend.simd,
        );
    });
    Ok(())
}

fn launch_canvas_attention(
    backend: &CpuBackend,
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let q = as_f32_slice(&inputs[0])?;
    let k_canvas = as_f32_slice(&inputs[3])?;
    let v_canvas = as_f32_slice(&inputs[4])?;
    let out = as_f32_slice_mut(
        outputs
            .first_mut()
            .ok_or_else(|| FellmError::other("canvas attention: no output"))?,
    )?;
    let n_heads = attrs.n_heads.max(1) as usize;
    let n_kv = attrs.n_kv_heads.max(1) as usize;
    let head_dim = attrs.head_dim.max(1) as usize;
    let row_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let rows = attrs.query_len.max(1) as usize;
    let prefix_len = attrs.past_len as usize;
    if q.len() != rows * row_width
        || out.len() != rows * row_width
        || k_canvas.len() < rows * kv_width
        || v_canvas.len() < rows * kv_width
    {
        return Err(FellmError::other("canvas attention: shape mismatch"));
    }
    let heads_per_kv = n_heads / n_kv;
    let window = attrs.attention_window as usize;
    let prefix_start = if window == 0 {
        0
    } else {
        prefix_len.saturating_sub(window)
    };
    let prefix_ctx = if attrs.block_size > 0 {
        paged_ctx::snapshot_paged_context()
    } else {
        None
    };
    let prefix_k = as_f32_slice(&inputs[1]).ok();
    let prefix_v = as_f32_slice(&inputs[2]).ok();
    backend.pool.install(|| {
        q.par_chunks_exact(row_width)
            .zip(out.par_chunks_exact_mut(row_width))
            .for_each(|(q_row, out_row)| {
                for h in 0..n_heads {
                    let kv_h = h / heads_per_kv;
                    let q_head = &q_row[h * head_dim..(h + 1) * head_dim];
                    let mut scores =
                        Vec::with_capacity(prefix_len.saturating_sub(prefix_start) + rows);
                    for t in prefix_start..prefix_len {
                        let mut score = 0.0f32;
                        if let Some(ref ctx) = prefix_ctx {
                            // SAFETY: the active paged context remains valid for the graph step.
                            let k_row = unsafe { ctx.k_row(attrs.layer_ord as usize, t) };
                            let k_head = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
                            for i in 0..head_dim {
                                score += q_head[i] * k_head[i].to_f32();
                            }
                        } else if let Some(prefix_k) = prefix_k {
                            let k_row = &prefix_k[t * kv_width + kv_h * head_dim
                                ..t * kv_width + (kv_h + 1) * head_dim];
                            for i in 0..head_dim {
                                score += q_head[i] * k_row[i];
                            }
                        }
                        scores.push(
                            score
                                * if attrs.scale > 0.0 {
                                    attrs.scale
                                } else {
                                    1.0 / (head_dim as f32).sqrt()
                                },
                        );
                    }
                    for canvas_row in 0..rows {
                        let k_row = &k_canvas[canvas_row * kv_width + kv_h * head_dim
                            ..canvas_row * kv_width + (kv_h + 1) * head_dim];
                        let mut score = 0.0f32;
                        for i in 0..head_dim {
                            score += q_head[i] * k_row[i];
                        }
                        scores.push(
                            score
                                * if attrs.scale > 0.0 {
                                    attrs.scale
                                } else {
                                    1.0 / (head_dim as f32).sqrt()
                                },
                        );
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denom = scores.iter().map(|s| (*s - max).exp()).sum::<f32>();
                    let out_head = &mut out_row[h * head_dim..(h + 1) * head_dim];
                    out_head.fill(0.0);
                    if denom > 0.0 {
                        for (index, score) in scores.iter().copied().enumerate() {
                            let weight = (score - max).exp() / denom;
                            if index < prefix_len.saturating_sub(prefix_start) {
                                let t = prefix_start + index;
                                if let Some(ref ctx) = prefix_ctx {
                                    // SAFETY: the active paged context remains valid for the graph step.
                                    let v_row = unsafe { ctx.v_row(attrs.layer_ord as usize, t) };
                                    let v_head = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
                                    for i in 0..head_dim {
                                        out_head[i] += weight * v_head[i].to_f32();
                                    }
                                } else if let Some(prefix_v) = prefix_v {
                                    let v_row = &prefix_v[t * kv_width + kv_h * head_dim
                                        ..t * kv_width + (kv_h + 1) * head_dim];
                                    for i in 0..head_dim {
                                        out_head[i] += weight * v_row[i];
                                    }
                                }
                            } else {
                                let canvas_row = index - prefix_len.saturating_sub(prefix_start);
                                let v_row = &v_canvas[canvas_row * kv_width + kv_h * head_dim
                                    ..canvas_row * kv_width + (kv_h + 1) * head_dim];
                                for i in 0..head_dim {
                                    out_head[i] += weight * v_row[i];
                                }
                            }
                        }
                    }
                }
            });
    });
    Ok(())
}

fn launch_add(inputs: &[TensorRef], outputs: &mut [TensorMut], simd: PulpDispatch) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("add: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let a = as_f32_slice(&inputs[0])?;
    let b = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    crate::kernels::simd_f32::add_f32(a, b, y, simd);
    Ok(())
}

fn launch_mul(inputs: &[TensorRef], outputs: &mut [TensorMut], simd: PulpDispatch) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("mul: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let a = as_f32_slice(&inputs[0])?;
    let b = as_f32_slice(&inputs[1])?;
    let y = as_f32_slice_mut(y_out)?;
    crate::kernels::simd_f32::mul_f32(a, b, y, simd);
    Ok(())
}

fn launch_sigmoid_gate(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(FellmError::other("sigmoid_gate: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().expect("checked");
    let value = as_f32_slice(&inputs[0])?;
    let gate = as_f32_slice(&inputs[1])?;
    let output = as_f32_slice_mut(y_out)?;
    if value.len() != gate.len() || value.len() != output.len() {
        return Err(FellmError::other("sigmoid_gate: shape mismatch"));
    }
    for ((output, &value), &gate) in output.iter_mut().zip(value).zip(gate) {
        *output = value * (1.0 / (1.0 + (-gate).exp()));
    }
    Ok(())
}

fn launch_reshape(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("reshape: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let src = as_bytes_slice(&inputs[0]);
    // SAFETY: y_out.byte_len is the valid target length.
    let dst = unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
    if dst.len() != src.len() {
        return Err(FellmError::other(format!(
            "reshape: byte length mismatch (src={}, dst={})",
            src.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(src);
    Ok(())
}

fn launch_cast(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("cast: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let src_dtype = inputs[0]
        .dtype()
        .ok_or_else(|| FellmError::other("cast src dtype"))?;
    let dst_dtype = attrs
        .cast_dtype()
        .ok_or_else(|| FellmError::other("cast: dst dtype unset"))?;

    // Phase 1: support f32<->f16, f32<->bf16, and quantized -> f32 (via dequant).
    match (src_dtype, dst_dtype) {
        (DType::F32, DType::F32) => {
            let src = as_f32_slice(&inputs[0])?;
            let dst = as_f32_slice_mut(y_out)?;
            dst.copy_from_slice(src);
        }
        (DType::F16, DType::F32) => {
            let bytes = as_bytes_slice(&inputs[0]);
            let src: &[half::f16] = bytemuck::cast_slice(bytes);
            let dst = as_f32_slice_mut(y_out)?;
            for i in 0..dst.len() {
                dst[i] = src[i].to_f32();
            }
        }
        (DType::BF16, DType::F32) => {
            let bytes = as_bytes_slice(&inputs[0]);
            let src: &[u16] = bytemuck::cast_slice(bytes);
            let dst = as_f32_slice_mut(y_out)?;
            for i in 0..dst.len() {
                dst[i] = f32::from_bits((u32::from(src[i])) << 16);
            }
        }
        (DType::F32, DType::F16) => {
            let src = as_f32_slice(&inputs[0])?;
            // SAFETY: y_out.byte_len valid.
            let dst_bytes =
                unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
            let dst: &mut [half::f16] = bytemuck::cast_slice_mut(dst_bytes);
            for i in 0..src.len() {
                dst[i] = half::f16::from_f32(src[i]);
            }
        }
        (DType::F32, DType::BF16) => {
            let src = as_f32_slice(&inputs[0])?;
            let dst_bytes =
                unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
            let dst: &mut [u16] = bytemuck::cast_slice_mut(dst_bytes);
            for i in 0..src.len() {
                dst[i] = (src[i].to_bits() >> 16) as u16;
            }
        }
        (q, DType::F32) if q.is_quantized() => {
            let dst = as_f32_slice_mut(y_out)?;
            let bytes = as_bytes_slice(&inputs[0]);
            crate::dequant::dequantize_row(q, bytes, dst, dst.len())?;
        }
        (a, b) => {
            return Err(FellmError::other(format!("cast: unsupported {a} -> {b}")));
        }
    }
    Ok(())
}

fn launch_concat(inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // Concat along last dim, contiguous f32.
    if outputs.is_empty() {
        return Err(FellmError::other("concat: no outputs"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let dst = as_f32_slice_mut(y_out)?;
    if inputs.is_empty() {
        return Err(FellmError::other("concat: no inputs"));
    }
    let rows = inputs[0].dims_slice()[..inputs[0].dims_slice().len().saturating_sub(1)]
        .iter()
        .product::<u64>() as usize;
    let rows = rows.max(1);
    let widths = inputs
        .iter()
        .map(|input| input.dims_slice().last().copied().unwrap_or(1) as usize)
        .collect::<Vec<_>>();
    let total_width = widths.iter().sum::<usize>();
    if dst.len() != rows * total_width {
        return Err(FellmError::other(format!(
            "concat: shape mismatch (rows={rows}, width={total_width}, dst={})",
            dst.len()
        )));
    }
    for row in 0..rows {
        let mut column = 0;
        for (input, &width) in inputs.iter().zip(&widths) {
            let values = as_f32_slice(input)?;
            if values.len() != rows * width {
                return Err(FellmError::other("concat: input shape mismatch"));
            }
            dst[row * total_width + column..row * total_width + column + width]
                .copy_from_slice(&values[row * width..(row + 1) * width]);
            column += width;
        }
    }
    Ok(())
}

fn launch_sample(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("sample: bad arity"));
    }
    let (y_out, _) = outputs.split_first_mut().unwrap();
    // We need a mutable f32 copy of logits to modify in place.
    let logits_in = as_f32_slice(&inputs[0])?;
    let mut work = logits_in.to_vec();
    let tok = sample(
        &mut work,
        attrs.temperature,
        attrs.top_k,
        attrs.top_p,
        attrs.seed,
    );
    // Write out as u32.
    // SAFETY: y_out.byte_len is valid and at least 4 bytes.
    let dst_bytes = unsafe { core::slice::from_raw_parts_mut(y_out.data, y_out.byte_len as usize) };
    if dst_bytes.len() < 4 {
        return Err(FellmError::other("sample: output too small"));
    }
    dst_bytes[..4].copy_from_slice(&tok.to_le_bytes());
    Ok(())
}

fn launch_kv_write(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    // inputs[0] = row [dim] f32
    // outputs[0] = kv_buf [max_seq, dim] f32 (aliased mutable storage) — unused when paged
    if inputs.is_empty() || outputs.is_empty() {
        return Err(FellmError::other("kv_write: bad arity"));
    }
    let row = as_f32_slice(&inputs[0])?;
    let pos = attrs.position as usize;
    let use_paged = attrs.block_size > 0 && paged_ctx::has_paged_context();

    if use_paged {
        let layer = attrs.layer_ord as usize;
        let is_v = attrs.kv_slot != 0;
        return paged_ctx::with_paged_context_mut(|ctx| {
            let ctx = ctx.ok_or_else(|| FellmError::other("kv_write: missing paged ctx"))?;
            if !row.len().is_multiple_of(ctx.tokens_stride)
                || row.len() / ctx.tokens_stride != ctx.batch_size()
            {
                return Err(FellmError::other(format!(
                    "kv_write: {} values do not match {} rows of stride {}",
                    row.len(),
                    ctx.batch_size(),
                    ctx.tokens_stride
                )));
            }
            let stride = ctx.tokens_stride;
            for batch_row in 0..ctx.batch_size() {
                let pos = ctx.row_positions[batch_row] as usize;
                let dst = unsafe { ctx.row_mut_for(batch_row, layer, pos, is_v) };
                let src = &row[batch_row * stride..(batch_row + 1) * stride];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = half::f16::from_f32(s);
                }
            }
            Ok(())
        });
    }

    let (buf_out, _) = outputs.split_first_mut().unwrap();
    let dims = buf_out.dims_slice();
    if dims.len() != 2 {
        return Err(FellmError::other("kv_write: kv_buf must be 2D"));
    }
    let max_seq = dims[0] as usize;
    let dim = dims[1] as usize;
    if row.len() != dim {
        return Err(FellmError::other(format!(
            "kv_write: row len {} != dim {dim}",
            row.len()
        )));
    }
    if pos >= max_seq {
        return Err(FellmError::other(format!(
            "kv_write: position {pos} >= max_seq {max_seq}"
        )));
    }
    let dst = as_f32_slice_mut(buf_out)?;
    dst[pos * dim..(pos + 1) * dim].copy_from_slice(row);
    Ok(())
}

fn launch_shortconv(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    // inputs: [x, in_proj, conv, out_proj]
    // outputs: [y, conv_state]
    if inputs.len() < 4 || outputs.len() < 2 {
        return Err(FellmError::other("shortconv: bad arity"));
    }
    let (y_out, rest) = outputs.split_first_mut().unwrap();
    let state_out = rest
        .first_mut()
        .ok_or_else(|| FellmError::other("shortconv: missing state output"))?;

    let x = as_f32_slice(&inputs[0])?;
    let in_proj = &inputs[1];
    let conv = as_f32_slice(&inputs[2])?;
    let out_proj = &inputs[3];
    let y = as_f32_slice_mut(y_out)?;
    let state = as_f32_slice_mut(state_out)?;

    let n_embd = if attrs.n_embd > 0 {
        attrs.n_embd as usize
    } else {
        x.len()
    };
    let conv_dims = inputs[2].dims_slice();
    let l_cache = if attrs.shortconv_l_cache > 0 {
        attrs.shortconv_l_cache as usize
    } else {
        *conv_dims
            .get(1)
            .ok_or_else(|| FellmError::other("shortconv: conv weight must be 2D"))? as usize
    };

    let in_proj_dtype = in_proj
        .dtype()
        .ok_or_else(|| FellmError::other("shortconv: in_proj dtype"))?;
    let out_proj_dtype = out_proj
        .dtype()
        .ok_or_else(|| FellmError::other("shortconv: out_proj dtype"))?;

    let state_elements = l_cache.saturating_sub(1) * n_embd;
    if !x.len().is_multiple_of(n_embd)
        || y.len() != x.len()
        || state.len() != x.len() / n_embd * state_elements
    {
        return Err(FellmError::other("shortconv: batched shape mismatch"));
    }
    for ((x_row, y_row), state_row) in x
        .chunks_exact(n_embd)
        .zip(y.chunks_exact_mut(n_embd))
        .zip(state.chunks_exact_mut(state_elements))
    {
        crate::kernels::shortconv::shortconv_decode(
            x_row,
            as_bytes_slice(in_proj),
            in_proj_dtype,
            conv,
            as_bytes_slice(out_proj),
            out_proj_dtype,
            state_row,
            y_row,
            n_embd,
            l_cache,
        )?;
    }
    Ok(())
}

fn launch_gated_delta_net(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    use crate::kernels::gated_delta_net::{
        GatedDeltaNetDimensions, GatedDeltaNetWeights, gated_delta_net_decode,
    };
    if inputs.len() != 10 || outputs.len() < 3 {
        return Err(FellmError::other("gated_delta_net: bad arity"));
    }
    let (model_output, states) = outputs.split_first_mut().unwrap();
    let (conv_output, ssm_outputs) = states.split_first_mut().unwrap();
    let ssm_output = ssm_outputs
        .first_mut()
        .ok_or_else(|| FellmError::other("gated_delta_net: missing recurrent state"))?;
    let x = as_f32_slice(&inputs[0])?;
    let output = as_f32_slice_mut(model_output)?;
    let conv_state = as_f32_slice_mut(conv_output)?;
    let recurrent_state = as_f32_slice_mut(ssm_output)?;
    let dtype = |index: usize| {
        inputs[index]
            .dtype()
            .ok_or_else(|| FellmError::other("gated_delta_net: invalid weight dtype"))
    };
    let dimensions = GatedDeltaNetDimensions {
        model: attrs.n_embd as usize,
        inner: attrs.gdn_inner_size as usize,
        key_heads: attrs.n_kv_heads as usize,
        value_heads: attrs.n_heads as usize,
        state_size: attrs.gdn_state_size as usize,
        conv_kernel: attrs.gdn_conv_kernel as usize,
        norm_epsilon: attrs.eps,
    };
    let weights = GatedDeltaNetWeights {
        qkv: (as_bytes_slice(&inputs[1]), dtype(1)?),
        z: (as_bytes_slice(&inputs[2]), dtype(2)?),
        beta: (as_bytes_slice(&inputs[3]), dtype(3)?),
        alpha: (as_bytes_slice(&inputs[4]), dtype(4)?),
        dt_bias: as_f32_slice(&inputs[5])?,
        decay: as_f32_slice(&inputs[6])?,
        conv: as_f32_slice(&inputs[7])?,
        norm: as_f32_slice(&inputs[8])?,
        output: (as_bytes_slice(&inputs[9]), dtype(9)?),
    };
    let conv_elements = dimensions.conv_kernel.saturating_sub(1)
        * (dimensions.inner + 2 * dimensions.key_heads * dimensions.state_size);
    let ssm_elements = dimensions.value_heads * dimensions.state_size * dimensions.state_size;
    if dimensions.model == 0
        || !x.len().is_multiple_of(dimensions.model)
        || output.len() != x.len()
        || conv_state.len() != x.len() / dimensions.model * conv_elements
        || recurrent_state.len() != x.len() / dimensions.model * ssm_elements
    {
        return Err(FellmError::other("gated_delta_net: batched shape mismatch"));
    }
    for (((x_row, output_row), conv_row), ssm_row) in x
        .chunks_exact(dimensions.model)
        .zip(output.chunks_exact_mut(dimensions.model))
        .zip(conv_state.chunks_exact_mut(conv_elements))
        .zip(recurrent_state.chunks_exact_mut(ssm_elements))
    {
        gated_delta_net_decode(x_row, &weights, conv_row, ssm_row, output_row, dimensions)?;
    }
    Ok(())
}

fn read_sparse_expert(
    storage: &mut CpuWeightStorage,
    bank: &SparseWeight,
    expert: usize,
    bytes_per: usize,
) -> Result<Vec<u8>> {
    let offset = bank.offset.saturating_add((expert * bytes_per) as u64);
    if offset.saturating_add(bytes_per as u64) > bank.offset.saturating_add(bank.len) {
        return Err(FellmError::other("moe: expert slice exceeds bank"));
    }
    let provider = storage
        .providers
        .get(&bank.path)
        .cloned()
        .ok_or_else(|| {
            FellmError::other(format!(
                "no storage provider for expert bank {}",
                bank.path.display()
            ))
        })?;
    let mut buf = vec![0u8; bytes_per];
    let started = std::time::Instant::now();
    provider
        .read_at(offset, &mut buf)
        .map_err(|error| FellmError::other(format!("expert slice read: {error}")))?;
    let nanos = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    storage.physical_reads = storage.physical_reads.saturating_add(1);
    storage.physical_bytes = storage.physical_bytes.saturating_add(bytes_per as u64);
    storage.expert_slice_reads = storage.expert_slice_reads.saturating_add(1);
    storage.expert_slice_bytes = storage.expert_slice_bytes.saturating_add(bytes_per as u64);
    storage.read_nanos = storage.read_nanos.saturating_add(nanos);
    Ok(buf)
}

fn launch_moe(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
    mut storage: Option<&mut CpuWeightStorage>,
) -> Result<()> {
    // inputs: [x, gate_inp, gate_exps, up_exps, down_exps, optional bias]
    // outputs: [y]
    if inputs.len() < 5 || outputs.is_empty() {
        return Err(FellmError::other("moe: bad arity"));
    }
    let n_expert_used_hint = attrs.n_expert_used.max(1) as usize;
    let (n_core, forced) = moe_forced_experts(inputs, n_expert_used_hint)?;
    let inputs = &inputs[..n_core];
    let forced_experts = forced.as_deref();
    let (y_out, _) = outputs.split_first_mut().unwrap();
    let x = as_f32_slice(&inputs[0])?;
    let gate_dtype = inputs[1]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: gate dtype"))?;
    let mut gate_owned;
    let gate_inp: &[f32] = if gate_dtype == DType::F32 {
        as_f32_slice(&inputs[1])?
    } else {
        let n = inputs[1].dims_slice().iter().product::<u64>() as usize;
        gate_owned = vec![0.0f32; n.max(1)];
        crate::dequant::dequantize_row(gate_dtype, as_bytes_slice(&inputs[1]), &mut gate_owned, n)?;
        &gate_owned
    };
    let y = as_f32_slice_mut(y_out)?;

    if inputs.len() >= 7 && inputs[4].dims_slice().len() == 2 {
        let packed_dims = inputs[2].dims_slice();
        let down_dims = inputs[3].dims_slice();
        let shared_dims = inputs[4].dims_slice();
        if packed_dims.len() != 3 || down_dims.len() != 3 || shared_dims.len() != 2 {
            return Err(FellmError::other(
                "moe gemma: invalid packed/shared dimensions",
            ));
        }
        let n_experts = attrs.n_experts.max(1) as usize;
        let n_expert_used = attrs.n_expert_used.max(1) as usize;
        let n_embd = attrs.n_embd.max(1) as usize;
        let n_ff = packed_dims[1] as usize / 2;
        let shared_ff = shared_dims[0] as usize;
        let bias = if inputs.len() > 7 {
            Some(as_f32_slice(&inputs[7])?)
        } else {
            None
        };
        let x_dims = inputs[0].dims_slice();
        let tokens = if x_dims.len() <= 1 {
            1
        } else {
            x_dims[..x_dims.len() - 1].iter().product::<u64>() as usize
        };
        if tokens > 1 {
            return crate::kernels::moe::moe_decode_gemma_batch(
                x,
                gate_inp,
                as_bytes_slice(&inputs[2]),
                inputs[2]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: packed dtype"))?,
                as_bytes_slice(&inputs[3]),
                inputs[3]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: down dtype"))?,
                as_bytes_slice(&inputs[4]),
                inputs[4]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: shared gate dtype"))?,
                as_bytes_slice(&inputs[5]),
                inputs[5]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: shared up dtype"))?,
                as_bytes_slice(&inputs[6]),
                inputs[6]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe gemma: shared down dtype"))?,
                bias,
                y,
                tokens,
                n_experts,
                n_expert_used,
                n_ff,
                shared_ff,
                n_embd,
                attrs.expert_gating_func,
                attrs.routed_scaling_factor,
                attrs.norm_topk_prob != 0,
            );
        }
        return crate::kernels::moe::moe_decode_gemma(
            x,
            gate_inp,
            as_bytes_slice(&inputs[2]),
            inputs[2]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: packed dtype"))?,
            as_bytes_slice(&inputs[3]),
            inputs[3]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: down dtype"))?,
            as_bytes_slice(&inputs[4]),
            inputs[4]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: shared gate dtype"))?,
            as_bytes_slice(&inputs[5]),
            inputs[5]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: shared up dtype"))?,
            as_bytes_slice(&inputs[6]),
            inputs[6]
                .dtype()
                .ok_or_else(|| FellmError::other("moe gemma: shared down dtype"))?,
            bias,
            y,
            n_experts,
            n_expert_used,
            n_ff,
            shared_ff,
            n_embd,
            attrs.expert_gating_func,
            attrs.routed_scaling_factor,
            attrs.norm_topk_prob != 0,
        );
    }

    let gate_dims = inputs[1].dims_slice();
    let expert_dims = inputs[2].dims_slice();
    let down_dims = inputs[4].dims_slice();
    if gate_dims.len() != 2 || expert_dims.len() != 3 || down_dims.len() != 3 {
        return Err(FellmError::other(
            "moe: expected 2D router and 3D expert weights",
        ));
    }

    let n_experts = if attrs.n_experts > 0 {
        attrs.n_experts as usize
    } else {
        gate_dims[0] as usize
    };
    let n_embd = if attrs.n_embd > 0 {
        attrs.n_embd as usize
    } else {
        gate_dims[1] as usize
    };
    let n_ff = expert_dims[1] as usize;
    let n_expert_used = if attrs.n_expert_used > 0 {
        attrs.n_expert_used as usize
    } else {
        1
    };

    if expert_dims[0] as usize != n_experts
        || expert_dims[2] as usize != n_embd
        || down_dims[0] as usize != n_experts
        || down_dims[1] as usize != n_embd
        || down_dims[2] as usize != n_ff
    {
        return Err(FellmError::other("moe: expert dimensions mismatch"));
    }

    let gate_exps_dtype = inputs[2]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: gate expert dtype"))?;
    let up_exps_dtype = inputs[3]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: up expert dtype"))?;
    let down_exps_dtype = inputs[4]
        .dtype()
        .ok_or_else(|| FellmError::other("moe: down expert dtype"))?;
    let shexp = inputs.len() >= 8 && inputs[5].dims_slice().len() == 2;
    let bias = if shexp && inputs.len() > 8 {
        Some(as_f32_slice(&inputs[8])?)
    } else if !shexp && inputs.len() > 5 {
        Some(as_f32_slice(&inputs[5])?)
    } else {
        None
    };

    if !x.len().is_multiple_of(n_embd) || y.len() != x.len() {
        return Err(FellmError::other("moe: batched shape mismatch"));
    }
    if let Some(storage) = storage.as_mut()
        && storage.sparse.contains_key(&inputs[2].logical_id)
    {
        let gate_bank = storage
            .sparse
            .get(&inputs[2].logical_id)
            .cloned()
            .ok_or_else(|| FellmError::other("moe: missing gate expert bank"))?;
        let up_bank = storage
            .sparse
            .get(&inputs[3].logical_id)
            .cloned()
            .ok_or_else(|| FellmError::other("moe: missing up expert bank"))?;
        let down_bank = storage
            .sparse
            .get(&inputs[4].logical_id)
            .cloned()
            .ok_or_else(|| FellmError::other("moe: missing down expert bank"))?;
        let gate_bpe = gate_exps_dtype.byte_size(n_ff * n_embd);
        let up_bpe = up_exps_dtype.byte_size(n_ff * n_embd);
        let down_bpe = down_exps_dtype.byte_size(n_embd * n_ff);
        for (x_row, y_row) in x.chunks_exact(n_embd).zip(y.chunks_exact_mut(n_embd)) {
            let selected = crate::kernels::moe::moe_route(
                x_row,
                gate_inp,
                bias,
                n_experts,
                n_expert_used,
                n_embd,
                attrs.expert_gating_func,
                attrs.routed_scaling_factor,
                attrs.norm_topk_prob != 0,
                forced_experts,
            )?;
            let mut gate_bufs = Vec::with_capacity(selected.len());
            let mut up_bufs = Vec::with_capacity(selected.len());
            let mut down_bufs = Vec::with_capacity(selected.len());
            let mut jobs = Vec::new();
            for &(expert, _) in &selected {
                jobs.push((&gate_bank, expert, gate_bpe));
                jobs.push((&up_bank, expert, up_bpe));
                jobs.push((&down_bank, expert, down_bpe));
            }
            jobs.sort_by_key(|job| job.0.offset.saturating_add((job.1 * job.2) as u64));
            let mut loaded = std::collections::HashMap::<(std::path::PathBuf, u64, usize), Vec<u8>>::new();
            for (bank, expert, bytes_per) in jobs {
                let buf = read_sparse_expert(storage, bank, expert, bytes_per)?;
                loaded.insert((bank.path.clone(), bank.offset, expert), buf);
            }
            for &(expert, _) in &selected {
                gate_bufs.push(
                    loaded
                        .remove(&(gate_bank.path.clone(), gate_bank.offset, expert))
                        .ok_or_else(|| FellmError::other("moe: gate slice"))?,
                );
                up_bufs.push(
                    loaded
                        .remove(&(up_bank.path.clone(), up_bank.offset, expert))
                        .ok_or_else(|| FellmError::other("moe: up slice"))?,
                );
                down_bufs.push(
                    loaded
                        .remove(&(down_bank.path.clone(), down_bank.offset, expert))
                        .ok_or_else(|| FellmError::other("moe: down slice"))?,
                );
            }
            let gate_refs: Vec<&[u8]> = gate_bufs.iter().map(Vec::as_slice).collect();
            let up_refs: Vec<&[u8]> = up_bufs.iter().map(Vec::as_slice).collect();
            let down_refs: Vec<&[u8]> = down_bufs.iter().map(Vec::as_slice).collect();
            crate::kernels::moe::moe_apply_selected(
                x_row,
                &selected,
                &gate_refs,
                gate_exps_dtype,
                &up_refs,
                up_exps_dtype,
                &down_refs,
                down_exps_dtype,
                y_row,
                n_ff,
                n_embd,
            )?;
            if shexp {
                let shexp_ff = inputs[5].dims_slice()[0] as usize;
                crate::kernels::moe::add_shared_expert(
                    x_row,
                    as_bytes_slice(&inputs[5]),
                    inputs[5]
                        .dtype()
                        .ok_or_else(|| FellmError::other("moe: shexp gate dtype"))?,
                    as_bytes_slice(&inputs[6]),
                    inputs[6]
                        .dtype()
                        .ok_or_else(|| FellmError::other("moe: shexp up dtype"))?,
                    as_bytes_slice(&inputs[7]),
                    inputs[7]
                        .dtype()
                        .ok_or_else(|| FellmError::other("moe: shexp down dtype"))?,
                    y_row,
                    shexp_ff,
                    n_embd,
                )?;
            }
        }
        return Ok(());
    }
    for (x_row, y_row) in x.chunks_exact(n_embd).zip(y.chunks_exact_mut(n_embd)) {
        crate::kernels::moe::moe_decode(
            x_row,
            gate_inp,
            as_bytes_slice(&inputs[2]),
            gate_exps_dtype,
            as_bytes_slice(&inputs[3]),
            up_exps_dtype,
            as_bytes_slice(&inputs[4]),
            down_exps_dtype,
            bias,
            y_row,
            n_experts,
            n_expert_used,
            n_ff,
            n_embd,
            attrs.expert_gating_func,
            attrs.routed_scaling_factor,
            attrs.norm_topk_prob != 0,
            forced_experts,
        )?;
        if shexp {
            let shexp_ff = inputs[5].dims_slice()[0] as usize;
            crate::kernels::moe::add_shared_expert(
                x_row,
                as_bytes_slice(&inputs[5]),
                inputs[5]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe: shexp gate dtype"))?,
                as_bytes_slice(&inputs[6]),
                inputs[6]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe: shexp up dtype"))?,
                as_bytes_slice(&inputs[7]),
                inputs[7]
                    .dtype()
                    .ok_or_else(|| FellmError::other("moe: shexp down dtype"))?,
                y_row,
                shexp_ff,
                n_embd,
            )?;
        }
    }
    Ok(())
}

fn launch_hyper_connection(attrs: &OpAttrs, inputs: &[TensorRef], outputs: &mut [TensorMut]) -> Result<()> {
    if inputs.len() < 4 || outputs.is_empty() {
        return Err(FellmError::other("hyper_connection: arity"));
    }
    let y = as_f32_slice_mut(&mut outputs[0])?;
    let x = as_f32_slice(&inputs[0])?;
    let residual = if attrs.kv_slot == 1 && inputs.len() > 4 {
        Some(as_f32_slice(&inputs[4])?)
    } else {
        None
    };
    crate::kernels::hyper_connection::hyper_connection(
        x,
        residual,
        as_bytes_slice(&inputs[1]),
        inputs[1]
            .dtype()
            .ok_or_else(|| FellmError::other("deepseek4 hc fn dtype"))?,
        as_f32_slice(&inputs[2])?,
        as_f32_slice(&inputs[3])?,
        y,
        attrs.n_embd.max(1) as usize,
        attrs.gdn_state_size.max(1) as usize,
        attrs.kv_slot,
        attrs.gdn_conv_kernel.max(1),
        attrs.eps,
        attrs.eps.max(1e-6),
    )
}

fn launch_mla_attention(
    attrs: &OpAttrs,
    inputs: &[TensorRef],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if inputs.len() < 11 || outputs.is_empty() {
        return Err(FellmError::other("mla_attention: arity"));
    }
    let y = as_f32_slice_mut(&mut outputs[0])?;
    let k_bytes = unsafe {
        core::slice::from_raw_parts_mut(inputs[10].data as *mut u8, inputs[10].byte_len as usize)
    };
    let k_out: &mut [f32] = bytemuck::try_cast_slice_mut(k_bytes)
        .map_err(|e| FellmError::other(format!("mla k cast: {e:?}")))?;
    let mut empty_c = [];
    let c_out: &mut [f32] = if inputs.len() > 11 {
        let c_bytes = unsafe {
            core::slice::from_raw_parts_mut(inputs[11].data as *mut u8, inputs[11].byte_len as usize)
        };
        bytemuck::try_cast_slice_mut(c_bytes)
            .map_err(|e| FellmError::other(format!("mla c cast: {e:?}")))?
    } else {
        &mut empty_c
    };
    let pair = |idx: usize| inputs.get(idx).and_then(|t| t.dtype().map(|d| (as_bytes_slice(t), d)));
    let extras = crate::kernels::mla::MlaExtras {
        compress_kv: pair(12),
        compress_gate: pair(13),
        compress_ape: inputs.get(14).and_then(|t| as_f32_slice(t).ok()),
        compress_norm: inputs.get(15).and_then(|t| as_f32_slice(t).ok()),
        compress_state_dim: inputs
            .get(12)
            .map(|t| t.dims_slice().first().copied().unwrap_or(0) as usize)
            .unwrap_or(0),
        indexer_q_b: pair(16),
        indexer_proj: pair(17),
        indexer_comp_kv: pair(18),
        indexer_comp_gate: pair(19),
        indexer_comp_ape: inputs.get(20).and_then(|t| as_f32_slice(t).ok()),
        indexer_comp_norm: inputs.get(21).and_then(|t| as_f32_slice(t).ok()),
        indexer_state_dim: inputs
            .get(18)
            .map(|t| t.dims_slice().first().copied().unwrap_or(0) as usize)
            .unwrap_or(0),
        indexer_heads: attrs.n_kv_heads as usize,
        indexer_head_dim: attrs.query_len as usize,
        indexer_top_k: attrs.kv_len as usize,
    };
    let x = as_f32_slice(&inputs[0])?;
    let d_model = attrs.n_embd.max(1) as usize;
    let rows = (x.len() / d_model.max(1)).max(1);
    let paged = paged_ctx::snapshot_paged_context();
    for row in 0..rows {
        let (position, past_len) = if rows == 1 {
            (attrs.position, attrs.past_len)
        } else {
            let position = paged
                .as_ref()
                .and_then(|ctx| ctx.row_rope_positions.get(row).copied())
                .unwrap_or_else(|| attrs.position.saturating_add(row as u32));
            (position, position)
        };
        crate::kernels::mla::mla_decode(
            &x[row * d_model..(row + 1) * d_model],
            as_bytes_slice(&inputs[1]),
            inputs[1]
                .dtype()
                .ok_or_else(|| FellmError::other("q_a dtype"))?,
            as_f32_slice(&inputs[2])?,
            as_bytes_slice(&inputs[3]),
            inputs[3]
                .dtype()
                .ok_or_else(|| FellmError::other("q_b dtype"))?,
            as_bytes_slice(&inputs[4]),
            inputs[4]
                .dtype()
                .ok_or_else(|| FellmError::other("kv dtype"))?,
            as_f32_slice(&inputs[5])?,
            as_bytes_slice(&inputs[6]),
            inputs[6]
                .dtype()
                .ok_or_else(|| FellmError::other("wo_a dtype"))?,
            as_bytes_slice(&inputs[7]),
            inputs[7]
                .dtype()
                .ok_or_else(|| FellmError::other("wo_b dtype"))?,
            as_f32_slice(&inputs[8])?,
            as_f32_slice(&inputs[9])?,
            k_out,
            c_out,
            &mut y[row * d_model..(row + 1) * d_model],
            d_model,
            attrs.n_heads.max(1) as usize,
            attrs.head_dim.max(1) as usize,
            attrs.rope_dim.max(1) as usize,
            attrs.gdn_inner_size.max(1) as usize,
            attrs.gdn_state_size.max(1) as usize,
            attrs.shortconv_l_cache.max(1) as usize,
            position,
            past_len,
            attrs.attention_window as usize,
            attrs.eps,
            attrs.block_size as usize,
            extras,
            attrs.layer_ord,
        )?;
    }
    Ok(())
}
