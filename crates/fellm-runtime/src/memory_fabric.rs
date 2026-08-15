//! Runtime ownership for the shared Memory Fabric plan and telemetry.

use fellm_gguf::GgufFile;
use fellm_graph::graph::{Graph, OpValue};
use fellm_graph::plan::ExecutionPlan;
use fellm_memory::{
    ExecutionGroup, FabricMetrics, FabricPlan, FabricPlanner, HardwareProfile, ModelProfile,
    TransferCapabilities, WeightDescriptor, WeightId,
};
use fellm_plugin_abi::DeviceMemoryInfo;
use fellm_plugin_abi::op::OpKind;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

const DEFAULT_H2D_BPS: u64 = 24 * 1024 * 1024 * 1024;
const DEFAULT_SSD_BPS: u64 = 3 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Default)]
pub struct MemoryFabricConfig {
    pub device_memory_limit: Option<u64>,
    pub host_memory_limit: Option<u64>,
    pub h2d_bytes_per_second: Option<u64>,
    pub storage_bytes_per_second: Option<u64>,
    pub storage_latency_micros: Option<u64>,
    pub storage_provider: fellm_memory::StorageProviderRequest,
    pub host_weight_cache: u64,
    pub storage_overlap: bool,
    pub router_trace_capacity: usize,
    pub disable_cpu_partitions: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryFabricSnapshot {
    pub plan: FabricPlan,
    pub metrics: FabricMetrics,
}

/// One model's joint plan. Allocation failures update pressure and produce a smaller plan.
pub struct MemoryFabric {
    config: MemoryFabricConfig,
    hardware: HardwareProfile,
    model: ModelProfile,
    weights: Arc<[WeightDescriptor]>,
    storage_objects: Arc<fellm_memory::StorageObjectIndex>,
    schedule_ops: Arc<[Option<OpKind>]>,
    expert_tracker: Arc<Mutex<fellm_memory::ExpertAccessTracker>>,
    expert_placements: Arc<RwLock<Vec<fellm_memory::ExpertPlacement>>>,
    expert_route_step: AtomicU64,
    expert_routes: Arc<Mutex<std::collections::VecDeque<fellm_memory::ExpertRouteTrace>>>,
    state: Arc<RwLock<MemoryFabricSnapshot>>,
}

impl MemoryFabric {
    #[must_use]
    pub const fn storage_overlap_enabled(&self) -> bool {
        self.config.storage_overlap
    }
    pub fn inspect_and_plan(
        gguf: &GgufFile,
        graph: &Graph,
        execution: &ExecutionPlan,
        device: Option<DeviceMemoryInfo>,
        kv_bytes: u64,
        activation_bytes: u64,
        config: &MemoryFabricConfig,
    ) -> Result<Self, fellm_memory::PlanningError> {
        let mut system = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        system.refresh_memory();
        let device_limit = config.device_memory_limit;
        let host_limit = config.host_memory_limit;
        let hardware = HardwareProfile {
            device_total: device.map_or(0, |info| info.total_bytes),
            device_available: device.map_or(0, |info| {
                device_limit.map_or(info.available_bytes, |limit| {
                    info.available_bytes.min(limit)
                })
            }),
            host_total: host_limit.map_or(system.total_memory(), |limit| {
                system.total_memory().min(limit)
            }),
            host_available: host_limit.map_or(system.available_memory(), |limit| {
                system.available_memory().min(limit)
            }),
            h2d_bytes_per_second: config.h2d_bytes_per_second.unwrap_or(DEFAULT_H2D_BPS),
            storage_bytes_per_second: config.storage_bytes_per_second.unwrap_or(DEFAULT_SSD_BPS),
            storage_latency: Duration::from_micros(config.storage_latency_micros.unwrap_or(100)),
            cpu_score: std::thread::available_parallelism()
                .map_or(1.0, |threads| threads.get() as f64),
            transfers: TransferCapabilities {
                mmap: true,
                async_file: true,
                direct_io: cfg!(any(target_os = "linux", windows)),
                // Capability means an operational provider, not merely OS/library potential.
                io_uring: false,
                gds: false,
            },
        };
        let (model, weights) = profile_model(
            gguf,
            graph,
            execution,
            kv_bytes,
            activation_bytes,
            hardware.cpu_score,
            config.disable_cpu_partitions,
        );
        let mut plan = FabricPlanner::plan(&hardware, &model)?;
        let cpu_storage_native = hardware.device_total == 0
            && matches!(
                config.storage_provider,
                fellm_memory::StorageProviderRequest::Buffered
                    | fellm_memory::StorageProviderRequest::Direct
                    | fellm_memory::StorageProviderRequest::Mmap
                    | fellm_memory::StorageProviderRequest::IoUring
            );
        if cpu_storage_native {
            let dense_group_bytes = model.groups.iter().map(|group| {
                group
                    .weights
                    .iter()
                    .filter_map(|id| weights.iter().find(|weight| weight.id == *id))
                    .filter(|weight| !fellm_memory::is_moe_expert_bank(&weight.name))
                    .map(|weight| weight.byte_len)
                    .sum::<u64>()
            });
            let slot_bytes = dense_group_bytes
                .max()
                .unwrap_or(4 << 20)
                .saturating_add(128 << 10)
                .next_multiple_of(4096);
            // Two physical slots are required for multi-weight atomic bundles even when
            // predictive overlap is disabled. Queue depth still describes I/O concurrency.
            let slots = 2;
            plan.budget.weights_host = config.host_weight_cache.min(model.weight_bytes);
            plan.budget.host_staging = slot_bytes.saturating_mul(slots);
            plan.host_buffer_count = slots as u8;
            plan.host_buffer_bytes = slot_bytes;
            plan.storage_queue_depth = if config.storage_overlap { 2 } else { 1 };
            for placement in &mut plan.placements {
                placement.class = fellm_memory::ResidencyClass::StorageStream;
            }
        }
        let dense_weights = weights
            .iter()
            .filter(|weight| !fellm_memory::is_moe_expert_bank(&weight.name))
            .cloned()
            .collect::<Vec<_>>();
        let storage_objects = fellm_memory::StorageObjectIndex::from_execution_groups(
            &dense_weights,
            &model.groups,
            64 * 1024,
            plan.host_buffer_bytes.max(1),
        )
        .map_err(|_| fellm_memory::PlanningError::NoHostStaging)?;
        if cpu_storage_native {
            let bundle = storage_objects
                .max_objects_per_group(&model.groups)
                .max(2);
            let slots = bundle.saturating_add(1).max(2).min(255);
            plan.host_buffer_count = slots as u8;
            plan.host_buffer_bytes = plan.host_buffer_bytes.max(1);
            plan.budget.host_staging = plan.host_buffer_bytes.saturating_mul(slots as u64);
            plan.storage_queue_depth = if config.storage_overlap {
                slots.min(8) as u16
            } else {
                1
            };
        }
        let metrics = FabricMetrics {
            backing_storage_bytes: model.weight_bytes,
            resident_device_bytes: plan.budget.weights_device,
            resident_host_bytes: plan.budget.weights_host,
            ..FabricMetrics::default()
        };
        let mut expert_tracker = fellm_memory::ExpertAccessTracker::new(0.995);
        for (position, &node_id) in execution.order.iter().enumerate() {
            let node = graph.node(node_id);
            if node.op != Some(OpKind::MoE) || node.attrs.n_experts == 0 {
                continue;
            }
            let expert_count = u64::from(node.attrs.n_experts);
            let bank_bytes = graph
                .inputs_slice(node_id)
                .iter()
                .filter_map(|&input| {
                    let input = graph.node(input);
                    let OpValue::Constant(tensor) = &input.value else {
                        return None;
                    };
                    (input.label.contains("exps")
                        && tensor.shape().dims().first().copied() == Some(expert_count))
                    .then(|| tensor.layout().byte_size() as u64)
                })
                .sum::<u64>();
            let bytes_per_expert = bank_bytes.div_ceil(expert_count).max(1);
            for expert in 0..node.attrs.n_experts {
                expert_tracker.register(
                    fellm_memory::ExpertId {
                        operation: position as u64,
                        expert,
                    },
                    bytes_per_expert,
                );
            }
        }
        Ok(Self {
            config: config.clone(),
            hardware,
            model,
            weights: weights.into(),
            storage_objects: Arc::new(storage_objects),
            schedule_ops: execution
                .order
                .iter()
                .map(|&node| graph.node(node).op)
                .collect(),
            expert_tracker: Arc::new(Mutex::new(expert_tracker)),
            expert_placements: Arc::new(RwLock::new(Vec::new())),
            expert_route_step: AtomicU64::new(0),
            expert_routes: Arc::new(Mutex::new(std::collections::VecDeque::with_capacity(
                config.router_trace_capacity.min(65_536),
            ))),
            state: Arc::new(RwLock::new(MemoryFabricSnapshot { plan, metrics })),
        })
    }

    pub fn snapshot(&self) -> MemoryFabricSnapshot {
        self.state.read().expect("memory fabric lock").clone()
    }

    /// Feed router selections back into the irregular-workload policy and refresh independent
    /// hot-VRAM / warm-RAM / cold-storage expert placements.
    pub fn observe_expert_routes(&self, operation: u64, experts: &[u32]) {
        self.observe_expert_route_batch(std::slice::from_ref(&(operation, experts.to_vec())));
    }

    /// Apply all router decisions from one forward step before refreshing placement once.
    pub fn observe_expert_route_batch(&self, routes: &[(u64, Vec<u32>)]) {
        if routes.is_empty() {
            return;
        }
        let request_step = self.expert_route_step.fetch_add(1, Ordering::Relaxed);
        if self.config.router_trace_capacity > 0 {
            let mut trace = self.expert_routes.lock().expect("expert route trace lock");
            for (operation, experts) in routes {
                while trace.len() >= self.config.router_trace_capacity {
                    trace.pop_front();
                }
                trace.push_back(fellm_memory::ExpertRouteTrace {
                    request_step,
                    operation: *operation,
                    experts: experts.clone(),
                });
            }
        }
        let mut tracker = self.expert_tracker.lock().expect("expert tracker lock");
        for (operation, experts) in routes {
            tracker.observe(
                experts
                    .iter()
                    .copied()
                    .map(|expert| fellm_memory::ExpertId {
                        operation: *operation,
                        expert,
                    }),
            );
        }
        let snapshot = tracker.snapshot();
        drop(tracker);
        let plan = self.snapshot().plan;
        let placements = fellm_memory::plan_expert_residency(
            &snapshot,
            plan.budget.weights_device,
            plan.budget.weights_host,
        );
        for temperature in [
            fellm_memory::ExpertTemperature::Hot,
            fellm_memory::ExpertTemperature::Warm,
            fellm_memory::ExpertTemperature::Cold,
        ] {
            let count = placements
                .iter()
                .filter(|placement| placement.temperature == temperature)
                .count();
            let tier = match temperature {
                fellm_memory::ExpertTemperature::Hot => "hot",
                fellm_memory::ExpertTemperature::Warm => "warm",
                fellm_memory::ExpertTemperature::Cold => "cold",
            };
            metrics::gauge!("fellm_memory_experts", "temperature" => tier).set(count as f64);
        }
        let mut current = self
            .expert_placements
            .write()
            .expect("expert placement lock");
        let changed = current.len() != placements.len()
            || current.iter().zip(&placements).any(|(old, new)| {
                old.id != new.id
                    || old.temperature != new.temperature
                    || old.residency != new.residency
            });
        if changed {
            tracing::debug!(
                hot = placements
                    .iter()
                    .filter(|placement| {
                        placement.temperature == fellm_memory::ExpertTemperature::Hot
                    })
                    .count(),
                warm = placements
                    .iter()
                    .filter(|placement| {
                        placement.temperature == fellm_memory::ExpertTemperature::Warm
                    })
                    .count(),
                cold = placements
                    .iter()
                    .filter(|placement| {
                        placement.temperature == fellm_memory::ExpertTemperature::Cold
                    })
                    .count(),
                "updated router-driven expert residency"
            );
        }
        *current = placements;
    }

    #[must_use]
    pub fn expert_placements(&self) -> Vec<fellm_memory::ExpertPlacement> {
        self.expert_placements
            .read()
            .expect("expert placement lock")
            .clone()
    }

    #[must_use]
    pub fn expert_route_trace(&self) -> Vec<fellm_memory::ExpertRouteTrace> {
        self.expert_routes
            .lock()
            .expect("expert route trace lock")
            .iter()
            .cloned()
            .collect()
    }

    #[must_use]
    pub fn simulate_expert_cache(
        &self,
        capacity_bytes: u64,
        policy: fellm_memory::ExpertCachePolicy,
    ) -> fellm_memory::CacheSimulation {
        let sizes = self
            .expert_tracker
            .lock()
            .expect("expert tracker lock")
            .snapshot()
            .into_iter()
            .map(|access| (access.id, access.byte_len))
            .collect::<std::collections::HashMap<_, _>>();
        let mut events = Vec::new();
        for route in self.expert_route_trace() {
            for expert in route.experts {
                let id = fellm_memory::ExpertId {
                    operation: route.operation,
                    expert,
                };
                events.push(fellm_memory::ExpertTraceEvent {
                    expert: id,
                    bytes: sizes.get(&id).copied().unwrap_or(1),
                    load_cost_nanos: 1,
                });
            }
        }
        fellm_memory::simulate_expert_cache(&events, capacity_bytes, policy)
    }

    /// Immutable logical weight catalog. Replica locations may change independently.
    #[must_use]
    pub fn weights(&self) -> &[WeightDescriptor] {
        &self.weights
    }

    /// Graph-ordered physical storage objects and their O(1) weight lookup.
    #[must_use]
    pub fn storage_objects(&self) -> &fellm_memory::StorageObjectIndex {
        &self.storage_objects
    }

    pub fn select_storage_provider(
        &self,
        device_consumer: bool,
    ) -> fellm_core::error::Result<fellm_memory::StorageProviderKind> {
        let plan = self.snapshot().plan;
        let direct_io_aligned = self.storage_objects.objects().iter().all(|object| {
            object.extent.offset.is_multiple_of(4096) && object.extent.alignment >= 4096
        });
        let committed = self
            .hardware
            .host_total
            .saturating_sub(self.hardware.host_available)
            .saturating_add(plan.budget.weights_host)
            .saturating_add(plan.budget.host_staging)
            .saturating_add(plan.budget.kv_host)
            .saturating_add(plan.budget.host_reserve);
        let host_pressure = committed as f64 / self.hardware.host_total.max(1) as f64;
        let reuse_count = self
            .model
            .groups
            .iter()
            .map(|group| group.reuse_count)
            .max()
            .unwrap_or(1);
        fellm_memory::select_storage_provider(
            self.config.storage_provider,
            self.hardware.transfers,
            fellm_memory::StorageWorkload {
                reuse_count,
                host_pressure,
                direct_io_aligned,
                device_consumer,
            },
        )
    }

    /// Resolve an execution tensor to its stable GGUF weight identity by storage extent.
    #[must_use]
    pub fn weight_id_for_tensor(&self, tensor: &fellm_core::tensor::Tensor) -> Option<WeightId> {
        if let Some((path, offset, len)) = tensor.file_extent() {
            return self.weights.iter().find(|weight| {
                weight.home.path == *path
                    && weight.home.offset == offset
                    && weight.byte_len == len as u64
            }).map(|weight| weight.id);
        }
        let (offset, len) = tensor.mmap_extent()?;
        self.weights
            .iter()
            .find(|weight| weight.home.offset == offset as u64 && weight.byte_len == len as u64)
            .map(|weight| weight.id)
    }

    /// Logical weights selected for the permanent device replica set.
    #[must_use]
    pub fn permanent_device_weights(&self) -> std::collections::HashSet<WeightId> {
        let plan = self.snapshot().plan;
        let groups = plan
            .placements
            .iter()
            .filter(|placement| placement.class == fellm_memory::ResidencyClass::PermanentDevice)
            .map(|placement| placement.group)
            .collect::<std::collections::HashSet<_>>();
        self.model
            .groups
            .iter()
            .filter(|group| groups.contains(&group.id))
            .flat_map(|group| group.weights.iter().copied())
            .collect()
    }

    /// Storage-native weights plus the bounded host staging geometry selected for them.
    #[must_use]
    pub fn storage_stream_configuration(&self) -> (Vec<WeightDescriptor>, usize, usize) {
        let plan = self.snapshot().plan;
        let groups = plan
            .placements
            .iter()
            .filter(|placement| placement.class == fellm_memory::ResidencyClass::StorageStream)
            .map(|placement| placement.group)
            .collect::<std::collections::HashSet<_>>();
        let ids = self
            .model
            .groups
            .iter()
            .filter(|group| groups.contains(&group.id))
            .flat_map(|group| group.weights.iter().copied())
            .collect::<std::collections::HashSet<_>>();
        (
            self.weights
                .iter()
                .filter(|weight| ids.contains(&weight.id))
                .cloned()
                .collect(),
            usize::from(plan.host_buffer_count),
            usize::try_from(plan.host_buffer_bytes).unwrap_or(usize::MAX),
        )
    }

    /// Logical weights whose scheduled partition is cheaper on CPU than transfer to GPU.
    #[must_use]
    pub fn cpu_compute_weights(&self) -> std::collections::HashSet<WeightId> {
        let plan = self.snapshot().plan;
        let mut groups = plan
            .placements
            .iter()
            .filter(|placement| placement.class == fellm_memory::ResidencyClass::CpuCompute)
            .map(|placement| placement.group)
            .collect::<Vec<_>>();
        groups.sort_unstable();
        if let Ok(limit) = std::env::var("FELLM_CPU_PARTITION_GROUP_LIMIT")
            && let Ok(limit) = limit.parse::<usize>()
        {
            groups.truncate(limit);
        }
        let groups = groups.into_iter().collect::<std::collections::HashSet<_>>();
        self.model
            .groups
            .iter()
            .filter(|group| groups.contains(&group.id))
            .flat_map(|group| group.weights.iter().copied())
            .collect()
    }

    /// Compiled schedule positions covered by planner-selected CPU partitions.
    #[must_use]
    pub fn cpu_compute_ops(&self) -> std::collections::HashSet<u64> {
        let plan = self.snapshot().plan;
        let groups = plan
            .placements
            .iter()
            .filter(|placement| placement.class == fellm_memory::ResidencyClass::CpuCompute)
            .map(|placement| placement.group)
            .collect::<std::collections::HashSet<_>>();
        let mut ops = self
            .model
            .groups
            .iter()
            .filter(|group| groups.contains(&group.id))
            .flat_map(|group| u64::from(group.first_op)..=u64::from(group.last_op))
            .filter(|&position| {
                self.schedule_ops[position as usize] == Some(OpKind::GateUpSwiGlu)
                    || (self.schedule_ops[position as usize] == Some(OpKind::MatMul)
                        && position > 0
                        && self.schedule_ops[position as usize - 1] == Some(OpKind::GateUpSwiGlu))
            })
            .collect::<Vec<_>>();
        ops.sort_unstable();
        ops.dedup();
        if let Ok(limit) = std::env::var("FELLM_CPU_PARTITION_OP_LIMIT")
            && let Ok(limit) = limit.parse::<usize>()
        {
            ops.truncate(limit);
        }
        tracing::debug!(
            operation_count = ops.len(),
            first_op = ops.first().copied(),
            last_op = ops.last().copied(),
            partition = "stateless-mlp",
            "selected graph-safe CPU execution partitions"
        );
        ops.into_iter().collect()
    }

    /// Replan after CUDA OOM or host pressure. The failed domain is reduced before planning.
    pub fn replan_after_pressure(
        &self,
        domain: fellm_memory::MemoryDomain,
        bytes: u64,
    ) -> Result<FabricPlan, fellm_memory::PlanningError> {
        let mut hardware = self.hardware.clone();
        match domain {
            fellm_memory::MemoryDomain::Device => {
                hardware.device_available = hardware.device_available.saturating_sub(bytes)
            }
            fellm_memory::MemoryDomain::Host | fellm_memory::MemoryDomain::HostPinned => {
                hardware.host_available = hardware.host_available.saturating_sub(bytes)
            }
            _ => {}
        }
        let plan = FabricPlanner::plan(&hardware, &self.model)?;
        let mut state = self.state.write().expect("memory fabric lock");
        state.plan = plan.clone();
        state.metrics.replans = state.metrics.replans.saturating_add(1);
        Ok(plan)
    }

    pub fn publish_metrics(&self) {
        let snapshot = self.snapshot();
        metrics::gauge!("fellm_memory_resident_bytes", "tier" => "device")
            .set(snapshot.metrics.resident_device_bytes as f64);
        metrics::gauge!("fellm_memory_resident_bytes", "tier" => "host_pinned")
            .set(snapshot.metrics.resident_pinned_bytes as f64);
        metrics::gauge!("fellm_memory_resident_bytes", "tier" => "host")
            .set(snapshot.metrics.resident_host_bytes as f64);
        metrics::gauge!("fellm_memory_backing_bytes", "tier" => "storage")
            .set(snapshot.metrics.backing_storage_bytes as f64);
        metrics::counter!("fellm_memory_prefetch_hits_total")
            .absolute(snapshot.metrics.prefetch_hits);
        metrics::counter!("fellm_memory_prefetch_misses_total")
            .absolute(snapshot.metrics.prefetch_misses);
        metrics::counter!("fellm_memory_replans_total").absolute(snapshot.metrics.replans);
    }
}

fn profile_model(
    gguf: &GgufFile,
    graph: &Graph,
    execution: &ExecutionPlan,
    kv_bytes: u64,
    activation_bytes: u64,
    cpu_score: f64,
    disable_cpu_partitions: bool,
) -> (ModelProfile, Vec<WeightDescriptor>) {
    let mut total = 0u64;
    let mut weights = Vec::new();
    for info in gguf.tensors() {
        let Ok(tensor) = gguf.tensor(&info.name) else {
            continue;
        };
        let Some((path, offset, len)) = tensor.file_extent() else {
            continue;
        };
        let bytes = len as u64;
        total = total.saturating_add(bytes);
        weights.push(WeightDescriptor {
            id: WeightId(tensor.logical_id()),
            name: info.name.clone(),
            home: fellm_memory::StorageExtent {
                provider: "gguf".into(),
                path: path.clone(),
                offset,
                len: bytes,
                alignment: gguf.alignment(),
            },
            byte_len: bytes,
            replicas: Vec::new(),
        });
    }
    let by_extent = weights
        .iter()
        .map(|weight| {
            (
                (
                    weight.home.path.clone(),
                    weight.home.offset,
                    weight.byte_len as usize,
                ),
                weight.id,
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let sizes = weights
        .iter()
        .map(|weight| (weight.id, weight.byte_len))
        .collect::<std::collections::HashMap<_, _>>();
    let mut groups = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (position, &node_id) in execution.order.iter().enumerate() {
        let mut consumed = Vec::new();
        for &input in graph.inputs_slice(node_id) {
            let OpValue::Constant(tensor) = &graph.node(input).value else {
                continue;
            };
            let key = if let Some((path, offset, len)) = tensor.file_extent() {
                (path.clone(), offset, len)
            } else if let Some((offset, len)) = tensor.mmap_extent() {
                (std::path::PathBuf::new(), offset as u64, len)
            } else {
                continue;
            };
            let Some(&id) = by_extent.get(&key) else {
                continue;
            };
            if !consumed.contains(&id) {
                consumed.push(id);
                seen.insert(id);
            }
        }
        if !consumed.is_empty() {
            let bytes = consumed.iter().fold(0u64, |total, id| {
                total.saturating_add(sizes.get(id).copied().unwrap_or(0))
            });
            groups.push(ExecutionGroup {
                id: groups.len() as u32,
                weights: consumed,
                byte_len: bytes,
                first_op: position as u32,
                last_op: position as u32,
                reuse_count: 1,
                cpu_compute_time: Some(Duration::ZERO),
            });
        }
    }
    // Any catalog weight absent from the primary step graph remains a cold storage group. This
    // covers optional architecture paths without pretending it belongs to the dense schedule.
    for weight in &weights {
        if !seen.contains(&weight.id) {
            groups.push(ExecutionGroup {
                id: groups.len() as u32,
                weights: vec![weight.id],
                byte_len: weight.byte_len,
                first_op: u32::MAX,
                last_op: u32::MAX,
                reuse_count: 0,
                cpu_compute_time: Some(Duration::ZERO),
            });
        }
    }
    let cpu_weight_bandwidth =
        (cpu_score.max(1.0) * 1024.0 * 1024.0 * 1024.0).min(80.0 * 1024.0 * 1024.0 * 1024.0);
    for group in &mut groups {
        group.cpu_compute_time = (!disable_cpu_partitions)
            .then(|| Duration::from_secs_f64(group.byte_len as f64 / cpu_weight_bandwidth));
    }
    (
        ModelProfile {
            weight_bytes: total,
            kv_bytes,
            activation_bytes,
            groups,
        },
        weights,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_fabric_defaults_are_automatic() {
        let config = MemoryFabricConfig::default();
        assert_eq!(config.device_memory_limit, None);
        assert!(!config.disable_cpu_partitions);
    }
}
