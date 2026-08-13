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
use std::sync::{Arc, RwLock};
use std::time::Duration;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

const DEFAULT_H2D_BPS: u64 = 24 * 1024 * 1024 * 1024;
const DEFAULT_SSD_BPS: u64 = 3 * 1024 * 1024 * 1024;
const TARGET_GROUP_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Default)]
pub struct MemoryFabricConfig {
    pub device_memory_limit: Option<u64>,
    pub host_memory_limit: Option<u64>,
    pub h2d_bytes_per_second: Option<u64>,
    pub storage_bytes_per_second: Option<u64>,
    pub storage_latency_micros: Option<u64>,
    pub disable_cpu_partitions: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryFabricSnapshot {
    pub plan: FabricPlan,
    pub metrics: FabricMetrics,
}

/// One model's joint plan. Allocation failures update pressure and produce a smaller plan.
pub struct MemoryFabric {
    hardware: HardwareProfile,
    model: ModelProfile,
    weights: Arc<[WeightDescriptor]>,
    schedule_ops: Arc<[Option<OpKind>]>,
    expert_tracker: Arc<Mutex<fellm_memory::ExpertAccessTracker>>,
    expert_placements: Arc<RwLock<Vec<fellm_memory::ExpertPlacement>>>,
    state: Arc<RwLock<MemoryFabricSnapshot>>,
}

impl MemoryFabric {
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
                direct_io: cfg!(target_os = "linux"),
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
        let plan = FabricPlanner::plan(&hardware, &model)?;
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
                    .then(|| tensor.as_bytes().len() as u64)
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
            hardware,
            model,
            weights: weights.into(),
            schedule_ops: execution
                .order
                .iter()
                .map(|&node| graph.node(node).op)
                .collect(),
            expert_tracker: Arc::new(Mutex::new(expert_tracker)),
            expert_placements: Arc::new(RwLock::new(Vec::new())),
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
            tracing::info!(
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

    /// Immutable logical weight catalog. Replica locations may change independently.
    #[must_use]
    pub fn weights(&self) -> &[WeightDescriptor] {
        &self.weights
    }

    /// Resolve an execution tensor to its stable GGUF weight identity by storage extent.
    #[must_use]
    pub fn weight_id_for_tensor(&self, tensor: &fellm_core::tensor::Tensor) -> Option<WeightId> {
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
        tracing::info!(
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
    for tensor in gguf.tensors() {
        let bytes = tensor.dtype.byte_size(tensor.shape.num_elements()) as u64;
        let absolute_offset = gguf
            .tensor_data_offset()
            .saturating_add(tensor.relative_offset);
        let weight_id = WeightId(absolute_offset.saturating_add(1));
        total = total.saturating_add(bytes);
        weights.push(WeightDescriptor {
            id: weight_id,
            name: tensor.name.clone(),
            home: fellm_memory::StorageExtent {
                provider: "gguf".into(),
                path: gguf
                    .source_path()
                    .map_or_else(std::path::PathBuf::new, std::path::Path::to_path_buf),
                offset: absolute_offset,
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
                (weight.home.offset as usize, weight.byte_len as usize),
                weight.id,
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut scheduled = std::collections::HashMap::<WeightId, (u64, u32, u32, u32)>::new();
    for (position, &node_id) in execution.order.iter().enumerate() {
        for &input in graph.inputs_slice(node_id) {
            let OpValue::Constant(tensor) = &graph.node(input).value else {
                continue;
            };
            let Some(extent) = tensor.mmap_extent() else {
                continue;
            };
            let Some(&id) = by_extent.get(&extent) else {
                continue;
            };
            let entry = scheduled.entry(id).or_insert((
                extent.1 as u64,
                position as u32,
                position as u32,
                0,
            ));
            entry.2 = position as u32;
            entry.3 = entry.3.saturating_add(1);
        }
    }
    // Any catalog weight absent from the primary step graph remains a cold storage group. This
    // covers optional architecture paths without pretending it belongs to the dense schedule.
    for weight in &weights {
        scheduled
            .entry(weight.id)
            .or_insert((weight.byte_len, u32::MAX, u32::MAX, 0));
    }
    let mut scheduled = scheduled.into_iter().collect::<Vec<_>>();
    scheduled.sort_by_key(|(id, (_, first, _, _))| (*first, *id));
    let mut groups = Vec::new();
    let mut current: Option<ExecutionGroup> = None;
    for (weight, (bytes, first, last, reuse)) in scheduled {
        if current.as_ref().is_some_and(|group| {
            group.byte_len > 0 && group.byte_len.saturating_add(bytes) > TARGET_GROUP_BYTES
        }) {
            groups.push(current.take().expect("checked execution group"));
        }
        let group = current.get_or_insert_with(|| ExecutionGroup {
            id: groups.len() as u32,
            weights: Vec::new(),
            byte_len: 0,
            first_op: first,
            last_op: last,
            reuse_count: 0,
            cpu_compute_time: Some(Duration::ZERO),
        });
        group.weights.push(weight);
        group.byte_len = group.byte_len.saturating_add(bytes);
        group.first_op = group.first_op.min(first);
        group.last_op = group.last_op.max(last);
        group.reuse_count = group.reuse_count.saturating_add(reuse.max(1));
    }
    if let Some(group) = current {
        groups.push(group);
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
