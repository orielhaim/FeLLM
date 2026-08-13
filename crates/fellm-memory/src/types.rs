use std::path::PathBuf;
use std::time::Duration;

/// Stable logical weight identity, independent of addresses and residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightId(pub u64);

/// Stable identity for one disposable resident replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplicaId(pub u64);

/// Physical memory domains understood by the fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    Device,
    HostPinned,
    Host,
    Storage,
    NonResident,
}

/// Authoritative byte extent in a storage provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageExtent {
    pub provider: String,
    pub path: PathBuf,
    pub offset: u64,
    pub len: u64,
    pub alignment: u64,
}

/// A currently materialized replica. Backing remains present when replicas are evicted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replica {
    pub id: ReplicaId,
    pub domain: MemoryDomain,
    pub offset: u64,
    pub len: u64,
}

/// One logical immutable model weight.
#[derive(Debug, Clone)]
pub struct WeightDescriptor {
    pub id: WeightId,
    pub name: String,
    pub home: StorageExtent,
    pub byte_len: u64,
    pub replicas: Vec<Replica>,
}

/// Architecture-neutral unit of transfer and prefetch derived from an access schedule.
#[derive(Debug, Clone)]
pub struct ExecutionGroup {
    pub id: u32,
    pub weights: Vec<WeightId>,
    pub byte_len: u64,
    pub first_op: u32,
    pub last_op: u32,
    pub reuse_count: u32,
    pub cpu_compute_time: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyClass {
    PermanentDevice,
    DeviceStream,
    HostResident,
    StorageStream,
    CpuCompute,
}

/// Router-observed temperature of an independently placeable MoE expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertTemperature {
    Cold,
    Warm,
    Hot,
}

/// Stable expert identity within a routed operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub operation: u64,
    pub expert: u32,
}

/// Decayed router evidence used by irregular-workload placement.
#[derive(Debug, Clone, Copy)]
pub struct ExpertAccess {
    pub id: ExpertId,
    pub score: f64,
    pub selections: u64,
    pub last_step: u64,
    pub byte_len: u64,
}

/// One expert's independently chosen backing/residency policy.
#[derive(Debug, Clone, Copy)]
pub struct ExpertPlacement {
    pub id: ExpertId,
    pub temperature: ExpertTemperature,
    pub residency: ResidencyClass,
}

#[derive(Debug, Clone)]
pub struct WeightPlacement {
    pub group: u32,
    pub class: ResidencyClass,
    pub bytes: u64,
    pub estimated_transfer: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TransferCapabilities {
    pub mmap: bool,
    pub async_file: bool,
    pub direct_io: bool,
    pub io_uring: bool,
    pub gds: bool,
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub device_total: u64,
    pub device_available: u64,
    pub host_total: u64,
    pub host_available: u64,
    pub h2d_bytes_per_second: u64,
    pub storage_bytes_per_second: u64,
    pub storage_latency: Duration,
    pub cpu_score: f64,
    pub transfers: TransferCapabilities,
}

#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub activation_bytes: u64,
    pub groups: Vec<ExecutionGroup>,
}

#[derive(Debug, Clone, Copy)]
pub struct ConsumerBudget {
    pub weights_device: u64,
    pub weights_host: u64,
    pub kv_device: u64,
    pub kv_host: u64,
    pub activation_device: u64,
    pub device_staging: u64,
    pub host_staging: u64,
    pub device_reserve: u64,
    pub host_reserve: u64,
}

#[derive(Debug, Clone)]
pub struct FabricPlan {
    pub budget: ConsumerBudget,
    pub placements: Vec<WeightPlacement>,
    pub device_buffer_count: u8,
    pub device_buffer_bytes: u64,
    pub host_buffer_count: u8,
    pub host_buffer_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferKind {
    StorageToHost,
    HostToDevice,
    StorageToDevice,
    DeviceToHost,
}

/// Fabric-level counters. Mutable consumers may layer their own metrics on top.
#[derive(Debug, Clone, Default)]
pub struct FabricMetrics {
    pub resident_device_bytes: u64,
    pub resident_pinned_bytes: u64,
    pub resident_host_bytes: u64,
    pub backing_storage_bytes: u64,
    pub transfer_bytes: [u64; 4],
    pub transfer_time: [Duration; 4],
    pub stall_time: Duration,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub evictions: u64,
    pub replans: u64,
}
