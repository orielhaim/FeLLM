//! Shared tier, residency, transfer, and budget primitives for FeLLM memory consumers.
//!
//! Storage is authoritative backing. Device and host allocations are disposable replicas or
//! bounded streaming working sets; neither a pointer nor a replica location identifies a weight.

mod cache_simulator;
mod experts;
mod planner;
mod provider;
mod provider_selection;
mod storage_objects;
mod types;

pub use cache_simulator::{
    CacheSimulation, ExpertCachePolicy, ExpertTraceEvent, simulate_expert_cache,
};
pub use experts::{ExpertAccessTracker, plan_expert_residency};
pub use planner::{FabricPlanner, PlanningError};
#[cfg(any(target_os = "linux", windows))]
pub use provider::DirectFileProvider;
pub use provider::{
    BoundedTransferPool, FileProvider, MmapProvider, PrefetchedRead, TransferPoolMetrics,
    TransferProvider, coalesce_extents,
};
pub use provider_selection::{
    StorageProviderKind, StorageProviderRequest, StorageWorkload, select_storage_provider,
};
pub use storage_objects::StorageObjectIndex;
pub use types::{
    ConsumerBudget, ExecutionGroup, ExpertAccess, ExpertId, ExpertPlacement, ExpertRouteTrace,
    ExpertTemperature, FabricMetrics, FabricPlan, HardwareProfile, MemoryDomain, ModelProfile,
    Replica, ReplicaId, ResidencyClass, StorageExtent, StorageObject, StorageObjectId,
    StorageObjectMember, StorageSlotState, TransferCapabilities, TransferKind, WeightDescriptor,
    WeightId, WeightPlacement,
};

/// True when a GGUF tensor is a routed MoE expert bank that should not be streamed whole.
#[must_use]
pub fn is_moe_expert_bank(name: &str) -> bool {
    let name = name.strip_suffix(".weight").unwrap_or(name);
    name.ends_with("gate_exps")
        || name.ends_with("up_exps")
        || name.ends_with("down_exps")
        || name.ends_with("gate_up_exps")
}
