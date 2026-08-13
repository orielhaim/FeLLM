//! Shared tier, residency, transfer, and budget primitives for FeLLM memory consumers.
//!
//! Storage is authoritative backing. Device and host allocations are disposable replicas or
//! bounded streaming working sets; neither a pointer nor a replica location identifies a weight.

mod experts;
mod planner;
mod provider;
mod types;

pub use experts::{ExpertAccessTracker, plan_expert_residency};
pub use planner::{FabricPlanner, PlanningError};
#[cfg(target_os = "linux")]
pub use provider::DirectFileProvider;
pub use provider::{
    BoundedTransferPool, FileProvider, MmapProvider, PrefetchedRead, TransferProvider,
    coalesce_extents,
};
pub use types::{
    ConsumerBudget, ExecutionGroup, ExpertAccess, ExpertId, ExpertPlacement, ExpertTemperature,
    FabricMetrics, FabricPlan, HardwareProfile, MemoryDomain, ModelProfile, Replica, ReplicaId,
    ResidencyClass, StorageExtent, TransferCapabilities, TransferKind, WeightDescriptor, WeightId,
    WeightPlacement,
};
