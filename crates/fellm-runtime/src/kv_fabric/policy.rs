//! Replaceable residency / value-cost policy abstractions.

use super::storage::PageMeta;
use super::types::{KvTier, PhysicalSlot, ResidencyPolicyKind};

/// Inputs for scoring one page's residency value.
#[derive(Debug, Clone, Copy)]
pub struct PageScoreInput<'a> {
    pub slot: PhysicalSlot,
    pub meta: &'a PageMeta,
    pub clock: u64,
    pub memory_pressure: f64,
    pub request_importance: f32,
}

/// Trait for residency eviction / keep decisions.
pub trait ResidencyPolicy: Send {
    fn kind(&self) -> ResidencyPolicyKind;

    /// Higher score = more valuable to keep resident. Evict lowest first.
    fn score(&self, input: &PageScoreInput<'_>) -> f64;

    /// Preferred target tier under pressure (may be Host, Disk, NotResident).
    fn demote_target(&self, meta: &PageMeta, pressure: f64) -> KvTier {
        let _ = meta;
        if pressure > 0.85 {
            KvTier::Host
        } else if pressure > 0.6 {
            KvTier::HostPinned
        } else {
            KvTier::Device
        }
    }
}

/// Primary production policy: value / cost.
///
/// ```text
/// value ≈ reuse_probability * recompute_cost * request_importance * sharing
///         / residency_cost
/// ```
#[derive(Debug, Default)]
pub struct ValueCostPolicy;

impl ResidencyPolicy for ValueCostPolicy {
    fn kind(&self) -> ResidencyPolicyKind {
        ResidencyPolicyKind::ValueCost
    }

    fn score(&self, input: &PageScoreInput<'_>) -> f64 {
        let age = input
            .clock
            .saturating_sub(input.meta.last_used)
            .saturating_add(1) as f64;
        let reuse = (input.meta.access_count.saturating_add(1) as f64) / age.sqrt();
        let recompute = f64::from(input.meta.recompute_cost).max(1.0);
        let sharing = f64::from(input.meta.share_count.max(input.meta.refcount).max(1));
        let importance = f64::from(input.request_importance).max(0.1);
        let residency_cost = 1.0 + input.memory_pressure;
        // Immutable shared pages are expensive to re-fetch — boost heavily.
        let immut_boost = if input.meta.immutable { 4.0 } else { 1.0 };
        (reuse * recompute * sharing * importance * immut_boost) / residency_cost
    }
}

/// LRU — available for experiments, not the primary policy.
#[derive(Debug, Default)]
pub struct LruPolicy;

impl ResidencyPolicy for LruPolicy {
    fn kind(&self) -> ResidencyPolicyKind {
        ResidencyPolicyKind::Lru
    }

    fn score(&self, input: &PageScoreInput<'_>) -> f64 {
        // Higher last_used → higher score (keep recent).
        input.meta.last_used as f64
    }
}

#[must_use]
pub fn policy_from_kind(kind: ResidencyPolicyKind) -> Box<dyn ResidencyPolicy> {
    match kind {
        ResidencyPolicyKind::ValueCost => Box::new(ValueCostPolicy),
        ResidencyPolicyKind::Lru => Box::new(LruPolicy),
    }
}
