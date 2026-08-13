use crate::{ExpertAccess, ExpertId, ExpertPlacement, ExpertTemperature, ResidencyClass};
use std::collections::HashMap;

/// Online router-frequency model. Scores decay by step so a formerly popular expert naturally
/// demotes without an LRU scan; the planner may periodically snapshot and re-place the bank.
pub struct ExpertAccessTracker {
    decay_per_step: f64,
    step: u64,
    experts: HashMap<ExpertId, ExpertAccess>,
}

impl ExpertAccessTracker {
    #[must_use]
    pub fn new(decay_per_step: f64) -> Self {
        Self {
            decay_per_step: decay_per_step.clamp(0.0, 1.0),
            step: 0,
            experts: HashMap::new(),
        }
    }

    pub fn register(&mut self, id: ExpertId, byte_len: u64) {
        self.experts.entry(id).or_insert(ExpertAccess {
            id,
            score: 0.0,
            selections: 0,
            last_step: self.step,
            byte_len,
        });
    }

    /// Record the experts selected by a router for one token/batch decision.
    pub fn observe(&mut self, selected: impl IntoIterator<Item = ExpertId>) {
        self.step = self.step.saturating_add(1);
        for id in selected {
            let Some(access) = self.experts.get_mut(&id) else {
                continue;
            };
            let elapsed = self.step.saturating_sub(access.last_step) as i32;
            access.score = access.score * self.decay_per_step.powi(elapsed) + 1.0;
            access.selections = access.selections.saturating_add(1);
            access.last_step = self.step;
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<ExpertAccess> {
        let mut experts = self.experts.values().copied().collect::<Vec<_>>();
        for access in &mut experts {
            let elapsed = self.step.saturating_sub(access.last_step) as i32;
            access.score *= self.decay_per_step.powi(elapsed);
        }
        experts.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        experts
    }
}

/// Independently fill hot VRAM and warm RAM expert working sets from router evidence. Remaining
/// experts stay cold on their authoritative storage backing.
#[must_use]
pub fn plan_expert_residency(
    accesses: &[ExpertAccess],
    mut device_bytes: u64,
    mut host_bytes: u64,
) -> Vec<ExpertPlacement> {
    let mut ranked = accesses.to_vec();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    let mut placements = Vec::with_capacity(ranked.len());
    for access in ranked {
        let (temperature, residency) = if access.score > 0.0 && access.byte_len <= device_bytes {
            device_bytes -= access.byte_len;
            (ExpertTemperature::Hot, ResidencyClass::PermanentDevice)
        } else if access.score > 0.0 && access.byte_len <= host_bytes {
            host_bytes -= access.byte_len;
            (ExpertTemperature::Warm, ResidencyClass::HostResident)
        } else {
            (ExpertTemperature::Cold, ResidencyClass::StorageStream)
        };
        placements.push(ExpertPlacement {
            id: access.id,
            temperature,
            residency,
        });
    }
    placements.sort_by_key(|placement| placement.id);
    placements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_evidence_promotes_hot_then_warm_experts() {
        let mut tracker = ExpertAccessTracker::new(0.9);
        for expert in 0..3 {
            tracker.register(
                ExpertId {
                    operation: 7,
                    expert,
                },
                100,
            );
        }
        tracker.observe([
            ExpertId {
                operation: 7,
                expert: 1,
            },
            ExpertId {
                operation: 7,
                expert: 2,
            },
        ]);
        tracker.observe([ExpertId {
            operation: 7,
            expert: 1,
        }]);
        let plan = plan_expert_residency(&tracker.snapshot(), 100, 100);
        assert_eq!(plan[1].temperature, ExpertTemperature::Hot);
        assert_eq!(plan[2].temperature, ExpertTemperature::Warm);
        assert_eq!(plan[0].temperature, ExpertTemperature::Cold);
    }
}
