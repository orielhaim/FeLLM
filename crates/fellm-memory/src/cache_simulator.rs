use crate::ExpertId;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCachePolicy {
    Lru,
    Lfu,
    Belady,
    ValueCost,
    RecencyFrequency,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertTraceEvent {
    pub expert: ExpertId,
    pub bytes: u64,
    pub load_cost_nanos: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheSimulation {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_requested: u64,
    pub bytes_loaded: u64,
    pub evictions: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    bytes: u64,
    frequency: u64,
    last_use: usize,
    load_cost_nanos: u64,
}

/// Replay router traces without coupling observations to the live cache implementation.
/// Capacity is byte-exact; an expert larger than the cache is served but never admitted.
pub fn simulate_expert_cache(
    trace: &[ExpertTraceEvent],
    capacity_bytes: u64,
    policy: ExpertCachePolicy,
) -> CacheSimulation {
    let future = build_future_uses(trace);
    let mut next_cursor = HashMap::<ExpertId, usize>::new();
    let mut resident = HashMap::<ExpertId, Entry>::new();
    let mut resident_bytes = 0u64;
    let mut result = CacheSimulation::default();

    for (position, event) in trace.iter().enumerate() {
        result.requests += 1;
        result.bytes_requested = result.bytes_requested.saturating_add(event.bytes);
        let cursor = next_cursor.entry(event.expert).or_default();
        *cursor += 1;
        if let Some(entry) = resident.get_mut(&event.expert) {
            result.hits += 1;
            entry.frequency = entry.frequency.saturating_add(1);
            entry.last_use = position;
            continue;
        }
        result.misses += 1;
        result.bytes_loaded = result.bytes_loaded.saturating_add(event.bytes);
        if event.bytes > capacity_bytes {
            continue;
        }
        while resident_bytes.saturating_add(event.bytes) > capacity_bytes {
            let Some(victim) = choose_victim(&resident, policy, position, &future, &next_cursor)
            else {
                break;
            };
            if let Some(entry) = resident.remove(&victim) {
                resident_bytes = resident_bytes.saturating_sub(entry.bytes);
                result.evictions += 1;
            }
        }
        resident.insert(
            event.expert,
            Entry {
                bytes: event.bytes,
                frequency: 1,
                last_use: position,
                load_cost_nanos: event.load_cost_nanos,
            },
        );
        resident_bytes = resident_bytes.saturating_add(event.bytes);
    }
    result
}

fn build_future_uses(trace: &[ExpertTraceEvent]) -> HashMap<ExpertId, VecDeque<usize>> {
    let mut future = HashMap::<ExpertId, VecDeque<usize>>::new();
    for (position, event) in trace.iter().enumerate() {
        future.entry(event.expert).or_default().push_back(position);
    }
    future
}

fn choose_victim(
    resident: &HashMap<ExpertId, Entry>,
    policy: ExpertCachePolicy,
    position: usize,
    future: &HashMap<ExpertId, VecDeque<usize>>,
    cursors: &HashMap<ExpertId, usize>,
) -> Option<ExpertId> {
    resident
        .iter()
        .min_by(|(left_id, left), (right_id, right)| {
            eviction_value(**left_id, **left, policy, position, future, cursors)
                .total_cmp(&eviction_value(
                    **right_id, **right, policy, position, future, cursors,
                ))
                .then_with(|| left_id.cmp(right_id))
        })
        .map(|(&id, _)| id)
}

fn eviction_value(
    id: ExpertId,
    entry: Entry,
    policy: ExpertCachePolicy,
    position: usize,
    future: &HashMap<ExpertId, VecDeque<usize>>,
    cursors: &HashMap<ExpertId, usize>,
) -> f64 {
    match policy {
        ExpertCachePolicy::Lru => entry.last_use as f64,
        ExpertCachePolicy::Lfu => entry.frequency as f64,
        ExpertCachePolicy::Belady => {
            let cursor = cursors.get(&id).copied().unwrap_or_default();
            let next = future
                .get(&id)
                .and_then(|uses| uses.get(cursor))
                .copied()
                .unwrap_or(usize::MAX);
            -(next as f64)
        }
        ExpertCachePolicy::ValueCost => {
            entry.frequency as f64 * entry.load_cost_nanos.max(1) as f64 / entry.bytes.max(1) as f64
        }
        ExpertCachePolicy::RecencyFrequency => {
            let age = position.saturating_sub(entry.last_use) as f64;
            entry.frequency as f64 / (age + 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(expert: u32) -> ExpertTraceEvent {
        ExpertTraceEvent {
            expert: ExpertId {
                operation: 0,
                expert,
            },
            bytes: 1,
            load_cost_nanos: 1,
        }
    }

    #[test]
    fn oracle_beats_lru_on_cyclic_irregular_trace() {
        let trace = [0, 1, 2, 0, 1, 3, 0, 1, 2, 3]
            .into_iter()
            .map(event)
            .collect::<Vec<_>>();
        let lru = simulate_expert_cache(&trace, 2, ExpertCachePolicy::Lru);
        let oracle = simulate_expert_cache(&trace, 2, ExpertCachePolicy::Belady);
        assert!(oracle.hits > lru.hits);
        assert_eq!(oracle.requests, trace.len() as u64);
    }

    #[test]
    fn oversized_experts_are_never_admitted() {
        let trace = [event(1), event(1)];
        let result = simulate_expert_cache(&trace, 0, ExpertCachePolicy::Lfu);
        assert_eq!(result.hits, 0);
        assert_eq!(result.bytes_loaded, 2);
    }
}
