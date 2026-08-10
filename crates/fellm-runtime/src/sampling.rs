//! Host-side sampling from a logit vector (backend-agnostic).

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
struct Entry {
    logit: f32,
    id: u32,
}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.logit == other.logit && self.id == other.id
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .logit
            .partial_cmp(&self.logit)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// Reusable storage for the per-token sampling hot path.
#[derive(Debug, Default)]
pub struct SamplingWorkspace {
    candidates: Vec<(f32, u32)>,
    heap: BinaryHeap<Entry>,
    recent: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions<'a> {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub seed: u64,
    pub repetition_penalty: f32,
    pub recent_tokens: &'a [u32],
}

/// Sample one token id from immutable logits using a temporary workspace.
#[must_use]
pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    repetition_penalty: f32,
    recent_tokens: &[u32],
) -> u32 {
    sample_with_workspace(
        logits,
        SamplingOptions {
            temperature,
            top_k,
            top_p,
            seed,
            repetition_penalty,
            recent_tokens,
        },
        &mut SamplingWorkspace::default(),
    )
}

/// Sample using caller-owned scratch storage; steady-state sampling performs no
/// candidate or heap allocation once the vocabulary/top-k capacity has grown.
#[must_use]
pub fn sample_with_workspace(
    logits: &[f32],
    options: SamplingOptions<'_>,
    workspace: &mut SamplingWorkspace,
) -> u32 {
    workspace.recent.clear();
    workspace
        .recent
        .extend(options.recent_tokens.iter().copied());
    workspace.recent.sort_unstable();
    workspace.recent.dedup();
    if options.temperature <= 0.0 || options.top_k == 1 {
        return argmax_adjusted(logits, options, &workspace.recent) as u32;
    }

    let vocab = logits.len();
    let k_cut = if options.top_k > 0 && (options.top_k as usize) < vocab {
        options.top_k as usize
    } else {
        vocab
    };

    select_top_k(logits, k_cut, options, workspace);
    let cand = &mut workspace.candidates;
    cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    cand.truncate(k_cut);

    let m = cand
        .iter()
        .map(|(v, _)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (v, _) in cand.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return argmax_adjusted(logits, options, &workspace.recent) as u32;
    }
    for (v, _) in cand.iter_mut() {
        *v /= sum;
    }

    if options.top_p < 1.0 && options.top_p > 0.0 {
        let mut cum = 0.0f32;
        let mut cut = cand.len();
        for (idx, (p, _)) in cand.iter().enumerate() {
            cum += *p;
            if cum >= options.top_p {
                cut = idx + 1;
                break;
            }
        }
        cand.truncate(cut);
        let s: f32 = cand.iter().map(|(p, _)| *p).sum();
        if s > 0.0 {
            for (p, _) in cand.iter_mut() {
                *p /= s;
            }
        }
    }

    let mixed = splitmix64(options.seed);
    let mut rng = ChaCha8Rng::seed_from_u64(mixed);
    let u: f32 = rng.random::<f32>();

    let mut acc = 0.0f32;
    for (p, i) in cand.iter() {
        acc += p;
        if u <= acc {
            return *i;
        }
    }
    cand.last().map_or(0, |(_, i)| *i)
}

fn select_top_k(
    logits: &[f32],
    k: usize,
    options: SamplingOptions<'_>,
    workspace: &mut SamplingWorkspace,
) {
    let SamplingWorkspace {
        candidates,
        heap,
        recent,
    } = workspace;
    candidates.clear();
    heap.clear();
    if k == 0 || logits.is_empty() {
        return;
    }
    let inv_temperature = 1.0 / options.temperature.max(f32::EPSILON);
    if k >= logits.len() {
        candidates.extend(
            logits
                .iter()
                .enumerate()
                .map(|(i, &value)| (value * inv_temperature, i as u32)),
        );
        for &token in recent.iter() {
            if let Some((value, _)) = candidates.get_mut(token as usize) {
                *value = repetition_adjust(*value, options.repetition_penalty);
            }
        }
        return;
    }
    let scan_k = k.saturating_add(recent.len()).min(logits.len());
    heap.reserve(scan_k.saturating_sub(heap.capacity()));
    for (i, &value) in logits.iter().enumerate() {
        let e = Entry {
            logit: value,
            id: i as u32,
        };
        if heap.len() < scan_k {
            heap.push(e);
        } else if let Some(top) = heap.peek()
            && value > top.logit
        {
            heap.pop();
            heap.push(e);
        }
    }
    candidates.extend(heap.iter().map(|entry| {
        let value = if recent.binary_search(&entry.id).is_ok() {
            repetition_adjust(entry.logit, options.repetition_penalty)
        } else {
            entry.logit
        };
        (value * inv_temperature, entry.id)
    }));
}

fn repetition_adjust(value: f32, penalty: f32) -> f32 {
    if penalty > 1.0 && penalty.is_finite() {
        if value < 0.0 {
            value * penalty
        } else {
            value / penalty
        }
    } else {
        value
    }
}

fn adjusted_logit(value: f32, token: usize, options: SamplingOptions<'_>, recent: &[u32]) -> f32 {
    let penalized = if options.repetition_penalty > 1.0
        && options.repetition_penalty.is_finite()
        && recent.binary_search(&(token as u32)).is_ok()
    {
        repetition_adjust(value, options.repetition_penalty)
    } else {
        value
    };
    penalized
}

fn argmax_adjusted(logits: &[f32], options: SamplingOptions<'_>, recent: &[u32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (token, &value) in logits.iter().enumerate() {
        let value = adjusted_logit(value, token, options, recent);
        if value > best_value {
            best = token;
            best_value = value;
        }
    }
    best
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_prefers_largest_logits() {
        let mut logits = vec![-10.0f32; 1000];
        logits[7] = 5.0;
        logits[3] = 4.0;
        logits[99] = 3.0;
        // With top_k=2 and high temperature the third-best should never win.
        for seed in 0..50 {
            let mut l = logits.clone();
            let id = sample(&mut l, 1.0, 2, 1.0, seed, 1.0, &[]);
            assert!(id == 7 || id == 3, "seed {seed} produced {id}");
        }
    }

    #[test]
    fn greedy_is_argmax() {
        let mut logits = vec![0.1f32, 0.9, 0.2];
        assert_eq!(sample(&mut logits, 0.0, 80, 1.0, 0, 1.0, &[]), 1);
    }
}
