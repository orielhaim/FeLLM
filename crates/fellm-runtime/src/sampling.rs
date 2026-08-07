//! Host-side sampling from a logit vector (backend-agnostic).

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Sample one token id from `logits`. `logits` is modified in place.
#[must_use]
pub fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    repetition_penalty: f32,
    recent_tokens: &[u32],
) -> u32 {
    apply_repetition_penalty(logits, repetition_penalty, recent_tokens);
    if temperature <= 0.0 || top_k == 1 {
        return argmax(logits) as u32;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }

    let vocab = logits.len();
    let k_cut = if top_k > 0 && (top_k as usize) < vocab {
        top_k as usize
    } else {
        vocab
    };

    let mut cand = select_top_k(logits, k_cut);
    cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let m = cand
        .iter()
        .map(|(v, _)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (v, _) in &mut cand {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(logits) as u32;
    }
    for (v, _) in &mut cand {
        *v /= sum;
    }

    if top_p < 1.0 && top_p > 0.0 {
        let mut cum = 0.0f32;
        let mut cut = cand.len();
        for (idx, (p, _)) in cand.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                cut = idx + 1;
                break;
            }
        }
        cand.truncate(cut);
        let s: f32 = cand.iter().map(|(p, _)| *p).sum();
        if s > 0.0 {
            for (p, _) in &mut cand {
                *p /= s;
            }
        }
    }

    let mixed = splitmix64(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(mixed);
    let u: f32 = rng.random::<f32>();

    let mut acc = 0.0f32;
    for (p, i) in &cand {
        acc += p;
        if u <= acc {
            return *i;
        }
    }
    cand.last().map_or(0, |(_, i)| *i)
}

fn select_top_k(logits: &[f32], k: usize) -> Vec<(f32, u32)> {
    if k == 0 || logits.is_empty() {
        return Vec::new();
    }
    if k >= logits.len() {
        return logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
    }

    // Min-heap ordered by logit value (smallest of the current top-k on top).
    #[derive(Clone, Copy)]
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
            // Reverse so BinaryHeap is a min-heap on logit.
            other
                .logit
                .partial_cmp(&self.logit)
                .unwrap_or(Ordering::Equal)
                .then_with(|| other.id.cmp(&self.id))
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k);
    for (i, &v) in logits.iter().enumerate() {
        let e = Entry {
            logit: v,
            id: i as u32,
        };
        if heap.len() < k {
            heap.push(e);
        } else if let Some(top) = heap.peek()
            && v > top.logit
        {
            heap.pop();
            heap.push(e);
        }
    }
    heap.into_iter().map(|e| (e.logit, e.id)).collect()
}

fn apply_repetition_penalty(logits: &mut [f32], penalty: f32, recent_tokens: &[u32]) {
    if penalty <= 1.0 || !penalty.is_finite() {
        return;
    }
    for &token in recent_tokens {
        let Some(logit) = logits.get_mut(token as usize) else {
            continue;
        };
        *logit = if *logit < 0.0 {
            *logit * penalty
        } else {
            *logit / penalty
        };
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
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
