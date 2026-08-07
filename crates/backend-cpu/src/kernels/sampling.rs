//! Sampling from a logit vector (CPU kernel path; shared logic lives in
//! `fellm_runtime::sampling` for the engine/scheduler).

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Sample one token id from `logits`. `logits` is modified in place.
pub fn sample(logits: &mut [f32], temperature: f32, top_k: u32, top_p: f32, seed: u64) -> u32 {
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
    fn greedy_picks_argmax() {
        let mut l = vec![0.1, 0.9, 0.5, 0.2];
        let id = sample(&mut l, 0.0, 0, 1.0, 42);
        assert_eq!(id, 1);
    }
}
