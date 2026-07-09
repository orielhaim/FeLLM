//! Sampling from a logit vector.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Sample one token id from `logits`. `logits` is modified in place.
pub fn sample(logits: &mut [f32], temperature: f32, top_k: u32, top_p: f32, seed: u64) -> u32 {
    if temperature <= 0.0 || top_k == 1 {
        return argmax(logits) as u32;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }

    // Numerically stable softmax over the full vocab (or a top-k slice).
    let vocab = logits.len();
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let k_cut = if top_k > 0 && (top_k as usize) < vocab {
        top_k as usize
    } else {
        vocab
    };
    order.truncate(k_cut);

    // Softmax over retained set.
    let m = order
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = order.iter().map(|&i| (i, (logits[i] - m).exp())).collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(logits) as u32;
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // top-p cutoff (probs already sorted descending because `order` was).
    if top_p < 1.0 && top_p > 0.0 {
        let mut cum = 0.0f32;
        let mut cut = probs.len();
        for (idx, (_, p)) in probs.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                cut = idx + 1;
                break;
            }
        }
        probs.truncate(cut);
        let s: f32 = probs.iter().map(|(_, p)| *p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= s;
        }
    }

    // Draw a single u32 from a freshly-seeded RNG. To make consecutive calls
    // produce different draws when only `seed` changes, we mix the seed with
    // a splitmix64 step so that a linear seed sequence becomes uniform-like.
    let mixed = splitmix64(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(mixed);
    let u: f32 = rng.random::<f32>();

    let mut acc = 0.0f32;
    for (i, p) in &probs {
        acc += p;
        if u <= acc {
            return *i as u32;
        }
    }
    probs.last().map(|(i, _)| *i as u32).unwrap_or(0)
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
