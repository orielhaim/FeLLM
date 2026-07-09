//! Sampling from a logit vector.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Sample one token id from `logits`. `logits` is modified in place.
pub fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
) -> u32 {
    if temperature <= 0.0 || (top_k == 1) {
        // Greedy.
        return argmax(logits) as u32;
    }
    // Apply temperature.
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }
    // top-k: keep only the k largest.
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    if top_k > 0 && (top_k as usize) < logits.len() {
        indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(core::cmp::Ordering::Equal));
        indices.truncate(top_k as usize);
    }
    // Softmax over the retained set.
    let m = indices.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i, (logits[i] - m).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }
    // top-p: keep smallest set of tokens whose cumulative probability >= top_p.
    if top_p < 1.0 && top_p > 0.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
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
    // Draw.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
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
