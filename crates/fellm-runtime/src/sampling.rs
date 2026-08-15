//! Host-side sampling from a logit vector (backend-agnostic).

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::Arc;

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
    counts: HashMap<u32, u32>,
}

/// Deterministic finite-state token grammar for guided decoding.
#[derive(Debug, Clone)]
pub struct TokenGrammar {
    start_state: usize,
    transitions: Vec<BTreeMap<u32, usize>>,
    accepting: Vec<bool>,
}

impl TokenGrammar {
    pub fn new(
        start_state: usize,
        transitions: Vec<BTreeMap<u32, usize>>,
        accepting: Vec<bool>,
    ) -> Result<Self, &'static str> {
        if transitions.is_empty()
            || transitions.len() != accepting.len()
            || start_state >= transitions.len()
            || transitions
                .iter()
                .flat_map(BTreeMap::values)
                .any(|&state| state >= transitions.len())
        {
            return Err("invalid token grammar state table");
        }
        Ok(Self {
            start_state,
            transitions,
            accepting,
        })
    }

    #[must_use]
    pub fn allows(&self, state: usize, token: u32) -> bool {
        self.transitions
            .get(state)
            .is_some_and(|transitions| transitions.contains_key(&token))
    }

    #[must_use]
    pub fn transition(&self, state: usize, token: u32) -> Option<usize> {
        self.transitions.get(state)?.get(&token).copied()
    }

    #[must_use]
    pub fn is_accepting(&self, state: usize) -> bool {
        self.accepting.get(state).copied().unwrap_or(false)
    }
}

/// Request-local sampler state. Speculative rounds checkpoint this state and
/// commit only the accepted/correction path.
#[derive(Debug, Clone, Default)]
pub struct SamplerState {
    history: Vec<u32>,
    draw_index: u64,
    grammar: Option<Arc<TokenGrammar>>,
    grammar_state: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerCheckpoint {
    history_len: usize,
    draw_index: u64,
    grammar_state: usize,
}

impl SamplerState {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: Vec::with_capacity(capacity),
            draw_index: 0,
            grammar: None,
            grammar_state: 0,
        }
    }

    #[must_use]
    pub fn with_grammar(capacity: usize, grammar: Option<Arc<TokenGrammar>>) -> Self {
        let grammar_state = grammar.as_ref().map_or(0, |grammar| grammar.start_state);
        Self {
            history: Vec::with_capacity(capacity),
            draw_index: 0,
            grammar,
            grammar_state,
        }
    }

    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    #[must_use]
    pub fn draw_index(&self) -> u64 {
        self.draw_index
    }

    #[must_use]
    pub fn checkpoint(&self) -> SamplerCheckpoint {
        SamplerCheckpoint {
            history_len: self.history.len(),
            draw_index: self.draw_index,
            grammar_state: self.grammar_state,
        }
    }

    pub fn commit_token(&mut self, token: u32) {
        if let Some(grammar) = &self.grammar {
            self.grammar_state = grammar
                .transition(self.grammar_state, token)
                .expect("committed token must satisfy the active grammar");
        }
        self.history.push(token);
        self.draw_index = self.draw_index.wrapping_add(1);
    }

    /// Seed repetition/frequency/presence history from the request prompt.
    ///
    /// Prompt tokens are context for penalty processors, but they are not
    /// generated draws and must not advance either the RNG position or an
    /// output grammar. Keeping this operation distinct from `commit_token`
    /// makes those three state machines impossible to conflate.
    pub fn prime_history(&mut self, tokens: &[u32]) {
        self.history.extend_from_slice(tokens);
    }

    pub fn rollback(&mut self, checkpoint: SamplerCheckpoint) {
        self.history.truncate(checkpoint.history_len);
        self.draw_index = checkpoint.draw_index;
        self.grammar_state = checkpoint.grammar_state;
    }

    #[must_use]
    pub fn grammar_view(&self) -> Option<(&TokenGrammar, usize)> {
        self.grammar
            .as_deref()
            .map(|grammar| (grammar, self.grammar_state))
    }

    #[must_use]
    pub fn grammar_is_accepting(&self) -> bool {
        self.grammar
            .as_ref()
            .is_some_and(|grammar| grammar.is_accepting(self.grammar_state))
    }
}

/// Processed categorical distribution after temperature, penalties, top-k and top-p.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedDistribution {
    probabilities: Vec<f32>,
}

impl ProcessedDistribution {
    #[must_use]
    pub fn probability(&self, token: u32) -> f32 {
        self.probabilities
            .get(token as usize)
            .copied()
            .unwrap_or(0.0)
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.probabilities
    }

    #[must_use]
    pub fn argmax(&self) -> u32 {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(token, _)| token as u32)
    }

    #[must_use]
    pub fn sample(&self, seed: u64) -> u32 {
        let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(seed));
        let draw = rng.random::<f32>();
        let mut cumulative = 0.0;
        for (token, probability) in self.probabilities.iter().enumerate() {
            cumulative += probability;
            if draw < cumulative {
                return token as u32;
            }
        }
        self.probabilities.len().saturating_sub(1) as u32
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions<'a> {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub logit_bias: &'a [(u32, f32)],
    pub grammar: Option<(&'a TokenGrammar, usize)>,
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
            min_p: 0.0,
            seed,
            repetition_penalty,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: &[],
            grammar: None,
            recent_tokens,
        },
        &mut SamplingWorkspace::default(),
    )
}

/// Highest-scoring vocabulary ids, ties broken by smaller id.
#[must_use]
pub fn top_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (id, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        if best.len() < k {
            best.push((id as u32, logit));
            if best.len() == k {
                best.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
            }
            continue;
        }
        if logit > best[k - 1].1 || (logit == best[k - 1].1 && (id as u32) < best[k - 1].0) {
            best[k - 1] = (id as u32, logit);
            best.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
    }
    best.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    best
}

/// Sample using caller-owned scratch storage; steady-state sampling performs no
/// candidate or heap allocation once the vocabulary/top-k capacity has grown.
#[must_use]
pub fn sample_with_workspace(
    logits: &[f32],
    options: SamplingOptions<'_>,
    workspace: &mut SamplingWorkspace,
) -> u32 {
    prepare_recent(options, workspace);
    if options.temperature <= 0.0 || options.top_k == 1 {
        return argmax_adjusted(logits, options, &workspace.recent, &workspace.counts) as u32;
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
        return argmax_adjusted(logits, options, &workspace.recent, &workspace.counts) as u32;
    }
    for (v, _) in cand.iter_mut() {
        *v /= sum;
    }

    if options.min_p > 0.0 && options.min_p < 1.0 {
        let threshold = cand.first().map_or(0.0, |(probability, _)| *probability) * options.min_p;
        cand.retain(|(probability, _)| *probability >= threshold);
        renormalize(cand);
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
        renormalize(cand);
    }

    cand.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(options.seed));
    let draw = rng.random::<f32>();
    let mut cumulative = 0.0;
    for &(probability, token) in cand.iter() {
        cumulative += probability;
        if draw < cumulative {
            return token;
        }
    }
    cand.last().map_or(0, |(_, token)| *token)
}

/// Produce the exact processed distribution used for sampling and speculative verification.
#[must_use]
pub fn distribution_with_workspace(
    logits: &[f32],
    options: SamplingOptions<'_>,
    workspace: &mut SamplingWorkspace,
) -> ProcessedDistribution {
    prepare_recent(options, workspace);
    if options.temperature <= 0.0 || options.top_k == 1 {
        let token = argmax_adjusted(logits, options, &workspace.recent, &workspace.counts);
        let mut probabilities = vec![0.0; logits.len()];
        if let Some(probability) = probabilities.get_mut(token) {
            *probability = 1.0;
        }
        return ProcessedDistribution { probabilities };
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
        let token = argmax_adjusted(logits, options, &workspace.recent, &workspace.counts);
        let mut probabilities = vec![0.0; logits.len()];
        if let Some(probability) = probabilities.get_mut(token) {
            *probability = 1.0;
        }
        return ProcessedDistribution { probabilities };
    }
    for (v, _) in cand.iter_mut() {
        *v /= sum;
    }

    if options.min_p > 0.0 && options.min_p < 1.0 {
        let threshold = cand.first().map_or(0.0, |(probability, _)| *probability) * options.min_p;
        cand.retain(|(probability, _)| *probability >= threshold);
        renormalize(cand);
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
        renormalize(cand);
    }

    let mut probabilities = vec![0.0; logits.len()];
    for &(probability, token) in cand.iter() {
        probabilities[token as usize] = probability;
    }
    ProcessedDistribution { probabilities }
}

fn prepare_recent(options: SamplingOptions<'_>, workspace: &mut SamplingWorkspace) {
    workspace.recent.clear();
    workspace
        .recent
        .extend(options.recent_tokens.iter().copied());
    workspace.recent.sort_unstable();
    workspace.recent.dedup();
    workspace.counts.clear();
    for &token in options.recent_tokens {
        *workspace.counts.entry(token).or_default() += 1;
    }
}

fn renormalize(candidates: &mut [(f32, u32)]) {
    let sum: f32 = candidates.iter().map(|(probability, _)| *probability).sum();
    if sum > 0.0 {
        for (probability, _) in candidates {
            *probability /= sum;
        }
    }
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
        counts,
    } = workspace;
    candidates.clear();
    heap.clear();
    if k == 0 || logits.is_empty() {
        return;
    }
    let inv_temperature = 1.0 / options.temperature.max(f32::EPSILON);
    if k >= logits.len() {
        candidates.extend(logits.iter().enumerate().map(|(i, &value)| {
            (
                adjusted_logit(value, i, options, recent, counts) * inv_temperature,
                i as u32,
            )
        }));
        return;
    }
    let scan_k = k.saturating_add(recent.len()).min(logits.len());
    heap.reserve(scan_k.saturating_sub(heap.capacity()));
    for (i, &raw_value) in logits.iter().enumerate() {
        let value = adjusted_logit(raw_value, i, options, recent, counts) * inv_temperature;
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
    candidates.extend(heap.iter().map(|entry| (entry.logit, entry.id)));
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

fn adjusted_logit(
    value: f32,
    token: usize,
    options: SamplingOptions<'_>,
    recent: &[u32],
    counts: &HashMap<u32, u32>,
) -> f32 {
    if options
        .grammar
        .is_some_and(|(grammar, state)| !grammar.allows(state, token as u32))
    {
        return f32::NEG_INFINITY;
    }
    let penalized = if options.repetition_penalty > 1.0
        && options.repetition_penalty.is_finite()
        && recent.binary_search(&(token as u32)).is_ok()
    {
        repetition_adjust(value, options.repetition_penalty)
    } else {
        value
    };
    let count = counts.get(&(token as u32)).copied().unwrap_or(0) as f32;
    let bias: f32 = options
        .logit_bias
        .iter()
        .filter(|(biased, _)| *biased == token as u32)
        .map(|(_, bias)| *bias)
        .sum();
    penalized
        - options.frequency_penalty * count
        - options.presence_penalty * f32::from(count > 0.0)
        + bias
}

fn argmax_adjusted(
    logits: &[f32],
    options: SamplingOptions<'_>,
    recent: &[u32],
    counts: &HashMap<u32, u32>,
) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (token, &value) in logits.iter().enumerate() {
        let value = adjusted_logit(value, token, options, recent, counts);
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

    #[test]
    fn top_logits_keeps_highest_scores() {
        let logits = [0.1f32, 5.0, 0.2, 4.0, f32::NAN];
        assert_eq!(top_logits(&logits, 2), vec![(1, 5.0), (3, 4.0)]);
    }

    #[test]
    fn sampler_state_rolls_back_history_and_rng_position() {
        let mut state = SamplerState::with_capacity(8);
        state.commit_token(4);
        let checkpoint = state.checkpoint();
        state.commit_token(7);
        state.commit_token(9);
        state.rollback(checkpoint);
        assert_eq!(state.history(), &[4]);
        assert_eq!(state.draw_index(), 1);
    }

    #[test]
    fn prompt_history_does_not_advance_rng_or_output_grammar() {
        let grammar = TokenGrammar::new(
            0,
            vec![BTreeMap::from([(9, 1)]), BTreeMap::new()],
            vec![false, true],
        )
        .unwrap();
        let mut state = SamplerState::with_grammar(8, Some(Arc::new(grammar)));
        state.prime_history(&[9, 4, 9]);

        assert_eq!(state.history(), &[9, 4, 9]);
        assert_eq!(state.draw_index(), 0);
        assert!(!state.grammar_is_accepting());

        state.commit_token(9);
        assert_eq!(state.draw_index(), 1);
        assert!(state.grammar_is_accepting());
    }

    #[test]
    fn exposed_distribution_matches_sampling_filters() {
        let logits = [5.0, 4.0, 3.0, 2.0];
        let options = SamplingOptions {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            seed: 3,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: &[],
            grammar: None,
            recent_tokens: &[],
        };
        let distribution =
            distribution_with_workspace(&logits, options, &mut SamplingWorkspace::default());
        assert!(distribution.probability(0) > 0.0);
        assert!(distribution.probability(1) > 0.0);
        assert_eq!(distribution.probability(2), 0.0);
        assert!((distribution.as_slice().iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    fn options<'a>(recent_tokens: &'a [u32]) -> SamplingOptions<'a> {
        SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 1,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: &[],
            grammar: None,
            recent_tokens,
        }
    }

    #[test]
    fn all_stateful_processors_change_the_exposed_distribution() {
        let logits = [3.0, 2.0, 1.0];
        let mut configured = options(&[0, 0]);
        configured.frequency_penalty = 1.0;
        configured.presence_penalty = 1.0;
        configured.logit_bias = &[(2, 5.0)];
        configured.min_p = 0.1;
        let distribution =
            distribution_with_workspace(&logits, configured, &mut SamplingWorkspace::default());
        assert_eq!(distribution.argmax(), 2);
        assert_eq!(distribution.probability(1), 0.0);
    }

    #[test]
    fn grammar_state_and_rng_rollback_together() {
        let grammar = Arc::new(
            TokenGrammar::new(
                0,
                vec![
                    BTreeMap::from([(1, 1)]),
                    BTreeMap::from([(2, 2)]),
                    BTreeMap::new(),
                ],
                vec![false, false, true],
            )
            .unwrap(),
        );
        let mut state = SamplerState::with_grammar(4, Some(grammar));
        let checkpoint = state.checkpoint();
        let (grammar, grammar_state) = state.grammar_view().unwrap();
        let mut guided = options(state.history());
        guided.grammar = Some((grammar, grammar_state));
        let first = distribution_with_workspace(
            &[10.0, 0.0, 20.0],
            guided,
            &mut SamplingWorkspace::default(),
        );
        assert_eq!(first.argmax(), 1);
        state.commit_token(1);
        assert_eq!(state.grammar_view().unwrap().1, 1);
        state.rollback(checkpoint);
        assert_eq!(state.grammar_view().unwrap().1, 0);
        assert_eq!(state.draw_index(), 0);
    }
}
