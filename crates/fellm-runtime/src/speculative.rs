//! Backend-independent speculative decoding contracts and lossless verification.
//!
//! Speculators propose; the runtime verifies.  Nothing in this module assumes
//! that a proposer is another language model or shares a device with the target.

use crate::engine::{DecodeSequence, Engine, GenParams};
use fellm_core::error::{FellmError, Result as FellmResult};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

/// Target state advanced provisionally while scoring a linear continuation.
#[derive(Debug)]
pub struct ProvisionalTargetVerification {
    pub distributions: Vec<ProbabilityDistribution>,
    pub kv_transaction: crate::kv_fabric::KvTransaction,
    pub(crate) recurrent_checkpoint: Option<crate::HybridConvState>,
    /// Recurrent state after each provisionally verified token. This permits
    /// committing an accepted prefix shorter than the proposed suffix.
    pub(crate) recurrent_prefixes: Vec<Option<crate::HybridConvState>>,
    /// Owned feature rows corresponding to each provisionally verified token.
    pub(crate) feature_rows:
        Vec<Vec<(fellm_plugin_abi::TargetFeature, fellm_core::tensor::Tensor)>>,
}

/// A normalized categorical distribution in target-token vocabulary order.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilityDistribution {
    probabilities: Vec<f32>,
}

impl ProbabilityDistribution {
    /// Normalize non-negative finite weights. Invalid entries are treated as zero.
    pub fn from_weights(weights: impl Into<Vec<f32>>) -> Result<Self, &'static str> {
        let mut probabilities = weights.into();
        let sum: f64 = probabilities
            .iter_mut()
            .map(|value| {
                if !value.is_finite() || *value < 0.0 {
                    *value = 0.0;
                }
                f64::from(*value)
            })
            .sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err("distribution has no finite positive mass");
        }
        for value in &mut probabilities {
            *value = (f64::from(*value) / sum) as f32;
        }
        Ok(Self { probabilities })
    }

    /// A deterministic distribution, used by the optimized greedy path.
    #[must_use]
    pub fn point_mass(vocabulary_size: usize, token: u32) -> Self {
        let mut probabilities = vec![0.0; vocabulary_size];
        if let Some(value) = probabilities.get_mut(token as usize) {
            *value = 1.0;
        }
        Self { probabilities }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

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
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(0, |(token, _)| token as u32)
    }

    fn positive_residual(&self, draft: &Self) -> Result<Self, &'static str> {
        if self.len() != draft.len() {
            return Err("target and draft vocabulary sizes differ");
        }
        let residual = Self::from_weights(
            self.probabilities
                .iter()
                .zip(&draft.probabilities)
                .map(|(p, q)| (p - q).max(0.0))
                .collect::<Vec<_>>(),
        );
        // If p == q, the residual is mathematically unused for every valid q
        // sample. A malformed zero-mass claimed token is rejected defensively;
        // falling back to p preserves the target distribution.
        residual.or_else(|_| Ok(self.clone()))
    }

    fn sample(&self, rng: &mut ChaCha8Rng) -> u32 {
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

impl From<&crate::sampling::ProcessedDistribution> for ProbabilityDistribution {
    fn from(distribution: &crate::sampling::ProcessedDistribution) -> Self {
        Self {
            probabilities: distribution.as_slice().to_vec(),
        }
    }
}

/// One proposed target-vocabulary token and the processed distribution it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftToken {
    pub token: u32,
    pub distribution: ProbabilityDistribution,
}

/// Initial topology. The enum is deliberately extensible to tree/DAG proposals.
#[derive(Debug, Clone, PartialEq)]
pub enum DraftProposal {
    Linear(LinearProposal),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearProposal {
    pub tokens: Vec<DraftToken>,
    /// Prefix-survival estimate by proposal position, when supplied by a speculator.
    pub prefix_survival: Option<Vec<f32>>,
}

/// Result of one verification round. `emitted` includes a correction or bonus token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationOutcome {
    pub accepted: usize,
    pub emitted: Vec<u32>,
    pub rejected_at: Option<usize>,
    pub used_bonus: bool,
    pub terminal: bool,
}

/// Runtime-owned modified rejection sampler with request-local RNG state.
#[derive(Debug, Clone)]
pub struct SpeculativeVerifier {
    rng: ChaCha8Rng,
}

impl SpeculativeVerifier {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Verify a linear proposal. `target[i]` is p at proposal position i and
    /// `target[K]` is the bonus-token distribution.
    pub fn verify(
        &mut self,
        proposal: &LinearProposal,
        target: &[ProbabilityDistribution],
    ) -> Result<VerificationOutcome, &'static str> {
        self.verify_with_stops(proposal, target, &[])
    }

    pub fn verify_with_stops(
        &mut self,
        proposal: &LinearProposal,
        target: &[ProbabilityDistribution],
        stop_tokens: &[u32],
    ) -> Result<VerificationOutcome, &'static str> {
        self.verify_with_terminal_position(proposal, target, stop_tokens, None)
    }

    pub fn verify_with_terminal_position(
        &mut self,
        proposal: &LinearProposal,
        target: &[ProbabilityDistribution],
        stop_tokens: &[u32],
        terminal_position: Option<usize>,
    ) -> Result<VerificationOutcome, &'static str> {
        if target.len() != proposal.tokens.len() + 1 {
            return Err("verification requires K target distributions plus one bonus distribution");
        }
        let mut emitted = Vec::with_capacity(proposal.tokens.len() + 1);
        for (position, draft) in proposal.tokens.iter().enumerate() {
            let p = &target[position];
            if p.len() != draft.distribution.len() {
                return Err("target and draft vocabulary sizes differ");
            }
            let qx = draft.distribution.probability(draft.token);
            let px = p.probability(draft.token);
            // q(x)=0 is malformed for a token claimed to be sampled from q. Reject
            // safely; the residual then equals p for that zero-mass coordinate.
            let acceptance = if qx > 0.0 { (px / qx).min(1.0) } else { 0.0 };
            if self.rng.random::<f32>() < acceptance {
                emitted.push(draft.token);
                if stop_tokens.contains(&draft.token) || terminal_position == Some(position) {
                    return Ok(VerificationOutcome {
                        accepted: position + 1,
                        emitted,
                        rejected_at: None,
                        used_bonus: false,
                        terminal: true,
                    });
                }
                continue;
            }
            let correction = p
                .positive_residual(&draft.distribution)?
                .sample(&mut self.rng);
            emitted.push(correction);
            return Ok(VerificationOutcome {
                accepted: position,
                emitted,
                rejected_at: Some(position),
                used_bonus: false,
                terminal: stop_tokens.contains(&correction),
            });
        }
        let bonus = target.last().expect("length checked").sample(&mut self.rng);
        emitted.push(bonus);
        Ok(VerificationOutcome {
            accepted: proposal.tokens.len(),
            emitted,
            rejected_at: None,
            used_bonus: true,
            terminal: stop_tokens.contains(&bonus),
        })
    }

    /// No-RNG fast path for temperature zero / top-k one.
    pub fn verify_greedy(&self, proposed: &[u32], target_argmax: &[u32]) -> VerificationOutcome {
        self.verify_greedy_with_stops(proposed, target_argmax, &[], None)
    }

    pub fn verify_greedy_with_stops(
        &self,
        proposed: &[u32],
        target_argmax: &[u32],
        stop_tokens: &[u32],
        terminal_position: Option<usize>,
    ) -> VerificationOutcome {
        assert_eq!(target_argmax.len(), proposed.len() + 1);
        let mut emitted = Vec::with_capacity(target_argmax.len());
        for (position, (&draft, &target)) in proposed.iter().zip(target_argmax).enumerate() {
            if draft != target {
                emitted.push(target);
                return VerificationOutcome {
                    accepted: position,
                    emitted,
                    rejected_at: Some(position),
                    used_bonus: false,
                    terminal: stop_tokens.contains(&target),
                };
            }
            emitted.push(draft);
            if stop_tokens.contains(&draft) || terminal_position == Some(position) {
                return VerificationOutcome {
                    accepted: position + 1,
                    emitted,
                    rejected_at: None,
                    used_bonus: false,
                    terminal: true,
                };
            }
        }
        let bonus = *target_argmax.last().expect("non-empty bonus");
        emitted.push(bonus);
        VerificationOutcome {
            accepted: proposed.len(),
            emitted,
            rejected_at: None,
            used_bonus: true,
            terminal: stop_tokens.contains(&bonus),
        }
    }
}

pub const SPECULATIVE_METRIC_BUCKETS: usize = 65;

#[derive(Debug, Clone, Copy)]
pub struct SpeculationMetrics {
    pub rounds: u64,
    pub proposed: u64,
    pub verified: u64,
    pub accepted: u64,
    pub emitted: u64,
    pub draft_time: Duration,
    pub verification_time: Duration,
    pub sampling_time: Duration,
    pub h2d_bytes: u64,
    pub storage_bytes: u64,
    pub target_forward_passes: u64,
    pub draft_forward_passes: u64,
    pub disabled_rounds: u64,
    pub scheduler_time: Duration,
    /// Accepted proposal tokens by zero-based proposal position.
    pub accepted_by_position: [u64; SPECULATIVE_METRIC_BUCKETS],
    /// Rounds by selected K; the last bucket includes larger values.
    pub chosen_k: [u64; SPECULATIVE_METRIC_BUCKETS],
    /// Rounds by accepted prefix length; the last bucket includes larger values.
    pub accepted_length: [u64; SPECULATIVE_METRIC_BUCKETS],
}

impl Default for SpeculationMetrics {
    fn default() -> Self {
        Self {
            rounds: 0,
            proposed: 0,
            verified: 0,
            accepted: 0,
            emitted: 0,
            draft_time: Duration::ZERO,
            verification_time: Duration::ZERO,
            sampling_time: Duration::ZERO,
            h2d_bytes: 0,
            storage_bytes: 0,
            target_forward_passes: 0,
            draft_forward_passes: 0,
            disabled_rounds: 0,
            scheduler_time: Duration::ZERO,
            accepted_by_position: [0; SPECULATIVE_METRIC_BUCKETS],
            chosen_k: [0; SPECULATIVE_METRIC_BUCKETS],
            accepted_length: [0; SPECULATIVE_METRIC_BUCKETS],
        }
    }
}

#[derive(Debug, Clone)]
struct RollingCostModel {
    baseline_seconds: f64,
    draft_seconds_per_token: f64,
    verify_seconds: Vec<f64>,
    transfer_seconds: Vec<f64>,
}

impl RollingCostModel {
    fn new(max_proposal_len: usize) -> Self {
        Self {
            baseline_seconds: 0.0,
            draft_seconds_per_token: 0.0,
            verify_seconds: vec![0.0; max_proposal_len.max(1)],
            transfer_seconds: vec![0.0; max_proposal_len.max(1)],
        }
    }

    fn observe(previous: f64, sample: f64) -> f64 {
        if !sample.is_finite() || sample <= 0.0 {
            previous
        } else if previous <= 0.0 {
            sample
        } else {
            0.8 * previous + 0.2 * sample
        }
    }

    fn observe_decode(&mut self, seconds: f64) {
        self.baseline_seconds = Self::observe(self.baseline_seconds, seconds);
    }

    fn observe_round(&mut self, proposal_len: usize, draft: Duration, verify: Duration) {
        if proposal_len == 0 {
            return;
        }
        let draft_per_token = draft.as_secs_f64() / proposal_len as f64;
        self.draft_seconds_per_token = Self::observe(self.draft_seconds_per_token, draft_per_token);
        let index = proposal_len.saturating_sub(1).min(self.verify_seconds.len() - 1);
        self.verify_seconds[index] = Self::observe(self.verify_seconds[index], verify.as_secs_f64());
    }
}

impl SpeculationMetrics {
    fn record_round(&mut self, chosen_k: usize, accepted: usize) {
        let k_bucket = chosen_k.min(SPECULATIVE_METRIC_BUCKETS - 1);
        self.chosen_k[k_bucket] += 1;
        if chosen_k == 0 {
            return;
        }
        let accepted_bucket = accepted.min(SPECULATIVE_METRIC_BUCKETS - 1);
        self.accepted_length[accepted_bucket] += 1;
        for position in 0..accepted.min(SPECULATIVE_METRIC_BUCKETS) {
            self.accepted_by_position[position] += 1;
        }
    }

    /// Empirical P(accepted prefix >= k) for k = 1..=limit.
    #[must_use]
    pub fn prefix_survival(&self, limit: usize) -> Vec<f64> {
        let rounds = self.rounds.saturating_sub(self.disabled_rounds).max(1) as f64;
        (0..limit)
            .map(|position| {
                self.accepted_by_position
                    .get(position)
                    .copied()
                    .unwrap_or(0) as f64
                    / rounds
            })
            .collect()
    }

    #[must_use]
    pub fn accepted_length_percentile(&self, percentile: f64) -> usize {
        let total = self.accepted_length.iter().sum::<u64>();
        if total == 0 {
            return 0;
        }
        let threshold = (percentile.clamp(0.0, 1.0) * total as f64).ceil() as u64;
        let mut cumulative = 0;
        for (length, &count) in self.accepted_length.iter().enumerate() {
            cumulative += count;
            if cumulative >= threshold.max(1) {
                return length;
            }
        }
        SPECULATIVE_METRIC_BUCKETS - 1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpeculationDecision {
    pub proposal_len: usize,
    pub expected_tokens_per_second: f64,
}

/// Cost-aware adaptive K policy. It may choose zero and therefore never forces speculation.
#[derive(Debug, Clone)]
pub struct AdaptiveSpeculationPolicy {
    pub max_proposal_len: usize,
    pub minimum_gain: f64,
}

/// Target capacity curve measured in completed verification steps per second
/// as a function of the physical token batch.
#[derive(Debug, Clone, Default)]
pub struct VerificationCapacityProfile {
    samples: Vec<(usize, f64)>,
}

impl VerificationCapacityProfile {
    pub fn new(mut samples: Vec<(usize, f64)>) -> FellmResult<Self> {
        samples.retain(|(batch, rate)| *batch > 0 && rate.is_finite() && *rate > 0.0);
        samples.sort_unstable_by_key(|(batch, _)| *batch);
        samples.dedup_by_key(|(batch, _)| *batch);
        if samples.is_empty() {
            return Err(FellmError::other(
                "verification capacity profile has no valid samples",
            ));
        }
        Ok(Self { samples })
    }

    #[must_use]
    pub fn steps_per_second(&self, batch: usize) -> f64 {
        let index = self.samples.partition_point(|(sample, _)| *sample <= batch);
        if index == 0 {
            self.samples[0].1
        } else {
            self.samples[index - 1].1
        }
    }
}

/// Lossless DSpark-style global prefix allocation. Conditional confidence is
/// converted to cumulative prefix survival, then marginal verification tokens
/// are admitted in descending value order. The first throughput regression
/// terminates the search, preserving the non-anticipating property when later
/// confidence depends on already sampled draft tokens.
#[must_use]
pub fn schedule_confident_prefixes(
    conditional_confidence: &[Vec<f32>],
    capacity: &VerificationCapacityProfile,
) -> Vec<usize> {
    let requests = conditional_confidence.len();
    if requests == 0 {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for (request, confidence) in conditional_confidence.iter().enumerate() {
        let mut survival = 1.0_f64;
        for (position, &conditional) in confidence.iter().enumerate() {
            survival *= f64::from(conditional.clamp(0.0, 1.0));
            if survival > 0.0 {
                candidates.push((survival, request, position + 1));
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });

    let mut lengths = vec![0; requests];
    let mut best_lengths = lengths.clone();
    let mut physical_batch = requests;
    let mut expected_tokens = requests as f64;
    let mut best_throughput = expected_tokens * capacity.steps_per_second(physical_batch);
    for (survival, request, position) in candidates {
        // Monotone survival makes the preceding position appear first. Keep the
        // invariant explicit for calibrated scores with equal/tied values.
        if position != lengths[request] + 1 {
            continue;
        }
        lengths[request] = position;
        physical_batch += 1;
        expected_tokens += survival;
        let throughput = expected_tokens * capacity.steps_per_second(physical_batch);
        if throughput <= best_throughput {
            break;
        }
        best_throughput = throughput;
        best_lengths.clone_from(&lengths);
    }
    best_lengths
}

#[derive(Debug, Clone)]
pub struct GenericDraftConfig {
    pub maximum_proposal_length: usize,
    pub initial_proposal_length: usize,
    pub minimum_gain: f64,
}

impl Default for GenericDraftConfig {
    fn default() -> Self {
        Self {
            maximum_proposal_length: 4,
            initial_proposal_length: 3,
            minimum_gain: 0.05,
        }
    }
}

/// Two independently planned ordinary FeLLM models coordinated by the common verifier.
pub struct GenericDraftRuntime {
    target: Engine,
    draft: Engine,
    config: GenericDraftConfig,
    metrics: SpeculationMetrics,
    next_proposal_length: usize,
    costs: RollingCostModel,
    policy: AdaptiveSpeculationPolicy,
}

/// Runtime coordinator for any request-owned [`fellm_plugin_abi::Speculator`].
/// The plugin only proposes scores; this type owns sampling, verification,
/// target state transactions, correction/bonus handling, and adaptive K.
pub struct PluginSpeculativeRuntime {
    target: Engine,
    speculator: Box<dyn fellm_plugin_abi::Speculator>,
    config: GenericDraftConfig,
    metrics: SpeculationMetrics,
    next_proposal_length: usize,
    costs: RollingCostModel,
    policy: AdaptiveSpeculationPolicy,
}

impl PluginSpeculativeRuntime {
    pub fn new(
        target: Engine,
        speculator: Box<dyn fellm_plugin_abi::Speculator>,
        config: GenericDraftConfig,
    ) -> FellmResult<Self> {
        speculator.validate_target(
            &target.spec().arch_id,
            u32::try_from(target.spec().vocab_size)
                .map_err(|_| FellmError::other("target vocabulary exceeds u32"))?,
        )?;
        if speculator.compatibility().topology != fellm_plugin_abi::ProposalTopology::Linear {
            return Err(FellmError::other(
                "this verifier currently accepts linear proposal frontiers",
            ));
        }
        for tap in &speculator.compatibility().required_features {
            if !target.settings().target_features.contains(&tap.feature) {
                return Err(FellmError::other(format!(
                    "speculator requires target feature {:?} that was not prepared",
                    tap.feature
                )));
            }
        }
        let maximum = config
            .maximum_proposal_length
            .min(speculator.compatibility().maximum_proposal_length as usize)
            .max(1);
        let next_proposal_length = config.initial_proposal_length.clamp(1, maximum);
        let minimum_gain = config.minimum_gain;
        Ok(Self {
            target,
            speculator,
            config: GenericDraftConfig {
                maximum_proposal_length: maximum,
                ..config
            },
            metrics: SpeculationMetrics::default(),
            next_proposal_length,
            costs: RollingCostModel::new(maximum),
            policy: AdaptiveSpeculationPolicy {
                max_proposal_len: maximum,
                minimum_gain,
            },
        })
    }

    #[must_use]
    pub fn target(&self) -> &Engine {
        &self.target
    }

    #[must_use]
    pub fn metrics(&self) -> SpeculationMetrics {
        self.metrics
    }

    pub fn generate_ids(&mut self, prompt_ids: &[u32], params: GenParams) -> FellmResult<Vec<u32>> {
        let speculator = &mut self.speculator;
        let mut sequence =
            self.target
                .prefill_sequence_observing(prompt_ids, |token, position, features| {
                    speculator.observe_committed(&fellm_plugin_abi::CommittedTargetToken {
                        token,
                        position: position as u32,
                        captured_features: features,
                    })
                })?;
        let result = self.generate_sequence(&mut sequence, prompt_ids, params);
        self.target.release_sequence(sequence);
        result
    }

    fn generate_sequence(
        &mut self,
        sequence: &mut DecodeSequence,
        prompt_ids: &[u32],
        params: GenParams,
    ) -> FellmResult<Vec<u32>> {
        let stop_tokens = self.target.stop_token_ids_pub();
        let mut prefix = prompt_ids.to_vec();
        let mut output = Vec::with_capacity(params.max_tokens as usize);
        let mut sampler_state = crate::sampling::SamplerState::with_grammar(
            params.max_tokens as usize,
            params.grammar.clone(),
        );
        sampler_state.prime_history(prompt_ids);
        let mut verifier = SpeculativeVerifier::new(params.seed ^ 0xa076_1d64_78bd_642f);
        let mut workspace = crate::sampling::SamplingWorkspace::default();

        while output.len() < params.max_tokens as usize && sequence.position < self.target.n_ctx() {
            let remaining = params.max_tokens as usize - output.len();
            let context_remaining = self.target.n_ctx().saturating_sub(sequence.position);
            let hard_limit = self
                .config
                .maximum_proposal_length
                .min(remaining.saturating_sub(1))
                .min(context_remaining.saturating_sub(1));
            let proposal_limit = schedule_verification_length(
                &self.policy,
                &self.costs,
                &self.metrics,
                hard_limit.min(self.next_proposal_length.max(1)).min(hard_limit),
                None,
            )
            .min(hard_limit);
            if proposal_limit == 0 {
                self.metrics.rounds += 1;
                self.metrics.disabled_rounds += 1;
                self.metrics.record_round(0, 0);
                let token = sample_logits(
                    sequence.logits(),
                    &params,
                    &sampler_state,
                    &mut workspace,
                    params.seed,
                )?;
                sampler_state.commit_token(token);
                prefix.push(token);
                output.push(token);
                self.metrics.emitted += 1;
                if stop_tokens.contains(&token)
                    || sampler_state.grammar_is_accepting()
                    || output.len() == params.max_tokens as usize
                {
                    break;
                }
                self.target.advance_sequence(sequence, token)?;
                let position = sequence.position - 1;
                let speculator = &mut self.speculator;
                self.target.with_target_features(|features| {
                    speculator.observe_committed(&fellm_plugin_abi::CommittedTargetToken {
                        token,
                        position: position as u32,
                        captured_features: features,
                    })
                })?;
                self.metrics.target_forward_passes += 1;
                if self.metrics.rounds.is_multiple_of(16) {
                    self.next_proposal_length = 1;
                }
                continue;
            }

            let round_started = std::time::Instant::now();
            let mut proposed_tokens = Vec::with_capacity(proposal_limit);
            let mut proposal = LinearProposal {
                tokens: Vec::with_capacity(proposal_limit),
                prefix_survival: Some(Vec::with_capacity(proposal_limit)),
            };
            let mut draft_sampler = sampler_state.clone();
            let draft_started = std::time::Instant::now();
            let speculator = &mut self.speculator;
            self.target.with_target_features(|features| {
                speculator.begin_round(&fellm_plugin_abi::SpeculatorContext {
                    prefix_tokens: &prefix,
                    proposed_tokens: &proposed_tokens,
                    maximum_length: proposal_limit as u32,
                    captured_features: features,
                })
            })?;
            let mut survival = 1.0_f32;
            let mut grammar_terminal_position = None;
            let proposal_result = (|| -> FellmResult<()> {
                for depth in 0..proposal_limit {
                    let frontier = self.target.with_target_features(|features| {
                        self.speculator
                            .propose_next(&fellm_plugin_abi::SpeculatorContext {
                                prefix_tokens: &prefix,
                                proposed_tokens: &proposed_tokens,
                                maximum_length: proposal_limit as u32,
                                captured_features: features,
                            })
                    })?;
                    let Some(node) = frontier.nodes.into_iter().next() else {
                        break;
                    };
                    if node.parent != depth.checked_sub(1).map(|parent| parent as u32) {
                        return Err(FellmError::other(
                            "speculator returned a non-linear proposal parent",
                        ));
                    }
                    let distribution = distribution_for_proposal_scores(
                        node.scores,
                        &params,
                        &draft_sampler,
                        &mut workspace,
                    )?;
                    let token = map_draft_token(
                        &self.speculator.compatibility().vocabulary,
                        sample_processed(&distribution, &params, &draft_sampler),
                    )?;
                    proposed_tokens.push(token);
                    proposal.tokens.push(DraftToken {
                        token,
                        distribution: (&distribution).into(),
                    });
                    survival *= node.confidence.unwrap_or(1.0).clamp(0.0, 1.0);
                    proposal
                        .prefix_survival
                        .as_mut()
                        .expect("initialized")
                        .push(survival);
                    draft_sampler.commit_token(token);
                    self.metrics.draft_forward_passes += 1;
                    if draft_sampler.grammar_is_accepting() {
                        grammar_terminal_position = Some(depth);
                        break;
                    }
                    if stop_tokens.contains(&token) {
                        break;
                    }
                }
                Ok(())
            })();
            if let Err(error) = proposal_result {
                self.speculator.abort_round();
                return Err(error);
            }
            self.metrics.draft_time += draft_started.elapsed();
            let draft_duration = draft_started.elapsed();
            if proposal.tokens.is_empty() {
                self.speculator.abort_round();
                self.next_proposal_length = 0;
                continue;
            }
            self.metrics.proposed += proposal.tokens.len() as u64;
            let scheduled = schedule_verification_length(
                &self.policy,
                &self.costs,
                &self.metrics,
                proposal.tokens.len(),
                proposal.prefix_survival.as_deref(),
            );
            if scheduled == 0 {
                self.speculator.abort_round();
                self.metrics.rounds += 1;
                self.metrics.disabled_rounds += 1;
                self.metrics.record_round(0, 0);
                let token = sample_logits(
                    sequence.logits(),
                    &params,
                    &sampler_state,
                    &mut workspace,
                    params.seed,
                )?;
                sampler_state.commit_token(token);
                prefix.push(token);
                output.push(token);
                self.metrics.emitted += 1;
                if stop_tokens.contains(&token)
                    || sampler_state.grammar_is_accepting()
                    || output.len() == params.max_tokens as usize
                {
                    break;
                }
                let started = std::time::Instant::now();
                self.target.advance_sequence(sequence, token)?;
                self.costs.observe_decode(started.elapsed().as_secs_f64());
                let position = sequence.position - 1;
                let speculator = &mut self.speculator;
                self.target.with_target_features(|features| {
                    speculator.observe_committed(&fellm_plugin_abi::CommittedTargetToken {
                        token,
                        position: position as u32,
                        captured_features: features,
                    })
                })?;
                self.metrics.target_forward_passes += 1;
                continue;
            }
            if scheduled < proposal.tokens.len() {
                proposal.tokens.truncate(scheduled);
                proposed_tokens.truncate(scheduled);
                if let Some(survival) = &mut proposal.prefix_survival {
                    survival.truncate(scheduled);
                }
            }

            let initial_logits = sequence.pending_logits.clone();
            let verification_start_position = sequence.position;
            let verification_started = std::time::Instant::now();
            let provisional = match self.target.verify_proposal(
                &mut sequence.cache,
                &mut sequence.recurrent,
                &initial_logits,
                &proposed_tokens,
                sequence.position,
                &params,
                &sampler_state,
            ) {
                Ok(value) => value,
                Err(error) => {
                    self.speculator.abort_round();
                    return Err(error);
                }
            };
            self.metrics.verification_time += verification_started.elapsed();
            self.metrics.target_forward_passes += proposed_tokens.len() as u64;
            self.metrics.verified += proposed_tokens.len() as u64;
            let outcome = match verify_linear_proposal(
                &mut verifier,
                &proposal,
                &provisional.distributions,
                &stop_tokens,
                grammar_terminal_position,
                params.is_greedy(),
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.target.finalize_verification(
                        &mut sequence.cache,
                        &mut sequence.recurrent,
                        provisional,
                        0,
                    )?;
                    self.speculator.abort_round();
                    return Err(FellmError::other(error));
                }
            };
            let accepted_feature_rows = provisional
                .feature_rows
                .iter()
                .take(outcome.accepted)
                .cloned()
                .collect::<Vec<_>>();
            if let Err(error) =
                self.speculator
                    .finish_round(&fellm_plugin_abi::SpeculatorRoundOutcome {
                        accepted: outcome.accepted,
                        emitted: &outcome.emitted,
                        terminal: outcome.terminal,
                    })
            {
                self.target.finalize_verification(
                    &mut sequence.cache,
                    &mut sequence.recurrent,
                    provisional,
                    0,
                )?;
                self.speculator.abort_round();
                return Err(error);
            }
            self.target.finalize_verification(
                &mut sequence.cache,
                &mut sequence.recurrent,
                provisional,
                outcome.accepted,
            )?;
            sequence.position += outcome.accepted;
            for (offset, (&token, features)) in proposed_tokens
                .iter()
                .zip(&accepted_feature_rows)
                .enumerate()
            {
                observe_owned_features(
                    self.speculator.as_mut(),
                    token,
                    verification_start_position + offset,
                    features,
                )?;
            }

            self.metrics.rounds += 1;
            self.metrics
                .record_round(proposal.tokens.len(), outcome.accepted);
            self.metrics.accepted += outcome.accepted as u64;
            self.metrics.emitted += outcome.emitted.len() as u64;
            for &token in &outcome.emitted {
                if output.len() == params.max_tokens as usize {
                    break;
                }
                sampler_state.commit_token(token);
                prefix.push(token);
                output.push(token);
                if sampler_state.grammar_is_accepting() {
                    break;
                }
            }
            if outcome.terminal
                || sampler_state.grammar_is_accepting()
                || output.len() == params.max_tokens as usize
            {
                break;
            }

            let final_token = *outcome.emitted.last().expect("verification always emits");
            let correction_started = std::time::Instant::now();
            self.target.advance_sequence(sequence, final_token)?;
            let final_position = sequence.position - 1;
            let speculator = &mut self.speculator;
            self.target.with_target_features(|features| {
                speculator.observe_committed(&fellm_plugin_abi::CommittedTargetToken {
                    token: final_token,
                    position: final_position as u32,
                    captured_features: features,
                })
            })?;
            self.metrics.target_forward_passes += 1;
            self.metrics.sampling_time += correction_started.elapsed();
            self.costs.observe_round(
                proposal.tokens.len(),
                draft_duration,
                verification_started.elapsed(),
            );
            self.costs
                .observe_decode(correction_started.elapsed().as_secs_f64());
            let round_seconds = round_started.elapsed().as_secs_f64();
            let speculative_rate = outcome.emitted.len() as f64 / round_seconds.max(f64::EPSILON);
            let baseline_rate = 1.0 / correction_started.elapsed().as_secs_f64().max(f64::EPSILON);
            if speculative_rate < baseline_rate * (1.0 + self.config.minimum_gain) {
                self.next_proposal_length = 0;
            } else if outcome.accepted == proposal.tokens.len() {
                self.next_proposal_length =
                    (proposal.tokens.len() + 1).min(self.config.maximum_proposal_length);
            } else {
                self.next_proposal_length = outcome
                    .accepted
                    .saturating_add(1)
                    .min(self.config.maximum_proposal_length);
            }
        }
        Ok(output)
    }
}

fn observe_owned_features(
    speculator: &mut dyn fellm_plugin_abi::Speculator,
    token: u32,
    position: usize,
    features: &[(fellm_plugin_abi::TargetFeature, fellm_core::tensor::Tensor)],
) -> FellmResult<()> {
    let captures = features
        .iter()
        .map(|(feature, tensor)| {
            // SAFETY: tensors in `features` own stable storage for this call.
            let view = unsafe {
                fellm_plugin_abi::TensorRef::from_raw(
                    tensor.dtype(),
                    tensor.shape().dims(),
                    tensor.layout().strides.as_slice(),
                    tensor.as_bytes().as_ptr(),
                    tensor.as_bytes().len(),
                )
            };
            fellm_plugin_abi::CapturedTargetFeature::new(
                *feature,
                view,
                fellm_plugin_abi::DeviceKind::Cpu,
            )
        })
        .collect::<Vec<_>>();
    speculator.observe_committed(&fellm_plugin_abi::CommittedTargetToken {
        token,
        position: position as u32,
        captured_features: &captures,
    })
}

impl GenericDraftRuntime {
    pub fn new(target: Engine, draft: Engine, config: GenericDraftConfig) -> FellmResult<Self> {
        validate_tokenizer_compatibility(&target, &draft)?;
        let maximum = config
            .maximum_proposal_length
            .min(target.settings().n_ubatch)
            .max(1);
        let next_proposal_length = config.initial_proposal_length.clamp(1, maximum);
        let minimum_gain = config.minimum_gain;
        Ok(Self {
            target,
            draft,
            config: GenericDraftConfig {
                maximum_proposal_length: maximum,
                ..config
            },
            metrics: SpeculationMetrics::default(),
            next_proposal_length,
            costs: RollingCostModel::new(maximum),
            policy: AdaptiveSpeculationPolicy {
                max_proposal_len: maximum,
                minimum_gain,
            },
        })
    }

    #[must_use]
    pub fn metrics(&self) -> SpeculationMetrics {
        self.metrics
    }

    #[must_use]
    pub fn target(&self) -> &Engine {
        &self.target
    }

    #[must_use]
    pub fn draft(&self) -> &Engine {
        &self.draft
    }

    /// Generate target-distributed tokens from an already-tokenized shared prompt.
    pub fn generate_ids(&mut self, prompt_ids: &[u32], params: GenParams) -> FellmResult<Vec<u32>> {
        let mut target_sequence = self.target.prefill_sequence(prompt_ids)?;
        let mut draft_sequence = match self.draft.prefill_sequence(prompt_ids) {
            Ok(sequence) => sequence,
            Err(error) => {
                self.target.release_sequence(target_sequence);
                return Err(error);
            }
        };
        let result = self.generate_sequences(
            &mut target_sequence,
            &mut draft_sequence,
            prompt_ids,
            params,
        );
        self.target.release_sequence(target_sequence);
        self.draft.release_sequence(draft_sequence);
        result
    }

    fn generate_sequences(
        &mut self,
        target_sequence: &mut DecodeSequence,
        draft_sequence: &mut DecodeSequence,
        prompt_ids: &[u32],
        params: GenParams,
    ) -> FellmResult<Vec<u32>> {
        let stop_tokens = self.target.stop_token_ids_pub();
        let mut sampler_state = crate::sampling::SamplerState::with_grammar(
            params.max_tokens as usize,
            params.grammar.clone(),
        );
        sampler_state.prime_history(prompt_ids);
        let mut verifier = SpeculativeVerifier::new(params.seed ^ 0xa076_1d64_78bd_642f);
        let mut output = Vec::with_capacity(params.max_tokens as usize);
        let mut workspace = crate::sampling::SamplingWorkspace::default();

        while output.len() < params.max_tokens as usize
            && target_sequence.position < self.target.n_ctx()
        {
            let remaining = params.max_tokens as usize - output.len();
            let context_remaining = self.target.n_ctx().saturating_sub(target_sequence.position);
            let hard_limit = self
                .config
                .maximum_proposal_length
                .min(remaining.saturating_sub(1))
                .min(context_remaining.saturating_sub(1));
            let mut proposal_length = schedule_verification_length(
                &self.policy,
                &self.costs,
                &self.metrics,
                hard_limit.min(self.next_proposal_length.max(1)),
                None,
            )
            .min(hard_limit);
            if proposal_length == 0 {
                self.metrics.rounds += 1;
                self.metrics.disabled_rounds += 1;
                self.metrics.record_round(0, 0);
                let token = sample_logits(
                    target_sequence.logits(),
                    &params,
                    &sampler_state,
                    &mut workspace,
                    params.seed,
                )?;
                sampler_state.commit_token(token);
                output.push(token);
                self.metrics.emitted += 1;
                metrics::counter!("fellm_speculative_rounds_total", "k" => "0").increment(1);
                if stop_tokens.contains(&token)
                    || sampler_state.grammar_is_accepting()
                    || output.len() == params.max_tokens as usize
                {
                    break;
                }
                let started = std::time::Instant::now();
                self.target.advance_sequence(target_sequence, token)?;
                self.draft.advance_sequence(draft_sequence, token)?;
                self.metrics.target_forward_passes += 1;
                self.metrics.draft_forward_passes += 1;
                self.metrics.verification_time += started.elapsed();
                // Periodically probe speculation again after a disabled interval.
                if self.metrics.rounds.is_multiple_of(16) {
                    self.next_proposal_length = 1;
                }
                continue;
            }

            let round_started = std::time::Instant::now();
            let draft_started = std::time::Instant::now();
            let mut draft_transaction = self.draft.begin_sequence_transaction(draft_sequence);
            let draft_recurrent_checkpoint = draft_sequence.recurrent.clone();
            let draft_start_position = draft_sequence.position;
            let mut draft_sampler = sampler_state.clone();
            let mut draft_logits = Vec::with_capacity(proposal_length + 1);
            let mut draft_recurrent_prefixes = Vec::with_capacity(proposal_length);
            draft_logits.push(draft_sequence.pending_logits.clone());
            let mut proposal = LinearProposal {
                tokens: Vec::with_capacity(proposal_length),
                prefix_survival: None,
            };
            let mut grammar_terminal_position = None;
            for _ in 0..proposal_length {
                let distribution = distribution_for_logits(
                    draft_sequence.logits(),
                    &params,
                    &draft_sampler,
                    &mut workspace,
                )?;
                let token = sample_processed(&distribution, &params, &draft_sampler);
                proposal.tokens.push(DraftToken {
                    token,
                    distribution: (&distribution).into(),
                });
                draft_sampler.commit_token(token);
                self.draft.advance_sequence(draft_sequence, token)?;
                draft_recurrent_prefixes.push(draft_sequence.recurrent.clone());
                self.metrics.draft_forward_passes += 1;
                draft_logits.push(draft_sequence.pending_logits.clone());
                if draft_sampler.grammar_is_accepting() {
                    grammar_terminal_position = Some(proposal.tokens.len() - 1);
                    break;
                }
                if stop_tokens.contains(&token) {
                    break;
                }
            }
            proposal_length = proposal.tokens.len();
            let draft_duration = draft_started.elapsed();
            self.metrics.draft_time += draft_duration;
            self.metrics.proposed += proposal_length as u64;

            let initial_target_logits = target_sequence.pending_logits.clone();
            let verification_started = std::time::Instant::now();
            let provisional = match self.target.verify_proposal(
                &mut target_sequence.cache,
                &mut target_sequence.recurrent,
                &initial_target_logits,
                &proposal
                    .tokens
                    .iter()
                    .map(|token| token.token)
                    .collect::<Vec<_>>(),
                target_sequence.position,
                &params,
                &sampler_state,
            ) {
                Ok(verification) => verification,
                Err(error) => {
                    self.draft.finalize_sequence_transaction(
                        draft_sequence,
                        &mut draft_transaction,
                        0,
                        draft_recurrent_checkpoint,
                    )?;
                    draft_sequence.pending_logits = draft_logits.remove(0);
                    return Err(error);
                }
            };
            self.metrics.verification_time += verification_started.elapsed();
            self.metrics.target_forward_passes += proposal_length as u64;
            self.metrics.verified += proposal_length as u64;
            let outcome = match verify_linear_proposal(
                &mut verifier,
                &proposal,
                &provisional.distributions,
                &stop_tokens,
                grammar_terminal_position,
                params.is_greedy(),
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.target.finalize_verification(
                        &mut target_sequence.cache,
                        &mut target_sequence.recurrent,
                        provisional,
                        0,
                    )?;
                    self.draft.finalize_sequence_transaction(
                        draft_sequence,
                        &mut draft_transaction,
                        0,
                        draft_recurrent_checkpoint,
                    )?;
                    draft_sequence.pending_logits = draft_logits.remove(0);
                    return Err(FellmError::other(error));
                }
            };
            self.target.finalize_verification(
                &mut target_sequence.cache,
                &mut target_sequence.recurrent,
                provisional,
                outcome.accepted,
            )?;
            target_sequence.position += outcome.accepted;
            self.draft.finalize_sequence_transaction(
                draft_sequence,
                &mut draft_transaction,
                outcome.accepted,
                draft_recurrent_checkpoint.clone(),
            )?;
            draft_sequence.recurrent = if outcome.accepted == 0 {
                draft_recurrent_checkpoint
            } else {
                draft_recurrent_prefixes[outcome.accepted - 1].clone()
            };
            draft_sequence.pending_logits = draft_logits[outcome.accepted].clone();
            draft_sequence.position = draft_start_position + outcome.accepted;

            self.metrics.rounds += 1;
            self.metrics.record_round(proposal_length, outcome.accepted);
            self.metrics.accepted += outcome.accepted as u64;
            self.metrics.emitted += outcome.emitted.len() as u64;
            metrics::counter!("fellm_speculative_draft_tokens_total", "state" => "proposed")
                .increment(proposal_length as u64);
            metrics::counter!("fellm_speculative_draft_tokens_total", "state" => "accepted")
                .increment(outcome.accepted as u64);
            metrics::histogram!("fellm_speculative_accepted_length")
                .record(outcome.accepted as f64);

            for &token in &outcome.emitted {
                if output.len() == params.max_tokens as usize {
                    break;
                }
                sampler_state.commit_token(token);
                output.push(token);
                if sampler_state.grammar_is_accepting() {
                    break;
                }
            }
            if outcome.terminal
                || sampler_state.grammar_is_accepting()
                || output.len() == params.max_tokens as usize
            {
                break;
            }

            let final_token = *outcome.emitted.last().expect("verification always emits");
            let correction_started = std::time::Instant::now();
            self.target.advance_sequence(target_sequence, final_token)?;
            self.draft.advance_sequence(draft_sequence, final_token)?;
            self.metrics.target_forward_passes += 1;
            self.metrics.draft_forward_passes += 1;
            let correction_time = correction_started.elapsed();
            self.metrics.sampling_time += correction_time;
            self.costs
                .observe_round(proposal_length, draft_duration, verification_started.elapsed());
            self.costs.observe_decode(correction_time.as_secs_f64());

            let round_seconds = round_started.elapsed().as_secs_f64();
            let speculative_rate = outcome.emitted.len() as f64 / round_seconds.max(f64::EPSILON);
            let baseline_rate = 1.0 / correction_time.as_secs_f64().max(f64::EPSILON);
            if speculative_rate < baseline_rate * (1.0 + self.config.minimum_gain) {
                self.next_proposal_length = 0;
            } else if outcome.accepted == proposal_length {
                self.next_proposal_length =
                    (proposal_length + 1).min(self.config.maximum_proposal_length);
            } else {
                self.next_proposal_length = outcome
                    .accepted
                    .saturating_add(1)
                    .min(self.config.maximum_proposal_length);
            }
        }
        Ok(output)
    }
}

fn distribution_for_logits(
    logits: &fellm_core::tensor::Tensor,
    params: &GenParams,
    state: &crate::sampling::SamplerState,
    workspace: &mut crate::sampling::SamplingWorkspace,
) -> FellmResult<crate::sampling::ProcessedDistribution> {
    Ok(crate::sampling::distribution_with_workspace(
        logits.as_slice::<f32>()?,
        crate::sampling::SamplingOptions {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            seed: params.seed.wrapping_add(state.draw_index()),
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            logit_bias: &params.logit_bias,
            grammar: state.grammar_view(),
            recent_tokens: state.history(),
        },
        workspace,
    ))
}

fn distribution_for_proposal_scores(
    scores: fellm_plugin_abi::ProposalScores,
    params: &GenParams,
    state: &crate::sampling::SamplerState,
    workspace: &mut crate::sampling::SamplingWorkspace,
) -> FellmResult<crate::sampling::ProcessedDistribution> {
    match scores {
        fellm_plugin_abi::ProposalScores::Logits(logits) => {
            Ok(crate::sampling::distribution_with_workspace(
                &logits,
                crate::sampling::SamplingOptions {
                    temperature: params.temperature,
                    top_k: params.top_k,
                    top_p: params.top_p,
                    min_p: params.min_p,
                    seed: params.seed.wrapping_add(state.draw_index()),
                    repetition_penalty: params.repetition_penalty,
                    frequency_penalty: params.frequency_penalty,
                    presence_penalty: params.presence_penalty,
                    logit_bias: &params.logit_bias,
                    grammar: state.grammar_view(),
                    recent_tokens: state.history(),
                },
                workspace,
            ))
        }
        fellm_plugin_abi::ProposalScores::Probabilities(probabilities) => {
            let logits = probabilities
                .into_iter()
                .map(|probability| {
                    if probability > 0.0 && probability.is_finite() {
                        probability.ln()
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>();
            distribution_for_proposal_scores(
                fellm_plugin_abi::ProposalScores::Logits(logits),
                params,
                state,
                workspace,
            )
        }
    }
}

fn sample_logits(
    logits: &fellm_core::tensor::Tensor,
    params: &GenParams,
    state: &crate::sampling::SamplerState,
    workspace: &mut crate::sampling::SamplingWorkspace,
    seed: u64,
) -> FellmResult<u32> {
    Ok(distribution_for_logits(logits, params, state, workspace)?
        .sample(seed.wrapping_add(state.draw_index())))
}

fn validate_tokenizer_compatibility(target: &Engine, draft: &Engine) -> FellmResult<()> {
    let target_tokenizer = target.tokenizer();
    let draft_tokenizer = draft.tokenizer();
    if target_tokenizer.vocab_size() != draft_tokenizer.vocab_size() {
        return Err(FellmError::other(format!(
            "speculative tokenizer vocabulary mismatch: target={} draft={}",
            target_tokenizer.vocab_size(),
            draft_tokenizer.vocab_size(),
        )));
    }
    if target_tokenizer.bos() != draft_tokenizer.bos()
        || target_tokenizer.eos() != draft_tokenizer.eos()
    {
        return Err(FellmError::other("speculative tokenizer BOS/EOS mismatch"));
    }
    for token in 0..target_tokenizer.vocab_size() as u32 {
        if target_tokenizer.vocabulary_piece(token) != draft_tokenizer.vocabulary_piece(token)
            || target_tokenizer.token_type(token) != draft_tokenizer.token_type(token)
        {
            return Err(FellmError::other(format!(
                "speculative tokenizer mapping mismatch at token {token}"
            )));
        }
    }
    Ok(())
}

fn schedule_verification_length(
    policy: &AdaptiveSpeculationPolicy,
    costs: &RollingCostModel,
    metrics: &SpeculationMetrics,
    limit: usize,
    prefix_survival: Option<&[f32]>,
) -> usize {
    if limit == 0 {
        return 0;
    }
    if costs.baseline_seconds <= 0.0 {
        return limit.min(policy.max_proposal_len);
    }
    let survival = if let Some(values) = prefix_survival {
        values
            .iter()
            .take(limit)
            .map(|&value| f64::from(value).clamp(0.0, 1.0))
            .collect::<Vec<_>>()
    } else {
        let empirical = metrics.prefix_survival(limit);
        if empirical.iter().all(|&value| value <= 0.0) {
            let mut survival = 1.0;
            (0..limit)
                .map(|_| {
                    survival *= 0.75;
                    survival
                })
                .collect()
        } else {
            empirical
        }
    };
    let verify = (0..limit)
        .map(|index| {
            let measured = costs
                .verify_seconds
                .get(index)
                .copied()
                .filter(|value| *value > 0.0)
                .unwrap_or(costs.baseline_seconds * (0.7 + 0.15 * (index + 1) as f64));
            measured
        })
        .collect::<Vec<_>>();
    let transfer = costs
        .transfer_seconds
        .iter()
        .copied()
        .take(limit)
        .chain(std::iter::repeat(0.0))
        .take(limit)
        .collect::<Vec<_>>();
    let baseline_tps = 1.0 / costs.baseline_seconds.max(f64::EPSILON);
    policy
        .choose(
            baseline_tps,
            costs.draft_seconds_per_token.max(f64::EPSILON),
            &verify,
            &survival,
            &transfer,
        )
        .proposal_len
}

fn map_draft_token(
    mapping: &fellm_plugin_abi::VocabularyMapping,
    token: u32,
) -> FellmResult<u32> {
    match mapping {
        fellm_plugin_abi::VocabularyMapping::Identity { vocabulary_size } => {
            if token < *vocabulary_size {
                Ok(token)
            } else {
                Err(FellmError::other("draft token is outside the shared vocabulary"))
            }
        }
        fellm_plugin_abi::VocabularyMapping::DraftToTarget(map) => map
            .get(token as usize)
            .copied()
            .ok_or_else(|| FellmError::other("draft token has no target vocabulary mapping")),
    }
}

fn verify_linear_proposal(
    verifier: &mut SpeculativeVerifier,
    proposal: &LinearProposal,
    target: &[ProbabilityDistribution],
    stop_tokens: &[u32],
    terminal_position: Option<usize>,
    greedy: bool,
) -> Result<VerificationOutcome, &'static str> {
    if greedy {
        let proposed = proposal
            .tokens
            .iter()
            .map(|token| token.token)
            .collect::<Vec<_>>();
        let argmax = target.iter().map(ProbabilityDistribution::argmax).collect::<Vec<_>>();
        Ok(verifier.verify_greedy_with_stops(
            &proposed,
            &argmax,
            stop_tokens,
            terminal_position,
        ))
    } else {
        verifier.verify_with_terminal_position(proposal, target, stop_tokens, terminal_position)
    }
}

fn sample_processed(
    distribution: &crate::sampling::ProcessedDistribution,
    params: &GenParams,
    state: &crate::sampling::SamplerState,
) -> u32 {
    if params.is_greedy() {
        distribution.argmax()
    } else {
        distribution.sample(params.seed ^ 0xe703_7ed1_a0b4_28db ^ state.draw_index())
    }
}

impl AdaptiveSpeculationPolicy {
    #[must_use]
    pub fn choose(
        &self,
        baseline_tokens_per_second: f64,
        draft_seconds_per_token: f64,
        verification_seconds: &[f64],
        prefix_survival: &[f64],
        transfer_seconds: &[f64],
    ) -> SpeculationDecision {
        let mut best = SpeculationDecision {
            proposal_len: 0,
            expected_tokens_per_second: baseline_tokens_per_second,
        };
        let limit = self
            .max_proposal_len
            .min(verification_seconds.len())
            .min(transfer_seconds.len());
        let mut expected_accepted = 0.0;
        for k in 1..=limit {
            expected_accepted += prefix_survival
                .get(k - 1)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let expected_emitted = expected_accepted + 1.0;
            let cost = draft_seconds_per_token * k as f64
                + verification_seconds[k - 1]
                + transfer_seconds[k - 1];
            let throughput = if cost > 0.0 {
                expected_emitted / cost
            } else {
                f64::INFINITY
            };
            if throughput > best.expected_tokens_per_second * (1.0 + self.minimum_gain.max(0.0)) {
                best = SpeculationDecision {
                    proposal_len: k,
                    expected_tokens_per_second: throughput,
                };
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(values: &[f32]) -> ProbabilityDistribution {
        ProbabilityDistribution::from_weights(values.to_vec()).unwrap()
    }

    #[test]
    fn greedy_handles_zero_partial_and_bonus() {
        let verifier = SpeculativeVerifier::new(1);
        assert_eq!(verifier.verify_greedy(&[1, 2], &[0, 2, 3]).accepted, 0);
        let partial = verifier.verify_greedy(&[1, 2], &[1, 0, 3]);
        assert_eq!((partial.accepted, partial.emitted), (1, vec![1, 0]));
        let all = verifier.verify_greedy(&[1, 2], &[1, 2, 3]);
        assert_eq!(
            (all.accepted, all.emitted, all.used_bonus),
            (2, vec![1, 2, 3], true)
        );
    }

    #[test]
    fn equal_distributions_accept_every_proposal() {
        let distribution = d(&[0.2, 0.8]);
        let proposal = LinearProposal {
            tokens: vec![DraftToken {
                token: 1,
                distribution: distribution.clone(),
            }],
            prefix_survival: None,
        };
        for seed in 0..100 {
            let result = SpeculativeVerifier::new(seed)
                .verify(&proposal, &[distribution.clone(), distribution.clone()])
                .unwrap();
            assert_eq!(result.accepted, 1);
            assert!(result.used_bonus);
        }
    }

    #[test]
    fn zero_probability_edges_reject_without_nan_or_invalid_token() {
        let p = d(&[1.0, 0.0]);
        let q = d(&[0.0, 1.0]);
        let proposal = LinearProposal {
            tokens: vec![DraftToken {
                token: 1,
                distribution: q,
            }],
            prefix_survival: None,
        };
        for seed in 0..100 {
            let result = SpeculativeVerifier::new(seed)
                .verify(&proposal, &[p.clone(), p.clone()])
                .unwrap();
            assert_eq!(result.accepted, 0);
            assert_eq!(result.emitted, vec![0]);
        }

        let malformed_q = d(&[1.0, 0.0]);
        let malformed = LinearProposal {
            tokens: vec![DraftToken {
                token: 1,
                distribution: malformed_q,
            }],
            prefix_survival: None,
        };
        let result = SpeculativeVerifier::new(4)
            .verify(&malformed, &[p.clone(), p])
            .unwrap();
        assert_eq!(result.emitted, vec![0]);
    }

    #[test]
    fn accepted_stop_token_suppresses_bonus() {
        let distribution = d(&[0.0, 1.0]);
        let proposal = LinearProposal {
            tokens: vec![DraftToken {
                token: 1,
                distribution: distribution.clone(),
            }],
            prefix_survival: None,
        };
        let result = SpeculativeVerifier::new(1)
            .verify_with_stops(&proposal, &[distribution.clone(), distribution], &[1])
            .unwrap();
        assert_eq!(result.emitted, vec![1]);
        assert_eq!(result.accepted, 1);
        assert!(result.terminal);
        assert!(!result.used_bonus);
    }

    #[test]
    fn rejection_sampling_converges_to_target_distribution() {
        let p = d(&[0.8, 0.2]);
        let q = d(&[0.1, 0.9]);
        let mut counts = [0_u32; 2];
        for seed in 0..50_000 {
            let mut draft_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x55aa);
            let token = q.sample(&mut draft_rng);
            let proposal = LinearProposal {
                tokens: vec![DraftToken {
                    token,
                    distribution: q.clone(),
                }],
                prefix_survival: None,
            };
            let output = SpeculativeVerifier::new(seed)
                .verify(&proposal, &[p.clone(), p.clone()])
                .unwrap();
            counts[output.emitted[0] as usize] += 1;
        }
        let observed = counts[0] as f64 / f64::from(counts.iter().sum::<u32>());
        assert!((observed - 0.8).abs() < 0.015, "observed {observed}");
    }

    #[test]
    fn scheduler_can_disable_unprofitable_speculation() {
        let policy = AdaptiveSpeculationPolicy {
            max_proposal_len: 4,
            minimum_gain: 0.05,
        };
        assert_eq!(
            policy
                .choose(100.0, 0.02, &[0.02; 4], &[0.9; 4], &[0.0; 4])
                .proposal_len,
            0
        );
        assert!(
            policy
                .choose(10.0, 0.001, &[0.01; 4], &[0.9; 4], &[0.0; 4])
                .proposal_len
                > 0
        );
    }

    #[test]
    fn confidence_scheduler_allocates_ragged_prefixes_globally() {
        let profile = VerificationCapacityProfile::new(vec![
            (3, 100.0),
            (4, 95.0),
            (5, 90.0),
            (6, 80.0),
            (7, 65.0),
        ])
        .unwrap();
        let lengths = schedule_confident_prefixes(
            &[vec![0.98, 0.95], vec![0.55, 0.2], vec![0.9, 0.8]],
            &profile,
        );
        assert_eq!(lengths.len(), 3);
        assert!(lengths[0] >= lengths[1]);
        assert!(lengths[2] >= lengths[1]);
    }

    #[test]
    fn confidence_scheduler_stops_before_token_dependent_future_can_bias_admission() {
        let profile =
            VerificationCapacityProfile::new(vec![(1, 1.0), (2, 0.5), (3, 0.45)]).unwrap();
        let high_future = schedule_confident_prefixes(&[vec![0.8, 0.9]], &profile);
        let low_future = schedule_confident_prefixes(&[vec![0.8, 0.0]], &profile);
        assert_eq!(high_future, vec![0]);
        assert_eq!(low_future, vec![0]);
    }

    #[test]
    fn vocabulary_mapping_is_explicit_and_not_identity_only() {
        let mapped = map_draft_token(
            &fellm_plugin_abi::VocabularyMapping::DraftToTarget(vec![7, 9]),
            1,
        )
        .unwrap();
        assert_eq!(mapped, 9);
        assert!(
            map_draft_token(
                &fellm_plugin_abi::VocabularyMapping::Identity {
                    vocabulary_size: 2
                },
                2
            )
            .is_err()
        );
    }

    #[test]
    fn greedy_stop_token_suppresses_bonus() {
        let outcome = SpeculativeVerifier::new(1).verify_greedy_with_stops(
            &[1, 2],
            &[1, 2, 3],
            &[2],
            None,
        );
        assert_eq!(outcome.emitted, vec![1, 2]);
        assert_eq!(outcome.accepted, 2);
        assert!(outcome.terminal);
        assert!(!outcome.used_bonus);
    }

    #[test]
    fn greedy_matches_rejection_sampling_on_point_masses() {
        let p0 = ProbabilityDistribution::point_mass(4, 1);
        let p1 = ProbabilityDistribution::point_mass(4, 2);
        let bonus = ProbabilityDistribution::point_mass(4, 3);
        let proposal = LinearProposal {
            tokens: vec![
                DraftToken {
                    token: 1,
                    distribution: p0.clone(),
                },
                DraftToken {
                    token: 0,
                    distribution: ProbabilityDistribution::point_mass(4, 0),
                },
            ],
            prefix_survival: None,
        };
        let greedy = SpeculativeVerifier::new(1).verify_greedy(&[1, 0], &[1, 2, 3]);
        let stochastic = SpeculativeVerifier::new(1)
            .verify(&proposal, &[p0, p1, bonus])
            .unwrap();
        assert_eq!(greedy.accepted, stochastic.accepted);
        assert_eq!(greedy.emitted, stochastic.emitted);
    }
}
