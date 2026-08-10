//! Reusable block-diffusion orchestration and entropy-bounded sampling.
//!
//! The driver is intentionally independent of a model family.  An
//! architecture supplies graph ids and the scheduler supplies graph outputs;
//! this module owns canvas state, temperature, acceptance, renoising,
//! self-conditioning payloads, adaptive stopping, and causal commit requests.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::{
    DriverAction, DriverEvent, GenerationDriver, GenerationRequest, GraphId, InputBinding,
    InputBindings, ModelProgram, StateBindings, TokenBatch,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Defaults recommended by the DiffusionGemma model card.
#[derive(Debug, Clone, Copy)]
pub struct BlockDiffusionConfig {
    /// Fixed canvas length.
    pub canvas_length: usize,
    /// Maximum denoising passes per canvas.
    pub max_denoising_steps: usize,
    /// Initial temperature.
    pub temperature_max: f32,
    /// Final temperature.
    pub temperature_min: f32,
    /// Entropy-bound acceptance budget.
    pub entropy_bound: f32,
    /// Mean entropy threshold for adaptive stopping.
    pub confidence_threshold: f32,
    /// Number of unchanged argmax canvases required for stopping.
    pub stability_threshold: usize,
    /// Number of logits retained for the next self-conditioning pass.
    /// `0` keeps the exact dense distribution.
    pub self_conditioning_top_k: usize,
}

impl Default for BlockDiffusionConfig {
    fn default() -> Self {
        Self {
            canvas_length: 256,
            max_denoising_steps: 48,
            temperature_max: 0.8,
            temperature_min: 0.4,
            entropy_bound: 0.1,
            confidence_threshold: 0.005,
            stability_threshold: 1,
            self_conditioning_top_k: 0,
        }
    }
}

/// Result of one sampler update.
#[derive(Debug, Clone, Default)]
pub struct SamplerStep {
    /// Updated canvas ids.
    pub canvas: Vec<u32>,
    /// Clean per-position argmax ids before entropy acceptance/renoising.
    /// These are the tokens committed when a denoising block finishes.
    pub argmax: Vec<u32>,
    /// Accepted positions.
    pub accepted: Vec<bool>,
    /// Per-position entropy.
    pub entropy: Vec<f32>,
    /// Mean entropy.
    pub mean_entropy: f32,
    /// Whether adaptive stopping conditions are satisfied.
    pub done: bool,
}

/// Correctness-first entropy-bounded discrete diffusion sampler.
pub struct EntropyBoundSampler {
    config: BlockDiffusionConfig,
    previous_argmax: Vec<u32>,
    stable_steps: usize,
    rng: ChaCha8Rng,
    /// Reused probability scratch; avoids one vocabulary-sized allocation per row.
    weights: Vec<f32>,
    positions: Vec<usize>,
}

impl EntropyBoundSampler {
    /// Create a deterministic sampler.
    #[must_use]
    pub fn new(config: BlockDiffusionConfig, seed: u64) -> Self {
        Self {
            config,
            previous_argmax: Vec::new(),
            stable_steps: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            weights: Vec::new(),
            positions: Vec::new(),
        }
    }

    /// Initialize a canvas from the uniform vocabulary state.
    pub fn initialize_canvas(&mut self, vocab_size: usize) -> Result<Vec<u32>> {
        if vocab_size == 0 || self.config.canvas_length == 0 {
            return Err(FellmError::other(
                "diffusion sampler has invalid canvas/vocabulary size",
            ));
        }
        self.previous_argmax.clear();
        self.stable_steps = 0;
        Ok((0..self.config.canvas_length)
            .map(|_| self.rng.random_range(0..vocab_size) as u32)
            .collect())
    }

    /// Apply temperature, sample every position, accept low-entropy positions,
    /// and uniformly renoise all rejected positions.
    pub fn step(
        &mut self,
        current: &[u32],
        logits: &[f32],
        vocab_size: usize,
        step: usize,
    ) -> Result<SamplerStep> {
        let mut output = SamplerStep::default();
        self.step_into(current, logits, vocab_size, step, &mut output)?;
        Ok(output)
    }

    /// Update a caller-owned result, reusing all canvas-sized scratch buffers.
    pub fn step_into(
        &mut self,
        current: &[u32],
        logits: &[f32],
        vocab_size: usize,
        step: usize,
        output: &mut SamplerStep,
    ) -> Result<()> {
        let n = self.config.canvas_length;
        if current.len() != n || logits.len() != n.saturating_mul(vocab_size) || vocab_size == 0 {
            return Err(FellmError::other(format!(
                "diffusion sampler shape mismatch: canvas={} logits={} expected={}x{}",
                current.len(),
                logits.len(),
                n,
                vocab_size
            )));
        }
        let temperature = self.temperature(step);
        output.argmax.resize(n, 0);
        output.entropy.resize(n, 0.0);
        for row in 0..n {
            let slice = &logits[row * vocab_size..(row + 1) * vocab_size];
            let (candidate, entropy) =
                sample_row(slice, temperature, &mut self.rng, &mut self.weights);
            output.argmax[row] = candidate;
            output.entropy[row] = entropy;
        }

        self.positions.clear();
        self.positions.extend(0..n);
        self.positions.sort_unstable_by(|&a, &b| {
            output.entropy[a]
                .total_cmp(&output.entropy[b])
                .then(a.cmp(&b))
        });
        output.accepted.clear();
        output.accepted.resize(n, false);
        let mut cumulative = 0.0f32;
        let mut max_entropy = 0.0f32;
        for &position in &self.positions {
            cumulative += output.entropy[position];
            max_entropy = max_entropy.max(output.entropy[position]);
            if cumulative - max_entropy <= self.config.entropy_bound {
                output.accepted[position] = true;
            }
        }

        output.canvas.clear();
        output.canvas.extend_from_slice(current);
        for (i, accepted) in output.accepted.iter().copied().enumerate() {
            output.canvas[i] = if accepted {
                output.argmax[i]
            } else {
                self.rng.random_range(0..vocab_size) as u32
            };
        }
        let same = self.previous_argmax == output.argmax;
        self.stable_steps = if same { self.stable_steps + 1 } else { 0 };
        self.previous_argmax.clear();
        self.previous_argmax.extend_from_slice(&output.argmax);
        output.mean_entropy = output.entropy.iter().sum::<f32>() / n.max(1) as f32;
        output.done = output.mean_entropy <= self.config.confidence_threshold
            && self.stable_steps >= self.config.stability_threshold.saturating_add(1);
        Ok(())
    }

    /// Linear temperature schedule, matching the HF implementation.
    #[must_use]
    pub fn temperature(&self, step: usize) -> f32 {
        let denominator = self.config.max_denoising_steps.max(1) as f32;
        let fraction = (step as f32 / denominator).clamp(0.0, 1.0);
        self.config.temperature_min
            + (self.config.temperature_max - self.config.temperature_min) * fraction
    }
}

fn sample_row(
    logits: &[f32],
    temperature: f32,
    rng: &mut ChaCha8Rng,
    weights: &mut Vec<f32>,
) -> (u32, f32) {
    let inv_temperature = 1.0 / temperature.max(f32::EPSILON);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    weights.clear();
    weights.extend(logits.iter().map(|v| ((*v - max) * inv_temperature).exp()));
    let sum = weights.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
        return (best as u32, 0.0);
    }
    for weight in weights.iter_mut() {
        *weight /= sum;
    }
    let entropy = weights
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    let needle = rng.random::<f32>();
    let mut cumulative = 0.0f32;
    for (i, probability) in weights.iter().copied().enumerate() {
        cumulative += probability;
        if needle <= cumulative {
            return (i as u32, entropy);
        }
    }
    ((weights.len() - 1) as u32, entropy)
}

/// Graph ids used by the generic block driver.
#[derive(Debug, Clone, Copy)]
pub struct BlockDiffusionGraphs {
    /// Causal prompt/incremental prefill graph.
    pub prefill: GraphId,
    /// Bidirectional canvas denoising graph.
    pub denoise: GraphId,
    /// Causal finalized-canvas commit graph.
    pub commit: GraphId,
}

/// Reusable block-diffusion generation driver.
pub struct BlockDiffusionDriver {
    config: BlockDiffusionConfig,
    graphs: BlockDiffusionGraphs,
    request: GenerationRequest,
    sampler: EntropyBoundSampler,
    sampler_step: SamplerStep,
    vocab_size: usize,
    canvas: Vec<u32>,
    emitted: Vec<u32>,
    commit_canvas: Vec<u32>,
    self_conditioning_logits: Vec<f32>,
    self_conditioning_indices: Vec<usize>,
    denoise_step: usize,
    awaiting_commit: bool,
    done: bool,
}

impl BlockDiffusionDriver {
    /// Construct a driver from a compiled multi-graph program.
    pub fn new(
        program: &ModelProgram,
        request: GenerationRequest,
        config: BlockDiffusionConfig,
        graphs: BlockDiffusionGraphs,
        vocab_size: usize,
    ) -> Result<Self> {
        if program
            .graphs
            .iter()
            .all(|graph| graph.id != graphs.prefill)
            || program
                .graphs
                .iter()
                .all(|graph| graph.id != graphs.denoise)
            || program.graphs.iter().all(|graph| graph.id != graphs.commit)
        {
            return Err(FellmError::other(
                "block-diffusion program is missing a required graph",
            ));
        }
        let mut sampler = EntropyBoundSampler::new(config, request.seed);
        let canvas = sampler.initialize_canvas(vocab_size)?;
        Ok(Self {
            config,
            graphs,
            request,
            sampler,
            sampler_step: SamplerStep::default(),
            vocab_size,
            canvas,
            emitted: Vec::new(),
            commit_canvas: Vec::new(),
            self_conditioning_logits: vec![
                0.0;
                config.canvas_length
                    * if config.self_conditioning_top_k == 0 {
                        vocab_size
                    } else {
                        config.self_conditioning_top_k.min(vocab_size) * 2
                    }
            ],
            self_conditioning_indices: Vec::with_capacity(vocab_size),
            denoise_step: 0,
            awaiting_commit: false,
            done: false,
        })
    }

    /// Current canvas, useful for diagnostics and device binding.
    #[must_use]
    pub fn canvas(&self) -> &[u32] {
        &self.canvas
    }

    /// Number of finalized tokens.
    #[must_use]
    pub fn emitted_tokens(&self) -> &[u32] {
        &self.emitted
    }
}

impl GenerationDriver for BlockDiffusionDriver {
    fn next_action(&mut self, event: DriverEvent) -> Result<DriverAction> {
        if self.done {
            return Ok(DriverAction::Done);
        }
        match event {
            DriverEvent::Started => Ok(DriverAction::InvokeGraph {
                graph: self.graphs.prefill,
                inputs: InputBindings {
                    inputs: vec![InputBinding {
                        name: "prompt_tokens".into(),
                        values: self.request.prompt.clone(),
                        float_values: Vec::new(),
                    }],
                },
                state: StateBindings {
                    states: vec![fellm_plugin_abi::StateBinding {
                        name: "prompt_kv_cache".into(),
                    }],
                },
            }),
            DriverEvent::GraphCompleted { graph, outputs: _ } if graph == self.graphs.prefill => {
                Ok(DriverAction::InvokeGraph {
                    graph: self.graphs.denoise,
                    inputs: InputBindings {
                        inputs: vec![InputBinding {
                            name: "canvas_tokens".into(),
                            values: self.canvas.clone(),
                            float_values: self.self_conditioning_logits.clone(),
                        }],
                    },
                    state: StateBindings {
                        states: vec![fellm_plugin_abi::StateBinding {
                            name: "prompt_kv_cache_read_only".into(),
                        }],
                    },
                })
            }
            DriverEvent::GraphCompleted { graph, outputs } if graph == self.graphs.denoise => {
                let output = outputs
                    .iter()
                    .find(|output| output.name == "logits")
                    .ok_or_else(|| FellmError::other("denoising graph did not return logits"))?;
                self.sampler.step_into(
                    &self.canvas,
                    &output.values,
                    output.cols,
                    self.denoise_step,
                    &mut self.sampler_step,
                )?;
                std::mem::swap(&mut self.canvas, &mut self.sampler_step.canvas);
                if self.config.self_conditioning_top_k == 0 {
                    self.self_conditioning_logits.clear();
                    self.self_conditioning_logits
                        .extend_from_slice(&output.values);
                } else {
                    sparse_self_conditioning_pairs_into(
                        &output.values,
                        self.sampler_step.argmax.len(),
                        output.cols,
                        self.config.self_conditioning_top_k,
                        &mut self.self_conditioning_logits,
                        &mut self.self_conditioning_indices,
                    )?;
                }
                self.denoise_step += 1;
                if self.sampler_step.done || self.denoise_step >= self.config.max_denoising_steps {
                    let remaining = self
                        .request
                        .max_tokens
                        .saturating_sub(self.emitted.len() as u32)
                        as usize;
                    let finalized = std::mem::take(&mut self.sampler_step.argmax);
                    let take = finalized.len().min(remaining);
                    let batch = finalized[..take].to_vec();
                    self.emitted.extend_from_slice(&batch);
                    self.commit_canvas = finalized;
                    self.awaiting_commit = true;
                    Ok(DriverAction::Emit(TokenBatch {
                        token_ids: batch,
                        commit_token_ids: self.commit_canvas.clone(),
                    }))
                } else {
                    Ok(DriverAction::InvokeGraph {
                        graph: self.graphs.denoise,
                        inputs: InputBindings {
                            inputs: vec![InputBinding {
                                name: "canvas_tokens".into(),
                                values: self.canvas.clone(),
                                float_values: self.self_conditioning_logits.clone(),
                            }],
                        },
                        state: StateBindings {
                            states: vec![
                                fellm_plugin_abi::StateBinding {
                                    name: "prompt_kv_cache_read_only".into(),
                                },
                                fellm_plugin_abi::StateBinding {
                                    name: "self_conditioning_logits".into(),
                                },
                            ],
                        },
                    })
                }
            }
            DriverEvent::CacheCommitted { .. } if self.awaiting_commit => {
                self.awaiting_commit = false;
                if self.emitted.len() as u32 >= self.request.max_tokens {
                    self.done = true;
                    Ok(DriverAction::Done)
                } else {
                    self.denoise_step = 0;
                    Ok(DriverAction::InvokeGraph {
                        graph: self.graphs.commit,
                        inputs: InputBindings {
                            inputs: vec![InputBinding {
                                name: "finalized_canvas".into(),
                                values: self.commit_canvas.clone(),
                                float_values: Vec::new(),
                            }],
                        },
                        state: StateBindings {
                            states: vec![fellm_plugin_abi::StateBinding {
                                name: "prompt_kv_cache_append".into(),
                            }],
                        },
                    })
                }
            }
            DriverEvent::GraphCompleted { graph, outputs: _ } if graph == self.graphs.commit => {
                self.canvas = self.sampler.initialize_canvas(self.vocab_size)?;
                self.self_conditioning_logits.fill(0.0);
                Ok(DriverAction::InvokeGraph {
                    graph: self.graphs.denoise,
                    inputs: InputBindings {
                        inputs: vec![InputBinding {
                            name: "canvas_tokens".into(),
                            values: self.canvas.clone(),
                            float_values: self.self_conditioning_logits.clone(),
                        }],
                    },
                    state: StateBindings {
                        states: vec![fellm_plugin_abi::StateBinding {
                            name: "prompt_kv_cache_read_only".into(),
                        }],
                    },
                })
            }
            DriverEvent::Cancelled => {
                self.done = true;
                Ok(DriverAction::Done)
            }
            _ => Err(FellmError::other(
                "unexpected event for block-diffusion driver",
            )),
        }
    }
}

#[cfg(test)]
fn sparse_self_conditioning_logits(
    logits: &[f32],
    rows: usize,
    vocab: usize,
    top_k: usize,
) -> Result<Vec<f32>> {
    if logits.len() != rows.saturating_mul(vocab) || vocab == 0 {
        return Err(FellmError::other("sparse self-conditioning shape mismatch"));
    }
    let keep = top_k.min(vocab);
    if keep == 0 {
        return Ok(vec![0.0; logits.len()]);
    }
    let mut sparse = vec![f32::NEG_INFINITY; logits.len()];
    for (row, source) in logits.chunks_exact(vocab).enumerate() {
        let mut indices: Vec<usize> = (0..vocab).collect();
        if keep < vocab {
            indices.select_nth_unstable_by(keep - 1, |&a, &b| {
                source[b].total_cmp(&source[a]).then(a.cmp(&b))
            });
        }
        for &token in &indices[..keep] {
            sparse[row * vocab + token] = source[token];
        }
    }
    Ok(sparse)
}

/// Reuse storage while packing retained `(token_id, logit)` pairs.
fn sparse_self_conditioning_pairs_into(
    logits: &[f32],
    rows: usize,
    vocab: usize,
    top_k: usize,
    packed: &mut Vec<f32>,
    indices: &mut Vec<usize>,
) -> Result<()> {
    if logits.len() != rows.saturating_mul(vocab) || vocab == 0 {
        return Err(FellmError::other("sparse self-conditioning shape mismatch"));
    }
    let keep = top_k.min(vocab);
    if keep == 0 {
        packed.clear();
        packed.resize(logits.len(), 0.0);
        return Ok(());
    }
    packed.clear();
    packed.resize(rows * keep * 2, 0.0);
    for (row, source) in logits.chunks_exact(vocab).enumerate() {
        indices.clear();
        indices.extend(0..vocab);
        if keep < vocab {
            indices.select_nth_unstable_by(keep - 1, |&a, &b| {
                source[b].total_cmp(&source[a]).then(a.cmp(&b))
            });
        }
        for (slot, &token) in indices[..keep].iter().enumerate() {
            let dst = (row * keep + slot) * 2;
            packed[dst] = token as f32;
            packed[dst + 1] = source[token];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_plugin_abi::{GraphOutput, GraphSpec};

    #[test]
    fn seeded_uniform_canvas_is_reproducible() {
        let config = BlockDiffusionConfig {
            canvas_length: 8,
            ..Default::default()
        };
        let a = EntropyBoundSampler::new(config, 7);
        let b = EntropyBoundSampler::new(config, 7);
        let mut a = a;
        let mut b = b;
        assert_eq!(
            a.initialize_canvas(32).unwrap(),
            b.initialize_canvas(32).unwrap()
        );
    }

    #[test]
    fn entropy_bound_accepts_low_entropy_positions_and_renoises_others() {
        let config = BlockDiffusionConfig {
            canvas_length: 2,
            entropy_bound: 0.01,
            ..Default::default()
        };
        let mut sampler = EntropyBoundSampler::new(config, 3);
        let current = sampler.initialize_canvas(4).unwrap();
        let logits = vec![20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sampler.step(&current, &logits, 4, 0).unwrap();
        assert!(result.accepted.iter().any(|v| *v));
        assert!(result.entropy.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_self_conditioning_keeps_only_top_logits() {
        let logits = vec![1.0, 4.0, 2.0, 3.0, 8.0, 7.0, 6.0, 5.0];
        let sparse = sparse_self_conditioning_logits(&logits, 2, 4, 2).unwrap();
        assert_eq!(sparse[1], 4.0);
        assert_eq!(sparse[3], 3.0);
        assert!(sparse[0].is_infinite() && sparse[0].is_sign_negative());
        assert_eq!(sparse[4], 8.0);
        assert_eq!(sparse[5], 7.0);
        assert!(sparse[6].is_infinite() && sparse[6].is_sign_negative());
    }

    #[test]
    fn driver_carries_self_conditioning_and_resets_after_commit() {
        let config = BlockDiffusionConfig {
            canvas_length: 2,
            max_denoising_steps: 2,
            self_conditioning_top_k: 2,
            ..Default::default()
        };
        let program = ModelProgram {
            architecture_id: "test".into(),
            graphs: vec![
                GraphSpec {
                    id: GraphId(0),
                    name: "prefill".into(),
                },
                GraphSpec {
                    id: GraphId(1),
                    name: "denoise".into(),
                },
                GraphSpec {
                    id: GraphId(2),
                    name: "commit".into(),
                },
            ],
        };
        let request = GenerationRequest {
            prompt: vec![11, 12],
            max_tokens: 4,
            seed: 9,
        };
        let mut driver = BlockDiffusionDriver::new(
            &program,
            request,
            config,
            BlockDiffusionGraphs {
                prefill: GraphId(0),
                denoise: GraphId(1),
                commit: GraphId(2),
            },
            4,
        )
        .unwrap();

        let prefill = driver.next_action(DriverEvent::Started).unwrap();
        assert!(matches!(
            prefill,
            DriverAction::InvokeGraph {
                graph: GraphId(0),
                ..
            }
        ));
        let denoise = driver
            .next_action(DriverEvent::GraphCompleted {
                graph: GraphId(0),
                outputs: Vec::new(),
            })
            .unwrap();
        let first_float_values = match denoise {
            DriverAction::InvokeGraph { inputs, .. } => inputs.inputs[0].float_values.clone(),
            _ => panic!("prefill did not transition to denoise"),
        };
        assert!(first_float_values.iter().all(|value| *value == 0.0));

        let logits = vec![4.0, 1.0, 3.0, 2.0, 1.0, 3.0, 4.0, 2.0];
        let second_denoise = driver
            .next_action(DriverEvent::GraphCompleted {
                graph: GraphId(1),
                outputs: vec![GraphOutput {
                    name: "logits".into(),
                    values: logits.clone(),
                    rows: 2,
                    cols: 4,
                }],
            })
            .unwrap();
        let second_float_values = match second_denoise {
            DriverAction::InvokeGraph { inputs, .. } => inputs.inputs[0].float_values.clone(),
            _ => panic!("first denoise did not schedule a second denoise"),
        };
        assert_eq!(second_float_values.len(), 2 * config.canvas_length * 2);
        assert!(second_float_values.iter().all(|value| value.is_finite()));
        assert_eq!(second_float_values[1], 4.0);

        let emit = driver
            .next_action(DriverEvent::GraphCompleted {
                graph: GraphId(1),
                outputs: vec![GraphOutput {
                    name: "logits".into(),
                    values: logits,
                    rows: 2,
                    cols: 4,
                }],
            })
            .unwrap();
        let commit_count = match emit {
            DriverAction::Emit(batch) => {
                assert_eq!(batch.commit_token_ids.len(), 2);
                batch.commit_token_ids.len()
            }
            _ => panic!("second denoise did not emit"),
        };
        let commit = driver
            .next_action(DriverEvent::CacheCommitted {
                token_count: commit_count,
            })
            .unwrap();
        assert!(matches!(
            commit,
            DriverAction::InvokeGraph {
                graph: GraphId(2),
                ..
            }
        ));

        let next_canvas = driver
            .next_action(DriverEvent::GraphCompleted {
                graph: GraphId(2),
                outputs: Vec::new(),
            })
            .unwrap();
        match next_canvas {
            DriverAction::InvokeGraph {
                graph: GraphId(1),
                inputs,
                ..
            } => {
                assert!(
                    inputs.inputs[0]
                        .float_values
                        .iter()
                        .all(|value| *value == 0.0)
                );
            }
            _ => panic!("commit did not start a fresh canvas"),
        }
    }
}
