//! The Engine: the top-level user-facing API.

use crate::executor::GraphExecutor;
use crate::kv_cache::KvCache;
use arch_llama::{LlamaArch, LlamaConfig};
use backend_cpu::CpuBackend;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_gguf::GgufFile;
use fellm_graph::plan::ExecutionPlan;
use fellm_plugin_abi::op::OpAttrs;
use fellm_tokenizer::{Tokenizer, load as load_tokenizer};
use std::path::Path;
use std::sync::Arc;

/// Sampling parameters.
#[derive(Debug, Clone, Copy)]
pub struct GenParams {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Softmax temperature (0.0 = greedy).
    pub temperature: f32,
    /// top-k (0 disables).
    pub top_k: u32,
    /// top-p / nucleus (>= 1.0 disables).
    pub top_p: f32,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// Builder for [`Engine`].
pub struct EngineBuilder {
    model_path: Option<String>,
    max_seq: Option<usize>,
}

impl EngineBuilder {
    /// New builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_path: None,
            max_seq: None,
        }
    }

    /// Set the model path.
    #[must_use]
    pub fn model(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Override max sequence length.
    #[must_use]
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    /// Finalize.
    pub fn build(self) -> Result<Engine> {
        let path = self
            .model_path
            .ok_or_else(|| FellmError::other("no model path"))?;
        Engine::open(Path::new(&path), self.max_seq)
    }
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The single-request inference engine.
pub struct Engine {
    #[allow(dead_code)]
    gguf: Arc<GgufFile>,
    tokenizer: Box<dyn Tokenizer>,
    arch: LlamaArch,
    backend: CpuBackend,
    config: LlamaConfig,
    kv: KvCache,
    #[allow(dead_code)]
    max_seq: usize,
}

impl Engine {
    /// Open a GGUF model file.
    pub fn open(path: &Path, max_seq_override: Option<usize>) -> Result<Self> {
        tracing::info!(path = ?path, "opening GGUF");
        let gguf = Arc::new(GgufFile::open(path)?);
        let tokenizer = load_tokenizer(&gguf)?;

        let arch_id = gguf.metadata.arch()?;
        tracing::info!(arch = arch_id, "detected architecture");
        if arch_id != "llama" {
            return Err(FellmError::UnsupportedArchitecture(arch_id.into()));
        }

        let arch = LlamaArch::new();
        let config = arch.config_from_gguf(&gguf)?;
        let max_seq = max_seq_override.unwrap_or(config.context_length.min(8192));

        let kv = KvCache::new(
            config.n_layers,
            max_seq,
            config.n_kv_heads,
            config.head_dim(),
        )?;

        Ok(Self {
            gguf,
            tokenizer,
            arch,
            backend: CpuBackend::new(),
            config,
            kv,
            max_seq,
        })
    }

    /// Tokenizer reference.
    #[must_use]
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Config reference.
    #[must_use]
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Generate tokens.
    pub fn generate(&mut self, prompt: &str, params: GenParams) -> Result<TokenStream> {
        let ids = self.tokenizer.encode(prompt, true)?;
        tracing::info!(n_tokens = ids.len(), "prompt tokenized");

        // Prefill: run each prompt token through the graph, appending to KV.
        let mut logits: Option<Tensor> = None;
        for (pos, &tok) in ids.iter().enumerate() {
            logits = Some(self.step(tok, pos, false)?);
        }

        // Prepare the stream state.
        let last_logits = logits.ok_or_else(|| FellmError::other("empty prompt"))?;
        let start_pos = ids.len();
        Ok(TokenStream {
            engine: self,
            params,
            pending_logits: Some(last_logits),
            emitted: 0,
            position: start_pos,
            finished: false,
        })
    }

    /// One forward step for token id `tok` at position `pos`.
    ///
    /// Returns the logits tensor for the next token.
    fn step(&mut self, tok: u32, pos: usize, _final_token: bool) -> Result<Tensor> {
        // Build a single-token graph specialized to this position.
        let graph = self
            .arch
            .build_step_graph(&self.gguf, &self.config, pos)?;
        let plan = ExecutionPlan::from_graph(&graph)?;
        let executor = GraphExecutor::new(&graph, &plan, &self.backend);

        // Wire the token id input.
        let tok_tensor = scalar_u32_tensor(tok);
        let mut exec = executor;
        exec.bind_input("token_id", tok_tensor);

        // Bind KV-cache tensors (K and V per layer as inputs; the graph writes
        // updated versions to outputs "k_layer_i" and "v_layer_i").
        for layer in 0..self.config.n_layers {
            let k_bytes = self.kv.layers_k_bytes(layer);
            let v_bytes = self.kv.layers_v_bytes(layer);
            let dim = self.kv.tokens_stride;
            let shape = Shape::new(&[self.kv.max_seq as u64, dim as u64])?;
            exec.bind_input(format!("k_in_{layer}"), tensor_from_bytes(k_bytes, DType::F32, shape.clone()));
            exec.bind_input(format!("v_in_{layer}"), tensor_from_bytes(v_bytes, DType::F32, shape));
        }
        // Also bind current position and past_len.
        exec.bind_input(
            "position",
            scalar_u32_tensor(pos as u32),
        );
        exec.bind_input("past_len", scalar_u32_tensor(pos as u32));

        let outs = exec.run()?;

        // Copy new K/V rows for this layer back into the cache.
        for layer in 0..self.config.n_layers {
            if let (Some(k_new), Some(v_new)) = (
                outs.get(&format!("k_out_{layer}")),
                outs.get(&format!("v_out_{layer}")),
            ) {
                let k_slice: &[f32] = k_new.as_slice()?;
                let v_slice: &[f32] = v_new.as_slice()?;
                self.kv.append(layer, k_slice, v_slice, pos);
            }
        }
        self.kv.advance();

        outs.get("logits")
            .cloned()
            .ok_or_else(|| FellmError::other("no logits output"))
    }
}

/// A streaming generator.
pub struct TokenStream<'a> {
    engine: &'a mut Engine,
    params: GenParams,
    pending_logits: Option<Tensor>,
    emitted: u32,
    position: usize,
    finished: bool,
}

impl<'a> TokenStream<'a> {
    /// Decode a token id to bytes.
    pub fn decode_token(&self, id: u32) -> Result<Vec<u8>> {
        self.engine.tokenizer.decode_token(id)
    }
}

impl<'a> Iterator for TokenStream<'a> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.emitted >= self.params.max_tokens {
            return None;
        }
        let logits_tensor = self.pending_logits.take()?;
        // Sample.
        let logits: &[f32] = match logits_tensor.as_slice::<f32>() {
            Ok(s) => s,
            Err(e) => {
                self.finished = true;
                return Some(Err(e));
            }
        };
        let mut work = logits.to_vec();
        let tok = backend_cpu::kernels::sampling::sample(
            &mut work,
            self.params.temperature,
            self.params.top_k,
            self.params.top_p,
            self.params.seed.wrapping_add(u64::from(self.emitted)),
        );
        self.emitted += 1;

        // Check EOS.
        if let Some(eos) = self.engine.tokenizer.eos()
            && tok == eos
        {
            self.finished = true;
            return Some(Ok(tok));
        }

        // If we have room, step once more to prepare next logits.
        if self.emitted < self.params.max_tokens && self.position + 1 < self.engine.kv.max_seq {
            match self.engine.step(tok, self.position, false) {
                Ok(next) => {
                    self.pending_logits = Some(next);
                    self.position += 1;
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        } else {
            self.finished = true;
        }
        Some(Ok(tok))
    }
}

// ---- small helpers ----

fn scalar_u32_tensor(v: u32) -> Tensor {
    let mut buf = AlignedBuffer::new_zeroed(4, 4);
    buf.as_mut_slice().copy_from_slice(&v.to_le_bytes());
    let layout = Layout::contiguous(DType::U32, Shape::new(&[1]).expect("valid"));
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    Tensor::from_storage(layout, storage)
}

fn tensor_from_bytes(bytes: &[u8], dtype: DType, shape: Shape) -> Tensor {
    let mut buf = AlignedBuffer::new_zeroed(bytes.len(), 64);
    buf.as_mut_slice().copy_from_slice(bytes);
    let layout = Layout::contiguous(dtype, shape);
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    Tensor::from_storage(layout, storage)
}

// Small extension trait so we can grab raw KV layer bytes.
trait KvBytes {
    fn layers_k_bytes(&self, layer: usize) -> &[u8];
    fn layers_v_bytes(&self, layer: usize) -> &[u8];
}

impl KvBytes for KvCache {
    fn layers_k_bytes(&self, layer: usize) -> &[u8] {
        // SAFETY: layer < n_layers; buffer alive for &self.
        // We reinterpret the &[f32] as &[u8] via bytemuck.
        let f32s: &[f32] = self.k(layer);
        bytemuck::cast_slice(f32s)
    }
    fn layers_v_bytes(&self, layer: usize) -> &[u8] {
        let f32s: &[f32] = self.v(layer);
        bytemuck::cast_slice(f32s)
    }
}
