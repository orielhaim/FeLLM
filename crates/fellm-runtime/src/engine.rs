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
use fellm_tokenizer::{Message, Tokenizer, load as load_tokenizer};
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

    /// Generate tokens from a raw prompt string (completion mode).
    ///
    /// Does **not** apply a chat template. Prefer [`Engine::chat`] for
    /// instruction-tuned models.
    pub fn generate(&mut self, prompt: &str, params: GenParams) -> Result<TokenStream> {
        self.kv.reset();
        let ids = self.tokenizer.encode(prompt, true)?;
        tracing::info!(n_tokens = ids.len(), "prompt tokenized");
        self.generate_from_ids(&ids, params)
    }

    /// Generate a reply to a chat conversation.
    ///
    /// Applies the model's GGUF `tokenizer.chat_template` when present
    /// (Llama 3 / Mistral / Qwen style). Falls back to joining message
    /// contents if the model has no template (base / completion models).
    pub fn chat(&mut self, messages: &[Message], params: GenParams) -> Result<TokenStream> {
        self.kv.reset();
        let prompt = match self.tokenizer.apply_chat_template(messages, true)? {
            Some(formatted) => {
                tracing::debug!(prompt = %formatted, "chat template applied");
                formatted
            }
            None => {
                // No template: concatenate contents (base model / raw completion).
                messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };
        // Template already includes BOS when the model defines one — do not
        // double-prepend. `encode` still dedupes if BOS is present.
        let ids = self.tokenizer.encode(&prompt, true)?;
        tracing::info!(n_tokens = ids.len(), "chat prompt tokenized");
        self.generate_from_ids(&ids, params)
    }

    /// Convenience: single-turn user chat.
    pub fn chat_user(&mut self, user: &str, params: GenParams) -> Result<TokenStream> {
        self.chat(
            &[Message {
                role: "user".into(),
                content: user.into(),
            }],
            params,
        )
    }

    fn generate_from_ids(&mut self, ids: &[u32], params: GenParams) -> Result<TokenStream> {
        if ids.is_empty() {
            return Err(FellmError::other("empty prompt"));
        }
        let stop_token_ids = self.stop_token_ids();
        let mut logits: Option<Tensor> = None;
        for (pos, &tok) in ids.iter().enumerate() {
            logits = Some(self.step(tok, pos, false)?);
        }
        let last_logits = logits.ok_or_else(|| FellmError::other("empty prompt"))?;
        let start_pos = ids.len();
        Ok(TokenStream {
            engine: self,
            params,
            pending_logits: Some(last_logits),
            emitted: 0,
            position: start_pos,
            finished: false,
            stop_token_ids,
        })
    }

    /// Tokens that should end generation (EOS / EOT and friends).
    fn stop_token_ids(&self) -> Vec<u32> {
        let mut stops = Vec::new();
        if let Some(eos) = self.tokenizer.eos() {
            stops.push(eos);
        }
        // Llama 3 Instruct often uses <|eot_id|> as eos already; also stop on
        // <|end_of_text|> if distinct and present in the rendered vocabulary
        // via a second encode of the surface form.
        if let Ok(ids) = self.tokenizer.encode("<|end_of_text|>", false)
            && ids.len() == 1
        {
            let id = ids[0];
            if !stops.contains(&id) {
                stops.push(id);
            }
        }
        if let Ok(ids) = self.tokenizer.encode("<|eot_id|>", false)
            && ids.len() == 1
        {
            let id = ids[0];
            if !stops.contains(&id) {
                stops.push(id);
            }
        }
        stops
    }

    /// One forward step for token id `tok` at position `pos`.
    fn step(&mut self, tok: u32, pos: usize, _final_token: bool) -> Result<Tensor> {
        let graph = self.arch.build_step_graph(&self.gguf, &self.config, pos)?;
        let plan = ExecutionPlan::from_graph(&graph)?;
        let mut exec = GraphExecutor::new(&graph, &plan, &self.backend);

        // Read-only inputs.
        exec.bind_input("token_id", scalar_u32_tensor(tok));
        exec.bind_input("position", scalar_u32_tensor(pos as u32));
        exec.bind_input("past_len", scalar_u32_tensor(pos as u32));

        // Mutable KV cache bindings. The graph's `KvWrite` op mutates these
        // in place at position `pos`, and Attention reads them afterward.
        let dim = self.kv.tokens_stride;
        let shape = Shape::new(&[self.kv.max_seq as u64, dim as u64])?;
        for layer in 0..self.config.n_layers {
            exec.bind_mutable(
                format!("k_in_{layer}"),
                crate::executor::MutableBinding {
                    dtype: DType::F32,
                    shape: shape.clone(),
                    buffer: self.kv.k_buffer(layer),
                },
            );
            exec.bind_mutable(
                format!("v_in_{layer}"),
                crate::executor::MutableBinding {
                    dtype: DType::F32,
                    shape: shape.clone(),
                    buffer: self.kv.v_buffer(layer),
                },
            );
        }

        let outs = exec.run()?;
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
    stop_token_ids: Vec<u32>,
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

        // Check stop tokens (EOS / EOT).
        if self.stop_token_ids.contains(&tok) {
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
