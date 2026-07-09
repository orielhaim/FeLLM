//! The Engine: the top-level user-facing API.

use crate::executor::GraphExecutor;
use crate::kv_cache::KvCache;
use arch_llama::parse_assistant_output;
use arch_llama::{LlamaArch, LlamaConfig, PositionNodes};
use backend_cpu::CpuBackend;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_gguf::GgufFile;
use fellm_graph::Graph;
use fellm_graph::plan::ExecutionPlan;
use fellm_tokenizer::{AssistantOutput, Message, Tokenizer, ToolDef, load as load_tokenizer};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default context size when the caller does not override
pub const DEFAULT_CTX_SIZE: usize = 8192;
/// Default evaluation batch size (`n_batch`): max prompt tokens scheduled per
/// prompt-processing pass.
pub const DEFAULT_BATCH_SIZE: usize = 2048;
/// Default physical batch size (`n_ubatch`): max prompt tokens processed in
/// one compute chunk.
pub const DEFAULT_UBATCH_SIZE: usize = 512;

/// Engine load / runtime settings (mirrors llama.cpp context params).
#[derive(Debug, Clone, Copy)]
pub struct EngineSettings {
    /// Context length (`n_ctx`). `None` → default [`DEFAULT_CTX_SIZE`].
    /// Use [`EngineSettings::ctx_from_model`] / CLI `0` to take the GGUF max.
    pub n_ctx: Option<usize>,
    /// When true, `n_ctx` is the model's GGUF `context_length` (auto-detect).
    pub n_ctx_from_model: bool,
    /// Evaluation batch size (`n_batch`): max prompt tokens to schedule during
    /// prompt processing.
    pub n_batch: usize,
    /// Physical batch size (`n_ubatch`): max prompt tokens per compute chunk.
    /// Must be `<= n_batch`.
    pub n_ubatch: usize,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            n_ctx: Some(DEFAULT_CTX_SIZE),
            n_ctx_from_model: false,
            n_batch: DEFAULT_BATCH_SIZE,
            n_ubatch: DEFAULT_UBATCH_SIZE,
        }
    }
}

impl EngineSettings {
    /// Use the model's GGUF-reported maximum context length.
    #[must_use]
    pub fn ctx_from_model(mut self) -> Self {
        self.n_ctx = None;
        self.n_ctx_from_model = true;
        self
    }

    /// Set an explicit context size (clamped to the model max at open).
    #[must_use]
    pub fn ctx_size(mut self, n: usize) -> Self {
        self.n_ctx = Some(n);
        self.n_ctx_from_model = false;
        self
    }

    /// Set evaluation batch size.
    #[must_use]
    pub fn batch_size(mut self, n: usize) -> Self {
        self.n_batch = n.max(1);
        self
    }

    /// Set physical batch size.
    #[must_use]
    pub fn ubatch_size(mut self, n: usize) -> Self {
        self.n_ubatch = n.max(1);
        self
    }

    /// Resolve `n_ctx` given the model's maximum from GGUF metadata.
    #[must_use]
    pub fn resolve_n_ctx(self, model_max: usize) -> usize {
        let model_max = model_max.max(1);
        if self.n_ctx_from_model {
            return model_max;
        }
        let requested = self.n_ctx.unwrap_or(DEFAULT_CTX_SIZE).max(1);
        requested.min(model_max)
    }

    /// Physical chunk size used during prompt processing.
    #[must_use]
    pub fn resolve_ubatch(self) -> usize {
        let n_batch = self.n_batch.max(1);
        self.n_ubatch.max(1).min(n_batch)
    }
}

/// Timing / throughput stats for one generation
///
/// Timers start **after** the model is loaded into memory (`Engine::open`).
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    /// Prompt tokens processed (prefill).
    pub prompt_tokens: u32,
    /// Generated tokens (decode), excluding the stop token if not counted as output.
    pub predicted_tokens: u32,
    /// Wall time for prompt processing (prefill), until logits for the first
    /// sample are ready.
    pub prompt_ms: f64,
    /// Wall time from start of generation until the first sampled token
    /// (includes prompt processing + first sample). Excludes model load.
    pub time_to_first_token_ms: f64,
    /// Wall time spent in the decode / token-generation loop after the first
    /// token (subsequent tokens only). Zero if fewer than 2 tokens emitted.
    pub predicted_ms: f64,
    /// Total wall time from generation start to finish.
    pub total_ms: f64,
}

impl GenStats {
    /// Prompt-processing throughput (tokens / second).
    #[must_use]
    pub fn prompt_tok_per_sec(&self) -> f64 {
        if self.prompt_ms <= 0.0 || self.prompt_tokens == 0 {
            return 0.0;
        }
        f64::from(self.prompt_tokens) / (self.prompt_ms / 1000.0)
    }

    /// Generation throughput (tokens / second) for tokens after the first.
    ///
    /// Matches common llama.cpp reporting: eval speed over predicted tokens
    /// after TTFT. If only one token was produced, returns 0.
    #[must_use]
    pub fn predicted_tok_per_sec(&self) -> f64 {
        let n = self.predicted_tokens.saturating_sub(1);
        if self.predicted_ms <= 0.0 || n == 0 {
            return 0.0;
        }
        f64::from(n) / (self.predicted_ms / 1000.0)
    }

    /// Overall generation throughput including the first token
    /// (`predicted_tokens / (total - prompt)` when possible).
    #[must_use]
    pub fn generation_tok_per_sec(&self) -> f64 {
        if self.predicted_tokens == 0 {
            return 0.0;
        }
        let gen_ms = (self.total_ms - self.prompt_ms).max(0.0);
        if gen_ms <= 0.0 {
            return 0.0;
        }
        f64::from(self.predicted_tokens) / (gen_ms / 1000.0)
    }
}

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
    settings: EngineSettings,
}

impl EngineBuilder {
    /// New builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_path: None,
            settings: EngineSettings::default(),
        }
    }

    /// Set the model path.
    #[must_use]
    pub fn model(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Override max sequence / context length (`n_ctx`).
    #[must_use]
    pub fn max_seq(mut self, n: usize) -> Self {
        self.settings = self.settings.ctx_size(n);
        self
    }

    /// Full engine settings (context, batch, ubatch).
    #[must_use]
    pub fn settings(mut self, settings: EngineSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Finalize.
    pub fn build(self) -> Result<Engine> {
        let path = self
            .model_path
            .ok_or_else(|| FellmError::other("no model path"))?;
        Engine::open_with(Path::new(&path), self.settings)
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
    #[allow(dead_code)]
    arch: LlamaArch,
    backend: CpuBackend,
    config: LlamaConfig,
    kv: KvCache,
    /// Resolved context length (`n_ctx`).
    max_seq: usize,
    /// Model's GGUF-reported maximum context.
    model_max_ctx: usize,
    /// Evaluation / physical batch settings.
    settings: EngineSettings,
    /// Cached per-token forward graph (built once at open).
    step_graph: Graph,
    /// Cached execution plan.
    step_plan: ExecutionPlan,
    /// Nodes whose attrs are patched each step.
    position_nodes: PositionNodes,
}

impl Engine {
    /// Open a GGUF model with default settings (ctx 8192, clamped to model max).
    pub fn open(path: &Path, max_seq_override: Option<usize>) -> Result<Self> {
        let mut settings = EngineSettings::default();
        if let Some(n) = max_seq_override {
            settings = settings.ctx_size(n);
        }
        Self::open_with(path, settings)
    }

    /// Open a GGUF model with explicit settings.
    pub fn open_with(path: &Path, settings: EngineSettings) -> Result<Self> {
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
        let model_max_ctx = config.context_length.max(1);
        let max_seq = settings.resolve_n_ctx(model_max_ctx);
        let n_ubatch = settings.resolve_ubatch();
        let n_batch = settings.n_batch.max(1);

        tracing::info!(
            n_ctx = max_seq,
            model_max_ctx,
            n_batch,
            n_ubatch,
            "context / batch settings"
        );

        let kv = KvCache::new(
            config.n_layers,
            max_seq,
            config.n_kv_heads,
            config.head_dim(),
        )?;

        tracing::info!("building step graph (once)");
        let step_graph = arch.build_graph(&gguf, &config)?;
        let step_plan = ExecutionPlan::from_graph(&step_graph)?;
        let position_nodes = LlamaArch::collect_position_nodes(&step_graph);
        tracing::info!(
            nodes = step_graph.node_count(),
            rope = position_nodes.rope.len(),
            "step graph ready"
        );

        Ok(Self {
            gguf,
            tokenizer,
            arch,
            backend: CpuBackend::new(),
            config,
            kv,
            max_seq,
            model_max_ctx,
            settings: EngineSettings {
                n_ctx: Some(max_seq),
                n_ctx_from_model: settings.n_ctx_from_model,
                n_batch,
                n_ubatch,
            },
            step_graph,
            step_plan,
            position_nodes,
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

    /// Resolved context length (`n_ctx`).
    #[must_use]
    pub fn n_ctx(&self) -> usize {
        self.max_seq
    }

    /// Model GGUF maximum context length.
    #[must_use]
    pub fn model_max_ctx(&self) -> usize {
        self.model_max_ctx
    }

    /// Active engine settings (resolved `n_ctx`).
    #[must_use]
    pub fn settings(&self) -> EngineSettings {
        self.settings
    }

    /// Generate tokens from a raw prompt string (completion mode).
    ///
    /// Does **not** apply a chat template. Prefer [`Engine::chat`] for
    /// instruction-tuned models.
    pub fn generate(&mut self, prompt: &str, params: GenParams) -> Result<TokenStream<'_>> {
        self.kv.reset();
        let ids = self.tokenizer.encode(prompt, true)?;
        tracing::info!(n_tokens = ids.len(), "prompt tokenized");
        self.generate_from_ids(&ids, params)
    }

    /// Generate a reply to a chat conversation (no tools).
    pub fn chat(&mut self, messages: &[Message], params: GenParams) -> Result<TokenStream<'_>> {
        self.chat_with_tools(messages, &[], params)
    }

    /// Generate a reply with an optional tool list.
    pub fn chat_with_tools(
        &mut self,
        messages: &[Message],
        tools: &[ToolDef],
        params: GenParams,
    ) -> Result<TokenStream<'_>> {
        self.kv.reset();
        let prompt = match self
            .tokenizer
            .apply_chat_template_with_tools(messages, tools, true)?
        {
            Some(formatted) => {
                tracing::debug!(prompt = %formatted, "chat template applied");
                formatted
            }
            None => messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        let ids = self.tokenizer.encode(&prompt, true)?;
        tracing::info!(n_tokens = ids.len(), "chat prompt tokenized");
        self.generate_from_ids(&ids, params)
    }

    /// Convenience: single-turn user chat.
    pub fn chat_user(&mut self, user: &str, params: GenParams) -> Result<TokenStream<'_>> {
        self.chat(&[Message::text("user", user)], params)
    }

    /// Run one chat turn and parse the assistant output for tool calls.
    ///
    /// Returns either plain text or a list of [`ToolCall`]s. Callers that
    /// execute tools should append `Message::assistant_tools(...)` and
    /// `Message::tool_result(...)` then call again.
    pub fn chat_turn(
        &mut self,
        messages: &[Message],
        tools: &[ToolDef],
        params: GenParams,
    ) -> Result<AssistantOutput> {
        let mut stream = self.chat_with_tools(messages, tools, params)?;
        let mut bytes = Vec::new();
        while let Some(tok_result) = stream.next() {
            let tok = tok_result?;
            bytes.extend_from_slice(&stream.decode_token(tok)?);
        }
        let text = String::from_utf8_lossy(&bytes).to_string();
        Ok(parse_assistant_output(&text))
    }

    fn generate_from_ids(&mut self, ids: &[u32], params: GenParams) -> Result<TokenStream<'_>> {
        if ids.is_empty() {
            return Err(FellmError::other("empty prompt"));
        }
        if ids.len() >= self.max_seq {
            return Err(FellmError::other(format!(
                "prompt length {} exceeds context size n_ctx={}",
                ids.len(),
                self.max_seq
            )));
        }

        let stop_token_ids = self.stop_token_ids();
        let n_batch = self.settings.n_batch.max(1);
        let n_ubatch = self.settings.resolve_ubatch();

        let gen_start = Instant::now();
        let mut logits: Option<Tensor> = None;
        let mut pos = 0usize;

        while pos < ids.len() {
            let scheduled = (ids.len() - pos).min(n_batch);
            let mut done = 0usize;
            while done < scheduled {
                let chunk = (scheduled - done).min(n_ubatch);
                for i in 0..chunk {
                    let abs = pos + done + i;
                    logits = Some(self.step(ids[abs], abs)?);
                }
                done += chunk;
            }
            pos += scheduled;
        }

        let prompt_elapsed = gen_start.elapsed();
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
            stats: GenStats {
                prompt_tokens: ids.len() as u32,
                predicted_tokens: 0,
                prompt_ms: duration_ms(prompt_elapsed),
                time_to_first_token_ms: 0.0,
                predicted_ms: 0.0,
                total_ms: 0.0,
            },
            gen_start,
            first_token_at: None,
            last_token_at: None,
        })
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        let mut stops = Vec::new();
        if let Some(eos) = self.tokenizer.eos() {
            stops.push(eos);
        }
        for surface in ["<|end_of_text|>", "<|eot_id|>", "<|eom_id|>"] {
            if let Ok(ids) = self.tokenizer.encode(surface, false)
                && ids.len() == 1
            {
                let id = ids[0];
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        stops
    }

    /// One forward step for token id `tok` at position `pos`.
    fn step(&mut self, tok: u32, pos: usize) -> Result<Tensor> {
        let pos_u32 = pos as u32;
        let mut exec = GraphExecutor::new(&self.step_graph, &self.step_plan, &self.backend);

        for &id in &self.position_nodes.rope {
            let mut a = self.step_graph.node(id).attrs;
            a.position = pos_u32;
            exec.set_attrs(id, a);
        }
        for &id in &self.position_nodes.kv_write {
            let mut a = self.step_graph.node(id).attrs;
            a.position = pos_u32;
            exec.set_attrs(id, a);
        }
        for &id in &self.position_nodes.attention {
            let mut a = self.step_graph.node(id).attrs;
            a.past_len = pos_u32;
            exec.set_attrs(id, a);
        }

        exec.bind_input("token_id", scalar_u32_tensor(tok));

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
    stats: GenStats,
    gen_start: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
}

impl<'a> TokenStream<'a> {
    /// Decode a token id to bytes.
    pub fn decode_token(&self, id: u32) -> Result<Vec<u8>> {
        self.engine.tokenizer.decode_token(id)
    }

    /// Snapshot of timing / throughput stats (updated as tokens are emitted).
    #[must_use]
    pub fn stats(&self) -> GenStats {
        let mut s = self.stats.clone();
        s.total_ms = duration_ms(self.gen_start.elapsed());
        if let Some(first) = self.first_token_at {
            s.time_to_first_token_ms = duration_ms(first.duration_since(self.gen_start));
            if let Some(last) = self.last_token_at
                && s.predicted_tokens > 1
            {
                s.predicted_ms = duration_ms(last.duration_since(first));
            }
        }
        s
    }

    /// Finalize stats after the stream is exhausted (or partially consumed).
    #[must_use]
    pub fn finish_stats(&mut self) -> GenStats {
        self.stats.total_ms = duration_ms(self.gen_start.elapsed());
        if let Some(first) = self.first_token_at {
            self.stats.time_to_first_token_ms = duration_ms(first.duration_since(self.gen_start));
            if let Some(last) = self.last_token_at
                && self.stats.predicted_tokens > 1
            {
                self.stats.predicted_ms = duration_ms(last.duration_since(first));
            }
        }
        self.stats.clone()
    }
}

impl<'a> Iterator for TokenStream<'a> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.emitted >= self.params.max_tokens {
            return None;
        }
        let logits_tensor = self.pending_logits.take()?;
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
        self.stats.predicted_tokens = self.emitted;

        let now = Instant::now();
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
            self.stats.time_to_first_token_ms = duration_ms(now.duration_since(self.gen_start));
        }
        self.last_token_at = Some(now);

        if self.stop_token_ids.contains(&tok) {
            self.finished = true;
            return Some(Ok(tok));
        }

        if self.emitted < self.params.max_tokens && self.position + 1 < self.engine.kv.max_seq {
            match self.engine.step(tok, self.position) {
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

fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn scalar_u32_tensor(v: u32) -> Tensor {
    let mut buf = AlignedBuffer::new_zeroed(4, 4);
    buf.as_mut_slice().copy_from_slice(&v.to_le_bytes());
    let layout = Layout::contiguous(DType::U32, Shape::new(&[1]).expect("valid"));
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    Tensor::from_storage(layout, storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ctx_is_8192_clamped_to_model() {
        let s = EngineSettings::default();
        assert_eq!(s.resolve_n_ctx(131_072), 8192);
        assert_eq!(s.resolve_n_ctx(4096), 4096);
    }

    #[test]
    fn ctx_zero_uses_model_max() {
        let s = EngineSettings::default().ctx_from_model();
        assert_eq!(s.resolve_n_ctx(131_072), 131_072);
    }

    #[test]
    fn ubatch_capped_by_batch() {
        let s = EngineSettings::default().batch_size(256).ubatch_size(512);
        assert_eq!(s.resolve_ubatch(), 256);
    }

    #[test]
    fn gen_stats_tok_per_sec() {
        let s = GenStats {
            prompt_tokens: 100,
            predicted_tokens: 50,
            prompt_ms: 1000.0,
            time_to_first_token_ms: 1050.0,
            predicted_ms: 2000.0,
            total_ms: 3000.0,
        };
        assert!((s.prompt_tok_per_sec() - 100.0).abs() < 1e-6);
        assert!((s.predicted_tok_per_sec() - 24.5).abs() < 1e-6);
        assert!((s.generation_tok_per_sec() - 25.0).abs() < 1e-6);
    }
}

// Re-exports live in lib.rs.
