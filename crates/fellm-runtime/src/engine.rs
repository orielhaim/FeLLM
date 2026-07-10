//! The Engine: the top-level user-facing API.

use crate::backend_select::{BackendPreference, BackendSelect};
use crate::compiled::CompiledStep;
use crate::executor::MutableBinding;
use crate::hybrid_state::HybridConvState;
use crate::kv_cache::KvCache;
use crate::paged::{CacheManager, SequenceCache};
use crate::sampling;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use fellm_gguf::GgufFile;
use fellm_graph::Graph;
use fellm_graph::plan::ExecutionPlan;
use fellm_model::{
    ModelSpec, StepBindings, build_step_graph, collect_step_bindings, parse_assistant_output,
};
use fellm_plugin_abi::{Backend, PagedKvContext, set_paged_context};
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
    /// Backend selection (`auto` / `cpu` / `cuda`) + CPU fallback policy.
    pub backend: BackendSelect,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            n_ctx: Some(DEFAULT_CTX_SIZE),
            n_ctx_from_model: false,
            n_batch: DEFAULT_BATCH_SIZE,
            n_ubatch: DEFAULT_UBATCH_SIZE,
            backend: BackendSelect::from_env(),
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

    /// Backend preference + fallback policy.
    #[must_use]
    pub fn backend_select(mut self, select: BackendSelect) -> Self {
        self.backend = select;
        self
    }

    /// Shorthand: force CPU / CUDA / auto.
    #[must_use]
    pub fn backend_preference(mut self, preference: BackendPreference) -> Self {
        self.backend.preference = preference;
        self
    }

    /// Allow (default) or deny falling back to CPU when CUDA fails.
    #[must_use]
    pub fn allow_cpu_fallback(mut self, allow: bool) -> Self {
        self.backend.allow_cpu_fallback = allow;
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
    backend: Box<dyn Backend>,
    model: LoadedModel,
    /// Evaluation / physical batch settings.
    settings: EngineSettings,
}

/// Runtime state + compiled step graph for one GGUF model.
struct LoadedModel {
    spec: ModelSpec,
    /// Shared physical KV pool + prefix/swap.
    cache: CacheManager,
    /// Active single-request sequence (CLI / default path).
    seq: SequenceCache,
    /// Dummy contiguous buffers kept so the graph can bind `k_in_*` / `v_in_*`
    /// (paged kernels ignore their contents when [`PagedKvContext`] is set).
    dummy_kv: KvCache,
    /// Fixed ShortConv state for hybrid models.
    conv: Option<HybridConvState>,
    max_seq: usize,
    model_max_ctx: usize,
    step_graph: Graph,
    step_plan: ExecutionPlan,
    bindings: StepBindings,
    /// Compiled step schedule, built once on first step and reused.
    compiled: Option<CompiledStep>,
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

        let spec = ModelSpec::from_gguf(&gguf)?;
        tracing::info!(
            arch = %spec.arch_id,
            n_layers = spec.n_layers,
            attn = spec.n_attn_layers(),
            conv = spec.n_conv_layers(),
            "probed model recipe from GGUF"
        );

        let model_max_ctx = spec.context_length.max(1);
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

        let mut model = LoadedModel::new(&gguf, spec, max_seq, model_max_ctx)?;

        let backend = settings.backend.resolve()?;
        tracing::info!(backend = backend.id(), "compute backend ready");

        // B2: size VRAM KV arena to match the host PhysicalPool.
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
            let pool = &model.cache.pool;
            let swap_blocks = (pool.n_blocks() / 2).max(1);
            if let Err(e) = cuda.init_kv_arena(
                pool.n_blocks(),
                model.spec.n_kv_heads.max(1),
                model.spec.head_dim.max(1),
                swap_blocks,
            ) {
                tracing::warn!(error = %e, "DeviceKvArena init failed; paged ops stay host-only");
            }
        }

        // Compile the reusable step schedule once now that the backend is known.
        model.compile_step(backend.as_ref())?;

        Ok(Self {
            gguf,
            tokenizer,
            backend,
            model,
            settings: EngineSettings {
                n_ctx: Some(max_seq),
                n_ctx_from_model: settings.n_ctx_from_model,
                n_batch,
                n_ubatch,
                backend: settings.backend,
            },
        })
    }

    /// Open with an explicit backend (tests / callers that already resolved one).
    pub fn open_with_backend(
        path: &Path,
        settings: EngineSettings,
        backend: Box<dyn Backend>,
    ) -> Result<Self> {
        let mut eng = Self::open_with(path, settings.backend_preference(BackendPreference::Cpu))?;
        eng.backend = backend;
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = eng
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
        {
            let pool = &eng.model.cache.pool;
            let _ = cuda.init_kv_arena(
                pool.n_blocks(),
                eng.model.spec.n_kv_heads.max(1),
                eng.model.spec.head_dim.max(1),
                (pool.n_blocks() / 2).max(1),
            );
        }
        // The step was compiled against the default backend; recompile so kernel
        // handles match the injected one.
        eng.model.compiled = None;
        eng.model.compile_step(eng.backend.as_ref())?;
        Ok(eng)
    }

    /// Active backend id (`"cpu"` / `"cuda"`).
    #[must_use]
    pub fn backend_id(&self) -> &'static str {
        self.backend.id()
    }

    /// Tokenizer reference.
    #[must_use]
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Probed model recipe.
    #[must_use]
    pub fn spec(&self) -> &ModelSpec {
        &self.model.spec
    }

    /// Resolved context length (`n_ctx`).
    #[must_use]
    pub fn n_ctx(&self) -> usize {
        self.model.max_seq
    }

    /// Model GGUF maximum context length.
    #[must_use]
    pub fn model_max_ctx(&self) -> usize {
        self.model.model_max_ctx
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
        self.model.reset();
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
        self.model.reset();
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
        if ids.len() >= self.model.max_seq {
            return Err(FellmError::other(format!(
                "prompt length {} exceeds context size n_ctx={}",
                ids.len(),
                self.model.max_seq
            )));
        }

        let stop_token_ids = self.stop_token_ids();
        let n_batch = self.settings.n_batch.max(1);
        let n_ubatch = self.settings.resolve_ubatch();

        let gen_start = Instant::now();
        let mut logits: Option<Tensor> = None;
        let mut pos = 0usize;
        let n_prompt = ids.len();

        while pos < n_prompt {
            let scheduled = (n_prompt - pos).min(n_batch);
            let mut done = 0usize;
            while done < scheduled {
                let chunk = (scheduled - done).min(n_ubatch);
                for i in 0..chunk {
                    let abs = pos + done + i;
                    // Only the last prompt token needs lm_head / logits.
                    let need_logits = abs + 1 == n_prompt;
                    logits = Some(self.step(ids[abs], abs, need_logits)?);
                }
                done += chunk;
            }
            pos += scheduled;
        }

        let prompt_elapsed = gen_start.elapsed();
        let last_logits = logits.ok_or_else(|| FellmError::other("empty prompt"))?;
        let start_pos = n_prompt;

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

    /// Public stop-token set for the scheduler.
    #[must_use]
    pub fn stop_token_ids_pub(&self) -> Vec<u32> {
        self.stop_token_ids()
    }

    /// One forward step for token id `tok` at position `pos`.
    ///
    /// When `compute_logits` is false, `final_norm` + `lm_head` are skipped
    /// (prefill mid-tokens still write KV / conv state).
    fn step(&mut self, tok: u32, pos: usize, compute_logits: bool) -> Result<Tensor> {
        self.model
            .step(self.backend.as_ref(), tok, pos, compute_logits)
    }

    /// Free blocks remaining in the physical pool.
    #[must_use]
    pub fn cache_free_blocks(&self) -> usize {
        self.model.cache.pool.free_count()
    }

    /// Attach radix-prefix blocks for `ids` onto `seq_cache`. Returns matched token count.
    pub fn attach_prefix(&mut self, ids: &[u32], seq_cache: &mut SequenceCache) -> usize {
        let matched = self
            .model
            .cache
            .prefix
            .attach_match(&mut self.model.cache.pool, seq_cache, ids);
        #[cfg(feature = "backend-cuda")]
        if matched > 0 {
            if let Some(cuda) = self
                .backend
                .as_any()
                .downcast_ref::<backend_cuda::CudaBackend>()
            {
                // Prefix reuses host-resident physical blocks; mirror once before decode.
                cuda.mark_kv_host_dirty();
            }
        }
        matched
    }

    /// Insert a completed prompt into the prefix tree.
    pub fn insert_prefix(&mut self, ids: &[u32], seq_cache: &SequenceCache) {
        self.model.cache.prefix.insert_prompt(ids, seq_cache);
    }

    /// Ensure `pos` is writable in `seq_cache` (alloc / CoW).
    pub fn ensure_seq_writable(&mut self, seq_cache: &mut SequenceCache, pos: usize) -> Result<()> {
        self.model.cache.ensure_writable(seq_cache, pos)
    }

    /// Release physical refs held by a sequence cache.
    pub fn release_seq_cache(&mut self, seq_cache: &mut SequenceCache) {
        self.model.cache.release_sequence(seq_cache);
    }

    /// Swap a sequence's blocks to secondary RAM.
    pub fn swap_out_sequence(&mut self, seq_cache: &mut SequenceCache) -> Result<()> {
        let mut buf = vec![0u8; self.model.cache.pool.block_bytes()];
        for layer in 0..seq_cache.n_layers() {
            for &phys in seq_cache.table(layer).blocks() {
                self.model.cache.pool.read_block_bytes(phys, &mut buf);
                self.model.cache.swap.swap_out(phys, &buf)?;
                // Keep refcount but free physical slot for reuse: copy to swap then
                // temporarily zero-ref free — for simplicity we leave physical allocated
                // until dec_ref; here we free by dec_ref after swap.
                // Actually: we need the physical id to remain in the table for swap_in.
                // So we only copy to swap and mark swapped; physical can be freed if we
                // remapped. Simpler approach: keep physical occupied but mark seq swapped
                // so scheduler doesn't run it — true free requires remapping.
                // For LRU free: dec_ref and clear table entries stored in swap map by phys id.
                let _ = phys;
            }
        }
        // Free physical blocks after copying.
        for layer in 0..seq_cache.n_layers() {
            for &phys in seq_cache.table(layer).blocks() {
                // Drop one ref (sequence's); if only this seq held it, returns to free list.
                self.model.cache.pool.dec_ref(phys);
            }
        }
        seq_cache.swapped = true;
        Ok(())
    }

    /// Restore a swapped sequence's blocks.
    pub fn swap_in_sequence(&mut self, seq_cache: &mut SequenceCache) -> Result<()> {
        let mut buf = vec![0u8; self.model.cache.pool.block_bytes()];
        for layer in 0..seq_cache.n_layers() {
            let n = seq_cache.table(layer).num_blocks();
            for logical in 0..n {
                let old_phys = seq_cache.table(layer).block(logical);
                let new_phys = self
                    .model
                    .cache
                    .pool
                    .alloc_block()
                    .ok_or_else(|| FellmError::other("swap_in: out of blocks"))?;
                self.model.cache.swap.swap_in(old_phys, &mut buf)?;
                self.model.cache.pool.write_block_bytes(new_phys, &buf);
                self.model.cache.pool.inc_ref(new_phys);
                *seq_cache.table_mut(layer).block_mut(logical) = new_phys;
            }
        }
        seq_cache.swapped = false;
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
        {
            cuda.mark_kv_host_dirty();
        }
        Ok(())
    }

    /// Run one forward step against an arbitrary sequence cache (multi-seq).
    pub fn step_sequence(
        &mut self,
        seq_cache: &mut SequenceCache,
        tok: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<Tensor> {
        // Temporarily swap active seq.
        std::mem::swap(&mut self.model.seq, seq_cache);
        let result = self
            .model
            .step(self.backend.as_ref(), tok, pos, compute_logits);
        std::mem::swap(&mut self.model.seq, seq_cache);
        result
    }
}

impl LoadedModel {
    fn new(gguf: &GgufFile, spec: ModelSpec, max_seq: usize, model_max_ctx: usize) -> Result<Self> {
        let n_attn = spec.n_attn_layers().max(1);
        let cache = CacheManager::with_capacity(
            max_seq,
            n_attn,
            spec.n_kv_heads.max(1),
            spec.head_dim.max(1),
            4,
        )?;
        let seq = cache.new_sequence(max_seq);
        let dummy_kv = KvCache::new(
            n_attn,
            max_seq.max(1),
            spec.n_kv_heads.max(1),
            spec.head_dim.max(1),
        )?;
        let conv = if spec.is_hybrid() {
            Some(HybridConvState::new(
                &spec.layer_kv_heads_for_state(),
                spec.d_model,
                spec.shortconv_l_cache,
            )?)
        } else {
            None
        };

        tracing::info!("building step graph (once)");
        let step_graph = build_step_graph(gguf, &spec)?;
        let step_plan = ExecutionPlan::from_graph(&step_graph)?;
        let bindings = collect_step_bindings(&step_graph);
        tracing::info!(
            nodes = step_graph.node_count(),
            rope = bindings.rope.len(),
            kv_write = bindings.kv_write.len(),
            attention = bindings.attention.len(),
            attn_layers = spec.n_attn_layers(),
            conv_layers = spec.n_conv_layers(),
            pool_blocks = cache.pool.n_blocks(),
            "step graph ready (paged KV)"
        );

        Ok(Self {
            spec,
            cache,
            seq,
            dummy_kv,
            conv,
            max_seq,
            model_max_ctx,
            step_graph,
            step_plan,
            bindings,
            compiled: None,
        })
    }

    /// Build the reusable [`CompiledStep`] once (resolves kernels + preallocates
    /// buffers). Idempotent: does nothing if already compiled.
    fn compile_step(&mut self, backend: &dyn Backend) -> Result<()> {
        if self.compiled.is_some() {
            return Ok(());
        }
        let mut mutable_inputs: std::collections::HashMap<String, MutableBinding> =
            std::collections::HashMap::new();

        let dim = self.dummy_kv.tokens_stride;
        let shape = Shape::new(&[self.dummy_kv.max_seq as u64, dim as u64])?;
        for layer in 0..self.spec.n_attn_layers() {
            mutable_inputs.insert(
                format!("k_in_{layer}"),
                MutableBinding {
                    dtype: DType::F32,
                    shape: shape.clone(),
                    buffer: self.dummy_kv.k_buffer(layer),
                },
            );
            mutable_inputs.insert(
                format!("v_in_{layer}"),
                MutableBinding {
                    dtype: DType::F32,
                    shape: shape.clone(),
                    buffer: self.dummy_kv.v_buffer(layer),
                },
            );
        }

        if let Some(state) = &self.conv {
            let conv_shape = Shape::new(&[state.conv_elements() as u64])?;
            for conv_ord in 0..state.conv_layer_ids.len() {
                mutable_inputs.insert(
                    format!("conv_in_{conv_ord}"),
                    MutableBinding {
                        dtype: DType::F32,
                        shape: conv_shape.clone(),
                        buffer: state.conv_buffer(conv_ord),
                    },
                );
            }
        }

        let compiled =
            CompiledStep::compile(&self.step_graph, &self.step_plan, backend, &mutable_inputs)?;
        self.compiled = Some(compiled);
        Ok(())
    }

    fn reset(&mut self) {
        self.cache.release_sequence(&mut self.seq);
        self.seq = self.cache.new_sequence(self.max_seq);
        if let Some(c) = &mut self.conv {
            c.reset();
        }
    }

    fn step(
        &mut self,
        backend: &dyn Backend,
        tok: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<Tensor> {
        self.cache.ensure_writable(&mut self.seq, pos)?;
        self.cache.tick();

        let n_logical = self.seq.table(0).num_blocks().max(1);
        let block_table = self.seq.flatten_block_tables();

        let (device_arena, device_arena_len) = {
            #[cfg(feature = "backend-cuda")]
            {
                if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
                    // One-shot H2D only when host KV was mutated outside GPU KvWrite
                    // (prefix attach / swap-in). Never re-upload the full arena every token.
                    let host = self.cache.pool.arena_bytes();
                    if let Err(e) = cuda.sync_kv_if_dirty(host) {
                        tracing::warn!(error = %e, "KV H2D sync failed");
                    }
                    if cuda.plugins_enabled() {
                        cuda.device_kv_ptr()
                            .unwrap_or((std::ptr::null_mut(), 0))
                    } else {
                        (std::ptr::null_mut(), 0)
                    }
                } else {
                    (std::ptr::null_mut(), 0)
                }
            }
            #[cfg(not(feature = "backend-cuda"))]
            {
                (std::ptr::null_mut(), 0usize)
            }
        };

        let (arena_ptr, arena_len) = self.cache.pool.arena_ptr_mut();
        set_paged_context(Some(PagedKvContext {
            arena: arena_ptr,
            arena_len,
            block_table: std::sync::Arc::<[u32]>::from(block_table),
            n_logical_blocks: n_logical,
            n_layers: self.cache.pool.n_layers(),
            tokens_stride: self.cache.pool.tokens_stride(),
            block_bytes: self.cache.pool.block_bytes(),
            block_size: crate::paged::BLOCK_SIZE,
            elem_bytes: fellm_plugin_abi::PAGED_KV_ELEM_BYTES,
            device_arena,
            device_arena_len,
        }));

        let result = self.step_inner(backend, tok, pos, compute_logits);
        // Plugin D2H of logits already drains the stream; synchronize is a no-op
        // when plugins are disabled (see CudaBackend::synchronize).
        let _ = backend.synchronize();
        set_paged_context(None);
        result
    }

    fn step_inner(
        &mut self,
        backend: &dyn Backend,
        tok: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<Tensor> {
        self.compile_step(backend)?;
        let pos_u32 = pos as u32;

        // Patch per-token attribute overrides into the compiled schedule, then
        // bind the token id and run. Buffers / kernels are already resolved.
        let step = self
            .compiled
            .as_mut()
            .ok_or_else(|| FellmError::other("step not compiled"))?;

        for &id in &self.bindings.rope {
            let mut a = self.step_graph.node(id).attrs;
            a.position = pos_u32;
            step.set_attrs(id, a);
        }
        for &id in &self.bindings.kv_write {
            let mut a = self.step_graph.node(id).attrs;
            a.position = pos_u32;
            a.block_size = 16;
            step.set_attrs(id, a);
        }
        for &id in &self.bindings.attention {
            let mut a = self.step_graph.node(id).attrs;
            a.past_len = pos_u32;
            a.block_size = 16;
            step.set_attrs(id, a);
        }

        step.bind_input("token_id", scalar_u32_tensor(tok));

        // B3: optionally capture this decode under a CUDA graph (once per past_len
        // bucket). Replay is not used yet — kernel attrs (token/past_len) change
        // every step and need cudaGraphExecKernelNodeSetParams (follow-up).
        #[cfg(feature = "backend-cuda")]
        if pos > 0 {
            if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
                if backend_cuda::CudaBackend::graphs_enabled() {
                    let max_ctx = self.max_seq as u32;
                    let mut captured: Option<Tensor> = None;
                    let capture_result = cuda.ensure_decode_graph(pos_u32, max_ctx, || {
                        captured = Some(step.run(backend, compute_logits)?);
                        Ok(())
                    });
                    match capture_result {
                        Ok(true) => {
                            if let Some(t) = captured {
                                return Ok(t);
                            }
                        }
                        Ok(false) => {}
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                past_len = pos_u32,
                                "CUDA graph capture skipped"
                            );
                        }
                    }
                }
            }
        }

        step.run(backend, compute_logits)
    }

    /// Access the shared cache manager (scheduler / multi-seq).
    #[allow(dead_code)]
    fn cache_mut(&mut self) -> &mut CacheManager {
        &mut self.cache
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
        let mut logits_owned = logits_tensor;
        let tok = match logits_owned.as_mut_slice::<f32>() {
            Ok(work) => sampling::sample(
                work,
                self.params.temperature,
                self.params.top_k,
                self.params.top_p,
                self.params.seed.wrapping_add(u64::from(self.emitted)),
            ),
            Err(_) => {
                // Fallback if storage is shared / non-owned (should not happen
                // on the compiled path).
                let logits = match logits_owned.as_slice::<f32>() {
                    Ok(s) => s,
                    Err(e) => {
                        self.finished = true;
                        return Some(Err(e));
                    }
                };
                let mut work = logits.to_vec();
                sampling::sample(
                    &mut work,
                    self.params.temperature,
                    self.params.top_k,
                    self.params.top_p,
                    self.params.seed.wrapping_add(u64::from(self.emitted)),
                )
            }
        };
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

        if self.emitted < self.params.max_tokens && self.position + 1 < self.engine.model.max_seq {
            match self.engine.step(tok, self.position, true) {
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
