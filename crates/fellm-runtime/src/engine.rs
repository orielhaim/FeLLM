//! The Engine: the top-level user-facing API.

use crate::architecture::{
    ArchitectureGenerationMode, ArchitecturePluginHandle, ArchitecturePreparation,
};
use crate::backend_select::{BackendPreference, BackendSelect};
use crate::compiled::CompiledStep;
#[cfg(feature = "backend-cuda")]
use crate::cuda_lowering::{LoweredDecodeGraph, compile_cuda_layout, lower_decode_graph};
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
#[cfg(feature = "backend-cuda")]
use fellm_graph::graph::OpValue;
use fellm_graph::plan::ExecutionPlan;
use fellm_model::{
    ModelSpec, StepBindings, build_step_graph, collect_step_bindings, parse_assistant_output,
};
use fellm_plugin_abi::{
    Backend, DriverAction, DriverEvent, GenerationRequest, GraphId, GraphOutput, PagedKvContext,
    set_paged_context,
};
use fellm_tokenizer::{AssistantOutput, Message, Tokenizer, ToolDef, load as load_tokenizer};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
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
    /// Softmax temperature. LFM2.5's recommended default is 0.2.
    pub temperature: f32,
    /// top-k. Zero disables the restriction.
    pub top_k: u32,
    /// top-p / nucleus (>= 1.0 disables).
    pub top_p: f32,
    /// RNG seed.
    pub seed: u64,
    /// Repetition penalty. Values <= 1.0 disable it.
    pub repetition_penalty: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.2,
            top_k: 80,
            top_p: 1.0,
            seed: 0,
            repetition_penalty: 1.05,
        }
    }
}

/// Builder for [`Engine`].
pub struct EngineBuilder {
    model_path: Option<String>,
    settings: EngineSettings,
    architecture: Option<ArchitecturePluginHandle>,
}

impl EngineBuilder {
    /// New builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_path: None,
            settings: EngineSettings::default(),
            architecture: None,
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

    /// Install an architecture plugin for model-family-specific preparation.
    #[must_use]
    pub fn architecture(mut self, plugin: ArchitecturePluginHandle) -> Self {
        self.architecture = Some(plugin);
        self
    }

    /// Finalize.
    pub fn build(self) -> Result<Engine> {
        let path = self
            .model_path
            .ok_or_else(|| FellmError::other("no model path"))?;
        Engine::open_with_architecture(Path::new(&path), self.settings, self.architecture)
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
    architecture: Option<ArchitecturePluginHandle>,
    architecture_program: Option<fellm_plugin_abi::ModelProgram>,
    /// Evaluation / physical batch settings.
    settings: EngineSettings,
}

/// Runtime state + compiled step graph for one GGUF model.
struct LoadedModel {
    spec: ModelSpec,
    architecture_mode: ArchitectureGenerationMode,
    /// Shared physical KV pool + prefix/swap.
    cache: CacheManager,
    /// Active single-request sequence (CLI / default path).
    seq: SequenceCache,
    /// Dummy contiguous buffers kept so the graph can bind `k_in_*` / `v_in_*`
    /// (paged kernels ignore their contents when [`PagedKvContext`] is set).
    dummy_kv: KvCache,
    /// Fixed `ShortConv` state for hybrid models.
    conv: Option<HybridConvState>,
    max_seq: usize,
    model_max_ctx: usize,
    step_graph: Graph,
    step_plan: ExecutionPlan,
    bindings: StepBindings,
    /// Compiled step schedule, built once on first step and reused.
    compiled: Option<CompiledStep>,
    /// Device-native physical plan. This owns stable decode arena/control
    /// allocations and is the replacement execution path for `CompiledStep`.
    #[cfg(feature = "backend-cuda")]
    cuda_decode: Option<CudaDecodePlan>,
    /// Optional full-canvas graph supplied by an architecture plugin.
    canvas_graph: Option<Graph>,
    canvas_plan: Option<ExecutionPlan>,
    canvas_bindings: StepBindings,
    compiled_canvas: Option<CompiledStep>,
    /// Reused self-conditioning input storage.  Sparse mode is normally only
    /// a few hundred KiB; dense mode remains available as an explicit fallback.
    self_conditioning_buffer: Option<Rc<RefCell<AlignedBuffer>>>,
}

#[cfg(feature = "backend-cuda")]
struct CudaDecodePlan {
    tensors:
        std::collections::HashMap<fellm_plugin_abi::PlanTensorId, fellm_plugin_abi::DeviceTensor>,
    _lowered: LoweredDecodeGraph,
    _physical: fellm_plugin_abi::PhysicalPlan,
    _device: backend_cuda::DecodeDeviceState,
    _model: backend_cuda::ModelImage,
}

impl Engine {
    pub fn open(path: &Path, max_seq_override: Option<usize>) -> Result<Self> {
        let mut settings = EngineSettings::default();
        if let Some(n) = max_seq_override {
            settings = settings.ctx_size(n);
        }
        Self::open_with(path, settings)
    }

    pub fn open_with(path: &Path, settings: EngineSettings) -> Result<Self> {
        Self::open_with_architecture(path, settings, None)
    }

    /// Open with an optional architecture plugin.
    pub fn open_with_architecture(
        path: &Path,
        settings: EngineSettings,
        architecture: Option<ArchitecturePluginHandle>,
    ) -> Result<Self> {
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

        let backend = settings.backend.resolve()?;
        let preparation = architecture
            .as_ref()
            .map(|plugin| plugin.prepare(&gguf, &spec, backend.as_ref()))
            .transpose()?
            .flatten();
        if spec.is_diffusion && preparation.is_none() {
            return Err(FellmError::other(
                "diffusion-gemma requires an architecture plugin; pass the DiffusionGemma plugin to EngineBuilder",
            ));
        }
        if let Some(preparation) = &preparation {
            tracing::info!(
                architecture = %preparation.program.architecture_id,
                graphs = preparation.program.graphs.len(),
                "architecture plugin program ready"
            );
        }
        let architecture_program = preparation.as_ref().map(|p| p.program.clone());
        let mut model = LoadedModel::new(&gguf, spec, max_seq, model_max_ctx, preparation)?;

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
            architecture,
            architecture_program,
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

    /// Add the model-family system message required by LFM2.5 when callers
    /// provide a user-only conversation. Other model families are unchanged.
    #[must_use]
    pub fn prepare_chat_messages(&self, messages: &[Message]) -> Vec<Message> {
        if self.model.spec.arch_id == "lfm2moe"
            && !messages.iter().any(|message| message.role == "system")
        {
            let mut prepared = Vec::with_capacity(messages.len() + 1);
            prepared.push(Message::text(
                "system",
                "You are a helpful assistant trained by Liquid AI.",
            ));
            prepared.extend_from_slice(messages);
            prepared
        } else {
            messages.to_vec()
        }
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
        let prepared_messages = self.prepare_chat_messages(messages);
        let prompt =
            match self
                .tokenizer
                .apply_chat_template_with_tools(&prepared_messages, tools, true)?
            {
                Some(formatted) => {
                    tracing::debug!(prompt = %formatted, "chat template applied");
                    formatted
                }
                None => prepared_messages
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

        if self.model.architecture_mode == ArchitectureGenerationMode::BlockDiffusion {
            return self.generate_diffusion_from_ids(ids, params);
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
            prefetched: std::collections::VecDeque::new(),
            emitted: 0,
            position: start_pos,
            finished: false,
            stop_token_ids,
            generated_tokens: Vec::with_capacity(params.max_tokens as usize),
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

    fn generate_diffusion_from_ids(
        &mut self,
        ids: &[u32],
        params: GenParams,
    ) -> Result<TokenStream<'_>> {
        let plugin = self
            .architecture
            .clone()
            .ok_or_else(|| FellmError::other("block-diffusion model has no architecture plugin"))?;
        let program = self.architecture_program.clone().ok_or_else(|| {
            FellmError::other("architecture plugin did not provide a generation program")
        })?;
        let gen_start = Instant::now();
        let n_batch = self.settings.n_batch.max(1);
        let n_ubatch = self.settings.resolve_ubatch();
        let mut pos = 0usize;
        while pos < ids.len() {
            let scheduled = (ids.len() - pos).min(n_batch);
            let mut done = 0usize;
            while done < scheduled {
                let chunk = (scheduled - done).min(n_ubatch);
                for i in 0..chunk {
                    let abs = pos + done + i;
                    let _ = self.step(ids[abs], abs, false)?;
                }
                done += chunk;
            }
            pos += scheduled;
        }
        let prompt_elapsed = gen_start.elapsed();
        let stop_token_ids = self.stop_token_ids();
        let mut emitted = Vec::new();
        let mut context_len = ids.len();
        let request = GenerationRequest {
            prompt: ids.to_vec(),
            max_tokens: params.max_tokens,
            seed: params.seed,
        };
        let mut driver = plugin.create_generation_driver(&program, request)?;
        let mut action = driver.next_action(DriverEvent::Started)?;
        while !matches!(action, DriverAction::Done) {
            action = match action {
                DriverAction::InvokeGraph { graph, .. } if graph == GraphId(0) => {
                    // The generic engine has already prefetched the prompt
                    // through its paged causal graph.  Notify the plugin's
                    // state machine without duplicating that work.
                    driver.next_action(DriverEvent::GraphCompleted {
                        graph,
                        outputs: Vec::new(),
                    })?
                }
                DriverAction::InvokeGraph { graph, inputs, .. } if graph == GraphId(1) => {
                    let canvas = inputs
                        .inputs
                        .iter()
                        .find(|binding| binding.name == "canvas_tokens")
                        .ok_or_else(|| {
                            FellmError::other("diffusion driver omitted canvas_tokens")
                        })?;
                    let logits = self.model.canvas_step(
                        self.backend.as_ref(),
                        &canvas.values,
                        &canvas.float_values,
                        context_len,
                    )?;
                    let values = logits.as_slice::<f32>()?.to_vec();
                    driver.next_action(DriverEvent::GraphCompleted {
                        graph,
                        outputs: vec![GraphOutput {
                            name: "logits".into(),
                            values,
                            rows: canvas.values.len(),
                            cols: self.model.spec.vocab_size,
                        }],
                    })?
                }
                DriverAction::Emit(batch) => {
                    let mut hit_stop = false;
                    for token in batch.token_ids {
                        if stop_token_ids.contains(&token) {
                            hit_stop = true;
                            break;
                        }
                        emitted.push(token);
                    }
                    if hit_stop || emitted.len() >= params.max_tokens as usize {
                        break;
                    }
                    // The driver supplies the complete finalized canvas in
                    // the action, while the visible token batch remains
                    // capped by max_tokens.
                    for token in batch.commit_token_ids {
                        self.step(token, context_len, false)?;
                        context_len += 1;
                        if context_len >= self.model.max_seq {
                            break;
                        }
                    }
                    if emitted.len() >= params.max_tokens as usize
                        || context_len >= self.model.max_seq
                    {
                        break;
                    }
                    driver.next_action(DriverEvent::CacheCommitted {
                        token_count: context_len,
                    })?
                }
                DriverAction::InvokeGraph { graph, .. } if graph == GraphId(2) => {
                    // The host performed the cache append carried by the
                    // preceding Emit action.  Complete the plugin graph
                    // transition without a second causal forward pass.
                    driver.next_action(DriverEvent::GraphCompleted {
                        graph,
                        outputs: Vec::new(),
                    })?
                }
                DriverAction::CommitCache(commit) => {
                    let token_count = commit.token_ids.len();
                    for token in commit.token_ids {
                        self.step(token, context_len, false)?;
                        context_len += 1;
                    }
                    driver.next_action(DriverEvent::CacheCommitted { token_count })?
                }
                DriverAction::InvokeGraph { graph, .. } => {
                    return Err(FellmError::other(format!(
                        "unsupported architecture graph id {}",
                        graph.0
                    )));
                }
                DriverAction::Done => DriverAction::Done,
            };
        }
        Ok(TokenStream {
            engine: self,
            params,
            pending_logits: None,
            prefetched: emitted.into_iter().collect(),
            emitted: 0,
            position: context_len,
            finished: false,
            stop_token_ids,
            generated_tokens: Vec::with_capacity(params.max_tokens as usize),
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
        let matched =
            self.model
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
    pub fn insert_prefix(&mut self, ids: &[u32], seq_cache: &SequenceCache) -> Result<()> {
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
            && cuda.plugins_enabled()
        {
            cuda.sync_kv_device_to_host(self.model.cache.pool.arena_bytes_mut())?;
        }
        self.model.cache.prefix.insert_prompt(ids, seq_cache);
        Ok(())
    }

    /// Ensure `pos` is writable in `seq_cache` (alloc / `CoW`).
    pub fn ensure_seq_writable(&mut self, seq_cache: &mut SequenceCache, pos: usize) -> Result<()> {
        self.model.cache.ensure_writable(seq_cache, pos)
    }

    /// Release physical refs held by a sequence cache.
    pub fn release_seq_cache(&mut self, seq_cache: &mut SequenceCache) {
        self.model.cache.release_sequence(seq_cache);
    }

    /// Swap a sequence's blocks to secondary RAM.
    pub fn swap_out_sequence(&mut self, seq_cache: &mut SequenceCache) -> Result<()> {
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
            && cuda.plugins_enabled()
        {
            cuda.sync_kv_device_to_host(self.model.cache.pool.arena_bytes_mut())?;
        }
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
    fn new(
        gguf: &GgufFile,
        spec: ModelSpec,
        max_seq: usize,
        model_max_ctx: usize,
        preparation: Option<ArchitecturePreparation>,
    ) -> Result<Self> {
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

        let step_graph = build_step_graph(gguf, &spec)?;
        let step_plan = ExecutionPlan::from_graph(&step_graph)?;
        let bindings = collect_step_bindings(&step_graph);
        let (architecture_mode, canvas_graph, canvas_plan, canvas_bindings) =
            if let Some(preparation) = preparation {
                let ArchitecturePreparation {
                    generation_mode,
                    canvas_graph,
                    ..
                } = preparation;
                let Some(graph) = canvas_graph else {
                    return Err(FellmError::other(
                        "architecture plugin did not provide its required graph",
                    ));
                };
                let plan = ExecutionPlan::from_graph(&graph)?;
                let bindings = collect_step_bindings(&graph);
                (generation_mode, Some(graph), Some(plan), bindings)
            } else {
                (
                    ArchitectureGenerationMode::Autoregressive,
                    None,
                    None,
                    StepBindings::default(),
                )
            };
        let self_conditioning_buffer = if canvas_graph.is_some() {
            let slots = fellm_model::diffusion_self_conditioning_slots(spec.vocab_size);
            Some(Rc::new(RefCell::new(AlignedBuffer::new_zeroed(
                spec.canvas_length.saturating_mul(slots).saturating_mul(4),
                64,
            ))))
        } else {
            None
        };
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
            architecture_mode,
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
            #[cfg(feature = "backend-cuda")]
            cuda_decode: None,
            canvas_graph,
            canvas_plan,
            canvas_bindings,
            compiled_canvas: None,
            self_conditioning_buffer,
        })
    }

    /// Build the reusable [`CompiledStep`] once (resolves kernels + preallocates
    /// buffers). Idempotent: does nothing if already compiled.
    fn compile_step(&mut self, backend: &dyn Backend) -> Result<()> {
        if self.compiled.is_some() {
            return Ok(());
        }
        #[cfg(feature = "backend-cuda")]
        if self.cuda_decode.is_none()
            && let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>()
        {
            let lowered = lower_decode_graph(&self.step_graph, &self.step_plan)?;
            let physical = compile_cuda_layout(&lowered)?;
            let blobs = self
                .step_plan
                .order
                .iter()
                .enumerate()
                .filter_map(|(index, &id)| match &self.step_graph.node(id).value {
                    OpValue::Constant(tensor) => Some(backend_cuda::ModelBlob {
                        tensor: fellm_plugin_abi::PlanTensorId(index as u32),
                        bytes: tensor.as_bytes(),
                        alignment: 128,
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let model = backend_cuda::ModelImage::upload(cuda.device_state(), &blobs)?;
            for blob in &blobs {
                let (device_ptr, len) = model.resolve(blob.tensor)?;
                debug_assert_eq!(len, blob.bytes.len());
                cuda.register_device_tensor(blob.bytes.as_ptr(), blob.bytes.len(), device_ptr)?;
            }
            let device =
                backend_cuda::DecodeDeviceState::new(cuda.device_state(), physical.arena_bytes)?;
            let tensors = device.arena.resolve(&physical, &lowered.tensors)?;
            tracing::info!(
                arena_bytes = physical.arena_bytes,
                tensor_count = lowered.tensors.len(),
                macro_ops = physical.operations.len(),
                model_image_bytes = model.byte_len(),
                "compiled device-native CUDA decode layout"
            );
            self.cuda_decode = Some(CudaDecodePlan {
                tensors,
                _lowered: lowered,
                _physical: physical,
                _device: device,
                _model: model,
            });
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

        if let Some(buffer) = &self.self_conditioning_buffer {
            let slots = fellm_model::diffusion_self_conditioning_slots(self.spec.vocab_size);
            mutable_inputs.insert(
                "self_conditioning_logits".into(),
                MutableBinding {
                    dtype: DType::F32,
                    shape: Shape::new(&[self.spec.canvas_length as u64, slots as u64])?,
                    buffer: buffer.clone(),
                },
            );
        }

        let compiled =
            CompiledStep::compile(&self.step_graph, &self.step_plan, backend, &mutable_inputs)?;
        self.compiled = Some(compiled);
        #[cfg(feature = "backend-cuda")]
        if let (Some(cuda), Some(plan), Some(compiled)) = (
            backend.as_any().downcast_ref::<backend_cuda::CudaBackend>(),
            self.cuda_decode.as_ref(),
            self.compiled.as_ref(),
        ) {
            let mut bound = std::collections::HashSet::new();
            for (id, host_ptr, byte_len) in compiled.arena_bindings() {
                let Some(device_tensor) = plan.tensors.get(&id) else {
                    continue;
                };
                // In-place semantic nodes share one host allocation. Register it once;
                // all aliases consequently use exactly one stable device address.
                if bound.insert(host_ptr as usize) {
                    cuda.register_device_tensor(host_ptr, byte_len, device_tensor.ptr)?;
                }
            }
        }
        if let (Some(graph), Some(plan)) = (&self.canvas_graph, &self.canvas_plan) {
            self.compiled_canvas = Some(CompiledStep::compile(
                graph,
                plan,
                backend,
                &mutable_inputs,
            )?);
        }
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
                        cuda.device_kv_ptr().unwrap_or((std::ptr::null_mut(), 0))
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

        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
            let (kv_write_block, kv_write_slot) = self.seq.table(0).locate(pos);
            cuda.update_step_params(&fellm_plugin_abi::DeviceStepParams {
                token_id: tok,
                position: pos as u32,
                sequence_length: pos.saturating_add(1) as u32,
                active_batch: 1,
                kv_write_block,
                kv_write_slot: kv_write_slot as u32,
                ..Default::default()
            })?;
        }

        backend.begin_step();
        let result = self.step_inner(backend, tok, pos, compute_logits);
        // CUDA launches are stream ordered. The LM-head result download is the
        // only host-visible boundary and synchronizes that transfer itself;
        // prefill/body-only steps deliberately remain enqueued.
        backend.end_step();
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

        step.run(backend, compute_logits)
    }

    fn canvas_step(
        &mut self,
        backend: &dyn Backend,
        canvas: &[u32],
        self_conditioning_logits: &[f32],
        prompt_len: usize,
    ) -> Result<Tensor> {
        self.compile_step(backend)?;
        let Some(graph) = &self.canvas_graph else {
            return Err(FellmError::other("canvas graph is not available"));
        };
        let n_logical = self.seq.table(0).num_blocks().max(1);
        let block_table = self.seq.flatten_block_tables();
        let (device_arena, device_arena_len) = {
            #[cfg(feature = "backend-cuda")]
            {
                if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
                    let host = self.cache.pool.arena_bytes();
                    cuda.sync_kv_if_dirty(host)?;
                    if cuda.plugins_enabled() {
                        cuda.device_kv_ptr().unwrap_or((std::ptr::null_mut(), 0))
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
        backend.begin_step();
        let result = (|| {
            let slots = fellm_model::diffusion_self_conditioning_slots(self.spec.vocab_size);
            if self_conditioning_logits.len() != canvas.len().saturating_mul(slots) {
                return Err(FellmError::other(format!(
                    "self-conditioning payload shape mismatch: got {} values, expected {}",
                    self_conditioning_logits.len(),
                    canvas.len().saturating_mul(slots)
                )));
            }
            if let Some(buffer) = &self.self_conditioning_buffer {
                let mut buffer = buffer.borrow_mut();
                buffer
                    .as_mut_slice()
                    .copy_from_slice(bytemuck::cast_slice(self_conditioning_logits));
            }
            let step = self
                .compiled_canvas
                .as_mut()
                .ok_or_else(|| FellmError::other("canvas graph not compiled"))?;
            for &id in &self.canvas_bindings.rope {
                let mut attrs = graph.node(id).attrs;
                attrs.position = prompt_len as u32;
                step.set_attrs(id, attrs);
            }
            for &id in &self.canvas_bindings.attention {
                let mut attrs = graph.node(id).attrs;
                attrs.past_len = prompt_len as u32;
                attrs.query_len = canvas.len() as u32;
                attrs.attention_mode = 1;
                step.set_attrs(id, attrs);
            }
            step.bind_input("canvas_tokens", u32_tensor(canvas)?);
            if self.self_conditioning_buffer.is_none() {
                step.bind_input(
                    "self_conditioning_logits",
                    f32_matrix_tensor(self_conditioning_logits, canvas.len(), slots)?,
                );
            }
            step.run(backend, true)
        })();
        let _ = backend.synchronize();
        backend.end_step();
        set_paged_context(None);
        result
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
    prefetched: std::collections::VecDeque<u32>,
    emitted: u32,
    position: usize,
    finished: bool,
    stop_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    stats: GenStats,
    gen_start: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
}

impl TokenStream<'_> {
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

impl Iterator for TokenStream<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.emitted >= self.params.max_tokens {
            return None;
        }
        if let Some(tok) = self.prefetched.pop_front() {
            self.emitted += 1;
            self.stats.predicted_tokens = self.emitted;
            let now = Instant::now();
            if self.first_token_at.is_none() {
                self.first_token_at = Some(now);
            }
            self.last_token_at = Some(now);
            if self.stop_token_ids.contains(&tok) || self.prefetched.is_empty() {
                self.finished = self.prefetched.is_empty();
            }
            return Some(Ok(tok));
        }
        let logits_tensor = self.pending_logits.take()?;
        let mut logits_owned = logits_tensor;
        let device_greedy = self.params.temperature <= 0.0
            && (self.params.repetition_penalty <= 1.0 || self.generated_tokens.is_empty());
        let device_token = if device_greedy {
            match tensor_f32_ffi_views(&mut logits_owned).and_then(|(input, _)| {
                self.engine.backend.sample_device(
                    input,
                    &fellm_plugin_abi::OpAttrs {
                        temperature: self.params.temperature,
                        top_k: self.params.top_k,
                        top_p: self.params.top_p,
                        seed: self.params.seed.wrapping_add(u64::from(self.emitted)),
                        ..Default::default()
                    },
                )
            }) {
                Ok(token) => token,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        } else {
            None
        };
        if device_token.is_none()
            && let Ok((input, output)) = tensor_f32_ffi_views(&mut logits_owned)
            && let Err(error) = self.engine.backend.materialize(input, output)
        {
            self.finished = true;
            return Some(Err(error));
        }
        let tok = if let Some(token) = device_token {
            token
        } else if let Ok(work) = logits_owned.as_mut_slice::<f32>() {
            sampling::sample(
                work,
                self.params.temperature,
                self.params.top_k,
                self.params.top_p,
                self.params.seed.wrapping_add(u64::from(self.emitted)),
                self.params.repetition_penalty,
                &self.generated_tokens,
            )
        } else {
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
                self.params.repetition_penalty,
                &self.generated_tokens,
            )
        };
        self.generated_tokens.push(tok);
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

fn tensor_f32_ffi_views(
    tensor: &mut Tensor,
) -> Result<(fellm_plugin_abi::TensorRef, fellm_plugin_abi::TensorMut)> {
    if tensor.dtype() != DType::F32 {
        return Err(FellmError::other("device sampling requires f32 logits"));
    }
    let dims = tensor.shape().dims().to_vec();
    let strides = tensor.shape().row_major_strides().as_slice().to_vec();
    let values = tensor.as_mut_slice::<f32>()?;
    let ptr = values.as_mut_ptr();
    let bytes = core::mem::size_of_val(values);
    Ok(unsafe {
        (
            fellm_plugin_abi::TensorRef::from_raw(
                DType::F32,
                &dims,
                &strides,
                ptr.cast_const().cast(),
                bytes,
            ),
            fellm_plugin_abi::TensorMut::from_raw(DType::F32, &dims, &strides, ptr.cast(), bytes),
        )
    })
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

fn u32_tensor(values: &[u32]) -> Result<Tensor> {
    let mut buf = AlignedBuffer::new_zeroed(values.len() * 4, 4);
    for (index, value) in values.iter().copied().enumerate() {
        let start = index * 4;
        buf.as_mut_slice()[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
    let layout = Layout::contiguous(
        DType::U32,
        Shape::new(&[values.len() as u64]).map_err(FellmError::from)?,
    );
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    Ok(Tensor::from_storage(layout, storage))
}

fn f32_matrix_tensor(values: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
    if values.len() != rows.saturating_mul(cols) {
        return Err(FellmError::other("f32 matrix tensor shape mismatch"));
    }
    let mut buf = AlignedBuffer::new_zeroed(values.len() * 4, 64);
    let bytes: &mut [u8] = buf.as_mut_slice();
    for (index, value) in values.iter().copied().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    let layout = Layout::contiguous(
        DType::F32,
        Shape::new(&[rows as u64, cols as u64]).map_err(FellmError::from)?,
    );
    let storage = Arc::new(Storage::Owned(Arc::new(buf)));
    Ok(Tensor::from_storage(layout, storage))
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
