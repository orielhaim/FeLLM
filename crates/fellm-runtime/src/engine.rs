//! The Engine: the top-level user-facing API.

use crate::architecture::{
    ArchitectureGenerationMode, ArchitecturePluginHandle, ArchitecturePreparation,
};
use crate::backend_select::{BackendPreference, BackendSelect};
use crate::compiled::{CompiledStep, MutableBinding};
#[cfg(feature = "backend-cuda")]
use crate::cuda_lowering::{LoweredDecodeGraph, compile_cuda_layout, lower_decode_graph};
use crate::hybrid_state::HybridConvState;
use crate::kv_fabric::{
    DummyKvBuffers, KvFabric, KvFabricConfig, KvMemoryPlan, KvSequence, STANDARD_PAGE_TOKENS,
};
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
    ModelSpec, StepBindings, build_batch_step_graph_with_features, build_step_graph_with_features,
    collect_step_bindings, parse_assistant_output,
};
use fellm_plugin_abi::op::OpKind;
use fellm_plugin_abi::{
    AttentionDispatch, AttentionKernelPath, AttentionPathKind, Backend, DriverAction, DriverEvent,
    GenerationRequest, GraphId, GraphOutput, PagedKvContext, PreRopeKeyStore, RetentionContext,
    SequenceAttentionState, set_attention_dispatch, set_paged_context, set_pre_rope_store,
};
use fellm_tokenizer::{
    AssistantOutput, ChatRenderOptions, Message, Tokenizer, ToolDef, load as load_tokenizer,
};
use std::cell::RefCell;
use std::collections::HashMap;
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
#[derive(Debug, Clone)]
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
    /// Explicit / automatic provider selection (attention, kv policy, config).
    pub providers: crate::providers::ProviderSelection,
    /// Optional plugin directory for dynamic capability plugins.
    pub plugin_dir: Option<std::path::PathBuf>,
    /// KV fabric configuration (mode, budgets, addressing, policies).
    pub kv_cache: KvFabricConfig,
    /// Hardware/planner overrides. Unset capacities are discovered automatically.
    pub memory_fabric: crate::memory_fabric::MemoryFabricConfig,
    /// Internal representations retained for the active speculator only.
    pub target_features: Vec<fellm_plugin_abi::TargetFeature>,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            n_ctx: Some(DEFAULT_CTX_SIZE),
            n_ctx_from_model: false,
            n_batch: DEFAULT_BATCH_SIZE,
            n_ubatch: DEFAULT_UBATCH_SIZE,
            backend: BackendSelect::from_env(),
            providers: crate::providers::ProviderSelection::new(),
            plugin_dir: None,
            kv_cache: KvFabricConfig::default(),
            memory_fabric: crate::memory_fabric::MemoryFabricConfig::default(),
            target_features: Vec::new(),
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

    /// Provider selection (attention / kv-policy / plugin config).
    #[must_use]
    pub fn providers(mut self, selection: crate::providers::ProviderSelection) -> Self {
        self.providers = selection;
        self
    }

    /// Directory of dynamic capability plugins.
    #[must_use]
    pub fn plugin_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.plugin_dir = Some(dir.into());
        self
    }

    #[must_use]
    pub fn kv_cache(mut self, config: KvFabricConfig) -> Self {
        self.kv_cache = config;
        self
    }

    /// Request internal target representations for a prepared speculator.
    #[must_use]
    pub fn target_features(
        mut self,
        features: impl IntoIterator<Item = fellm_plugin_abi::TargetFeature>,
    ) -> Self {
        self.target_features = features.into_iter().collect();
        self.target_features
            .sort_unstable_by_key(|feature| match feature {
                fellm_plugin_abi::TargetFeature::EmbeddingOutput => 0,
                fellm_plugin_abi::TargetFeature::LayerHiddenState(layer) => layer + 1,
                fellm_plugin_abi::TargetFeature::FinalHiddenState => u32::MAX,
            });
        self.target_features.dedup();
        self
    }

    /// Resolve `n_ctx` given the model's maximum from GGUF metadata.
    #[must_use]
    pub fn resolve_n_ctx(&self, model_max: usize) -> usize {
        let model_max = model_max.max(1);
        if self.n_ctx_from_model {
            return model_max;
        }
        let requested = self.n_ctx.unwrap_or(DEFAULT_CTX_SIZE).max(1);
        requested.min(model_max)
    }

    /// Physical chunk size used during prompt processing.
    #[must_use]
    pub fn resolve_ubatch(&self) -> usize {
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
#[derive(Debug, Clone)]
pub struct GenParams {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Softmax temperature. LFM2.5's recommended default is 0.2.
    pub temperature: f32,
    /// top-k. Zero disables the restriction.
    pub top_k: u32,
    /// top-p / nucleus (>= 1.0 disables).
    pub top_p: f32,
    /// Minimum probability relative to the most likely token (0 disables).
    pub min_p: f32,
    /// RNG seed.
    pub seed: u64,
    /// Repetition penalty. Values <= 1.0 disable it.
    pub repetition_penalty: f32,
    /// OpenAI-style count-scaled frequency penalty.
    pub frequency_penalty: f32,
    /// OpenAI-style one-time presence penalty.
    pub presence_penalty: f32,
    /// Sparse token-id logit adjustments.
    pub logit_bias: Arc<[(u32, f32)]>,
    /// Optional transactional finite-state token grammar.
    pub grammar: Option<Arc<sampling::TokenGrammar>>,
    /// Scheduler priority; larger values are selected first within a work class.
    pub priority: i32,
    /// Chat-template `enable_thinking`. `None` uses the tokenizer default
    /// (off when the GGUF template supports the switch).
    pub enable_thinking: Option<bool>,
}

impl GenParams {
    /// True when the processed distribution is a point mass, so greedy
    /// speculative verification is distributionally exact.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.2,
            top_k: 80,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            repetition_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Arc::from([]),
            grammar: None,
            priority: 0,
            enable_thinking: None,
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
    model: LoadedModel,
    // Declared after `model` so CUDA graphs/device allocations are destroyed
    // before the backend unloads their plugin module and context.
    backend: Box<dyn Backend>,
    architecture: Option<ArchitecturePluginHandle>,
    architecture_program: Option<fellm_plugin_abi::ModelProgram>,
    /// Evaluation / physical batch settings.
    settings: EngineSettings,
    /// Capability providers (attention, KV policy) prepared at open.
    providers: crate::providers::ProviderManager,
    /// Per-request sequence attention state (views + policy metadata).
    seq_attn: SequenceAttentionState,
    /// Tokens generated since last KV policy compression window.
    tokens_since_compress: u32,
}

/// Runtime state + compiled step graph for one GGUF model.
struct LoadedModel {
    spec: ModelSpec,
    /// Joint weights/KV/activation plan and cross-tier telemetry.
    memory_fabric: crate::memory_fabric::MemoryFabric,
    architecture_mode: ArchitectureGenerationMode,
    /// KV fabric (logical identity + residency + sharing).
    cache: KvFabric,
    /// Active single-request sequence (CLI / default path).
    seq: KvSequence,
    /// Dummy contiguous buffers kept so the graph can bind `k_in_*` / `v_in_*`
    /// (paged kernels ignore their contents when [`PagedKvContext`] is set).
    dummy_kv: DummyKvBuffers,
    /// Fixed `ShortConv` state for hybrid models.
    conv: Option<HybridConvState>,
    max_seq: usize,
    model_max_ctx: usize,
    step_graph: Graph,
    step_plan: ExecutionPlan,
    bindings: StepBindings,
    target_features: Vec<fellm_plugin_abi::TargetFeature>,
    /// Compiled step schedule, built once on first step and reused.
    compiled: Option<CompiledStep>,
    /// Reusable physical-batch graphs keyed by row count (`<= n_ubatch`).
    batches: HashMap<usize, CompiledBatch>,
    /// Worst-case physical-batch arena reserved in KV budget calculations.
    batch_activation_reserve: usize,
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

struct CompiledBatch {
    graph: Graph,
    bindings: StepBindings,
    step: CompiledStep,
    conv_buffers: Vec<Rc<RefCell<AlignedBuffer>>>,
    ssm_buffers: Vec<Rc<RefCell<AlignedBuffer>>>,
}

/// One token row in a physical inference batch.
#[derive(Debug, Clone, Copy)]
pub struct BatchToken {
    /// Index into the sequence-cache slice passed to [`Engine::step_batch`].
    pub sequence: usize,
    /// Input token id.
    pub token: u32,
    /// Absolute model position used by RoPE.
    pub position: usize,
    /// Whether the caller needs this row's logits.
    pub compute_logits: bool,
}

/// Independently owned autoregressive sequence state, used by generic draft
/// models as well as target verification.
pub struct DecodeSequence {
    pub(crate) cache: KvSequence,
    pub(crate) recurrent: Option<HybridConvState>,
    pub(crate) pending_logits: Tensor,
    pub(crate) position: usize,
}

impl DecodeSequence {
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    #[must_use]
    pub fn logits(&self) -> &Tensor {
        &self.pending_logits
    }

    #[must_use]
    pub fn kv_len(&self) -> usize {
        self.cache.len_tokens
    }
}

#[cfg(feature = "backend-cuda")]
struct CudaDecodePlan {
    tensors:
        std::collections::HashMap<fellm_plugin_abi::PlanTensorId, fellm_plugin_abi::DeviceTensor>,
    _lowered: LoweredDecodeGraph,
    _physical: fellm_plugin_abi::PhysicalPlan,
    device: backend_cuda::DecodeDeviceState,
    _weights: Option<backend_cuda::CudaWeightFabric>,
    graph: Option<backend_cuda::CudaGraphExec>,
    full_step_warmed: bool,
    graph_replay_safe: bool,
}

impl Engine {
    /// Parsed GGUF backing this engine. Model-native speculators use this to
    /// Arc-share checkpoint tensors rather than reopening or duplicating them.
    #[must_use]
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

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
        let backend = settings.backend.resolve()?;
        let storage_native_cpu = backend.id() == "cpu";
        let mut settings = settings;
        let mut probe_features = settings.target_features.clone();
        probe_features.extend([
            fellm_plugin_abi::TargetFeature::EmbeddingOutput,
            fellm_plugin_abi::TargetFeature::LayerHiddenState(0),
            fellm_plugin_abi::TargetFeature::FinalHiddenState,
        ]);
        settings = settings.target_features(probe_features);
        let gguf = Arc::new(if storage_native_cpu {
            GgufFile::open_storage_native(path)?
        } else {
            GgufFile::open(path)?
        });
        let tokenizer = load_tokenizer(&gguf)?;

        let spec = ModelSpec::from_gguf(&gguf)?;
        tracing::info!(
            arch = %spec.arch_id,
            n_layers = spec.n_layers,
            "loaded model"
        );

        let model_max_ctx = spec.context_length.max(1);
        let max_seq = settings.resolve_n_ctx(model_max_ctx);
        let n_ubatch = settings.resolve_ubatch();
        let n_batch = settings.n_batch.max(1);

        tracing::info!(
            n_ctx = max_seq,
            n_batch,
            n_ubatch,
            "context"
        );
        tracing::info!(backend = backend.id(), "backend");

        // Discover dynamic plugins and prepare attention / KV-policy providers.
        let mut providers = crate::providers::ProviderManager::new(settings.providers.clone());
        providers.load_plugins(settings.plugin_dir.as_deref())?;
        let prep = providers.prepare(
            backend.as_ref(),
            spec.n_heads.max(1) as u32,
            spec.n_kv_heads.max(1) as u32,
            spec.head_dim.max(1) as u32,
            spec.n_layers.max(1) as u32,
        )?;
        tracing::info!(
            attention = %prep.attention_name,
            kv_policy = %prep.kv_policy_name,
            "providers"
        );
        tracing::debug!(
            attention_id = prep.attention_id.0,
            kv_policy_id = prep.kv_policy_id.0,
            "provider ids"
        );
        for note in &prep.report.notes {
            tracing::debug!(%note, "provider selection");
        }
        // Install prepared attention paths so launchers dispatch FA2/FA3 kernels.
        install_attention_dispatch(&providers, backend.id());
        // Pre-RoPE key store for sequence-state policies.
        set_pre_rope_store(Some(PreRopeKeyStore::new(
            spec.n_attn_layers().max(1),
            spec.n_kv_heads.max(1),
            spec.head_dim.max(1),
            max_seq,
        )));

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
            tracing::debug!(
                architecture = %preparation.program.architecture_id,
                graphs = preparation.program.graphs.len(),
                "architecture plugin program ready"
            );
        }
        let architecture_program = preparation.as_ref().map(|p| p.program.clone());
        let n_attn_layers = spec.n_attn_layers().max(1);
        let backend_memory = backend.memory_info();
        let accelerator_memory = (backend.id() == "cuda").then_some(backend_memory).flatten();
        let mut model = LoadedModel::new(
            &gguf,
            spec,
            max_seq,
            model_max_ctx,
            preparation,
            &settings.kv_cache,
            backend_memory,
            accelerator_memory,
            n_ubatch,
            &settings.memory_fabric,
            &settings.target_features,
        )?;

        // B2: size VRAM KV arena to match the host fabric arena.
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
            cuda.configure_cpu_partitions(
                model.memory_fabric.cpu_compute_weights(),
                model.memory_fabric.cpu_compute_ops(),
            );
            let n_pages = model.cache.n_pages();
            let host_pages = (n_pages / 2).max(1);
            if let Err(e) = cuda.init_kv_arena(
                n_pages,
                model.spec.n_kv_heads.max(1),
                model.spec.head_dim.max(1),
                host_pages,
            ) {
                tracing::warn!(error = %e, "DeviceKvArena init failed; fabric ops stay host-only");
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
                providers: settings.providers.clone(),
                plugin_dir: settings.plugin_dir.clone(),
                kv_cache: settings.kv_cache.clone(),
                memory_fabric: settings.memory_fabric.clone(),
                target_features: settings.target_features.clone(),
            },
            providers,
            seq_attn: SequenceAttentionState::new(n_attn_layers),
            tokens_since_compress: 0,
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
            let n_pages = eng.model.cache.n_pages();
            let _ = cuda.init_kv_arena(
                n_pages,
                eng.model.spec.n_kv_heads.max(1),
                eng.model.spec.head_dim.max(1),
                (n_pages / 2).max(1),
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

    fn chat_render_options(&self, params: &GenParams) -> ChatRenderOptions {
        let enable_thinking = params.enable_thinking.or_else(|| {
            self.tokenizer.supports_thinking().then_some(false)
        });
        ChatRenderOptions { enable_thinking }
    }

    fn invalidate_hybrid_device_mirrors(&self) {
        #[cfg(feature = "backend-cuda")]
        if let (Some(conv), Some(cuda)) = (
            &self.model.conv,
            self.backend
                .as_any()
                .downcast_ref::<backend_cuda::CudaBackend>(),
        ) {
            for buffer in conv.conv.iter().chain(conv.ssm.iter()) {
                let host = buffer.borrow();
                let bytes = host.as_slice();
                if bytes.is_empty() {
                    continue;
                }
                cuda.plugins().invalidate_f32_host(
                    bytes.as_ptr() as *const f32,
                    bytes.len(),
                );
            }
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
    pub fn settings(&self) -> &EngineSettings {
        &self.settings
    }

    /// Prepared capability providers (attention, KV policy).
    #[must_use]
    pub fn providers(&self) -> &crate::providers::ProviderManager {
        &self.providers
    }

    /// Run the active sequence-state policy: score → select → compact → reclaim.
    pub fn maybe_compress_kv(&mut self) -> Result<()> {
        let policy_name = self
            .providers
            .prepared()
            .map(|p| p.kv_policy_name.as_str())
            .unwrap_or("kv.full");
        let Some(policy) = self.providers.capabilities().sequence_policy(policy_name) else {
            return Ok(());
        };
        // Sync live index + views from KvSequence before scoring.
        self.sync_seq_attn_from_cache();
        let config = self.settings.providers.config.clone();
        // Pre-RoPE packed in **live private candidate order** only (no ghost abs).
        let private = self.seq_attn.live_private_positions();
        let pre_rope = fellm_plugin_abi::pre_rope_gather(0, &private);
        let ctx = RetentionContext {
            state: &self.seq_attn,
            layer: None,
            current_pos: self
                .model
                .seq
                .absolute_pos
                .saturating_sub(1)
                .max(self.model.seq.len_tokens.saturating_sub(1)) as u32,
            tokens_since_window: self.tokens_since_compress,
            pre_rope_keys: pre_rope.as_deref(),
            head_dim: self.model.spec.head_dim as u32,
            n_kv_heads: self.model.spec.n_kv_heads as u32,
            n_heads: self.model.spec.n_heads as u32,
            config: &config,
        };
        if !policy.should_compress(&ctx) {
            return Ok(());
        }
        let mut plan = policy.plan_retention(&ctx)?;
        // Guard: never retain abs ids outside the live set at plan time.
        let live_set: std::collections::BTreeSet<u32> = self
            .seq_attn
            .live_retained_positions()
            .into_iter()
            .collect();
        plan.retain_positions
            .retain(|p| live_set.contains(p) || *p < self.seq_attn.shared_prefix_len);
        plan.retain_positions.sort_unstable();
        plan.retain_positions.dedup();

        let free_before = self.model.cache.free_count();
        let dense_before = self.model.seq.len_tokens;
        let live_before = self.seq_attn.live_retained_positions();
        if plan.compact && !plan.retain_positions.is_empty() {
            let reclaimed = self.model.cache.compact_sequence_to_positions(
                &mut self.model.seq,
                &plan.retain_positions,
                crate::kv_fabric::BLOCK_SIZE,
            );
            // Refresh densified live index + views; mark host as layout owner.
            self.sync_seq_attn_from_cache();
            self.seq_attn.layout_owner = fellm_plugin_abi::LayoutOwner::HostDensified;
            plan.host_densified = true;
            // Prune pre-RoPE ghosts so the next window cannot score evicted abs.
            let live_now = self.seq_attn.live_retained_positions();
            fellm_plugin_abi::pre_rope_prune(0, &live_now);
            let stats = policy.apply_plan(&mut self.seq_attn, &plan)?;
            tracing::info!(
                retained = self.model.seq.len_tokens,
                dense_before,
                reclaimed,
                free_before,
                free_after = self.model.cache.free_count(),
                live_before = live_before.len(),
                live_after = live_now.len(),
                policy_retained = stats.retained_count,
                "sequence-state policy compacted KV (densify + rebuild + prune pre-rope)"
            );
            #[cfg(feature = "backend-cuda")]
            if let Some(cuda) = self
                .backend
                .as_any()
                .downcast_ref::<backend_cuda::CudaBackend>()
            {
                cuda.mark_kv_host_dirty();
            }
        }
        self.tokens_since_compress = 0;
        Ok(())
    }

    /// Push live KvSequence state into `seq_attn` (single live-index contract).
    fn sync_seq_attn_from_cache(&mut self) {
        let seq = &self.model.seq;
        self.seq_attn.logical_len = seq.absolute_pos.max(seq.len_tokens) as u32;
        self.seq_attn.shared_prefix_len = seq.shared_prefix_len as u32;
        self.seq_attn.dense_len = seq.len_tokens as u32;
        // Live absolute positions: original_positions when compressed, else 0..len.
        self.seq_attn.live_positions = if seq.is_compressed() {
            seq.original_positions.clone()
        } else {
            (0..seq.len_tokens as u32).collect()
        };
        while self.seq_attn.layer_views.len() < seq.n_layers() {
            self.seq_attn.layer_views.push(Default::default());
        }
        for layer in 0..seq.n_layers() {
            let view = &mut self.seq_attn.layer_views[layer];
            view.layer = layer as u32;
            view.dense_block_table = self.model.cache.physical_block_table_layer(seq, layer);
            view.dense_seq_len = seq.len_tokens as u32;
            view.entries.clear();
            if seq.is_compressed() {
                for (dense_i, &abs) in seq.original_positions.iter().enumerate() {
                    let Ok((phys, slot)) = self.model.cache.locate(seq, layer, dense_i) else {
                        continue;
                    };
                    view.entries.push(fellm_plugin_abi::RetainedEntry {
                        logical_pos: abs,
                        physical_block: phys.0,
                        physical_slot: slot as u16,
                        kv_head: u16::MAX,
                        layer: layer as u16,
                    });
                }
            }
            view.ownership = Some(if seq.shared_prefix_len > 0 {
                fellm_plugin_abi::StateOwnership::SharedImmutable
            } else {
                fellm_plugin_abi::StateOwnership::RequestPrivate
            });
        }
        // Default layout owner is host densified after compact; otherwise policy view.
        if seq.is_compressed() {
            self.seq_attn.layout_owner = fellm_plugin_abi::LayoutOwner::HostDensified;
        }
    }

    /// Generate tokens from a raw prompt string (completion mode).
    ///
    /// Does **not** apply a chat template. Prefer [`Engine::chat`] for
    /// instruction-tuned models.
    pub fn generate(&mut self, prompt: &str, params: GenParams) -> Result<TokenStream<'_>> {
        self.model.reset();
        self.invalidate_hybrid_device_mirrors();
        let ids = self.tokenizer.encode(prompt, true)?;
        self.log_prompt_tokens(prompt, &ids);
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
        self.invalidate_hybrid_device_mirrors();
        let prepared_messages = self.prepare_chat_messages(messages);
        let prompt = match self.tokenizer.apply_chat_template_with_options(
            &prepared_messages,
            tools,
            true,
            self.chat_render_options(&params),
        )? {
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
        self.log_prompt_tokens(&prompt, &ids);
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
                    let need_logits = abs + 1 == n_prompt;
                    crate::activation::set_probe_step(abs == 0 || need_logits);
                    logits = Some(self.step(ids[abs], abs, need_logits)?);
                    crate::activation::set_probe_step(false);
                    if abs == 0 {
                        self.probe_named_activation("feature.embedding", true)?;
                        self.probe_named_activation("feature.layer.0", true)?;
                    }
                    if need_logits {
                        let _ = self.probe_named_activation("feature.final", false);
                    }
                }
                done += chunk;
            }
            pos += scheduled;
        }

        let prompt_elapsed = gen_start.elapsed();
        let last_logits = logits.ok_or_else(|| FellmError::other("empty prompt"))?;
        {
            let values = last_logits.as_slice::<f32>()?;
            crate::activation::require_nonzero("logits", values)?;
        }
        let start_pos = n_prompt;
        let mut sampler_state = sampling::SamplerState::with_grammar(
            params.max_tokens as usize,
            params.grammar.clone(),
        );
        sampler_state.prime_history(ids);

        Ok(TokenStream {
            engine: self,
            params,
            pending_logits: Some(last_logits),
            prefetched: std::collections::VecDeque::new(),
            emitted: 0,
            position: start_pos,
            finished: false,
            stop_token_ids,
            sampler_state,
            sampling: sampling::SamplingWorkspace::default(),
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
        let mut sampler_state = sampling::SamplerState::with_grammar(
            params.max_tokens as usize,
            params.grammar.clone(),
        );
        sampler_state.prime_history(ids);
        Ok(TokenStream {
            engine: self,
            params,
            pending_logits: None,
            prefetched: emitted.into_iter().collect(),
            emitted: 0,
            position: context_len,
            finished: false,
            stop_token_ids,
            sampler_state,
            sampling: sampling::SamplingWorkspace::default(),
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

    fn log_prompt_tokens(&self, prompt: &str, ids: &[u32]) {
        let pieces: Vec<String> = ids
            .iter()
            .map(|&id| {
                self.tokenizer
                    .vocabulary_piece(id)
                    .unwrap_or("?")
                    .to_string()
            })
            .collect();
        let types: Vec<i32> = ids
            .iter()
            .map(|&id| self.tokenizer.token_type(id).unwrap_or(-1))
            .collect();
        tracing::debug!(
            n_tokens = ids.len(),
            bos = self.tokenizer.bos(),
            eos = self.tokenizer.eos(),
            bos_str = self.tokenizer.bos_str().unwrap_or(""),
            starts_with_bos = ids.first().copied() == self.tokenizer.bos(),
            ids = ?ids,
            pieces = ?pieces,
            token_types = ?types,
            prompt_chars = prompt.chars().count(),
            "prompt tokenized"
        );
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

    /// Free pages remaining in the fabric device arena.
    #[must_use]
    pub fn cache_free_blocks(&self) -> usize {
        self.model.cache.free_count()
    }

    /// Whether the fabric can currently admit `need_pages` of work.
    #[must_use]
    pub fn can_admit_pages(&self, need_pages: usize) -> bool {
        self.model.cache.can_admit(need_pages)
    }

    /// Pages required for a sequence to absorb `extra_tokens`.
    #[must_use]
    pub fn pages_needed_for(&self, seq: &KvSequence, extra_tokens: usize) -> usize {
        self.model.cache.pages_needed_for(seq, extra_tokens)
    }

    /// Residency / cost signals for scheduling policy.
    #[must_use]
    pub fn residency_signals_for(
        &self,
        seq: &KvSequence,
        extra_tokens: usize,
        priority: i32,
        prefix_hit_tokens: usize,
    ) -> crate::kv_fabric::ResidencySignals {
        self.model
            .cache
            .residency_signals(seq, extra_tokens, priority, prefix_hit_tokens)
    }

    /// Current fabric memory pressure in `[0, 1]`.
    #[must_use]
    pub fn model_cache_pressure(&self) -> f64 {
        self.model.cache.memory_pressure()
    }

    /// Value/cost keep-score for a sequence (higher = prefer keep resident).
    /// Used as the primary preemption ranking input.
    #[must_use]
    pub fn sequence_keep_value(&self, seq: &KvSequence, priority: i32, last_used: u64) -> f64 {
        self.model
            .cache
            .sequence_keep_value(seq, priority, last_used)
    }

    #[must_use]
    pub fn backend_memory_info(&self) -> Option<fellm_plugin_abi::DeviceMemoryInfo> {
        self.backend.memory_info()
    }

    /// Joint weights/KV/activation placement and fabric telemetry.
    #[must_use]
    pub fn memory_fabric_snapshot(&self) -> crate::memory_fabric::MemoryFabricSnapshot {
        self.model.memory_fabric.snapshot()
    }

    /// Materialize one explicitly requested feature from the most recent
    /// single-token target step. No feature is retained unless requested in settings.
    pub fn capture_target_feature(
        &self,
        feature: fellm_plugin_abi::TargetFeature,
    ) -> Result<Tensor> {
        if !self.model.target_features.contains(&feature) {
            return Err(FellmError::other(format!(
                "target feature {feature:?} was not requested during engine preparation"
            )));
        }
        let name = target_feature_output_name(feature);
        self.model
            .compiled
            .as_ref()
            .ok_or_else(|| FellmError::other("target step has not executed"))?
            .materialize_named_output(self.backend.as_ref(), &name)
    }

    fn probe_named_activation(&self, name: &str, require_nonzero: bool) -> Result<()> {
        let compiled = self
            .model
            .compiled
            .as_ref()
            .ok_or_else(|| FellmError::other("target step has not executed"))?;
        let view = match compiled.named_output_ref(name) {
            Ok(view) => view,
            Err(_) => return Ok(()),
        };
        if view.data.is_null() || view.byte_len < 4 {
            if require_nonzero {
                return Err(FellmError::other(format!(
                    "{name} is missing; inference result is invalid"
                )));
            }
            return Ok(());
        }
        let values = unsafe {
            std::slice::from_raw_parts(view.data.cast::<f32>(), (view.byte_len as usize) / 4)
        };
        if require_nonzero {
            crate::activation::require_nonzero(name, values)?;
        } else {
            crate::activation::log_activation(name, values);
        }
        Ok(())
    }

    /// Borrow all requested target features in-place for one synchronous
    /// speculator call. The references cannot escape the callback or overlap a
    /// subsequent target execution that reuses the activation arena.
    pub fn with_target_features<R>(
        &self,
        callback: impl FnOnce(&[fellm_plugin_abi::CapturedTargetFeature<'_>]) -> Result<R>,
    ) -> Result<R> {
        let compiled = self
            .model
            .compiled
            .as_ref()
            .ok_or_else(|| FellmError::other("target step has not executed"))?;
        let device = match self.backend.capabilities().device_kind {
            fellm_plugin_abi::DeviceKind::Cpu => fellm_plugin_abi::DeviceKind::Cpu,
            fellm_plugin_abi::DeviceKind::Gpu => fellm_plugin_abi::DeviceKind::Gpu,
        };
        let mut captures = Vec::with_capacity(self.model.target_features.len());
        for &feature in &self.model.target_features {
            captures.push(fellm_plugin_abi::CapturedTargetFeature::new(
                feature,
                compiled.named_output_ref(&target_feature_output_name(feature))?,
                device,
            ));
        }
        callback(&captures)
    }

    /// Publish live cross-tier weight and transfer-provider telemetry.
    pub fn publish_memory_fabric_metrics(&self) {
        self.model.memory_fabric.publish_metrics();
        if let Some(cpu) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cpu::CpuBackend>()
        {
            let snapshot = cpu.storage_metrics();
            metrics::counter!("fellm_memory_transfer_bytes_total", "path" => "storage_to_host_weights")
                .absolute(snapshot.physical_bytes);
            metrics::counter!("fellm_memory_mmap_execution_bytes_total")
                .absolute(snapshot.mmap_execution_bytes);
            metrics::gauge!("fellm_memory_stall_seconds", "provider" => "cpu_storage")
                .set(snapshot.storage_stall_nanos as f64 / 1_000_000_000.0);
        }
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
        {
            let snapshot = cuda.metrics();
            metrics::gauge!("fellm_memory_resident_bytes", "tier" => "device_weights")
                .set(snapshot.weight_resident_bytes as f64);
            metrics::counter!("fellm_memory_transfer_bytes_total", "path" => "host_to_device_weights")
                .absolute(snapshot.weight_h2d_bytes);
            metrics::counter!("fellm_memory_prefetch_hits_total")
                .absolute(snapshot.weight_prefetch_hits);
            metrics::counter!("fellm_memory_prefetch_misses_total")
                .absolute(snapshot.weight_prefetch_misses);
            metrics::counter!("fellm_memory_evictions_total", "consumer" => "weights")
                .absolute(snapshot.weight_evictions);
            metrics::counter!("fellm_memory_transfer_bytes_total", "path" => "storage_to_host_weights")
                .absolute(snapshot.storage_read_bytes);
            metrics::counter!("fellm_memory_prefetch_hits_total", "provider" => "storage")
                .absolute(snapshot.storage_prefetch_hits);
            metrics::counter!("fellm_memory_prefetch_misses_total", "provider" => "storage")
                .absolute(snapshot.storage_prefetch_misses);
            metrics::gauge!("fellm_memory_stall_seconds", "provider" => "storage")
                .set(snapshot.storage_wait_nanos as f64 / 1_000_000_000.0);
            metrics::counter!("fellm_memory_cpu_partition_ops_total")
                .absolute(snapshot.cpu_partition_count);
        }
    }

    /// Emit the concrete residency and transfer counters used by the completed run.
    pub fn log_memory_fabric_runtime(&self, _evaluated_tokens: u64) {
        if let Some(cpu) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cpu::CpuBackend>()
        {
            let snapshot = cpu.storage_metrics();
            let mut system = sysinfo::System::new_with_specifics(
                sysinfo::RefreshKind::nothing()
                    .with_processes(sysinfo::ProcessRefreshKind::nothing().with_memory()),
            );
            let resident_set_bytes = sysinfo::get_current_pid()
                .ok()
                .and_then(|pid| {
                    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                    system.process(pid).map(sysinfo::Process::memory)
                })
                .unwrap_or(0);
            tracing::debug!(
                storage_read_bytes = snapshot.physical_bytes,
                buffered_storage_bytes = snapshot.buffered_storage_bytes,
                direct_storage_bytes = snapshot.direct_storage_bytes,
                storage_physical_reads = snapshot.physical_reads,
                storage_bytes_per_token =
                    snapshot.physical_bytes as f64 / _evaluated_tokens.max(1) as f64,
                mmap_execution_bytes = snapshot.mmap_execution_bytes,
                storage_wait_nanos = snapshot.storage_stall_nanos,
                io_compute_overlap_percent = snapshot.io_compute_overlap_percent,
                expert_slice_reads = snapshot.expert_slice_reads,
                expert_slice_bytes = snapshot.expert_slice_bytes,
                avg_storage_read_bytes = snapshot.avg_read_bytes,
                resident_dense_bytes = snapshot.resident_dense_bytes,
                resident_set_bytes,
                peak_rss_bytes = process_peak_rss_bytes(),
                zero_mmap_weight_execution = snapshot.mmap_execution_bytes == 0,
                "CPU Memory Fabric runtime counters"
            );
            tracing::info!(
                storage_read_gb = snapshot.physical_bytes as f64 / 1e9,
                storage_reads = snapshot.physical_reads,
                rss_gb = resident_set_bytes as f64 / 1e9,
                "storage"
            );
        }
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
        {
            let snapshot = cuda.metrics();
            tracing::debug!(
                device_weight_resident_bytes = self
                    .model
                    .memory_fabric
                    .snapshot()
                    .plan
                    .budget
                    .weights_device
                    .saturating_add(snapshot.weight_resident_bytes),
                weight_h2d_bytes = snapshot.weight_h2d_bytes,
                weight_prefetch_hits = snapshot.weight_prefetch_hits,
                weight_prefetch_misses = snapshot.weight_prefetch_misses,
                weight_evictions = snapshot.weight_evictions,
                storage_read_bytes = snapshot.storage_read_bytes,
                storage_useful_bytes = snapshot.storage_useful_bytes,
                storage_physical_reads = snapshot.storage_physical_reads,
                storage_failed_reads = snapshot.storage_failed_reads,
                storage_read_latency_p50_nanos = snapshot.storage_latency_p50_nanos,
                storage_read_latency_p95_nanos = snapshot.storage_latency_p95_nanos,
                storage_read_amplification = snapshot.storage_read_bytes as f64
                    / snapshot.storage_useful_bytes.max(1) as f64,
                storage_bytes_per_token =
                    snapshot.storage_read_bytes as f64 / _evaluated_tokens.max(1) as f64,
                weight_h2d_bytes_per_token =
                    snapshot.weight_h2d_bytes as f64 / _evaluated_tokens.max(1) as f64,
                storage_wait_nanos = snapshot.storage_wait_nanos,
                storage_prefetch_hits = snapshot.storage_prefetch_hits,
                storage_prefetch_misses = snapshot.storage_prefetch_misses,
                storage_resident_hits = snapshot.storage_resident_hits,
                storage_required_requests = snapshot.storage_required_requests,
                storage_true_resident_hit_rate = snapshot.storage_resident_hits as f64
                    / snapshot.storage_required_requests.max(1) as f64,
                storage_prefetch_precision = snapshot.storage_prefetch_hits as f64
                    / snapshot
                        .storage_prefetch_hits
                        .saturating_add(snapshot.storage_prefetch_misses)
                        .max(1) as f64,
                cpu_partition_ops = snapshot.cpu_partition_count,
                cpu_fallback_ops = snapshot.cpu_fallback_count,
                "Memory Fabric runtime counters"
            );
        }
    }

    #[must_use]
    pub fn model_bytes(&self) -> u64 {
        self.gguf.tensors().fold(0u64, |total, tensor| {
            total.saturating_add(tensor.dtype.byte_size(tensor.shape.num_elements()) as u64)
        })
    }

    #[must_use]
    pub fn activation_arena_bytes(&self) -> usize {
        self.model
            .step_plan
            .memory
            .arena_bytes
            .saturating_add(
                self.model
                    .canvas_plan
                    .as_ref()
                    .map_or(0, |plan| plan.memory.arena_bytes),
            )
            .saturating_add(self.model.batch_activation_reserve)
    }

    #[must_use]
    pub fn cache_total_blocks(&self) -> usize {
        self.model.cache.n_pages()
    }

    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        self.model
            .cache
            .n_pages()
            .saturating_mul(self.model.cache.page_bytes())
    }

    #[must_use]
    pub fn prefix_cache_stats(&self) -> crate::kv_fabric::PrefixCacheStats {
        self.model.cache.prefix_stats()
    }

    /// Fabric metrics snapshot (residency, migrations, prefix, pressure).
    #[must_use]
    pub fn fabric_metrics(&self) -> crate::kv_fabric::FabricMetrics {
        self.model.cache.metrics()
    }

    /// Reclaim idle cached prefixes, preserving every active sequence reference.
    pub fn evict_prefixes_for_blocks(&mut self, required_free: usize) -> usize {
        self.model.cache.evict_shared_until(required_free)
    }

    /// Attach content-addressed prefix for `ids` onto `seq_cache`. Returns matched tokens.
    ///
    /// Matched tokens become **immutable shared prefix** (`shared_prefix_len`);
    /// compression policies must not reclaim them in place.
    pub fn attach_prefix(&mut self, ids: &[u32], seq_cache: &mut KvSequence) -> usize {
        let matched = self.model.cache.attach_prefix(ids, seq_cache);
        #[cfg(feature = "backend-cuda")]
        if matched > 0 {
            if let Some(cuda) = self
                .backend
                .as_any()
                .downcast_ref::<backend_cuda::CudaBackend>()
            {
                cuda.mark_kv_host_dirty();
            }
        }
        matched
    }

    /// Insert a completed prompt into the content-addressed shared store.
    pub fn insert_prefix(&mut self, ids: &[u32], seq_cache: &KvSequence) -> Result<()> {
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
            && cuda.plugins_enabled()
        {
            cuda.sync_kv_device_to_host(self.model.cache.arena_bytes_mut())?;
        }
        self.model.cache.insert_prefix(ids, seq_cache);
        Ok(())
    }

    /// Ensure `pos` is writable in `seq_cache` (alloc / `CoW`).
    pub fn ensure_seq_writable(&mut self, seq_cache: &mut KvSequence, pos: usize) -> Result<()> {
        match self.model.cache.ensure_writable(seq_cache, pos) {
            Ok(()) => Ok(()),
            Err(first) => {
                let required = self.model.spec.n_attn_layers().max(1);
                if self.evict_prefixes_for_blocks(required) == 0 {
                    return Err(first);
                }
                self.model.cache.ensure_writable(seq_cache, pos)
            }
        }
    }

    /// Release physical refs held by a sequence cache.
    pub fn release_seq_cache(&mut self, seq_cache: &mut KvSequence) {
        self.model.cache.release_sequence(seq_cache);
    }

    /// Allocate isolated recurrent state for one scheduled sequence.
    pub fn new_hybrid_state(&self) -> Result<Option<HybridConvState>> {
        self.model
            .spec
            .is_hybrid()
            .then(|| {
                HybridConvState::new(
                    &self.model.spec.layer_kv_heads_for_state(),
                    self.model.spec.d_model,
                    self.model.spec.shortconv_l_cache,
                    self.model.spec.gdn_conv_kernel,
                    self.model.spec.gdn_inner_size,
                    self.model.spec.gdn_key_heads,
                    self.model.spec.gdn_value_heads,
                    self.model.spec.gdn_state_size,
                )
            })
            .transpose()
    }

    /// Migrate a sequence's pages to host tier (preempt / residency demotion).
    pub fn swap_out_sequence(&mut self, seq_cache: &mut KvSequence) -> Result<()> {
        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = self
            .backend
            .as_any()
            .downcast_ref::<backend_cuda::CudaBackend>()
            && cuda.plugins_enabled()
        {
            cuda.sync_kv_device_to_host(self.model.cache.arena_bytes_mut())?;
        }
        let bytes = self.model.cache.migrate_out(seq_cache)?;
        metrics::counter!("fellm_kv_migrations_total").increment(1);
        metrics::counter!("fellm_kv_migration_bytes_total").increment(bytes);
        Ok(())
    }

    /// Restore a non-resident sequence's pages to the compute tier.
    pub fn swap_in_sequence(&mut self, seq_cache: &mut KvSequence) -> Result<()> {
        metrics::counter!("fellm_kv_swap_in_total").increment(1);
        let bytes = self.model.cache.migrate_in(seq_cache)?;
        metrics::counter!("fellm_kv_migrations_total").increment(1);
        metrics::counter!("fellm_kv_migration_bytes_total").increment(bytes);
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
        seq_cache: &mut KvSequence,
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

    /// Run one forward step with request-owned recurrent state.
    pub fn step_sequence_state(
        &mut self,
        seq_cache: &mut KvSequence,
        conv: &mut Option<HybridConvState>,
        tok: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<Tensor> {
        std::mem::swap(&mut self.model.seq, seq_cache);
        if let (Some(bound), Some(request)) = (&mut self.model.conv, conv.as_ref()) {
            copy_hybrid_state(request, bound)?;
        }
        let result = self
            .model
            .step(self.backend.as_ref(), tok, pos, compute_logits);
        if let (Some(bound), Some(request)) = (&self.model.conv, conv.as_mut()) {
            // Export even on an execution error: the caller's transaction
            // checkpoint decides whether this provisional mutation is kept.
            copy_hybrid_state(bound, request)?;
        }
        std::mem::swap(&mut self.model.seq, seq_cache);
        result
    }

    /// Execute token rows from several sequence caches as one graph batch.
    /// Rows may also refer to the same sequence, enabling chunked prefill.
    pub fn step_batch(
        &mut self,
        sequences: &mut [&mut KvSequence],
        conv_states: &mut [&mut Option<HybridConvState>],
        rows: &[BatchToken],
    ) -> Result<Vec<Option<Tensor>>> {
        self.model.step_batch(
            &self.gguf,
            self.backend.as_ref(),
            sequences,
            conv_states,
            rows,
        )
    }

    /// Prefill an isolated sequence through the normal model/KV infrastructure.
    pub fn prefill_sequence(&mut self, token_ids: &[u32]) -> Result<DecodeSequence> {
        self.prefill_sequence_observing(token_ids, |_, _, _| Ok(()))
    }

    /// Prefill while synchronously exposing requested feature taps after every
    /// committed prompt token. This is used to catch up stateful speculators
    /// without retaining the entire prompt activation history.
    pub fn prefill_sequence_observing(
        &mut self,
        token_ids: &[u32],
        mut observe: impl FnMut(
            u32,
            usize,
            &[fellm_plugin_abi::CapturedTargetFeature<'_>],
        ) -> Result<()>,
    ) -> Result<DecodeSequence> {
        if token_ids.is_empty() {
            return Err(FellmError::other("empty prompt"));
        }
        if token_ids.len() >= self.model.max_seq {
            return Err(FellmError::other("prompt exceeds model context"));
        }
        let mut cache = self.model.cache.new_sequence(self.model.max_seq);
        let mut recurrent = self.new_hybrid_state()?;
        let mut pending_logits = None;
        for (position, &token) in token_ids.iter().enumerate() {
            pending_logits = Some(self.step_sequence_state(
                &mut cache,
                &mut recurrent,
                token,
                position,
                position + 1 == token_ids.len(),
            )?);
            self.with_target_features(|features| observe(token, position, features))?;
        }
        Ok(DecodeSequence {
            cache,
            recurrent,
            pending_logits: pending_logits.expect("non-empty prompt"),
            position: token_ids.len(),
        })
    }

    /// Consume one committed token and replace the sequence's next-token logits.
    pub fn advance_sequence(&mut self, sequence: &mut DecodeSequence, token: u32) -> Result<()> {
        if sequence.position >= self.model.max_seq {
            return Err(FellmError::other("sequence reached model context limit"));
        }
        sequence.pending_logits = self.step_sequence_state(
            &mut sequence.cache,
            &mut sequence.recurrent,
            token,
            sequence.position,
            true,
        )?;
        sequence.position += 1;
        Ok(())
    }

    #[must_use]
    pub fn begin_sequence_transaction(&self, sequence: &DecodeSequence) -> crate::KvTransaction {
        self.model.cache.begin_transaction(&sequence.cache)
    }

    /// Finalize a provisional sequence prefix and restore its logical cursor.
    pub fn finalize_sequence_transaction(
        &mut self,
        sequence: &mut DecodeSequence,
        transaction: &mut crate::KvTransaction,
        accepted_tokens: usize,
        recurrent_checkpoint: Option<HybridConvState>,
    ) -> Result<()> {
        let start = transaction.start_len();
        if accepted_tokens == 0 {
            self.model
                .cache
                .rollback_transaction(&mut sequence.cache, transaction)?;
            sequence.recurrent = recurrent_checkpoint;
        } else {
            self.model.cache.commit_transaction(
                &mut sequence.cache,
                transaction,
                accepted_tokens,
            )?;
        }
        sequence.position = start.saturating_add(accepted_tokens);
        Ok(())
    }

    /// Release all physical state owned by an isolated sequence.
    pub fn release_sequence(&mut self, mut sequence: DecodeSequence) {
        self.model.cache.release_sequence(&mut sequence.cache);
    }

    /// Score a proposed continuation in one physical target batch where the
    /// architecture supports it. KV writes remain provisional until finalized.
    pub fn verify_proposal(
        &mut self,
        seq_cache: &mut KvSequence,
        conv_state: &mut Option<HybridConvState>,
        initial_logits: &Tensor,
        proposed_tokens: &[u32],
        start_position: usize,
        params: &GenParams,
        sampler_state: &sampling::SamplerState,
    ) -> Result<crate::speculative::ProvisionalTargetVerification> {
        if proposed_tokens.is_empty() {
            return Err(FellmError::other("cannot verify an empty proposal"));
        }
        let transaction = self.model.cache.begin_transaction(seq_cache);
        let recurrent_checkpoint = conv_state.clone();
        let mut recurrent_prefixes = Vec::new();
        let mut feature_rows = Vec::with_capacity(proposed_tokens.len());
        let outputs = {
            let packed = false;
            if packed {
                let rows: Vec<BatchToken> = proposed_tokens
                    .iter()
                    .enumerate()
                    .map(|(offset, &token)| BatchToken {
                        sequence: 0,
                        token,
                        position: start_position + offset,
                        compute_logits: true,
                    })
                    .collect();
                match self.model.step_batch(
                    &self.gguf,
                    self.backend.as_ref(),
                    &mut [seq_cache],
                    &mut [conv_state],
                    &rows,
                ) {
                    Ok(outputs) => {
                        for _ in &rows {
                            recurrent_prefixes.push(conv_state.clone());
                        }
                        let k = rows.len();
                        if let Some(batch) = self.model.batches.get(&k) {
                            let mut split = vec![Vec::new(); k];
                            for &feature in &self.model.target_features {
                                let name = target_feature_output_name(feature);
                                let packed_feat = batch
                                    .step
                                    .materialize_named_output(self.backend.as_ref(), &name)?;
                                for row in 0..k {
                                    split[row].push((feature, packed_feat.row(row)?));
                                }
                            }
                            feature_rows = split;
                        }
                        outputs
                    }
                    Err(error) => {
                        let mut transaction = transaction;
                        let _ = self
                            .model
                            .cache
                            .rollback_transaction(seq_cache, &mut transaction);
                        *conv_state = recurrent_checkpoint;
                        return Err(error);
                    }
                }
            } else {
            let mut outputs = Vec::with_capacity(proposed_tokens.len());
            for (offset, &token) in proposed_tokens.iter().enumerate() {
                match self.step_sequence_state(
                    seq_cache,
                    conv_state,
                    token,
                    start_position + offset,
                    true,
                ) {
                    Ok(logits) => {
                        outputs.push(Some(logits));
                        recurrent_prefixes.push(conv_state.clone());
                        feature_rows.push(
                            self.model
                                .target_features
                                .iter()
                                .copied()
                                .map(|feature| {
                                    self.capture_target_feature(feature)
                                        .map(|tensor| (feature, tensor))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    Err(error) => {
                        let mut transaction = transaction;
                        let _ = self
                            .model
                            .cache
                            .rollback_transaction(seq_cache, &mut transaction);
                        *conv_state = recurrent_checkpoint;
                        return Err(error);
                    }
                }
            }
            outputs
            }
        };

        let mut speculative_sampler = sampler_state.clone();
        let mut workspace = sampling::SamplingWorkspace::default();
        let first = sampling::distribution_with_workspace(
            initial_logits.as_slice::<f32>()?,
            sampling::SamplingOptions {
                temperature: params.temperature,
                top_k: params.top_k,
                top_p: params.top_p,
                min_p: params.min_p,
                seed: params.seed.wrapping_add(speculative_sampler.draw_index()),
                repetition_penalty: params.repetition_penalty,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
                logit_bias: &params.logit_bias,
                grammar: speculative_sampler.grammar_view(),
                recent_tokens: speculative_sampler.history(),
            },
            &mut workspace,
        );
        let mut distributions = Vec::with_capacity(proposed_tokens.len() + 1);
        distributions.push((&first).into());
        for (&proposed, output) in proposed_tokens.iter().zip(outputs) {
            speculative_sampler.commit_token(proposed);
            let logits = output
                .ok_or_else(|| FellmError::other("target verification omitted requested logits"))?;
            let distribution = sampling::distribution_with_workspace(
                logits.as_slice::<f32>()?,
                sampling::SamplingOptions {
                    temperature: params.temperature,
                    top_k: params.top_k,
                    top_p: params.top_p,
                    min_p: params.min_p,
                    seed: params.seed.wrapping_add(speculative_sampler.draw_index()),
                    repetition_penalty: params.repetition_penalty,
                    frequency_penalty: params.frequency_penalty,
                    presence_penalty: params.presence_penalty,
                    logit_bias: &params.logit_bias,
                    grammar: speculative_sampler.grammar_view(),
                    recent_tokens: speculative_sampler.history(),
                },
                &mut workspace,
            );
            distributions.push((&distribution).into());
        }
        Ok(crate::speculative::ProvisionalTargetVerification {
            distributions,
            kv_transaction: transaction,
            recurrent_checkpoint,
            recurrent_prefixes,
            feature_rows,
        })
    }

    /// Score ragged linear continuations for several requests in one physical
    /// target batch. Each request retains an independent KV transaction, so a
    /// later acceptance phase may commit different prefix lengths safely.
    pub fn verify_proposals_ragged(
        &mut self,
        sequences: &mut [DecodeSequence],
        proposed_tokens: &[Vec<u32>],
        params: &[GenParams],
        sampler_states: &[sampling::SamplerState],
    ) -> Result<Vec<crate::speculative::ProvisionalTargetVerification>> {
        let request_count = sequences.len();
        if request_count == 0
            || proposed_tokens.len() != request_count
            || params.len() != request_count
            || sampler_states.len() != request_count
        {
            return Err(FellmError::other(
                "ragged verification request arrays must have the same non-zero length",
            ));
        }
        if proposed_tokens.iter().any(Vec::is_empty) {
            return Err(FellmError::other(
                "ragged verification cannot contain an empty proposal",
            ));
        }
        let mut results = Vec::with_capacity(request_count);
        for index in 0..request_count {
            let sequence = &mut sequences[index];
            results.push(self.verify_proposal(
                &mut sequence.cache,
                &mut sequence.recurrent,
                &sequence.pending_logits,
                &proposed_tokens[index],
                sequence.position,
                &params[index],
                &sampler_states[index],
            )?);
        }
        Ok(results)
    }

    /// Commit the accepted target KV prefix or restore all provisional state.
    pub fn finalize_verification(
        &mut self,
        seq_cache: &mut KvSequence,
        conv_state: &mut Option<HybridConvState>,
        mut verification: crate::speculative::ProvisionalTargetVerification,
        accepted_tokens: usize,
    ) -> Result<()> {
        if accepted_tokens > verification.distributions.len().saturating_sub(1) {
            return Err(FellmError::other(
                "accepted speculative prefix exceeds proposal length",
            ));
        }
        if !verification.recurrent_prefixes.is_empty() {
            *conv_state = if accepted_tokens == 0 {
                verification.recurrent_checkpoint.clone()
            } else {
                verification.recurrent_prefixes[accepted_tokens - 1].clone()
            };
        }
        if accepted_tokens == 0 {
            self.model
                .cache
                .rollback_transaction(seq_cache, &mut verification.kv_transaction)
        } else {
            self.model.cache.commit_transaction(
                seq_cache,
                &mut verification.kv_transaction,
                accepted_tokens,
            )
        }
    }
}

#[cfg(windows)]
fn process_peak_rss_bytes() -> usize {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut counters = PROCESS_MEMORY_COUNTERS::default();
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        0
    } else {
        counters.PeakWorkingSetSize
    }
}

#[cfg(not(windows))]
fn process_peak_rss_bytes() -> usize {
    0
}

impl LoadedModel {
    #[allow(dead_code)]
    fn batch_feature_rows(
        &self,
        backend: &dyn Backend,
        rows: usize,
    ) -> Result<Vec<Vec<(fellm_plugin_abi::TargetFeature, Tensor)>>> {
        if self.target_features.is_empty() {
            return Ok(vec![Vec::new(); rows]);
        }
        let batch = self
            .batches
            .get(&rows)
            .ok_or_else(|| FellmError::other("batch feature graph is not compiled"))?;
        let mut result = vec![Vec::with_capacity(self.target_features.len()); rows];
        for &feature in &self.target_features {
            let tensor = batch
                .step
                .materialize_named_output(backend, &target_feature_output_name(feature))?;
            for (row, target) in result.iter_mut().enumerate() {
                target.push((feature, tensor.row(row)?));
            }
        }
        Ok(result)
    }

    fn new(
        gguf: &GgufFile,
        spec: ModelSpec,
        max_seq: usize,
        model_max_ctx: usize,
        preparation: Option<ArchitecturePreparation>,
        kv_config: &KvFabricConfig,
        memory_info: Option<fellm_plugin_abi::DeviceMemoryInfo>,
        accelerator_memory: Option<fellm_plugin_abi::DeviceMemoryInfo>,
        physical_batch: usize,
        memory_fabric_config: &crate::memory_fabric::MemoryFabricConfig,
        target_features: &[fellm_plugin_abi::TargetFeature],
    ) -> Result<Self> {
        let n_attn = spec.n_attn_layers().max(1);
        let step_graph = build_step_graph_with_features(gguf, &spec, target_features)?;
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
        let weights_bytes = gguf.tensors().fold(0u64, |total, tensor| {
            total.saturating_add(tensor.dtype.byte_size(tensor.shape.num_elements()) as u64)
        });
        let batch_activation_reserve = if spec.is_hybrid()
            || physical_batch <= 1
            || !gguf.has_tensor("blk.0.attn_q.weight")
        {
            0
        } else {
            let graph =
                build_batch_step_graph_with_features(gguf, &spec, physical_batch, target_features)?;
            ExecutionPlan::from_graph(&graph)?
                .memory
                .arena_bytes
                .saturating_mul(2)
        };
        let activation_bytes = (step_plan.memory.arena_bytes as u64)
            .saturating_add(
                canvas_plan
                    .as_ref()
                    .map_or(0, |plan| plan.memory.arena_bytes as u64),
            )
            .saturating_add(
                self_conditioning_buffer
                    .as_ref()
                    .map_or(0, |buffer| buffer.borrow().len() as u64),
            )
            .saturating_add(batch_activation_reserve as u64);
        let page_bytes = crate::kv_fabric::storage::PageArena::page_bytes_for_dims(
            spec.n_kv_heads.max(1),
            spec.head_dim.max(1),
            STANDARD_PAGE_TOKENS,
            kv_config.default_encoding,
        );
        let desired_kv_bytes = (max_seq.div_ceil(STANDARD_PAGE_TOKENS) as u64)
            .saturating_mul(n_attn as u64)
            .saturating_mul(page_bytes as u64);
        let memory_fabric = crate::memory_fabric::MemoryFabric::inspect_and_plan(
            gguf,
            &step_graph,
            &step_plan,
            accelerator_memory,
            desired_kv_bytes,
            activation_bytes,
            memory_fabric_config,
        )
        .map_err(|error| FellmError::other(format!("memory fabric planning failed: {error:?}")))?;
        let fabric_snapshot = memory_fabric.snapshot();
        let mut resolved_kv_config = kv_config.clone();
        if resolved_kv_config.device_budget.is_none() {
            resolved_kv_config.device_budget = Some(fabric_snapshot.plan.budget.kv_device);
        }
        if resolved_kv_config.host_budget.is_none() || resolved_kv_config.host_budget == Some(0) {
            resolved_kv_config.host_budget = Some(fabric_snapshot.plan.budget.kv_host);
        }
        let memory_plan = KvMemoryPlan::resolve(
            &resolved_kv_config,
            memory_info,
            weights_bytes,
            activation_bytes,
            page_bytes,
            n_attn,
            if accelerator_memory.is_some() {
                crate::kv_fabric::KvExecutionMemory::Accelerator
            } else {
                crate::kv_fabric::KvExecutionMemory::Host
            },
        )?;
        let cache = KvFabric::new_full_attention(
            resolved_kv_config.clone(),
            memory_plan.execution_pages,
            n_attn,
            spec.n_kv_heads.max(1),
            spec.head_dim.max(1),
            memory_plan.overflow_host_pages,
        )?;
        let seq = cache.new_sequence(max_seq);
        let dummy_kv = DummyKvBuffers::new(
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
                spec.gdn_conv_kernel,
                spec.gdn_inner_size,
                spec.gdn_key_heads,
                spec.gdn_value_heads,
                spec.gdn_state_size,
            )?)
        } else {
            None
        };
        tracing::info!(
            weights_gb = weights_bytes as f64 / 1e9,
            kv_mb = fabric_snapshot.plan.budget.kv_host as f64 / 1e6,
            staging_mb = fabric_snapshot.plan.budget.host_staging as f64 / 1e6,
            "memory"
        );
        tracing::debug!(
            weight_device_bytes = fabric_snapshot.plan.budget.weights_device,
            weight_host_bytes = fabric_snapshot.plan.budget.weights_host,
            weight_storage_bytes = weights_bytes,
            device_staging_bytes = fabric_snapshot.plan.budget.device_staging,
            host_staging_bytes = fabric_snapshot.plan.budget.host_staging,
            kv_device_bytes = fabric_snapshot.plan.budget.kv_device,
            kv_host_bytes = fabric_snapshot.plan.budget.kv_host,
            execution_groups = fabric_snapshot.plan.placements.len(),
            permanent_groups =
                fabric_snapshot
                    .plan
                    .placements
                    .iter()
                    .filter(|placement| placement.class
                        == fellm_memory::ResidencyClass::PermanentDevice)
                    .count(),
            host_stream_groups = fabric_snapshot
                .plan
                .placements
                .iter()
                .filter(|placement| placement.class == fellm_memory::ResidencyClass::HostResident)
                .count(),
            storage_stream_groups = fabric_snapshot
                .plan
                .placements
                .iter()
                .filter(|placement| placement.class == fellm_memory::ResidencyClass::StorageStream)
                .count(),
            cpu_compute_groups = fabric_snapshot
                .plan
                .placements
                .iter()
                .filter(|placement| placement.class == fellm_memory::ResidencyClass::CpuCompute)
                .count(),
            device_buffers = fabric_snapshot.plan.device_buffer_count,
            host_buffers = fabric_snapshot.plan.host_buffer_count,
            storage_queue_depth = fabric_snapshot.plan.storage_queue_depth,
            "selected automatic Memory Fabric plan"
        );
        memory_fabric.publish_metrics();
        tracing::debug!(
            weights_bytes = memory_plan.weights_bytes,
            activation_bytes = memory_plan.activation_bytes,
            kv_page_bytes = memory_plan.page_bytes,
            execution_memory = ?memory_plan.execution_memory,
            kv_execution_pages = memory_plan.execution_pages,
            kv_execution_bytes = memory_plan.execution_bytes,
            overflow_host_bytes = memory_plan.overflow_host_bytes,
            mode = ?kv_config.mode,
            addressing = ?kv_config.addressing,
            remaining_reserve_bytes = ?memory_plan.remaining_reserve_bytes,
            "resolved KV fabric memory budget"
        );
        tracing::debug!(
            nodes = step_graph.node_count(),
            rope = bindings.rope.len(),
            kv_write = bindings.kv_write.len(),
            attention = bindings.attention.len(),
            attn_layers = spec.n_attn_layers(),
            conv_layers = spec.n_conv_layers(),
            fabric_pages = cache.n_pages(),
            "step graph ready (KV fabric)"
        );

        Ok(Self {
            spec,
            memory_fabric,
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
            target_features: target_features.to_vec(),
            compiled: None,
            batches: HashMap::new(),
            batch_activation_reserve,
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
                    OpValue::Constant(tensor) => Some(backend_cuda::WeightBlob {
                        id: self
                            .memory_fabric
                            .weight_id_for_tensor(tensor)
                            .unwrap_or(fellm_memory::WeightId((1u64 << 63) | index as u64)),
                        tensor: fellm_plugin_abi::PlanTensorId(index as u32),
                        bytes: tensor.as_bytes(),
                        alignment: 128,
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let storage_tensor_refs = blobs
                .iter()
                .map(|blob| {
                    // SAFETY: model storage owns every blob for the engine lifetime.
                    unsafe {
                        fellm_plugin_abi::TensorRef::from_raw(
                            DType::U8,
                            &[blob.bytes.len() as u64],
                            &[1],
                            blob.bytes.as_ptr(),
                            blob.bytes.len(),
                        )
                    }
                    .with_logical_id(blob.id.0)
                })
                .collect::<Vec<_>>();
            let fabric_plan = self.memory_fabric.snapshot().plan;
            let (storage_weights, host_buffers, host_buffer_bytes) =
                self.memory_fabric.storage_stream_configuration();
            cuda.configure_weight_storage(
                &storage_weights,
                self.memory_fabric.storage_objects(),
                &storage_tensor_refs,
                self.memory_fabric.select_storage_provider(true)?,
                host_buffers,
                host_buffer_bytes,
            )?;
            cuda.set_weight_cache_budget(
                fabric_plan.budget.device_staging.max(1),
                u32::from(fabric_plan.device_buffer_count.max(1)),
            )?;
            let all_permanent = fabric_plan
                .placements
                .iter()
                .all(|placement| placement.class == fellm_memory::ResidencyClass::PermanentDevice);
            cuda.set_weight_streaming_enabled(!all_permanent);
            let permanent_ids = self.memory_fabric.permanent_device_weights();
            let resident_blobs = blobs
                .iter()
                .filter(|blob| permanent_ids.contains(&blob.id))
                .map(|blob| backend_cuda::WeightBlob {
                    id: blob.id,
                    tensor: blob.tensor,
                    bytes: blob.bytes,
                    alignment: blob.alignment,
                })
                .collect::<Vec<_>>();
            let weights = if resident_blobs.is_empty() {
                None
            } else {
                match backend_cuda::CudaWeightFabric::materialize(
                    cuda.device_state(),
                    &resident_blobs,
                ) {
                    Ok(weights) => Some(weights),
                    Err(error) => {
                        let attempted = resident_blobs.iter().fold(0u64, |total, blob| {
                            total.saturating_add(blob.bytes.len() as u64)
                        });
                        let replanned = self
                            .memory_fabric
                            .replan_after_pressure(fellm_memory::MemoryDomain::Device, attempted)
                            .map_err(|plan_error| {
                                FellmError::other(format!(
                                    "CUDA weight allocation failed ({error}); pressure replan failed: {plan_error:?}"
                                ))
                            })?;
                        tracing::warn!(
                            error = %error,
                            device_staging_bytes = replanned.budget.device_staging,
                            "CUDA allocation pressure demoted weights to bounded streaming"
                        );
                        None
                    }
                }
            };
            if let Some(resident) = &weights {
                // Establish ownership across the host and dynamically loaded plugin streams.
                cuda.synchronize()?;
                for blob in &resident_blobs {
                    let (device_ptr, len) = resident.resolve(blob.tensor)?;
                    debug_assert_eq!(len, blob.bytes.len());
                    cuda.register_device_tensor(blob.bytes.as_ptr(), blob.bytes.len(), device_ptr)?;
                }
            }
            let page_table_capacity = self
                .cache
                .n_layers()
                .saturating_mul(self.max_seq.div_ceil(crate::kv_fabric::BLOCK_SIZE));
            let device = backend_cuda::DecodeDeviceState::new(
                cuda.device_state(),
                physical.arena_bytes,
                page_table_capacity,
                self.cache.n_layers(),
            )?;
            let tensors = device.arena.resolve(&physical, &lowered.tensors)?;
            let graph_replay_safe = all_permanent
                && weights.is_some()
                && lowered
                    .operations
                    .iter()
                    .find(|operation| operation.kind == fellm_plugin_abi::MacroOpKind::Embedding)
                    .is_some_and(|operation| {
                        operation.inputs.iter().any(|tensor| {
                            lowered.tensors.iter().any(|desc| {
                                desc.id == *tensor
                                    && desc.storage == fellm_plugin_abi::StorageClass::Model
                                    && matches!(
                                        desc.dtype,
                                        DType::Q4K | DType::Q5K | DType::Q6K | DType::Q8_0
                                    )
                            })
                        })
                    });
            tracing::debug!(
                tensor_count = lowered.tensors.len(),
                macro_ops = physical.operations.len(),
                weight_device_bytes = weights
                    .as_ref()
                    .map_or(0, backend_cuda::CudaWeightFabric::byte_len),
                weight_strategy = if all_permanent {
                    "permanent"
                } else if weights.is_some() {
                    "permanent-plus-predictive-window"
                } else {
                    "predictive-window"
                },
                graph_replay_safe,
                "compiled device-native CUDA decode layout"
            );
            self.cuda_decode = Some(CudaDecodePlan {
                tensors,
                _lowered: lowered,
                _physical: physical,
                device,
                _weights: weights,
                graph: None,
                full_step_warmed: false,
                graph_replay_safe,
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
                if state.ssm_elements() > 0 {
                    mutable_inputs.insert(
                        format!("ssm_in_{conv_ord}"),
                        MutableBinding {
                            dtype: DType::F32,
                            shape: Shape::new(&[state.ssm_elements() as u64])?,
                            buffer: state.ssm_buffer(conv_ord),
                        },
                    );
                }
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
        if let Some(cpu) = backend.as_any().downcast_ref::<backend_cpu::CpuBackend>() {
            let (storage_weights, host_buffers, host_buffer_bytes) =
                self.memory_fabric.storage_stream_configuration();
            if !storage_weights.is_empty() {
                cpu.configure_weight_storage(
                    &storage_weights,
                    self.memory_fabric.storage_objects(),
                    self.memory_fabric.select_storage_provider(false)?,
                    host_buffers,
                    host_buffer_bytes,
                    self.memory_fabric.storage_overlap_enabled(),
                )?;
            }
        }
        #[cfg(feature = "backend-cuda")]
        if let (Some(cuda), Some(plan), Some(compiled)) = (
            backend.as_any().downcast_ref::<backend_cuda::CudaBackend>(),
            self.cuda_decode.as_ref(),
            self.compiled.as_ref(),
        ) {
            for (id, host_ptr, byte_len) in compiled.arena_bindings() {
                let Some(device_tensor) = plan.tensors.get(&id) else {
                    continue;
                };
                // One arena offset can host differently-sized tensors at
                // non-overlapping lifetimes; register each typed byte view.
                cuda.register_device_tensor(host_ptr, byte_len, device_tensor.ptr)?;
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

    fn compile_batch(&mut self, gguf: &GgufFile, backend: &dyn Backend, rows: usize) -> Result<()> {
        if self.batches.contains_key(&rows) {
            return Ok(());
        }
        if self.batches.len() >= 2
            && let Some(evicted) = self.batches.keys().copied().next()
        {
            self.batches.remove(&evicted);
        }
        let graph =
            build_batch_step_graph_with_features(gguf, &self.spec, rows, &self.target_features)?;
        let plan = ExecutionPlan::from_graph(&graph)?;
        let bindings = collect_step_bindings(&graph);
        let mut mutable_inputs = HashMap::new();
        let mut conv_buffers = Vec::new();
        let mut ssm_buffers = Vec::new();
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
        if self.spec.is_hybrid() {
            let elements = self.spec.recurrent_conv_elements();
            let ssm_elements = self.spec.recurrent_ssm_elements();
            for conv_ord in 0..self.spec.n_conv_layers() {
                let buffer = Rc::new(RefCell::new(AlignedBuffer::new_zeroed(
                    rows.saturating_mul(elements).saturating_mul(4),
                    64,
                )));
                mutable_inputs.insert(
                    format!("conv_in_{conv_ord}"),
                    MutableBinding {
                        dtype: DType::F32,
                        shape: Shape::new(&[rows as u64, elements as u64])?,
                        buffer: buffer.clone(),
                    },
                );
                conv_buffers.push(buffer);
                if ssm_elements > 0 {
                    let ssm_buffer = Rc::new(RefCell::new(AlignedBuffer::new_zeroed(
                        rows.saturating_mul(ssm_elements).saturating_mul(4),
                        64,
                    )));
                    mutable_inputs.insert(
                        format!("ssm_in_{conv_ord}"),
                        MutableBinding {
                            dtype: DType::F32,
                            shape: Shape::new(&[rows as u64, ssm_elements as u64])?,
                            buffer: ssm_buffer.clone(),
                        },
                    );
                    ssm_buffers.push(ssm_buffer);
                }
            }
        }
        let step = CompiledStep::compile(&graph, &plan, backend, &mutable_inputs)?;
        self.batches.insert(
            rows,
            CompiledBatch {
                graph,
                bindings,
                step,
                conv_buffers,
                ssm_buffers,
            },
        );
        Ok(())
    }

    fn step_batch(
        &mut self,
        gguf: &GgufFile,
        backend: &dyn Backend,
        sequences: &mut [&mut KvSequence],
        conv_states: &mut [&mut Option<HybridConvState>],
        rows: &[BatchToken],
    ) -> Result<Vec<Option<Tensor>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        self.compile_batch(gguf, backend, rows.len())?;
        if self.spec.is_hybrid() {
            let elements = self.spec.recurrent_conv_elements();
            let ssm_elements = self.spec.recurrent_ssm_elements();
            let batch = self.batches.get_mut(&rows.len()).expect("compiled batch");
            for (conv_ord, packed) in batch.conv_buffers.iter().enumerate() {
                let mut packed = packed.borrow_mut();
                for (batch_row, row) in rows.iter().enumerate() {
                    let state = conv_states
                        .get(row.sequence)
                        .and_then(|state| state.as_ref())
                        .ok_or_else(|| FellmError::other("missing per-sequence ShortConv state"))?;
                    let source = state.conv_buffer(conv_ord);
                    let source = source.borrow();
                    let start = batch_row * elements * 4;
                    packed.as_mut_slice()[start..start + elements * 4]
                        .copy_from_slice(source.as_slice());
                }
            }
            for (conv_ord, packed) in batch.ssm_buffers.iter().enumerate() {
                let mut packed = packed.borrow_mut();
                for (batch_row, row) in rows.iter().enumerate() {
                    let state = conv_states[row.sequence]
                        .as_ref()
                        .ok_or_else(|| FellmError::other("missing per-sequence recurrent state"))?;
                    let source = state.ssm_buffer(conv_ord);
                    let source = source.borrow();
                    let start = batch_row * ssm_elements * 4;
                    packed.as_mut_slice()[start..start + ssm_elements * 4]
                        .copy_from_slice(source.as_slice());
                }
            }
        }
        let mut row_positions = Vec::with_capacity(rows.len());
        let mut row_lengths = Vec::with_capacity(rows.len());
        let mut rope_positions = Vec::with_capacity(rows.len());
        for row in rows {
            let sequence = sequences
                .get_mut(row.sequence)
                .ok_or_else(|| FellmError::other("batch sequence index out of bounds"))?;
            let kv_position = if sequence.is_compressed() {
                sequence.kv_write_index()
            } else {
                row.position
            };
            self.cache.ensure_writable(sequence, kv_position)?;
            if sequence.is_compressed() {
                if sequence.original_positions.len() == kv_position {
                    sequence.original_positions.push(row.position as u32);
                } else if kv_position < sequence.original_positions.len() {
                    sequence.original_positions[kv_position] = row.position as u32;
                } else {
                    return Err(FellmError::other(
                        "compressed batch KV position skipped its live index",
                    ));
                }
            }
            sequence.absolute_pos = row.position + 1;
            row_positions.push(kv_position as u32);
            row_lengths.push(sequence.len_tokens as u32);
            rope_positions.push(row.position as u32);
        }
        self.cache.tick();

        let n_logical = rows
            .iter()
            .map(|row| sequences[row.sequence].layer_map(0).num_pages())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut block_tables = Vec::with_capacity(rows.len() * self.cache.n_layers() * n_logical);
        for row in rows {
            let sequence = &sequences[row.sequence];
            block_tables.extend(
                self.cache
                    .physical_block_table_padded(sequence, n_logical)?,
            );
        }
        let (device_arena, device_arena_len) = {
            #[cfg(feature = "backend-cuda")]
            {
                if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
                    cuda.sync_kv_if_dirty(self.cache.arena_bytes())?;
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
        let (arena, arena_len) = self.cache.arena_ptr_mut();
        set_paged_context(Some(PagedKvContext {
            arena,
            arena_len,
            block_table: Arc::from(block_tables),
            n_logical_blocks: n_logical,
            n_layers: self.cache.n_layers(),
            tokens_stride: self.cache.tokens_stride(),
            block_bytes: self.cache.page_bytes(),
            block_size: crate::kv_fabric::BLOCK_SIZE,
            elem_bytes: fellm_plugin_abi::PAGED_KV_ELEM_BYTES,
            device_arena,
            device_arena_len,
            device_block_table: std::ptr::null_mut(),
            n_device_block_table: 0,
            device_logical_stride: 0,
            row_positions: Arc::from(row_positions),
            row_lengths: Arc::from(row_lengths),
            row_rope_positions: Arc::from(rope_positions),
        }));

        backend.begin_step();
        let result = (|| {
            let batch = self.batches.get_mut(&rows.len()).expect("compiled batch");
            for &id in &batch.bindings.rope {
                let attrs = batch.graph.node(id).attrs;
                batch.step.set_attrs(id, attrs);
            }
            for &id in &batch.bindings.kv_write {
                let mut attrs = batch.graph.node(id).attrs;
                attrs.block_size = crate::kv_fabric::BLOCK_SIZE as u32;
                batch.step.set_attrs(id, attrs);
            }
            for &id in &batch.bindings.attention {
                let mut attrs = batch.graph.node(id).attrs;
                if batch.graph.node(id).op != Some(OpKind::MlaAttention) {
                    attrs.block_size = crate::kv_fabric::BLOCK_SIZE as u32;
                    attrs.query_len = rows.len() as u32;
                    attrs.custom_op_id = fellm_plugin_abi::attention_dispatch().prefill.as_u32();
                }
                batch.step.set_attrs(id, attrs);
            }
            batch.step.bind_input(
                "token_id",
                u32_tensor(&rows.iter().map(|row| row.token).collect::<Vec<_>>())?,
            );
            let logits = batch
                .step
                .run(backend, rows.iter().any(|row| row.compute_logits))?;
            let outputs = rows
                .iter()
                .enumerate()
                .map(|(index, row)| row.compute_logits.then(|| logits.row(index)).transpose())
                .collect::<Result<Vec<_>>>()?;
            if self.spec.is_hybrid() {
                let elements = self.spec.recurrent_conv_elements();
                let ssm_elements = self.spec.recurrent_ssm_elements();
                for (conv_ord, packed) in batch.conv_buffers.iter().enumerate() {
                    let packed = packed.borrow();
                    for (batch_row, row) in rows.iter().enumerate() {
                        let state = conv_states[row.sequence].as_mut().expect("validated state");
                        let target = state.conv_buffer(conv_ord);
                        let mut target = target.borrow_mut();
                        let start = batch_row * elements * 4;
                        target
                            .as_mut_slice()
                            .copy_from_slice(&packed.as_slice()[start..start + elements * 4]);
                    }
                }
                for (conv_ord, packed) in batch.ssm_buffers.iter().enumerate() {
                    let packed = packed.borrow();
                    for (batch_row, row) in rows.iter().enumerate() {
                        let state = conv_states[row.sequence].as_mut().expect("validated state");
                        let target = state.ssm_buffer(conv_ord);
                        let mut target = target.borrow_mut();
                        let start = batch_row * ssm_elements * 4;
                        target
                            .as_mut_slice()
                            .copy_from_slice(&packed.as_slice()[start..start + ssm_elements * 4]);
                    }
                }
            }
            Ok(outputs)
        })();
        backend.end_step();
        set_paged_context(None);
        result
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
        // Absolute generation position for RoPE; dense storage index for KV.
        self.seq.absolute_pos = pos + 1;
        let kv_pos = if self.seq.is_compressed() {
            self.seq.kv_write_index()
        } else {
            pos
        };
        self.cache.ensure_writable(&mut self.seq, kv_pos)?;
        // CUDA compilation owns the fixed-capacity device page table used by
        // the context installed below. The call is idempotent after token one.
        self.compile_step(backend)?;
        if self.seq.is_compressed() {
            // Track absolute identity of the newly written dense slot.
            if self.seq.original_positions.len() == kv_pos {
                self.seq.original_positions.push(pos as u32);
            } else if kv_pos < self.seq.original_positions.len() {
                self.seq.original_positions[kv_pos] = pos as u32;
            } else {
                while self.seq.original_positions.len() < kv_pos {
                    self.seq
                        .original_positions
                        .push(self.seq.original_positions.len() as u32);
                }
                self.seq.original_positions.push(pos as u32);
            }
        }
        self.cache.tick();

        let n_logical = self.seq.layer_map(0).num_pages().max(1);
        let block_table = self.cache.physical_block_table(&self.seq);

        #[cfg(feature = "backend-cuda")]
        let (device_block_table, n_device_block_table, device_logical_stride) =
            if let (Some(cuda), Some(decode)) = (
                backend.as_any().downcast_ref::<backend_cuda::CudaBackend>(),
                self.cuda_decode.as_mut(),
            ) {
                decode.device.sync_page_table(
                    cuda.device_state(),
                    &block_table,
                    self.cache.n_layers(),
                    n_logical,
                )?;
                let (pointer, len) = decode.device.page_table_ptr();
                (pointer, len, decode.device.page_table_stride())
            } else {
                (std::ptr::null_mut(), 0, 0)
            };
        #[cfg(not(feature = "backend-cuda"))]
        let (device_block_table, n_device_block_table, device_logical_stride) =
            (std::ptr::null_mut(), 0usize, 0usize);

        let (device_arena, device_arena_len) = {
            #[cfg(feature = "backend-cuda")]
            {
                if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
                    // One-shot H2D only when host KV was mutated outside GPU KvWrite
                    // (prefix attach / migrate-in). Never re-upload the full arena every token.
                    let host = self.cache.arena_bytes();
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

        let (arena_ptr, arena_len) = self.cache.arena_ptr_mut();
        set_paged_context(Some(PagedKvContext {
            arena: arena_ptr,
            arena_len,
            block_table: std::sync::Arc::<[u32]>::from(block_table),
            n_logical_blocks: n_logical,
            n_layers: self.cache.n_layers(),
            tokens_stride: self.cache.tokens_stride(),
            block_bytes: self.cache.page_bytes(),
            block_size: crate::kv_fabric::BLOCK_SIZE,
            elem_bytes: fellm_plugin_abi::PAGED_KV_ELEM_BYTES,
            device_arena,
            device_arena_len,
            device_block_table,
            n_device_block_table,
            device_logical_stride,
            row_positions: std::sync::Arc::from([kv_pos as u32]),
            row_lengths: std::sync::Arc::from([self.seq.len_tokens as u32]),
            row_rope_positions: std::sync::Arc::from([pos as u32]),
        }));

        #[cfg(feature = "backend-cuda")]
        if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
            // Dense storage index after compress (not absolute generation pos).
            let (phys, kv_write_slot) = self.cache.locate(&self.seq, 0, kv_pos)?;
            cuda.update_step_params(&fellm_plugin_abi::DeviceStepParams {
                token_id: tok,
                position: pos as u32,
                sequence_length: self.seq.len_tokens as u32,
                active_batch: 1,
                kv_write_block: phys.0,
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

        // Dense storage length drives attention work after compression.
        let dense_len = self.seq.len_tokens.max(1);
        let dense_past = dense_len.saturating_sub(1) as u32;
        let kv_write_pos = if self.seq.is_compressed() {
            // Current token was just ensured at dense end-1 after write path;
            // ensure_writable set len_tokens = kv_pos+1, so write index was dense_len-1.
            dense_past
        } else {
            pos_u32
        };

        // Rope: absolute position for correct rotations; preserve pre-RoPE K flag.
        for &id in &self.bindings.rope {
            let mut a = self.step_graph.node(id).attrs;
            a.position = pos_u32;
            step.set_attrs(id, a);
        }
        for &id in &self.bindings.kv_write {
            let mut a = self.step_graph.node(id).attrs;
            a.position = kv_write_pos;
            a.block_size = 16;
            step.set_attrs(id, a);
        }
        let dispatch = fellm_plugin_abi::attention_dispatch();
        for &id in &self.bindings.attention {
            let mut a = self.step_graph.node(id).attrs;
            a.position = pos_u32;
            a.past_len = dense_past;
            // Paged full-attention overlays must not clobber MLA fields that reuse
            // `block_size` (compress ratio), `query_len` (indexer head dim), and
            // `kv_len` (indexer top-k).
            if self.step_graph.node(id).op != Some(OpKind::MlaAttention) {
                a.block_size = 16;
                a.query_len = 1;
                a.kv_len = dense_len as u32;
                a.custom_op_id = dispatch.decode.as_u32();
            }
            step.set_attrs(id, a);
        }

        step.bind_input("token_id", scalar_u32_tensor(tok));

        #[cfg(feature = "backend-cuda")]
        if compute_logits
            // Per-op profiling synchronizes around every launch. CUDA forbids
            // those synchronization calls while a stream is being captured,
            // so profiling deliberately exercises the uncaptured schedule.
            && std::env::var_os("FELLM_PROFILE_OPS").is_none()
            && let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>()
            && let Some(decode) = self.cuda_decode.as_mut()
            && decode.graph_replay_safe
        {
            if let Some(graph) = &decode.graph {
                let boundary_profile = std::env::var_os("FELLM_PROFILE_CUDA_BOUNDARY").is_some();
                let launched_at = boundary_profile.then(Instant::now);
                graph.launch()?;
                let launch_us = launched_at.map(|started| started.elapsed().as_micros());
                let materialized_at = boundary_profile.then(Instant::now);
                let result = step.materialize_result(backend, true);
                if boundary_profile {
                    tracing::debug!(
                        materialize_us = materialized_at
                            .map(|started| started.elapsed().as_micros())
                            .unwrap_or_default(),
                        "CUDA decode host boundary profile"
                    );
                }
                return result;
            }
            if decode.full_step_warmed {
                let capture = cuda.begin_graph_capture()?;
                step.enqueue(backend, true)?;
                decode.graph = Some(capture.finish()?);
                tracing::debug!(
                    macro_ops = decode._physical.operations.len(),
                    "captured stable CUDA decode graph"
                );
                return step.materialize_result(backend, true);
            }
            decode.full_step_warmed = true;
        }

        let result = step.run(backend, compute_logits);
        self.memory_fabric
            .observe_expert_route_batch(&fellm_plugin_abi::take_expert_routes());
        result
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
        let n_logical = self.seq.layer_map(0).num_pages().max(1);
        let block_table = self.cache.physical_block_table(&self.seq);
        let (device_arena, device_arena_len) = {
            #[cfg(feature = "backend-cuda")]
            {
                if let Some(cuda) = backend.as_any().downcast_ref::<backend_cuda::CudaBackend>() {
                    let host = self.cache.arena_bytes();
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
        let (arena_ptr, arena_len) = self.cache.arena_ptr_mut();
        set_paged_context(Some(PagedKvContext {
            arena: arena_ptr,
            arena_len,
            block_table: std::sync::Arc::<[u32]>::from(block_table),
            n_logical_blocks: n_logical,
            n_layers: self.cache.n_layers(),
            tokens_stride: self.cache.tokens_stride(),
            block_bytes: self.cache.page_bytes(),
            block_size: crate::kv_fabric::BLOCK_SIZE,
            elem_bytes: fellm_plugin_abi::PAGED_KV_ELEM_BYTES,
            device_arena,
            device_arena_len,
            device_block_table: std::ptr::null_mut(),
            n_device_block_table: 0,
            device_logical_stride: 0,
            row_positions: std::sync::Arc::from([prompt_len as u32]),
            row_lengths: std::sync::Arc::from([prompt_len as u32]),
            row_rope_positions: std::sync::Arc::from([prompt_len as u32]),
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
    sampler_state: sampling::SamplerState,
    sampling: sampling::SamplingWorkspace,
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
        // CompiledStep returns an owned, host-authoritative logits tensor.  It
        // has already crossed the device boundary exactly once.  Re-running
        // `materialize` here gives the detached allocation a cache identity
        // based only on a recyclable host address and can overwrite fresh
        // logits with a stale device mirror from an earlier token.
        let tok = {
            let logits = match logits_tensor.as_slice::<f32>() {
                Ok(logits) => logits,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            };
            if !logits.iter().any(|value| value.is_finite()) {
                self.finished = true;
                return Some(Err(FellmError::other(
                    "logits are not finite; refusing to sample token 0",
                )));
            }
            if crate::activation::ActivationStats::collect(logits).all_zero() {
                self.finished = true;
                return Some(Err(FellmError::other(
                    "all output logits are zero; inference result is invalid",
                )));
            }
            if self.emitted == 0 {
                let top = sampling::top_logits(logits, 8);
                let top_fmt: Vec<String> = top
                    .iter()
                    .map(|(id, logit)| {
                        let piece = self
                            .engine
                            .tokenizer
                            .vocabulary_piece(*id)
                            .unwrap_or("?");
                        format!("{id}:{logit:.4}:{piece}")
                    })
                    .collect();
                let bos = self.engine.tokenizer.bos().unwrap_or(0) as usize;
                let eos = self.engine.tokenizer.eos().unwrap_or(1) as usize;
                tracing::debug!(
                    vocab = logits.len(),
                    finite = logits.iter().filter(|v| v.is_finite()).count(),
                    max = logits
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(f32::NEG_INFINITY, f32::max),
                    min = logits
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(f32::INFINITY, f32::min),
                    bos_logit = logits.get(bos).copied().unwrap_or(f32::NAN),
                    eos_logit = logits.get(eos).copied().unwrap_or(f32::NAN),
                    top = %top_fmt.join(" | "),
                    "first-step logits"
                );
            }
            sampling::sample_with_workspace(
                logits,
                sampling::SamplingOptions {
                    temperature: self.params.temperature,
                    top_k: self.params.top_k,
                    top_p: self.params.top_p,
                    min_p: self.params.min_p,
                    seed: self
                        .params
                        .seed
                        .wrapping_add(self.sampler_state.draw_index()),
                    repetition_penalty: self.params.repetition_penalty,
                    frequency_penalty: self.params.frequency_penalty,
                    presence_penalty: self.params.presence_penalty,
                    logit_bias: &self.params.logit_bias,
                    grammar: self.sampler_state.grammar_view(),
                    recent_tokens: self.sampler_state.history(),
                },
                &mut self.sampling,
            )
        };
        self.sampler_state.commit_token(tok);
        self.emitted += 1;
        self.stats.predicted_tokens = self.emitted;

        let now = Instant::now();
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
            self.stats.time_to_first_token_ms = duration_ms(now.duration_since(self.gen_start));
        }
        self.last_token_at = Some(now);

        if self.stop_token_ids.contains(&tok) || self.sampler_state.grammar_is_accepting() {
            self.finished = true;
            return Some(Ok(tok));
        }

        if self.emitted < self.params.max_tokens && self.position + 1 < self.engine.model.max_seq {
            match self.engine.step(tok, self.position, true) {
                Ok(next) => {
                    self.pending_logits = Some(next);
                    self.position += 1;
                    self.engine.tokens_since_compress =
                        self.engine.tokens_since_compress.saturating_add(1);
                    // Sequence-state policy: score → compact → reclaim physical blocks.
                    if let Err(e) = self.engine.maybe_compress_kv() {
                        self.finished = true;
                        return Some(Err(e));
                    }
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

/// Install process-wide attention dispatch from prepared provider plans.
fn install_attention_dispatch(providers: &crate::providers::ProviderManager, backend_id: &str) {
    let mut d = AttentionDispatch {
        decode: AttentionKernelPath::Fa2Decode,
        prefill: AttentionKernelPath::Fa2Prefill,
        q_tile: 16,
        kv_tile: 16,
        pipeline_stages: 2,
        provider_id: 0,
    };
    if let Some(prep) = providers.prepared() {
        d.provider_id = prep.attention_id.0;
        if prep.attention_name.contains("host") || backend_id == "cpu" {
            d.decode = AttentionKernelPath::HostFa2;
            d.prefill = AttentionKernelPath::HostFa2;
        }
        // Path kind: Prefill=1, Decode=2 (see AttentionPathKind).
        if let Some(plan) = providers.attention_plan(AttentionPathKind::Decode as u8) {
            d.decode =
                AttentionKernelPath::from_prepared(plan.plan_handle, plan.kernel_variant, false);
            if backend_id == "cpu" {
                d.decode = AttentionKernelPath::HostFa2;
            }
            // Decode tile encoding: high 16 = Br, low 16 = Bc for host; CUDA uses style bit.
            if plan.kernel_variant > 0xFFFF {
                d.q_tile = ((plan.kernel_variant >> 16) & 0xFFFF) as u32;
                d.kv_tile = (plan.kernel_variant & 0xFFFF) as u32;
            }
        }
        if let Some(plan) = providers.attention_plan(AttentionPathKind::Prefill as u8) {
            d.prefill =
                AttentionKernelPath::from_prepared(plan.plan_handle, plan.kernel_variant, true);
            if backend_id == "cpu" {
                d.prefill = AttentionKernelPath::HostFa2;
            }
        }
    }
    set_attention_dispatch(d);
    tracing::debug!(
        decode = ?d.decode,
        prefill = ?d.prefill,
        provider_id = d.provider_id,
        "attention dispatch installed"
    );
}

fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn target_feature_output_name(feature: fellm_plugin_abi::TargetFeature) -> String {
    match feature {
        fellm_plugin_abi::TargetFeature::EmbeddingOutput => "feature.embedding".into(),
        fellm_plugin_abi::TargetFeature::LayerHiddenState(layer) => {
            format!("feature.layer.{layer}")
        }
        fellm_plugin_abi::TargetFeature::FinalHiddenState => "feature.final".into(),
    }
}

fn copy_hybrid_state(source: &HybridConvState, target: &mut HybridConvState) -> Result<()> {
    if source.conv.len() != target.conv.len() || source.ssm.len() != target.ssm.len() {
        return Err(FellmError::other(
            "hybrid recurrent state topology mismatch",
        ));
    }
    for (source, target) in source.conv.iter().zip(&target.conv) {
        let source = source.borrow();
        let mut target = target.borrow_mut();
        if source.len() != target.len() {
            return Err(FellmError::other("hybrid convolution state size mismatch"));
        }
        target.as_mut_slice().copy_from_slice(source.as_slice());
    }
    for (source, target) in source.ssm.iter().zip(&target.ssm) {
        let source = source.borrow();
        let mut target = target.borrow_mut();
        if source.len() != target.len() {
            return Err(FellmError::other("hybrid matrix state size mismatch"));
        }
        target.as_mut_slice().copy_from_slice(source.as_slice());
    }
    Ok(())
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
