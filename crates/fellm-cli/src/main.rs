//! `fellm` CLI.

mod config;

use clap::{Parser, Subcommand};
use fellm_architecture_diffusion_gemma::DiffusionGemmaPlugin;
use fellm_gguf::GgufFile;
use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_abi::capability::{CapabilityKind, PluginConfig, ProviderSelection};
use fellm_plugin_host::PluginHost;
use fellm_runtime::{
    BackendPreference, BackendSelect, ChatRenderOptions, Engine, EngineSettings, GenParams,
};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "fellm", version, about = "FeLLM inference engine")]
struct Cli {
    /// Configuration file. Defaults to `fellm.toml` in the current directory.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[arg(long, global = true)]
    log: Option<String>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run generation on a GGUF model.
    Run(RunArgs),
    /// Inspect GGUF metadata and tensors.
    Inspect(InspectArgs),
    /// Discover and inspect dynamically loaded plugins / providers.
    Plugins(PluginsCmd),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Path to a GGUF file.
    model: Option<PathBuf>,
    /// Compatible smaller GGUF model used as an ordinary independent drafter.
    #[arg(long = "speculative-draft")]
    speculative_draft: Option<PathBuf>,
    /// Native or checkpoint-backed speculator (`mtp`, `dspark`).
    #[arg(long)]
    speculator: Option<String>,
    /// Optional checkpoint/model owned by the selected speculator.
    #[arg(long = "speculator-model")]
    speculator_model: Option<PathBuf>,
    /// Maximum dynamically selected speculative proposal length.
    #[arg(long = "speculative-max-tokens")]
    speculative_max_tokens: Option<usize>,
    /// Disable speculation when estimated speedup is below this fraction.
    #[arg(long = "speculative-min-gain")]
    speculative_min_gain: Option<f64>,
    /// Independent backend placement for the draft model.
    #[arg(long = "speculative-draft-backend")]
    speculative_draft_backend: Option<String>,
    /// Speculative method: `auto`, `mtp`, `dspark`, or `off`.
    #[arg(long)]
    speculative: Option<String>,
    /// Prompt / user message string.
    #[arg(long)]
    prompt: Option<String>,
    /// Optional system message (chat mode only).
    #[arg(long)]
    system: Option<String>,
    /// Force raw completion (skip chat template even if the model has one).
    #[arg(long)]
    completion: Option<bool>,
    /// Max tokens to generate.
    #[arg(long)]
    max_tokens: Option<u32>,
    /// Sampling temperature.
    #[arg(long)]
    temperature: Option<f32>,
    /// top-k (0 disables).
    #[arg(long)]
    top_k: Option<u32>,
    /// top-p (>= 1.0 disables).
    #[arg(long)]
    top_p: Option<f32>,
    /// Minimum probability relative to the most likely token (0 disables).
    #[arg(long)]
    min_p: Option<f32>,
    /// Count-scaled token frequency penalty.
    #[arg(long)]
    frequency_penalty: Option<f32>,
    /// One-time token presence penalty.
    #[arg(long)]
    presence_penalty: Option<f32>,
    /// Sparse token logit adjustment as TOKEN_ID=BIAS; repeatable.
    #[arg(long = "logit-bias", value_name = "TOKEN_ID=BIAS")]
    logit_bias: Vec<String>,
    /// RNG seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Repetition penalty (1.0 disables).
    #[arg(long)]
    repetition_penalty: Option<f32>,
    /// Context size (`n_ctx`). Default 8192, clamped to the model maximum.
    /// Pass `0` to use the model's GGUF-reported maximum context length.
    #[arg(long = "ctx-size", short = 'c')]
    ctx_size: Option<usize>,
    /// Evaluation batch size (`n_batch`).
    #[arg(long = "batch-size", short = 'b')]
    batch_size: Option<usize>,
    /// Physical batch size (`n_ubatch`).
    #[arg(long = "ubatch-size")]
    ubatch_size: Option<usize>,
    /// Exact device KV arena bytes (0 selects automatic device/system budgeting).
    #[arg(long = "kv-device-budget", alias = "kv-cache-bytes")]
    kv_device_budget: Option<u64>,
    /// Fraction of available memory considered by automatic KV budgeting.
    #[arg(long)]
    kv_memory_fraction: Option<f64>,
    /// Host residency-tier bytes for KV migration / preempt.
    #[arg(long = "kv-host-budget", alias = "kv-swap-bytes")]
    kv_host_budget: Option<u64>,
    /// Bytes kept free as a safety reserve.
    #[arg(long)]
    kv_safety_reserve_bytes: Option<u64>,
    /// KV fabric mode: `auto`, `exact`, or `elastic`.
    #[arg(long = "kv-mode")]
    kv_mode: Option<String>,
    /// KV addressing strategy: `block_table` or `virtual_memory`.
    #[arg(long = "kv-addressing")]
    kv_addressing: Option<String>,
    /// Enable or disable content-addressed prefix sharing.
    #[arg(long)]
    kv_prefix_sharing: Option<bool>,
    /// Enable or disable fabric prefetch of non-resident pages.
    #[arg(long)]
    kv_prefetch: Option<bool>,
    /// Optional max sequence length override (alias of `--ctx-size`).
    #[arg(long, hide = true)]
    max_seq: Option<usize>,
    /// Compute backend provider preference: `auto`, `cpu`, or `cuda`.
    /// Also set via `FELLM_BACKEND`.
    #[arg(long)]
    backend: Option<String>,
    /// Explicit attention provider name (e.g. `attention.host_tiled`).
    /// Fails if missing or unsupported — never silently substitutes.
    #[arg(long)]
    attention: Option<String>,
    /// Explicit sequence-state / KV policy provider (e.g. `kv.full`, `kv.triattention`).
    #[arg(long = "kv-policy")]
    kv_policy: Option<String>,
    /// Plugin-specific configuration as `key=value` or `provider.key=value`.
    /// Repeatable. Validated before inference.
    #[arg(long = "plugin-config", value_name = "KEY=VALUE")]
    plugin_config: Vec<String>,
    /// Directory of dynamic plugins (default: `plugins/` or `FELLM_PLUGIN_DIR`).
    #[arg(long = "plugin-dir")]
    plugin_dir: Option<PathBuf>,
    /// Enable or disable CPU fallback when CUDA is unavailable.
    #[arg(long)]
    cpu_fallback: Option<bool>,
    /// Override detected available VRAM for planning.
    #[arg(long)]
    device_memory_limit: Option<u64>,
    /// Override detected available system RAM for planning.
    #[arg(long)]
    host_memory_limit: Option<u64>,
    #[arg(long)]
    h2d_bytes_per_second: Option<u64>,
    #[arg(long)]
    storage_bytes_per_second: Option<u64>,
    #[arg(long)]
    storage_latency_micros: Option<u64>,
    /// Storage provider: auto, page-cache, mmap-copy, buffered, direct, io-uring, or gds.
    #[arg(long)]
    storage_provider: Option<String>,
    /// Resident CPU weight-cache bytes. Zero is valid for storage-native execution.
    #[arg(long)]
    host_weight_cache: Option<u64>,
    /// Enable predictive CPU storage reads while the current group computes.
    #[arg(long)]
    storage_overlap: Option<bool>,
    /// Maximum router decisions retained independently for offline cache simulation.
    #[arg(long)]
    router_trace_capacity: Option<usize>,
    /// Disable CPU weight partitions even when the planner would use them.
    #[arg(long)]
    disable_cpu_partitions: Option<bool>,
    /// Enable GGUF chat-template thinking (`enable_thinking=true`), like llama.cpp `--think`.
    #[arg(long, conflicts_with = "no_think")]
    think: bool,
    /// Disable thinking when the template supports it (`enable_thinking=false`).
    #[arg(long = "no-think", conflicts_with = "think")]
    no_think: bool,
    /// Echo the prompt on stdout before generated tokens.
    #[arg(long = "echo-prompt")]
    echo_prompt: bool,
}

#[derive(clap::Args, Debug)]
struct InspectArgs {
    /// Path to a GGUF file.
    #[arg(long)]
    model: PathBuf,
}

#[derive(clap::Args, Debug)]
struct PluginsCmd {
    #[command(subcommand)]
    action: PluginsAction,
}

#[derive(Subcommand, Debug)]
enum PluginsAction {
    /// List discovered plugins and capability providers.
    List {
        /// Optional plugin directory.
        #[arg(long = "plugin-dir")]
        plugin_dir: Option<PathBuf>,
        /// Filter by capability (`attention`, `sequence_state_policy`, …).
        #[arg(long)]
        capability: Option<String>,
    },
    /// Inspect one provider by name.
    Inspect {
        /// Provider name (e.g. `attention.host_tiled`, `kv.triattention`).
        name: String,
        /// Optional plugin directory.
        #[arg(long = "plugin-dir")]
        plugin_dir: Option<PathBuf>,
    },
}

impl PluginsCmd {
    fn apply_config(&mut self, config: config::PluginsConfig) {
        let configured = config.plugin_dir;
        match &mut self.action {
            PluginsAction::List { plugin_dir, .. } | PluginsAction::Inspect { plugin_dir, .. } => {
                if plugin_dir.is_none() {
                    *plugin_dir = configured;
                }
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let config_path = cli
        .config
        .clone()
        .unwrap_or_else(|| PathBuf::from("fellm.toml"));
    let config = match config::FellmConfig::load(&config_path, cli.config.is_some()) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    };
    init_tracing(
        cli.log
            .as_deref()
            .or(config.log.as_deref())
            .unwrap_or("info"),
    );

    let result = match cli.cmd {
        Cmd::Run(a) => resolve_run(a, config.run, config.memory).and_then(run),
        Cmd::Inspect(a) => inspect(a),
        Cmd::Plugins(mut p) => {
            p.apply_config(config.plugins);
            plugins_cmd(p)
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct ResolvedRunArgs {
    model: PathBuf,
    speculative_draft: Option<PathBuf>,
    speculator: Option<String>,
    speculator_model: Option<PathBuf>,
    speculative_max_tokens: usize,
    speculative_min_gain: f64,
    speculative_draft_backend: Option<String>,
    speculative: Option<String>,
    prompt: String,
    system: Option<String>,
    completion: bool,
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    seed: u64,
    repetition_penalty: f32,
    ctx_size: usize,
    batch_size: usize,
    ubatch_size: usize,
    kv_device_budget: u64,
    kv_memory_fraction: f64,
    kv_host_budget: u64,
    kv_safety_reserve_bytes: u64,
    kv_mode: String,
    kv_addressing: String,
    kv_prefix_sharing: bool,
    kv_prefetch: bool,
    backend: String,
    attention: Option<String>,
    kv_policy: Option<String>,
    plugin_config: Vec<String>,
    plugin_dir: Option<PathBuf>,
    cpu_fallback: bool,
    think: bool,
    no_think: bool,
    echo_prompt: bool,
    memory_fabric: fellm_runtime::MemoryFabricConfig,
}

fn parse_logit_bias(values: Vec<String>) -> fellm_core::error::Result<Vec<(u32, f32)>> {
    values
        .into_iter()
        .map(|value| {
            let (token, bias) = value.split_once('=').ok_or_else(|| {
                fellm_core::error::FellmError::other(format!(
                    "logit bias must be TOKEN_ID=BIAS, got '{value}'"
                ))
            })?;
            let token = token.parse::<u32>().map_err(|error| {
                fellm_core::error::FellmError::other(format!(
                    "invalid logit-bias token '{token}': {error}"
                ))
            })?;
            let bias = bias.parse::<f32>().map_err(|error| {
                fellm_core::error::FellmError::other(format!(
                    "invalid logit-bias value '{bias}': {error}"
                ))
            })?;
            if !bias.is_finite() {
                return Err(fellm_core::error::FellmError::other(
                    "logit bias must be finite",
                ));
            }
            Ok((token, bias))
        })
        .collect()
}

fn resolve_run(
    cli: RunArgs,
    file: config::RunConfig,
    memory: config::MemoryConfig,
) -> fellm_core::error::Result<ResolvedRunArgs> {
    macro_rules! value {
        ($field:ident, $default:expr) => {
            cli.$field.or(file.$field).unwrap_or($default)
        };
    }
    let model = cli.model.or(file.model).ok_or_else(|| {
        fellm_core::error::FellmError::other("model must be set in [run].model or on the CLI")
    })?;
    let prompt = cli.prompt.or(file.prompt).ok_or_else(|| {
        fellm_core::error::FellmError::other("prompt must be set in [run].prompt or with --prompt")
    })?;
    let plugin_config = if cli.plugin_config.is_empty() {
        file.plugin_config.unwrap_or_default()
    } else {
        cli.plugin_config
    };
    let logit_bias = parse_logit_bias(if cli.logit_bias.is_empty() {
        file.logit_bias.unwrap_or_default()
    } else {
        cli.logit_bias
    })?;
    let storage_provider = cli
        .storage_provider
        .or(memory.storage_provider)
        .unwrap_or_else(|| "auto".into())
        .parse()?;
    Ok(ResolvedRunArgs {
        model,
        speculative_draft: cli.speculative_draft.or(file.speculative_draft),
        speculator: cli.speculator.or(file.speculator),
        speculator_model: cli.speculator_model.or(file.speculator_model),
        speculative_max_tokens: cli
            .speculative_max_tokens
            .or(file.speculative_max_tokens)
            .unwrap_or(4),
        speculative_min_gain: cli
            .speculative_min_gain
            .or(file.speculative_min_gain)
            .unwrap_or(0.05),
        speculative_draft_backend: cli
            .speculative_draft_backend
            .or(file.speculative_draft_backend),
        speculative: cli.speculative.or(file.speculative),
        prompt,
        system: cli.system.or(file.system),
        completion: value!(completion, false),
        max_tokens: value!(max_tokens, 128),
        temperature: value!(temperature, 0.2),
        top_k: value!(top_k, 80),
        top_p: value!(top_p, 1.0),
        min_p: value!(min_p, 0.0),
        frequency_penalty: value!(frequency_penalty, 0.0),
        presence_penalty: value!(presence_penalty, 0.0),
        logit_bias,
        seed: value!(seed, 0),
        repetition_penalty: value!(repetition_penalty, 1.05),
        ctx_size: cli
            .max_seq
            .or(cli.ctx_size)
            .or(file.ctx_size)
            .unwrap_or(8192),
        batch_size: value!(batch_size, 2048),
        ubatch_size: value!(ubatch_size, 512),
        kv_device_budget: value!(kv_device_budget, 0),
        kv_memory_fraction: value!(kv_memory_fraction, 0.25),
        kv_host_budget: value!(kv_host_budget, 0),
        kv_safety_reserve_bytes: value!(kv_safety_reserve_bytes, 2 * 1024 * 1024 * 1024),
        kv_mode: value!(kv_mode, String::from("auto")),
        kv_addressing: value!(kv_addressing, String::from("block_table")),
        kv_prefix_sharing: cli
            .kv_prefix_sharing
            .or(file.kv_prefix_sharing)
            .unwrap_or(true),
        kv_prefetch: cli.kv_prefetch.or(file.kv_prefetch).unwrap_or(true),
        backend: value!(backend, String::from("auto")),
        attention: cli.attention.or(file.attention),
        kv_policy: cli.kv_policy.or(file.kv_policy),
        plugin_config,
        plugin_dir: cli.plugin_dir.or(file.plugin_dir),
        cpu_fallback: cli.cpu_fallback.or(file.cpu_fallback).unwrap_or(true),
        think: cli.think,
        no_think: cli.no_think,
        echo_prompt: cli.echo_prompt,
        memory_fabric: fellm_runtime::MemoryFabricConfig {
            device_memory_limit: cli.device_memory_limit.or(memory.device_memory_limit),
            host_memory_limit: cli.host_memory_limit.or(memory.host_memory_limit),
            h2d_bytes_per_second: cli.h2d_bytes_per_second.or(memory.h2d_bytes_per_second),
            storage_bytes_per_second: cli
                .storage_bytes_per_second
                .or(memory.storage_bytes_per_second),
            storage_latency_micros: cli.storage_latency_micros.or(memory.storage_latency_micros),
            storage_provider,
            host_weight_cache: cli
                .host_weight_cache
                .or(memory.host_weight_cache)
                .unwrap_or(0),
            storage_overlap: cli
                .storage_overlap
                .or(memory.storage_overlap)
                .unwrap_or(true),
            router_trace_capacity: cli
                .router_trace_capacity
                .or(memory.router_trace_capacity)
                .unwrap_or(65_536),
            disable_cpu_partitions: cli
                .disable_cpu_partitions
                .or(memory.disable_cpu_partitions)
                .unwrap_or(false),
        },
    })
}

#[cfg(test)]
mod config_precedence_tests {
    use super::*;

    fn empty_cli() -> RunArgs {
        RunArgs {
            model: None,
            speculative_draft: None,
            speculator: None,
            speculator_model: None,
            speculative_max_tokens: None,
            speculative_min_gain: None,
            speculative_draft_backend: None,
            speculative: None,
            prompt: None,
            system: None,
            completion: None,
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: Vec::new(),
            seed: None,
            repetition_penalty: None,
            ctx_size: None,
            batch_size: None,
            ubatch_size: None,
            kv_device_budget: None,
            kv_memory_fraction: None,
            kv_host_budget: None,
            kv_safety_reserve_bytes: None,
            kv_mode: None,
            kv_addressing: None,
            kv_prefix_sharing: None,
            kv_prefetch: None,
            max_seq: None,
            backend: None,
            attention: None,
            kv_policy: None,
            plugin_config: Vec::new(),
            plugin_dir: None,
            cpu_fallback: None,
            device_memory_limit: None,
            host_memory_limit: None,
            h2d_bytes_per_second: None,
            storage_bytes_per_second: None,
            storage_latency_micros: None,
            storage_provider: None,
            host_weight_cache: None,
            storage_overlap: None,
            router_trace_capacity: None,
            disable_cpu_partitions: None,
            think: false,
            no_think: false,
            echo_prompt: false,
        }
    }

    #[test]
    fn cli_overrides_file_and_file_overrides_builtin_fallback() {
        let mut cli = empty_cli();
        cli.model = Some("cli.gguf".into());
        cli.prompt = Some("cli prompt".into());
        cli.backend = Some("cuda".into());
        let file = config::RunConfig {
            model: Some("file.gguf".into()),
            prompt: Some("file prompt".into()),
            backend: Some("cpu".into()),
            max_tokens: Some(7),
            ..config::RunConfig::default()
        };
        let resolved = resolve_run(cli, file, config::MemoryConfig::default()).unwrap();
        assert_eq!(resolved.model, PathBuf::from("cli.gguf"));
        assert_eq!(resolved.prompt, "cli prompt");
        assert_eq!(resolved.backend, "cuda");
        assert_eq!(resolved.max_tokens, 7);
        assert_eq!(resolved.ctx_size, 8192);
    }
}

fn init_tracing(filter: &str) {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));

    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
        tracing_subscriber::registry()
            .with(fmt::layer().with_target(false).with_filter(filter))
            .with(tracing_tracy::TracyLayer::default())
            .init();
        return;
    }

    #[cfg(not(feature = "tracy"))]
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
}

fn build_selection(args: &ResolvedRunArgs) -> fellm_core::error::Result<ProviderSelection> {
    let mut sel = ProviderSelection::new();
    if let Some(a) = &args.attention {
        sel.attention = Some(a.clone());
    }
    if let Some(k) = &args.kv_policy {
        sel.kv_policy = Some(k.clone());
    }
    if !args.plugin_config.is_empty() {
        sel.config = PluginConfig::from_pairs(&args.plugin_config)
            .map_err(|e| fellm_core::error::FellmError::other(e))?;
    }
    Ok(sel)
}

fn run(args: ResolvedRunArgs) -> fellm_core::error::Result<()> {
    if args.speculative.as_deref() == Some("off")
        && (args.speculative_draft.is_some() || args.speculator.is_some())
    {
        return Err(fellm_core::error::FellmError::other(
            "--speculative off cannot be combined with a speculator or draft model",
        ));
    }
    if args.speculative_draft.is_some()
        && matches!(args.speculator.as_deref(), Some("mtp" | "dspark"))
    {
        return Err(fellm_core::error::FellmError::other(
            "choose either --speculative-draft or --speculator, not both",
        ));
    }
    let mut speculator_kind = args.speculator.clone();
    if args.speculative.as_deref() == Some("auto") && speculator_kind.is_none() {
        let gguf = GgufFile::open(&args.model)?;
        let spec = fellm_runtime::ModelSpec::from_gguf(&gguf)?;
        if spec.n_mtp_layers > 0 {
            speculator_kind = Some("mtp".into());
        } else if args.speculator_model.is_some() {
            speculator_kind = Some("dspark".into());
        }
    } else if let Some(kind) = args.speculative.as_deref() {
        if kind != "auto" && kind != "off" && speculator_kind.is_none() {
            speculator_kind = Some(kind.to_owned());
        }
    }
    if speculator_kind.as_deref() == Some("off") {
        speculator_kind = None;
    }
    let preference = BackendPreference::parse(&args.backend)?;
    let select = BackendSelect::new(preference, args.cpu_fallback);
    let providers = build_selection(&args)?;
    let kv_cache = fellm_runtime::KvFabricConfig {
        mode: fellm_runtime::KvMode::parse(&args.kv_mode).unwrap_or_default(),
        device_budget: (args.kv_device_budget > 0).then_some(args.kv_device_budget),
        host_budget: Some(args.kv_host_budget),
        addressing: fellm_runtime::KvAddressing::parse(&args.kv_addressing).unwrap_or_default(),
        prefix_sharing: args.kv_prefix_sharing,
        prefetch: args.kv_prefetch,
        memory_fraction: args.kv_memory_fraction,
        safety_reserve_bytes: args.kv_safety_reserve_bytes,
        ..fellm_runtime::KvFabricConfig::default()
    };
    let mut settings = EngineSettings::default()
        .batch_size(args.batch_size)
        .ubatch_size(args.ubatch_size)
        .backend_select(select)
        .providers(providers)
        .kv_cache(kv_cache);
    settings.memory_fabric = args.memory_fabric;
    if speculator_kind.as_deref() == Some("mtp") {
        settings = settings.target_features([fellm_plugin_abi::TargetFeature::FinalHiddenState]);
    }
    let mut dspark_checkpoint = None;
    let mut dspark_support_gguf = None;
    if speculator_kind.as_deref() == Some("dspark") {
        let path = args.speculator_model.as_ref().ok_or_else(|| {
            fellm_core::error::FellmError::other(
                "--speculator dspark requires --speculator-model <checkpoint-directory-or-gguf>",
            )
        })?;
        if path.is_dir() {
            let checkpoint = Arc::new(fellm_dspark::DsparkCheckpoint::open(path)?);
            settings = settings.target_features(
                checkpoint
                    .config
                    .required_features()
                    .into_iter()
                    .map(|tap| tap.feature),
            );
            dspark_checkpoint = Some(checkpoint);
        } else {
            let support = fellm_gguf::GgufFile::open(path)?;
            let layers = support
                .metadata
                .get_u32_array("dspark.target_layer_ids")
                .map(|values| values.to_vec())
                .unwrap_or_else(|_| vec![40, 41, 42]);
            settings = settings.target_features(
                layers
                    .into_iter()
                    .map(fellm_plugin_abi::TargetFeature::LayerHiddenState),
            );
            dspark_support_gguf = Some(path.clone());
        }
    }

    if let Some(dir) = &args.plugin_dir {
        settings = settings.plugin_dir(dir.clone());
    }

    let ctx = args.ctx_size;
    settings = if ctx == 0 {
        settings.ctx_from_model()
    } else {
        settings.ctx_size(ctx)
    };

    let mut engine = Engine::open_with_architecture(
        &args.model,
        settings.clone(),
        Some(Arc::new(DiffusionGemmaPlugin)),
    )?;

    if args.think && args.no_think {
        return Err(fellm_core::error::FellmError::other(
            "--think and --no-think cannot be used together",
        ));
    }
    let params = GenParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
        repetition_penalty: args.repetition_penalty,
        frequency_penalty: args.frequency_penalty,
        presence_penalty: args.presence_penalty,
        logit_bias: Arc::from(args.logit_bias),
        priority: 0,
        enable_thinking: if args.think {
            Some(true)
        } else if args.no_think {
            Some(false)
        } else {
            None
        },
        ..GenParams::default()
    };

    let use_chat = !args.completion && engine.tokenizer().chat_template().is_some();

    let chat_options = ChatRenderOptions {
        enable_thinking: params.enable_thinking.or_else(|| {
            engine.tokenizer().supports_thinking().then_some(false)
        }),
    };
    let encode_chat = |engine: &Engine,
                       prompt: &str,
                       system: Option<&str>|
     -> fellm_core::error::Result<Vec<u32>> {
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(fellm_runtime::Message::text("system", system.to_string()));
        }
        messages.push(fellm_runtime::Message::text("user", prompt.to_string()));
        let prepared = engine.prepare_chat_messages(&messages);
        let rendered = engine
            .tokenizer()
            .apply_chat_template_with_options(&prepared, &[], true, chat_options.clone())?
            .unwrap_or_else(|| prompt.to_string());
        engine.tokenizer().encode(&rendered, true)
    };

    if let Some(speculator_kind) = &speculator_kind {
        if speculator_kind != "mtp" && speculator_kind != "dspark" {
            return Err(fellm_core::error::FellmError::other(format!(
                "unknown speculator '{speculator_kind}' (expected mtp or dspark)"
            )));
        }
        if speculator_kind == "mtp" && args.speculator_model.is_some() {
            return Err(fellm_core::error::FellmError::other(
                "native MTP uses tensors embedded in the target; --speculator-model is not applicable",
            ));
        }
        let prompt_ids = if use_chat {
            encode_chat(&engine, &args.prompt, args.system.as_deref())?
        } else {
            engine.tokenizer().encode(&args.prompt, true)?
        };
        let speculator_backend = if let Some(backend) = &args.speculative_draft_backend {
            BackendSelect::new(BackendPreference::parse(backend)?, args.cpu_fallback)
        } else {
            select
        };
        let speculator: Box<dyn fellm_plugin_abi::Speculator> = match speculator_kind.as_str() {
            "mtp" => Box::new(fellm_mtp::MtpSpeculator::from_model(
                engine.gguf(),
                engine.spec(),
                speculator_backend,
                engine.n_ctx(),
                args.speculative_max_tokens as u32,
            )?),
            "dspark" => {
                if let Some(checkpoint) = dspark_checkpoint {
                    Box::new(fellm_dspark::DsparkSpeculator::from_checkpoint(
                        checkpoint,
                        speculator_backend,
                        engine.n_ctx(),
                    )?)
                } else {
                    let path = dspark_support_gguf.expect("DSpark GGUF path prepared");
                    Box::new(fellm_dspark::DsparkSpeculator::from_support_gguf(
                        path,
                        engine.gguf(),
                        engine.spec(),
                        speculator_backend,
                        engine.n_ctx(),
                    )?)
                }
            }
            _ => unreachable!("validated speculator kind"),
        };
        if args.echo_prompt {
            print!("{}", args.prompt);
        }
        std::io::stdout().flush().ok();
        let started = std::time::Instant::now();
        let mut runtime = fellm_runtime::PluginSpeculativeRuntime::new(
            engine,
            speculator,
            fellm_runtime::GenericDraftConfig {
                maximum_proposal_length: args.speculative_max_tokens,
                initial_proposal_length: args.speculative_max_tokens.min(3),
                minimum_gain: args.speculative_min_gain,
            },
        )?;
        let tokens = runtime.generate_ids(&prompt_ids, params)?;
        let stop_ids = runtime.target().stop_token_ids_pub();
        let mut bytes = Vec::new();
        for token in &tokens {
            if !stop_ids.contains(token) {
                bytes.extend_from_slice(&runtime.target().tokenizer().decode_token(*token)?);
            }
        }
        println!("{}", String::from_utf8_lossy(&bytes));
        let metrics = runtime.metrics();
        eprintln!(
            "speculative: method={} rounds={} proposed={} verified={} accepted={} emitted={} acceptance={:.2}% accepted_p50={} accepted_p95={} k0={:.2}% draft_ms={:.2} verify_ms={:.2} sampling_ms={:.2} elapsed={:.2}ms",
            speculator_kind,
            metrics.rounds,
            metrics.proposed,
            metrics.verified,
            metrics.accepted,
            metrics.emitted,
            metrics.accepted as f64 / metrics.proposed.max(1) as f64 * 100.0,
            metrics.accepted_length_percentile(0.50),
            metrics.accepted_length_percentile(0.95),
            metrics.disabled_rounds as f64 / metrics.rounds.max(1) as f64 * 100.0,
            metrics.draft_time.as_secs_f64() * 1000.0,
            metrics.verification_time.as_secs_f64() * 1000.0,
            metrics.sampling_time.as_secs_f64() * 1000.0,
            started.elapsed().as_secs_f64() * 1000.0,
        );
        runtime.target().publish_memory_fabric_metrics();
        return Ok(());
    }

    if let Some(draft_path) = &args.speculative_draft {
        let mut draft_settings = settings;
        if let Some(backend) = &args.speculative_draft_backend {
            draft_settings.backend =
                BackendSelect::new(BackendPreference::parse(backend)?, args.cpu_fallback);
        }
        let draft = Engine::open_with_architecture(
            draft_path,
            draft_settings,
            Some(Arc::new(DiffusionGemmaPlugin)),
        )?;
        let prompt_ids = if use_chat {
            encode_chat(&engine, &args.prompt, args.system.as_deref())?
        } else {
            engine.tokenizer().encode(&args.prompt, true)?
        };
        if args.echo_prompt {
            print!("{}", args.prompt);
        }
        std::io::stdout().flush().ok();
        let started = std::time::Instant::now();
        let mut runtime = fellm_runtime::GenericDraftRuntime::new(
            engine,
            draft,
            fellm_runtime::GenericDraftConfig {
                maximum_proposal_length: args.speculative_max_tokens,
                initial_proposal_length: args.speculative_max_tokens.min(3),
                minimum_gain: args.speculative_min_gain,
            },
        )?;
        let tokens = runtime.generate_ids(&prompt_ids, params)?;
        let mut bytes = Vec::new();
        let stop_ids = runtime.target().stop_token_ids_pub();
        for token in &tokens {
            if !stop_ids.contains(token) {
                bytes.extend_from_slice(&runtime.target().tokenizer().decode_token(*token)?);
            }
        }
        println!("{}", String::from_utf8_lossy(&bytes));
        let metrics = runtime.metrics();
        eprintln!(
            "speculative: method=draft rounds={} proposed={} verified={} accepted={} emitted={} acceptance={:.2}% accepted_p50={} accepted_p95={} k0={:.2}% draft_ms={:.2} verify_ms={:.2} sampling_ms={:.2} elapsed={:.2}ms",
            metrics.rounds,
            metrics.proposed,
            metrics.verified,
            metrics.accepted,
            metrics.emitted,
            metrics.accepted as f64 / metrics.proposed.max(1) as f64 * 100.0,
            metrics.accepted_length_percentile(0.50),
            metrics.accepted_length_percentile(0.95),
            metrics.disabled_rounds as f64 / metrics.rounds.max(1) as f64 * 100.0,
            metrics.draft_time.as_secs_f64() * 1000.0,
            metrics.verification_time.as_secs_f64() * 1000.0,
            metrics.sampling_time.as_secs_f64() * 1000.0,
            started.elapsed().as_secs_f64() * 1000.0,
        );
        runtime.target().publish_memory_fabric_metrics();
        runtime.draft().publish_memory_fabric_metrics();
        return Ok(());
    }

    if args.echo_prompt {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }

    let mut stdout = std::io::stdout().lock();
    let stop_ids = {
        let mut s = Vec::new();
        if let Some(eos) = engine.tokenizer().eos() {
            s.push(eos);
        }
        s
    };
    let mut stream = if use_chat {
        let mut messages = Vec::new();
        if let Some(sys) = &args.system {
            messages.push(fellm_runtime::Message::text("system", sys.clone()));
        }
        messages.push(fellm_runtime::Message::text("user", args.prompt.clone()));
        engine.chat(&messages, params)?
    } else {
        engine.generate(&args.prompt, params)?
    };

    let mut byte_buf: Vec<u8> = Vec::new();
    while let Some(tok_result) = stream.next() {
        let tok = tok_result?;
        if stop_ids.contains(&tok) {
            continue;
        }
        let bytes = stream.decode_token(tok)?;
        if bytes.is_empty() {
            continue;
        }
        byte_buf.extend_from_slice(&bytes);
        let s = flush_valid_utf8_prefix(&mut byte_buf);
        stdout.write_all(s.as_bytes()).ok();
        stdout.flush().ok();
    }
    if !byte_buf.is_empty() {
        stdout
            .write_all(String::from_utf8_lossy(&byte_buf).as_bytes())
            .ok();
    }
    println!();

    let stats = stream.finish_stats();
    eprintln!(
        "prompt eval: {} tok in {:.2}ms ({:.2} tok/s)",
        stats.prompt_tokens,
        stats.prompt_ms,
        stats.prompt_tok_per_sec()
    );
    eprintln!(
        "eval: {} tok in {:.2}ms ({:.2} tok/s)",
        stats.predicted_tokens,
        (stats.total_ms - stats.prompt_ms).max(0.0),
        stats.generation_tok_per_sec()
    );
    eprintln!("TTFT: {:.2}ms", stats.time_to_first_token_ms);
    eprintln!("total: {:.2}ms", stats.total_ms);
    drop(stream);
    engine.publish_memory_fabric_metrics();
    engine.log_memory_fabric_runtime(u64::from(
        stats.prompt_tokens.saturating_add(stats.predicted_tokens),
    ));

    Ok(())
}

fn flush_valid_utf8_prefix(buf: &mut Vec<u8>) -> String {
    let mut cut = buf.len();
    while cut > 0 {
        if std::str::from_utf8(&buf[..cut]).is_ok() {
            break;
        }
        cut -= 1;
    }
    let s = String::from_utf8_lossy(&buf[..cut]).to_string();
    buf.drain(..cut);
    s
}

fn inspect(args: InspectArgs) -> fellm_core::error::Result<()> {
    let g = GgufFile::open_storage_native(&args.model)?;
    println!("architecture     : {}", g.metadata.arch().unwrap_or("?"));
    println!("tensor_data_off  : {}", g.tensor_data_offset());
    println!("alignment        : {}", g.alignment());
    println!("tensors          : {}", g.tensor_infos.len());
    println!("metadata entries : {}", g.metadata.iter().count());
    println!();
    println!("-- metadata --");
    for (k, v) in g.metadata.iter() {
        println!("  {k} = {v:?}");
    }
    println!();
    println!("-- tensors --");
    for ti in g.tensors() {
        println!(
            "  {:>40}  {:>6}  {}",
            ti.name,
            ti.dtype.to_string(),
            ti.shape
        );
    }
    Ok(())
}

fn open_plugin_host(dir: Option<&PathBuf>) -> fellm_core::error::Result<PluginHost> {
    let mut host = PluginHost::new();
    let ctx = HostContext::new(0, 0, std::ptr::null_mut(), "cpu");
    host.load_dir(dir.map(PathBuf::as_path), &ctx)?;
    Ok(host)
}

fn plugins_cmd(cmd: PluginsCmd) -> fellm_core::error::Result<()> {
    match cmd.action {
        PluginsAction::List {
            plugin_dir,
            capability,
        } => {
            let host = open_plugin_host(plugin_dir.as_ref())?;
            let cap_filter = capability
                .as_deref()
                .map(|s| {
                    CapabilityKind::parse(s).ok_or_else(|| {
                        fellm_core::error::FellmError::other(format!(
                            "unknown capability '{s}' (try attention, sequence_state_policy, backend, …)"
                        ))
                    })
                })
                .transpose()?;

            println!("loaded dynamic libraries: {}", host.plugin_count());
            for p in host.plugin_paths() {
                println!("  lib: {}", p.display());
            }
            println!();
            println!(
                "{:<32} {:<22} {:<10} {:>8}  {}",
                "NAME", "CAPABILITY", "VERSION", "PRIORITY", "SUMMARY"
            );
            println!("{}", "-".repeat(100));
            let mut list = host.capabilities().list();
            list.sort_by_key(|p| (p.descriptor.capability.name(), p.descriptor.name.as_str()));
            for p in list {
                if let Some(cf) = cap_filter {
                    if p.descriptor.capability != cf {
                        continue;
                    }
                }
                println!(
                    "{:<32} {:<22} {:<10} {:>8}  {}",
                    p.descriptor.name,
                    p.descriptor.capability.name(),
                    p.descriptor.version.to_string(),
                    p.descriptor.priority,
                    p.descriptor.summary
                );
            }
            Ok(())
        }
        PluginsAction::Inspect { name, plugin_dir } => {
            let host = open_plugin_host(plugin_dir.as_ref())?;
            let Some(p) = host.capabilities().get(&name) else {
                return Err(fellm_core::error::FellmError::other(format!(
                    "provider '{name}' not found; run `fellm plugins list`"
                )));
            };
            let d = &p.descriptor;
            println!("name        : {}", d.name);
            println!("capability  : {}", d.capability);
            println!("version     : {}", d.version);
            println!("priority    : {}", d.priority);
            println!("prepared_id : {}", p.id.0);
            println!(
                "source      : {}",
                p.source.as_deref().unwrap_or("dynamic/unknown")
            );
            println!("summary     : {}", d.summary);
            println!("provides    : {}", d.provides);
            println!("requires    : {}", d.requires);
            if !d.metadata.is_empty() {
                println!("metadata:");
                for (k, v) in &d.metadata {
                    println!("  {k} = {v}");
                }
            }
            Ok(())
        }
    }
}
