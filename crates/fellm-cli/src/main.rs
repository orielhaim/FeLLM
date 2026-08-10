//! `fellm` CLI.

use clap::{Parser, Subcommand};
use fellm_architecture_diffusion_gemma::DiffusionGemmaPlugin;
use fellm_gguf::GgufFile;
use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_abi::capability::{CapabilityKind, PluginConfig, ProviderSelection};
use fellm_plugin_host::PluginHost;
use fellm_runtime::{BackendPreference, BackendSelect, Engine, EngineSettings, GenParams};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "fellm", version, about = "FeLLM inference engine")]
struct Cli {
    #[arg(long, default_value = "info", global = true)]
    log: String,

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
    #[arg(long)]
    model: PathBuf,
    /// Prompt / user message string.
    #[arg(long)]
    prompt: String,
    /// Optional system message (chat mode only).
    #[arg(long)]
    system: Option<String>,
    /// Force raw completion (skip chat template even if the model has one).
    #[arg(long, default_value_t = false)]
    completion: bool,
    /// Max tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,
    /// Sampling temperature.
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,
    /// top-k (0 disables).
    #[arg(long, default_value_t = 80)]
    top_k: u32,
    /// top-p (>= 1.0 disables).
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Repetition penalty (1.0 disables).
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f32,
    /// Context size (`n_ctx`). Default 8192, clamped to the model maximum.
    /// Pass `0` to use the model's GGUF-reported maximum context length.
    #[arg(long = "ctx-size", short = 'c', default_value_t = 8192)]
    ctx_size: usize,
    /// Evaluation batch size (`n_batch`).
    #[arg(long = "batch-size", short = 'b', default_value_t = 2048)]
    batch_size: usize,
    /// Physical batch size (`n_ubatch`).
    #[arg(long = "ubatch-size", default_value_t = 512)]
    ubatch_size: usize,
    /// Exact KV arena bytes (0 selects automatic device/system budgeting).
    #[arg(long, default_value_t = 0)]
    kv_cache_bytes: u64,
    /// Fraction of available memory considered by automatic KV budgeting.
    #[arg(long, default_value_t = 0.25)]
    kv_memory_fraction: f64,
    /// Host swap-tier bytes reserved for paged KV.
    #[arg(long, default_value_t = 0)]
    kv_swap_bytes: u64,
    /// Bytes kept free as a safety reserve.
    #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024)]
    kv_safety_reserve_bytes: u64,
    /// Optional max sequence length override (alias of `--ctx-size`).
    #[arg(long, hide = true)]
    max_seq: Option<usize>,
    /// Compute backend provider preference: `auto`, `cpu`, or `cuda`.
    /// Also set via `FELLM_BACKEND`.
    #[arg(long, default_value = "auto")]
    backend: String,
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
    /// Disable CPU fallback when CUDA is requested/auto but unavailable.
    #[arg(long, default_value_t = false)]
    no_cpu_fallback: bool,
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

fn main() {
    let cli = Cli::parse();
    init_tracing(&cli.log);

    let result = match cli.cmd {
        Cmd::Run(a) => run(a),
        Cmd::Inspect(a) => inspect(a),
        Cmd::Plugins(p) => plugins_cmd(p),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
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
    fmt().with_env_filter(filter).with_target(false).init();
}

fn build_selection(args: &RunArgs) -> fellm_core::error::Result<ProviderSelection> {
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

fn run(args: RunArgs) -> fellm_core::error::Result<()> {
    let preference = BackendPreference::parse(&args.backend)?;
    let select = BackendSelect::new(preference, !args.no_cpu_fallback);
    let providers = build_selection(&args)?;
    let kv_cache = fellm_runtime::KvCacheConfig {
        budget_bytes: (args.kv_cache_bytes > 0).then_some(args.kv_cache_bytes),
        memory_fraction: args.kv_memory_fraction,
        safety_reserve_bytes: args.kv_safety_reserve_bytes,
        swap_bytes: args.kv_swap_bytes,
        ..fellm_runtime::KvCacheConfig::default()
    };
    let mut settings = EngineSettings::default()
        .batch_size(args.batch_size)
        .ubatch_size(args.ubatch_size)
        .backend_select(select)
        .providers(providers)
        .kv_cache(kv_cache);

    if let Some(dir) = &args.plugin_dir {
        settings = settings.plugin_dir(dir.clone());
    }

    let ctx = args.max_seq.unwrap_or(args.ctx_size);
    settings = if ctx == 0 {
        settings.ctx_from_model()
    } else {
        settings.ctx_size(ctx)
    };

    let mut engine = Engine::open_with_architecture(
        &args.model,
        settings,
        Some(Arc::new(DiffusionGemmaPlugin)),
    )?;

    if let Some(prep) = engine.providers().prepared() {
        eprintln!(
            "providers: attention={} (id={}) kv-policy={} (id={})",
            prep.attention_name, prep.attention_id.0, prep.kv_policy_name, prep.kv_policy_id.0
        );
    }

    let params = GenParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        repetition_penalty: args.repetition_penalty,
        priority: 0,
    };

    let use_chat = !args.completion && engine.tokenizer().chat_template().is_some();

    print!("{}", args.prompt);
    std::io::stdout().flush().ok();

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
    let g = GgufFile::open(&args.model)?;
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
