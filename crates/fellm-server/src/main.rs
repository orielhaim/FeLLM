mod openai;
mod routes;
mod state;
mod worker;

use clap::Parser;
use fellm_runtime::{BackendPreference, BackendSelect, EngineSettings, GenParams};
use state::AppState;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Parser, Debug)]
#[command(
    name = "fellm-server",
    version,
    about = "OpenAI-compatible HTTP server for FeLLM"
)]
struct Args {
    /// Path to a GGUF model file.
    #[arg(long)]
    model: PathBuf,

    /// Model id advertised in API responses (default: file stem).
    #[arg(long)]
    model_id: Option<String>,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Context size (`n_ctx`). Pass `0` for the model's GGUF maximum.
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

    /// Host residency-tier bytes for KV migration / preempt.
    #[arg(long = "kv-host-budget", alias = "kv-swap-bytes", default_value_t = 0)]
    kv_host_budget: u64,

    /// KV fabric mode: `auto`, `exact`, or `elastic`.
    #[arg(long = "kv-mode", default_value = "auto")]
    kv_mode: String,

    /// KV addressing: `block_table` or `virtual_memory`.
    #[arg(long = "kv-addressing", default_value = "block_table")]
    kv_addressing: String,

    /// Bytes kept free as a safety reserve.
    #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024)]
    kv_safety_reserve_bytes: u64,

    /// Default max tokens when the request omits it.
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,

    /// Default sampling temperature.
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    /// Default top-k (0 disables).
    #[arg(long, default_value_t = 80)]
    top_k: u32,

    /// Default top-p (>= 1.0 disables).
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Default RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Default repetition penalty (1.0 disables).
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f32,

    /// Log filter (e.g. `info`, `debug`).
    #[arg(long, default_value = "info")]
    log: String,

    /// Compute backend: `auto` (default), `cpu`, or `cuda`.
    /// Also set via `FELLM_BACKEND`. CUDA requires `--features backend-cuda`.
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Disable CPU fallback when CUDA is unavailable.
    /// Also set via `FELLM_CPU_FALLBACK=0`.
    #[arg(long, default_value_t = false)]
    no_cpu_fallback: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    init_tracing(&args.log);

    let model_id = args.model_id.clone().unwrap_or_else(|| {
        args.model
            .file_stem()
            .map_or_else(|| "fellm".into(), |s| s.to_string_lossy().into_owned())
    });

    let preference = BackendPreference::parse(&args.backend).unwrap_or_else(|e| {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    });
    let select = BackendSelect::new(preference, !args.no_cpu_fallback);
    let kv_cache = fellm_runtime::KvFabricConfig {
        mode: fellm_runtime::KvMode::parse(&args.kv_mode).unwrap_or_default(),
        device_budget: (args.kv_cache_bytes > 0).then_some(args.kv_cache_bytes),
        host_budget: Some(args.kv_host_budget),
        addressing: fellm_runtime::KvAddressing::parse(&args.kv_addressing).unwrap_or_default(),
        memory_fraction: args.kv_memory_fraction,
        safety_reserve_bytes: args.kv_safety_reserve_bytes,
        ..fellm_runtime::KvFabricConfig::default()
    };
    let mut settings = EngineSettings::default()
        .batch_size(args.batch_size)
        .ubatch_size(args.ubatch_size)
        .backend_select(select)
        .kv_cache(kv_cache);
    settings = if args.ctx_size == 0 {
        settings.ctx_from_model()
    } else {
        settings.ctx_size(args.ctx_size)
    };

    let defaults = GenParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        repetition_penalty: args.repetition_penalty,
        priority: 0,
        ..GenParams::default()
    };

    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .unwrap_or_else(|error| {
            eprintln!("fatal: install Prometheus recorder: {error}");
            std::process::exit(1);
        });
    let (task_tx, task_rx) = mpsc::channel(64);
    if let Err(e) = worker::spawn_worker(&args.model, settings, task_rx) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
    let state = AppState {
        task_tx,
        model_id: model_id.clone(),
        defaults,
        metrics,
    };

    let app = routes::router(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("fatal: invalid bind address: {e}");
            std::process::exit(1);
        });

    tracing::info!(%addr, model = %model_id, "listening");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("fatal: bind {addr}: {e}");
            std::process::exit(1);
        });

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap_or_else(|e| {
            eprintln!("fatal: server error: {e}");
            std::process::exit(1);
        });
}

fn init_tracing(level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_filter(filter),
            )
            .with(tracing_tracy::TracyLayer::default())
            .init();
        return;
    }

    #[cfg(not(feature = "tracy"))]
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
