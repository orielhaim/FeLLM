//! `fellm` CLI.

use clap::{Parser, Subcommand};
use fellm_gguf::GgufFile;
use fellm_runtime::{Engine, GenParams};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "fellm", version, about = "FeLLM inference engine (Phase 1)")]
struct Cli {
    /// Log level filter.
    #[arg(long, default_value = "info", global = true)]
    log: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate tokens from a prompt.
    Run(RunArgs),
    /// Print GGUF metadata and tensor list.
    Inspect(InspectArgs),
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
    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// top-k (0 disables).
    #[arg(long, default_value_t = 0)]
    top_k: u32,
    /// top-p (>= 1.0 disables).
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Optional max sequence length override.
    #[arg(long)]
    max_seq: Option<usize>,
}

#[derive(clap::Args, Debug)]
struct InspectArgs {
    /// Path to a GGUF file.
    #[arg(long)]
    model: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    init_tracing(&cli.log);

    let result = match cli.cmd {
        Cmd::Run(a) => run(a),
        Cmd::Inspect(a) => inspect(a),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn init_tracing(filter: &str) {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(false).init();
}

fn run(args: RunArgs) -> fellm_core::error::Result<()> {
    let mut engine = Engine::open(&args.model, args.max_seq)?;
    let params = GenParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };

    let use_chat = !args.completion && engine.tokenizer().chat_template().is_some();

    // Echo the user-visible prompt (not the templated internals).
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
            messages.push(fellm_runtime::Message {
                role: "system".into(),
                content: sys.clone(),
            });
        }
        messages.push(fellm_runtime::Message {
            role: "user".into(),
            content: args.prompt.clone(),
        });
        engine.chat(&messages, params)?
    } else {
        engine.generate(&args.prompt, params)?
    };

    let mut byte_buf: Vec<u8> = Vec::new();
    while let Some(tok_result) = stream.next() {
        let tok = tok_result?;
        // Stop tokens decode to empty; skip echoing them.
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
    Ok(())
}

fn flush_valid_utf8_prefix(buf: &mut Vec<u8>) -> String {
    // Find the longest prefix of buf that's valid UTF-8.
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
