use fellm_runtime::{
    BackendPreference, Engine, EngineSettings, GenParams, KvFabricConfig, Scheduler, SequenceEvent,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: continuous_batch_smoke <model.gguf>")?;
    let preference = match std::env::args().nth(2).as_deref() {
        Some("cuda") => BackendPreference::Cuda,
        _ => BackendPreference::Cpu,
    };
    let settings = EngineSettings::default()
        .ctx_size(512)
        .batch_size(32)
        .ubatch_size(8)
        .backend_preference(preference)
        .allow_cpu_fallback(false)
        .kv_cache(KvFabricConfig {
            device_budget: Some(256 * 1024 * 1024),
            ..KvFabricConfig::default()
        });
    let mut engine = Engine::open_with(Path::new(&model), settings)?;
    let recurrent = engine.spec().is_hybrid();
    let mut scheduler = Scheduler::new();
    let prompts = [
        "List several primary and secondary colors in order.",
        "List several common animals in alphabetical order.",
    ];
    for prompt in prompts {
        let tokens = engine.tokenizer().encode(prompt, true)?;
        scheduler.enqueue_ids(
            &mut engine,
            tokens,
            GenParams {
                max_tokens: 4,
                temperature: 0.0,
                ..GenParams::default()
            },
            false,
            false,
        )?;
    }

    let mut completed = 0;
    let mut iterations = 0;
    let mut max_plan_items = 0;
    let mut max_plan_tokens = 0;
    let mut saw_chunk = false;
    while completed < prompts.len() && iterations < 128 {
        if let Some(event) = scheduler.poll_event(&mut engine) {
            match event {
                SequenceEvent::Done { .. } => completed += 1,
                SequenceEvent::Error { message, .. } => return Err(message.into()),
                SequenceEvent::Token { .. } => {}
            }
        }
        let plan = scheduler.last_plan();
        max_plan_items = max_plan_items.max(plan.items.len());
        max_plan_tokens = max_plan_tokens.max(plan.scheduled_tokens);
        saw_chunk |= plan
            .items
            .iter()
            .any(|item| item.token_count > 1 && item.token_count <= 8);
        iterations += 1;
    }
    if completed != prompts.len() || max_plan_items < 2 || (!recurrent && !saw_chunk) {
        return Err(format!(
            "batch proof failed: completed={completed} max_items={max_plan_items} saw_chunk={saw_chunk}"
        )
        .into());
    }
    println!(
        "completed={completed} iterations={iterations} max_plan_items={max_plan_items} max_plan_tokens={max_plan_tokens} chunked_prefill={saw_chunk} recurrent={recurrent}"
    );
    Ok(())
}
