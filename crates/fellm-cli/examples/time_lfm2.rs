use fellm_runtime::{Engine, EngineSettings, GenParams};
use std::time::Instant;

fn main() {
    let settings = EngineSettings::default().ctx_size(512);
    let mut engine = Engine::open_with(
        std::path::Path::new("models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf"),
        settings,
    )
    .unwrap();
    let t0 = Instant::now();
    let mut stream = engine
        .generate(
            "hi",
            GenParams {
                max_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
    println!("prefill+first_logits {:?}", t0.elapsed());
    let t1 = Instant::now();
    let tok = stream.next().unwrap().unwrap();
    println!("first_token={} {:?}", tok, t1.elapsed());
    println!("stats={:?}", stream.finish_stats());
}
