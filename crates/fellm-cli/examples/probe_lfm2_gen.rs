use fellm_runtime::{Engine, EngineSettings, GenParams, Message};
use std::time::Instant;

fn main() {
    let settings = EngineSettings::default().ctx_size(2048);
    let mut engine = Engine::open_with(
        std::path::Path::new("models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf"),
        settings,
    )
    .unwrap();
    let params = GenParams {
        max_tokens: 48,
        temperature: 0.0,
        ..Default::default()
    };
    let msgs = [Message::text("user", "who are you")];
    let mut stream = engine.chat(&msgs, params).unwrap();
    let mut i = 0u32;
    while let Some(r) = stream.next() {
        let tok = r.unwrap();
        let b = stream.decode_token(tok).unwrap();
        let s = String::from_utf8_lossy(&b);
        println!("[{i}] id={tok} bytes={:?} text={:?}", b, s);
        i += 1;
    }
}
