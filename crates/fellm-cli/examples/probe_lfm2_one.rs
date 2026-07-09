use fellm_runtime::{Engine, EngineSettings, GenParams, Message};

fn main() {
    let settings = EngineSettings::default().ctx_size(512);
    let mut engine = Engine::open_with(
        std::path::Path::new("models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf"),
        settings,
    )
    .unwrap();
    // Force completion with exact chat prompt
    let prompt = "<|startoftext|><|im_start|>user\nwho are you<|im_end|>\n<|im_start|>assistant\n";
    let mut stream = engine
        .generate(
            prompt,
            GenParams {
                max_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
    // peek via generating one token - also print top logits by hacking through next
    let tok = stream.next().unwrap().unwrap();
    let b = stream.decode_token(tok).unwrap();
    println!("tok={} text={:?}", tok, String::from_utf8_lossy(&b));
}
