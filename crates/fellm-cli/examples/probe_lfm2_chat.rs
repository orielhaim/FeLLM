use fellm_gguf::GgufFile;
use fellm_tokenizer::{Message, load};
use std::time::Instant;

fn main() {
    let g = GgufFile::open("models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf").unwrap();
    let tok = load(&g).unwrap();
    let msgs = [Message::text("user", "who are you")];
    let t0 = Instant::now();
    let prompt = tok.apply_chat_template(&msgs, true).unwrap().unwrap();
    println!("template_ms={:?} chars={} ", t0.elapsed(), prompt.len());
    println!("---PROMPT---\n{prompt}\n---END---");
    let ids = tok.encode(&prompt, true).unwrap();
    println!("n_tokens={}", ids.len());
    // show first few decoded specials
    for (i, id) in ids.iter().take(20).enumerate() {
        let b = tok.decode_token(*id).unwrap();
        let s = String::from_utf8_lossy(&b);
        println!("[{i}] id={id} {:?}", s);
    }
}
