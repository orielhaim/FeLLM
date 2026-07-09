use fellm_gguf::GgufFile;
use fellm_tokenizer::{Message, ToolDef, load};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf".into());
    let g = GgufFile::open(&path).unwrap();
    let tok = load(&g).unwrap();

    let msgs = [Message::text("user", "who are you")];
    let p = tok.apply_chat_template(&msgs, true).unwrap().unwrap();
    println!("--- plain chat ({} chars) ---\n{p}", p.len());
    let ids = tok.encode(&p, true).unwrap();
    println!("n_tokens={}", ids.len());

    let tools = [ToolDef {
        name: "get_weather".into(),
        description: "Get weather info".into(),
        parameters_json:
            r#"{"type":"dict","required":["city"],"properties":{"city":{"type":"string"}}}"#.into(),
    }];
    let msgs2 = [Message::text("user", "What is the weather in SF?")];
    let p2 = tok
        .apply_chat_template_with_tools(&msgs2, &tools, true)
        .unwrap()
        .unwrap();
    println!("--- with tools ({} chars) ---\n{p2}", p2.len());
    let ids2 = tok.encode(&p2, true).unwrap();
    println!("n_tokens={}", ids2.len());
}
