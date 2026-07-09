use fellm_gguf::GgufFile;
fn main() {
    let g = GgufFile::open("models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf").unwrap();
    let tmpl = g.metadata.get_string("tokenizer.chat_template").unwrap();
    std::fs::write("target/chat_template_lfm2.jinja", tmpl).unwrap();
    println!("len={}", tmpl.len());
    println!("{tmpl}");
}
