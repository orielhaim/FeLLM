fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf".into());
    let g = fellm_gguf::GgufFile::open(&path).unwrap();
    println!("arch={}", g.metadata.arch().unwrap_or("?"));
    for ti in g.tensors() {
        println!("{}\t{}\t{}", ti.name, ti.dtype, ti.shape);
    }
}
