use fellm_gguf::GgufFile;
fn main() {
    let g = GgufFile::open("models/LFM2.5-8B-A1B-UD-Q4_K_S.gguf").unwrap();
    for name in [
        "blk.0.shortconv.conv.weight",
        "blk.2.ffn_gate_inp.weight",
        "blk.2.ffn_gate_exps.weight",
        "blk.2.ffn_down_exps.weight",
        "token_embd_norm.weight",
        "token_embd.weight",
    ] {
        let t = g.tensor(name).unwrap();
        println!(
            "{name}: dtype={} shape={} nbytes={}",
            t.dtype(),
            t.shape,
            t.as_bytes().len()
        );
    }
}
