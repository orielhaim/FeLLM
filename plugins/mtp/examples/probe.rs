use fellm_gguf::GgufFile;
use fellm_model::ModelSpec;
use fellm_runtime::architecture::ModelSpeculatorPlugin;
use std::path::PathBuf;

fn main() -> fellm_core::error::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| fellm_core::error::FellmError::other("usage: probe MODEL.gguf"))?;
    let gguf = GgufFile::open(&path)?;
    let spec = ModelSpec::from_gguf(&gguf)?;
    let prepared = fellm_mtp::MtpPlugin
        .prepare(&gguf, &spec)?
        .ok_or_else(|| fellm_core::error::FellmError::other("model has no MTP modules"))?;
    println!(
        "architecture={} trunk_layers={} mtp_stages={} vocab={} max_proposal={}",
        spec.arch_id,
        spec.n_layers,
        prepared.graphs.len(),
        spec.vocab_size,
        prepared.compatibility.maximum_proposal_length
    );
    for (stage, graph) in prepared.graphs.iter().enumerate() {
        println!("stage={stage} graph_nodes={}", graph.node_count());
    }
    Ok(())
}
