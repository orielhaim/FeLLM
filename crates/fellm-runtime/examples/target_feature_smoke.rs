use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::TargetFeature;
use fellm_runtime::{Engine, EngineSettings};
use std::path::Path;

fn main() -> Result<()> {
    let model = std::env::args()
        .nth(1)
        .ok_or_else(|| FellmError::other("usage: target_feature_smoke MODEL.gguf"))?;
    let requested = [
        TargetFeature::EmbeddingOutput,
        TargetFeature::FinalHiddenState,
    ];
    let settings = EngineSettings::default()
        .ctx_size(128)
        .target_features(requested);
    let mut engine = Engine::open_with_architecture(Path::new(&model), settings, None)?;
    let prompt = engine.tokenizer().encode("feature capture", true)?;
    let sequence = engine.prefill_sequence(&prompt)?;
    let embedding = engine.capture_target_feature(TargetFeature::EmbeddingOutput)?;
    let final_hidden = engine.capture_target_feature(TargetFeature::FinalHiddenState)?;
    if embedding.shape().num_elements() != engine.spec().d_model
        || final_hidden.shape().num_elements() != engine.spec().d_model
    {
        return Err(FellmError::other("captured feature shape mismatch"));
    }
    engine.release_sequence(sequence);
    println!(
        "captured {}-element embedding and final hidden state",
        engine.spec().d_model
    );
    Ok(())
}
