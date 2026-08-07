//! DiffusionGemma architecture plugin.
//!
//! This crate is the model-family implementation.  The generic runtime only
//! receives the prepared canvas graph and the block-diffusion program from
//! this plugin; it does not inspect `diffusion-gemma` tensor names.

#![deny(missing_docs)]

use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;
use fellm_model::{DEFAULT_DIFFUSION_SELF_COND_TOP_K, ModelSpec, build_diffusion_canvas_graph};
use fellm_plugin_abi::{
    ArchitectureConfig, ArchitectureProvider, BackendCapabilities, GenerationDriver,
    GenerationRequest, GraphId, GraphSpec, ModelProgram, ModelSource,
};
use fellm_runtime::architecture::{
    ArchitectureGenerationMode, ArchitecturePlugin, ArchitecturePreparation, backend_capabilities,
    source_from_gguf,
};
use fellm_runtime::block_diffusion::{
    BlockDiffusionConfig, BlockDiffusionDriver, BlockDiffusionGraphs,
};

/// The statically linked DiffusionGemma architecture plugin.
#[derive(Debug, Default, Clone, Copy)]
pub struct DiffusionGemmaPlugin;

impl DiffusionGemmaPlugin {
    const ID: &'static str = "diffusion-gemma";
}

impl ArchitectureProvider for DiffusionGemmaPlugin {
    fn architecture_id(&self) -> &str {
        Self::ID
    }

    fn probe(&self, source: &ModelSource) -> Result<ArchitectureConfig> {
        if source.architecture_id != Self::ID {
            return Err(FellmError::UnsupportedArchitecture(
                source.architecture_id.clone(),
            ));
        }
        for required in [
            "token_embd.weight",
            "output_norm.weight",
            "self_cond_pre_norm.weight",
            "self_cond_gate.weight",
            "self_cond_up.weight",
            "self_cond_down.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_inp.weight",
        ] {
            if !source.tensors.iter().any(|(name, _)| name == required) {
                return Err(FellmError::other(format!(
                    "diffusion-gemma missing required tensor {required}"
                )));
            }
        }
        Ok(ArchitectureConfig {
            architecture_id: Self::ID.into(),
            data: "{\"canvas_length\":256,\"text_only\":true}".into(),
        })
    }

    fn compile(
        &self,
        _source: &ModelSource,
        config: &ArchitectureConfig,
        backend: &BackendCapabilities,
    ) -> Result<ModelProgram> {
        if config.architecture_id != Self::ID {
            return Err(FellmError::other("DiffusionGemma config/provider mismatch"));
        }
        tracing::debug!(device_kind = ?backend.caps.device_kind, "compiling DiffusionGemma plugin program");
        Ok(ModelProgram {
            architecture_id: Self::ID.into(),
            graphs: vec![
                GraphSpec {
                    id: GraphId(0),
                    name: "causal_prefill".into(),
                },
                GraphSpec {
                    id: GraphId(1),
                    name: "bidirectional_canvas_denoise".into(),
                },
                GraphSpec {
                    id: GraphId(2),
                    name: "causal_canvas_commit".into(),
                },
            ],
        })
    }

    fn create_generation_driver(
        &self,
        program: &ModelProgram,
        request: GenerationRequest,
    ) -> Result<Box<dyn GenerationDriver>> {
        let max_denoising_steps = std::env::var("FELLM_DIFFUSION_STEPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|steps| *steps > 0)
            .unwrap_or(BlockDiffusionConfig::default().max_denoising_steps);
        let self_conditioning_top_k = std::env::var("FELLM_DIFFUSION_SELF_COND_TOP_K")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_DIFFUSION_SELF_COND_TOP_K);
        let config = BlockDiffusionConfig {
            max_denoising_steps,
            self_conditioning_top_k,
            ..Default::default()
        };
        BlockDiffusionDriver::new(
            program,
            request,
            config,
            BlockDiffusionGraphs {
                prefill: GraphId(0),
                denoise: GraphId(1),
                commit: GraphId(2),
            },
            262_144,
        )
        .map(|driver| Box::new(driver) as Box<dyn GenerationDriver>)
    }
}

impl ArchitecturePlugin for DiffusionGemmaPlugin {
    fn architecture_id(&self) -> &str {
        Self::ID
    }

    fn prepare(
        &self,
        gguf: &GgufFile,
        spec: &ModelSpec,
        backend: &dyn fellm_plugin_abi::Backend,
    ) -> Result<Option<ArchitecturePreparation>> {
        if spec.arch_id != Self::ID {
            return Ok(None);
        }
        let source = source_from_gguf(gguf);
        let config = self.probe(&source)?;
        let program = self.compile(&source, &config, &backend_capabilities(backend))?;
        let canvas_graph = build_diffusion_canvas_graph(gguf, spec)?;
        Ok(Some(ArchitecturePreparation {
            program,
            generation_mode: ArchitectureGenerationMode::BlockDiffusion,
            canvas_graph: Some(canvas_graph),
        }))
    }

    fn create_generation_driver(
        &self,
        program: &ModelProgram,
        request: GenerationRequest,
    ) -> Result<Box<dyn GenerationDriver>> {
        <Self as ArchitectureProvider>::create_generation_driver(self, program, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_compiles_three_graph_program() {
        let source = ModelSource {
            architecture_id: "diffusion-gemma".into(),
            tensors: [
                "token_embd.weight",
                "output_norm.weight",
                "self_cond_pre_norm.weight",
                "self_cond_gate.weight",
                "self_cond_up.weight",
                "self_cond_down.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_gate_inp.weight",
            ]
            .into_iter()
            .map(|name| (name.into(), String::new()))
            .collect(),
            ..Default::default()
        };
        let plugin = DiffusionGemmaPlugin;
        let config = plugin.probe(&source).unwrap();
        let program = plugin
            .compile(&source, &config, &BackendCapabilities::default())
            .unwrap();
        assert_eq!(program.graphs.len(), 3);
    }
}
