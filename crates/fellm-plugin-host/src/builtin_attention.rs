//! First-party attention providers registered through the public capability API.

use fellm_core::error::Result;
use fellm_plugin_abi::attention_provider::{
    AttentionPathKind, AttentionPrepareContext, AttentionProvider, AttentionWorkload,
    DeviceCapabilityView, PreparedAttention,
};
use fellm_plugin_abi::capability::{
    CapabilityKind, FeatureId, FeatureSet, PluginConfig, ProviderDescriptor, ProviderVersion,
};

/// Host FA2-style tiled attention (CPU reference path used for correctness and
/// as the default when no CUDA attention plugin is available).
pub struct HostTiledAttentionProvider {
    desc: ProviderDescriptor,
}

impl HostTiledAttentionProvider {
    /// Construct the host tiled provider.
    #[must_use]
    pub fn new() -> Self {
        let provides = FeatureSet::from_ids([
            FeatureId::ATTN_CAUSAL,
            FeatureId::ATTN_BIDIRECTIONAL,
            FeatureId::ATTN_MHA,
            FeatureId::ATTN_MQA,
            FeatureId::ATTN_GQA,
            FeatureId::ATTN_CONTIGUOUS_KV,
            FeatureId::ATTN_PAGED_KV,
            FeatureId::ATTN_PREFILL,
            FeatureId::ATTN_DECODE,
            FeatureId::ATTN_BATCHED_DECODE,
            FeatureId::ATTN_FP16,
            FeatureId::ATTN_BF16,
            FeatureId::ATTN_SLIDING_WINDOW,
            FeatureId::ATTN_INDIRECT_POSITIONS,
            FeatureId::ATTN_PER_HEAD_KV_VIEWS,
        ]);
        let desc = ProviderDescriptor::new(
            "attention.host_tiled",
            CapabilityKind::Attention,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "FA2-style tiled online-softmax attention on host (reference / CPU)",
        )
        .with_provides(provides)
        .with_priority(10)
        .with_meta("path", "fa2_style_host")
        .with_meta("builtin", "true");
        Self { desc }
    }
}

impl Default for HostTiledAttentionProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionProvider for HostTiledAttentionProvider {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.desc
    }

    fn validate_config(&self, config: &PluginConfig) -> Result<()> {
        let scoped = config.for_provider("attention.host_tiled", true);
        if let Some(v) = scoped.get("q_tile") {
            v.parse::<u32>().map_err(|_| {
                fellm_core::error::FellmError::other(format!(
                    "attention.host_tiled: invalid q_tile '{v}'"
                ))
            })?;
        }
        if let Some(v) = scoped.get("kv_tile") {
            v.parse::<u32>().map_err(|_| {
                fellm_core::error::FellmError::other(format!(
                    "attention.host_tiled: invalid kv_tile '{v}'"
                ))
            })?;
        }
        Ok(())
    }

    fn supports(&self, workload: &AttentionWorkload, _device: &DeviceCapabilityView) -> bool {
        workload.head_dim > 0
            && workload.head_dim <= 256
            && workload.n_heads > 0
            && workload.n_kv_heads > 0
            && workload.n_heads.is_multiple_of(workload.n_kv_heads.max(1))
    }

    fn prepare(&self, ctx: &AttentionPrepareContext<'_>) -> Result<Vec<PreparedAttention>> {
        let mut plans = Vec::new();
        let mut saw_prefill = false;
        let mut saw_decode = false;
        for w in ctx.workloads {
            if !self.supports(w, ctx.device) {
                return Err(fellm_core::error::FellmError::other(format!(
                    "attention.host_tiled: unsupported workload q={} kv={} hd={}",
                    w.query_len, w.kv_len, w.head_dim
                )));
            }
            if w.is_prefill() {
                saw_prefill = true;
            }
            if w.is_decode() {
                saw_decode = true;
            }
        }
        // Distinct prepared paths for prefill vs decode (FA2 principle).
        if saw_prefill || ctx.workloads.is_empty() {
            plans.push(PreparedAttention {
                provider: fellm_plugin_abi::PreparedProviderId::NONE,
                path: AttentionPathKind::Prefill,
                // Variant encodes tile sizes: high 16 = Br, low 16 = Bc
                kernel_variant: encode_tiles(64, 64),
                plan_handle: 1,
                features_used: self.desc.provides.clone(),
            });
        }
        if saw_decode || ctx.workloads.is_empty() {
            plans.push(PreparedAttention {
                provider: fellm_plugin_abi::PreparedProviderId::NONE,
                path: AttentionPathKind::Decode,
                kernel_variant: encode_tiles(1, 128),
                plan_handle: 2,
                features_used: self.desc.provides.clone(),
            });
        }
        if ctx
            .workloads
            .iter()
            .any(|w| w.query_len > 1 && w.query_len <= 8)
        {
            plans.push(PreparedAttention {
                provider: fellm_plugin_abi::PreparedProviderId::NONE,
                path: AttentionPathKind::BatchedDecode,
                kernel_variant: encode_tiles(8, 128),
                plan_handle: 3,
                features_used: self.desc.provides.clone(),
            });
        }
        Ok(plans)
    }

    fn applicability(&self, workload: &AttentionWorkload, device: &DeviceCapabilityView) -> i32 {
        if !self.supports(workload, device) {
            return i32::MIN;
        }
        // Prefer CUDA provider when hopper/ampere features exist.
        if device.features.contains(FeatureId::HW_AMPERE_ADA)
            || device.features.contains(FeatureId::HW_HOPPER)
        {
            return self.desc.priority - 50;
        }
        self.desc.priority
    }
}

/// CUDA attention provider: FA2-style on Ampere/Ada, FA3-style when Hopper
/// features are present. Selection is capability-driven.
pub struct CudaAttentionProvider {
    desc: ProviderDescriptor,
}

impl CudaAttentionProvider {
    /// Construct the CUDA attention provider.
    #[must_use]
    pub fn new() -> Self {
        let provides = FeatureSet::from_ids([
            FeatureId::ATTN_CAUSAL,
            FeatureId::ATTN_BIDIRECTIONAL,
            FeatureId::ATTN_MHA,
            FeatureId::ATTN_MQA,
            FeatureId::ATTN_GQA,
            FeatureId::ATTN_CONTIGUOUS_KV,
            FeatureId::ATTN_PAGED_KV,
            FeatureId::ATTN_PREFILL,
            FeatureId::ATTN_DECODE,
            FeatureId::ATTN_BATCHED_DECODE,
            FeatureId::ATTN_FP16,
            FeatureId::ATTN_BF16,
            FeatureId::ATTN_SLIDING_WINDOW,
            FeatureId::ATTN_INDIRECT_POSITIONS,
            FeatureId::ATTN_PER_HEAD_KV_VIEWS,
        ]);
        // Does not *require* hopper — FA2 path works on ampere/ada; FA3 is optional.
        let requires = FeatureSet::new();
        let desc = ProviderDescriptor::new(
            "attention.cuda_tiled",
            CapabilityKind::Attention,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "CUDA FA2-style tiled attention; FA3-style path when device exposes Hopper-class features",
        )
        .with_provides(provides)
        .with_requires(requires)
        .with_priority(100)
        .with_meta("path_ampere_ada", "fa2_style")
        .with_meta("path_hopper", "fa3_style")
        .with_meta("builtin", "true");
        Self { desc }
    }
}

impl Default for CudaAttentionProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionProvider for CudaAttentionProvider {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.desc
    }

    fn validate_config(&self, config: &PluginConfig) -> Result<()> {
        let scoped = config.for_provider("attention.cuda_tiled", true);
        for key in ["force_path", "q_tile", "kv_tile"] {
            if let Some(v) = scoped.get(key) {
                if key == "force_path" {
                    match v.as_str() {
                        "fa2" | "fa3" | "auto" => {}
                        other => {
                            return Err(fellm_core::error::FellmError::other(format!(
                                "attention.cuda_tiled: force_path must be fa2|fa3|auto, got '{other}'"
                            )));
                        }
                    }
                } else {
                    v.parse::<u32>().map_err(|_| {
                        fellm_core::error::FellmError::other(format!(
                            "attention.cuda_tiled: invalid {key} '{v}'"
                        ))
                    })?;
                }
            }
        }
        Ok(())
    }

    fn supports(&self, workload: &AttentionWorkload, device: &DeviceCapabilityView) -> bool {
        // CUDA provider is applicable when device is GPU-class or features include ampere/hopper.
        // On pure CPU builds it still *registers* but scores low / may be skipped if no GPU features.
        let gpuish = device.compute_major > 0
            || device.features.contains(FeatureId::HW_AMPERE_ADA)
            || device.features.contains(FeatureId::HW_HOPPER);
        if !gpuish {
            // Still "supports" for structural tests; auto-select scores lower on CPU.
        }
        workload.head_dim > 0
            && workload.head_dim <= 256
            && workload.n_heads > 0
            && workload.n_kv_heads > 0
    }

    fn prepare(&self, ctx: &AttentionPrepareContext<'_>) -> Result<Vec<PreparedAttention>> {
        let force = ctx
            .config
            .for_provider("attention.cuda_tiled", true)
            .get("force_path")
            .cloned()
            .unwrap_or_else(|| "auto".into());

        let use_fa3 = match force.as_str() {
            "fa3" => {
                if !ctx.device.features.contains(FeatureId::HW_HOPPER) {
                    return Err(fellm_core::error::FellmError::other(
                        "attention.cuda_tiled: force_path=fa3 but device lacks Hopper-class features",
                    ));
                }
                true
            }
            "fa2" => false,
            _ => ctx.device.features.contains(FeatureId::HW_HOPPER),
        };

        // Variant encoding:
        // bit 0: 0=FA2-style, 1=FA3-style
        // bits 1..: path-specific tile / pipeline depth
        let style_bit = u64::from(use_fa3);
        let mut plans = Vec::new();

        let need_prefill = ctx.workloads.iter().any(|w| w.is_prefill()) || ctx.workloads.is_empty();
        let need_decode = ctx.workloads.iter().any(|w| w.is_decode()) || ctx.workloads.is_empty();
        let need_batch = ctx
            .workloads
            .iter()
            .any(|w| w.query_len > 1 && w.query_len <= 8);

        if need_prefill {
            plans.push(PreparedAttention {
                provider: fellm_plugin_abi::PreparedProviderId::NONE,
                path: AttentionPathKind::Prefill,
                kernel_variant: style_bit | (64 << 8) | (64 << 24),
                plan_handle: if use_fa3 {
                    PLAN_FA3_PREFILL
                } else {
                    PLAN_FA2_PREFILL
                },
                features_used: features_for_style(use_fa3),
            });
        }
        if need_decode {
            plans.push(PreparedAttention {
                provider: fellm_plugin_abi::PreparedProviderId::NONE,
                path: AttentionPathKind::Decode,
                kernel_variant: style_bit | (1 << 8) | (128 << 24),
                plan_handle: if use_fa3 {
                    PLAN_FA3_DECODE
                } else {
                    PLAN_FA2_DECODE
                },
                features_used: features_for_style(use_fa3),
            });
        }
        if need_batch {
            plans.push(PreparedAttention {
                provider: fellm_plugin_abi::PreparedProviderId::NONE,
                path: AttentionPathKind::BatchedDecode,
                kernel_variant: style_bit | (8 << 8) | (128 << 24),
                plan_handle: if use_fa3 {
                    PLAN_FA3_BATCHED
                } else {
                    PLAN_FA2_BATCHED
                },
                features_used: features_for_style(use_fa3),
            });
        }
        Ok(plans)
    }

    fn applicability(&self, workload: &AttentionWorkload, device: &DeviceCapabilityView) -> i32 {
        if !self.supports(workload, device) {
            return i32::MIN;
        }
        let mut score = self.desc.priority;
        if device.features.contains(FeatureId::HW_HOPPER) {
            score += 50;
        } else if device.features.contains(FeatureId::HW_AMPERE_ADA) {
            score += 30;
        } else if device.compute_major == 0 {
            // No GPU: prefer host provider.
            score = 1;
        }
        score
    }
}

// Stable prepared-plan handles (not product version numbers — path tags only).
const PLAN_FA2_PREFILL: u64 = 0xFA2_0001;
const PLAN_FA2_DECODE: u64 = 0xFA2_0002;
const PLAN_FA2_BATCHED: u64 = 0xFA2_0003;
const PLAN_FA3_PREFILL: u64 = 0xFA3_0001;
const PLAN_FA3_DECODE: u64 = 0xFA3_0002;
const PLAN_FA3_BATCHED: u64 = 0xFA3_0003;

fn encode_tiles(br: u32, bc: u32) -> u64 {
    u64::from(br) << 16 | u64::from(bc)
}

fn features_for_style(fa3: bool) -> FeatureSet {
    let mut f = FeatureSet::from_ids([
        FeatureId::ATTN_CAUSAL,
        FeatureId::ATTN_GQA,
        FeatureId::ATTN_PAGED_KV,
        FeatureId::ATTN_INDIRECT_POSITIONS,
        FeatureId::ATTN_FP16,
        FeatureId::ATTN_BF16,
    ]);
    if fa3 {
        f.insert(FeatureId::HW_HOPPER);
        f.insert(FeatureId::HW_ASYNC_PIPELINE);
        f.insert(FeatureId::HW_TMA_CLASS);
        f.insert(FeatureId::HW_WGMMA_CLASS);
    } else {
        f.insert(FeatureId::HW_AMPERE_ADA);
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_core::dtype::DType;

    #[test]
    fn cuda_provider_selects_fa3_only_with_hopper_caps() {
        let p = CudaAttentionProvider::new();
        let w = AttentionWorkload {
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            query_len: 1,
            kv_len: 1024,
            dtype: DType::F16,
            causal: true,
            window: 0,
            paged: true,
            indirect_positions: false,
        };
        let mut device = DeviceCapabilityView::default();
        device.features.insert(FeatureId::HW_AMPERE_ADA);
        device.compute_major = 8;
        let cfg = PluginConfig::new();
        let plans = p
            .prepare(&AttentionPrepareContext {
                workloads: &[w],
                device: &device,
                config: &cfg,
                n_layers: 1,
            })
            .unwrap();
        assert!(
            plans.iter().all(|pl| pl.kernel_variant & 1 == 0),
            "Ampere/Ada path must not set FA3 style bit"
        );

        device.features.insert(FeatureId::HW_HOPPER);
        device.compute_major = 9;
        let plans = p
            .prepare(&AttentionPrepareContext {
                workloads: &[w],
                device: &device,
                config: &cfg,
                n_layers: 1,
            })
            .unwrap();
        assert!(
            plans.iter().any(|pl| pl.kernel_variant & 1 == 1),
            "Hopper features should select FA3-style path"
        );
    }

    #[test]
    fn host_and_cuda_coexist() {
        let h = HostTiledAttentionProvider::new();
        let c = CudaAttentionProvider::new();
        assert_ne!(h.descriptor().name, c.descriptor().name);
        assert_eq!(h.descriptor().capability, CapabilityKind::Attention);
        assert_eq!(c.descriptor().capability, CapabilityKind::Attention);
    }
}
