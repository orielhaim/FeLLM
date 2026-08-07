//! Provider selection and preparation for the engine.
//!
//! Owns a [`fellm_plugin_host::CapabilityRegistry`] (and optional dynamic
//! plugin load) so CLI/settings can pin attention and KV policies by name.

use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::attention_provider::{
    AttentionWorkload, DeviceCapabilityView, PreparedAttention,
};
use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_abi::capability::{
    CapabilityKind, FeatureSet, NegotiationReport, PreparedProviderId,
};
pub use fellm_plugin_abi::capability::{PluginConfig, ProviderSelection};
use fellm_plugin_abi::traits::Backend;
use fellm_plugin_host::{CapabilityRegistry, PluginHost};
use std::path::Path;
use std::sync::Arc;

/// Resolved providers ready for steady-state execution.
#[derive(Debug, Clone)]
pub struct PreparedProviders {
    /// Negotiation report (selected names + notes).
    pub report: NegotiationReport,
    /// Prepared attention provider id (hot-path key).
    pub attention_id: PreparedProviderId,
    /// Prepared KV policy id.
    pub kv_policy_id: PreparedProviderId,
    /// Selected attention provider name.
    pub attention_name: String,
    /// Selected KV policy name.
    pub kv_policy_name: String,
}

/// Engine-side provider manager.
pub struct ProviderManager {
    host: PluginHost,
    selection: ProviderSelection,
    prepared: Option<PreparedProviders>,
}

impl ProviderManager {
    /// Create with builtins only (no dynamic dir load yet).
    #[must_use]
    pub fn new(selection: ProviderSelection) -> Self {
        Self {
            host: PluginHost::new(),
            selection,
            prepared: None,
        }
    }

    /// Load dynamic plugins from `dir` (or `FELLM_PLUGIN_DIR` / `plugins/`).
    ///
    /// Ensures first-party TriAttention is available on the public
    /// `SequenceStatePolicy` contract. Dynamic `cdylib` name registration is
    /// idempotent with the rlib trait object.
    pub fn load_plugins(&mut self, dir: Option<&Path>) -> Result<()> {
        let ctx = HostContext::new(0, 0, std::ptr::null_mut(), "cpu");
        self.host.load_dir(dir, &ctx)?;
        let tri = Arc::new(fellm_triattention::TriAttentionPolicy::new());
        if self.host.capabilities().get("kv.triattention").is_none() {
            let _ = self.host.capabilities_mut().register_sequence_policy(tri);
        } else if self
            .host
            .capabilities()
            .sequence_policy("kv.triattention")
            .is_none()
        {
            self.host
                .capabilities_mut()
                .inject_sequence_policy("kv.triattention", tri)?;
        }
        Ok(())
    }

    /// Capability registry.
    #[must_use]
    pub fn capabilities(&self) -> &CapabilityRegistry {
        self.host.capabilities()
    }

    /// Mutable capability registry.
    pub fn capabilities_mut(&mut self) -> &mut CapabilityRegistry {
        self.host.capabilities_mut()
    }

    /// Plugin host.
    #[must_use]
    pub fn host(&self) -> &PluginHost {
        &self.host
    }

    /// Active selection.
    #[must_use]
    pub fn selection(&self) -> &ProviderSelection {
        &self.selection
    }

    /// Prepared result after [`Self::prepare`].
    #[must_use]
    pub fn prepared(&self) -> Option<&PreparedProviders> {
        self.prepared.as_ref()
    }

    /// Register an extra sequence policy (e.g. static TriAttention in tests).
    pub fn register_sequence_policy(
        &mut self,
        policy: Arc<dyn fellm_plugin_abi::SequenceStatePolicy>,
    ) -> Result<PreparedProviderId> {
        self.host
            .capabilities_mut()
            .register_sequence_policy(policy)
    }

    /// Prepare providers against the active backend and model shape.
    pub fn prepare(
        &mut self,
        backend: &dyn Backend,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_layers: u32,
    ) -> Result<&PreparedProviders> {
        let caps = backend.capabilities();
        let features = caps.feature_set();
        let device = DeviceCapabilityView {
            features: features.clone(),
            smem_per_sm: caps.smem_per_sm,
            compute_major: caps.compute_major,
            compute_minor: caps.compute_minor,
        };

        // Representative workloads for path preparation.
        let workloads = [
            AttentionWorkload {
                n_heads,
                n_kv_heads,
                head_dim,
                query_len: 1,
                kv_len: 1024,
                dtype: DType::F16,
                causal: true,
                window: 0,
                paged: true,
                indirect_positions: true,
            },
            AttentionWorkload {
                n_heads,
                n_kv_heads,
                head_dim,
                query_len: 128,
                kv_len: 128,
                dtype: DType::F16,
                causal: true,
                window: 0,
                paged: true,
                indirect_positions: true,
            },
            AttentionWorkload {
                n_heads,
                n_kv_heads,
                head_dim,
                query_len: 4,
                kv_len: 512,
                dtype: DType::F16,
                causal: true,
                window: 0,
                paged: true,
                indirect_positions: true,
            },
        ];

        let report = self.host.capabilities_mut().prepare(
            &self.selection,
            &features,
            &device,
            &workloads,
            n_layers,
        )?;

        let attention_name = report
            .selected
            .get(&CapabilityKind::Attention)
            .cloned()
            .ok_or_else(|| FellmError::other("no attention provider selected"))?;
        let kv_policy_name = report
            .selected
            .get(&CapabilityKind::SequenceStatePolicy)
            .cloned()
            .ok_or_else(|| FellmError::other("no kv policy selected"))?;

        let attention_id = self
            .host
            .capabilities()
            .id_of(&attention_name)
            .ok_or_else(|| FellmError::other("attention id missing"))?;
        let kv_policy_id = self
            .host
            .capabilities()
            .id_of(&kv_policy_name)
            .ok_or_else(|| FellmError::other("kv policy id missing"))?;

        // Ensure prepared attention plans exist for decode/prefill when Rust provider.
        let _decode: Option<&PreparedAttention> =
            self.host.capabilities().prepared_attention(attention_id, 2); // Decode = 2

        self.prepared = Some(PreparedProviders {
            report,
            attention_id,
            kv_policy_id,
            attention_name,
            kv_policy_name,
        });
        Ok(self.prepared.as_ref().expect("just set"))
    }

    /// Lookup prepared attention plan by path kind raw value.
    #[must_use]
    pub fn attention_plan(&self, path: u8) -> Option<&PreparedAttention> {
        let prep = self.prepared.as_ref()?;
        self.host
            .capabilities()
            .prepared_attention(prep.attention_id, path)
    }
}

/// Build a device feature set for pure host/CPU tests.
#[must_use]
pub fn cpu_feature_set() -> FeatureSet {
    backend_cpu::CpuBackend::new().capabilities().feature_set()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_plugin_abi::capability::FeatureId;

    #[test]
    fn auto_select_host_attention_on_cpu() {
        let mut mgr = ProviderManager::new(ProviderSelection::new());
        let backend = backend_cpu::CpuBackend::new();
        let prep = mgr.prepare(&backend, 32, 8, 64, 16).unwrap();
        assert_eq!(prep.attention_name, "attention.host_tiled");
        assert_eq!(prep.kv_policy_name, "kv.full");
        // Hot path uses ids, not names.
        assert_ne!(prep.attention_id, PreparedProviderId::NONE);
        assert!(mgr.attention_plan(1).is_some() || mgr.attention_plan(2).is_some());
    }

    #[test]
    fn explicit_missing_provider_fails() {
        let mut sel = ProviderSelection::new();
        sel.attention = Some("attention.does_not_exist".into());
        let mut mgr = ProviderManager::new(sel);
        let backend = backend_cpu::CpuBackend::new();
        let err = mgr.prepare(&backend, 8, 8, 64, 4).unwrap_err();
        assert!(err.to_string().contains("does_not_exist"));
    }

    #[test]
    fn explicit_incompatible_kv_fails_negotiation() {
        // Register a fake attention without indirect positions, then select triattention-like.
        let mut mgr = ProviderManager::new(ProviderSelection::new());
        // Force a dense-only attention by preparing with a custom selection after
        // registering a dense-only descriptor.
        use fellm_plugin_abi::capability::{
            CapabilityKind, FeatureSet, ProviderDescriptor, ProviderVersion,
        };
        let dense = ProviderDescriptor::new(
            "attention.dense_only_test",
            CapabilityKind::Attention,
            ProviderVersion::default(),
            "dense only",
        )
        .with_provides(FeatureSet::from_ids([FeatureId::ATTN_PAGED_KV]));
        mgr.capabilities_mut()
            .register_descriptor(dense, Some("test".into()))
            .unwrap();

        // Register a compress policy that needs indirect.
        let compress = ProviderDescriptor::new(
            "kv.compress_test",
            CapabilityKind::SequenceStatePolicy,
            ProviderVersion::default(),
            "needs indirect",
        )
        .with_provides(FeatureSet::from_ids([
            FeatureId::KV_MUTABLE_REMAP,
            FeatureId::KV_LOGICAL_POSITIONS,
        ]));
        mgr.capabilities_mut()
            .register_descriptor(compress, Some("test".into()))
            .unwrap();

        let mut sel = ProviderSelection::new();
        sel.attention = Some("attention.dense_only_test".into());
        sel.kv_policy = Some("kv.compress_test".into());
        mgr.selection = sel;
        let backend = backend_cpu::CpuBackend::new();
        let err = mgr.prepare(&backend, 8, 8, 64, 2).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("indirect") || msg.contains("compose") || msg.contains("missing"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn plugin_config_validation_rejects_bad_budget() {
        let mut sel = ProviderSelection::new();
        sel.config =
            PluginConfig::from_pairs(["attention.host_tiled.q_tile=not_a_number"]).unwrap();
        let mut mgr = ProviderManager::new(sel);
        // Pin host attention so config is validated.
        mgr.selection.attention = Some("attention.host_tiled".into());
        let backend = backend_cpu::CpuBackend::new();
        let err = mgr.prepare(&backend, 8, 8, 64, 2).unwrap_err();
        assert!(err.to_string().contains("q_tile") || err.to_string().contains("config"));
    }

    #[test]
    fn triattention_policy_composes_with_host_attention() {
        let mut mgr = ProviderManager::new(ProviderSelection::new());
        mgr.load_plugins(None).unwrap();
        let mut sel = ProviderSelection::new();
        sel.attention = Some("attention.host_tiled".into());
        sel.kv_policy = Some("kv.triattention".into());
        mgr.selection = sel;
        let backend = backend_cpu::CpuBackend::new();
        let prep = mgr.prepare(&backend, 32, 8, 64, 8).unwrap();
        assert_eq!(prep.kv_policy_name, "kv.triattention");
        assert_eq!(prep.attention_name, "attention.host_tiled");
        assert!(
            mgr.capabilities()
                .sequence_policy("kv.triattention")
                .is_some()
        );
    }
}
