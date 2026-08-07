//! Multi-capability provider registry: discovery, selection, negotiation.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::attention_provider::{
    AttentionPrepareContext, AttentionProvider, AttentionWorkload, DeviceCapabilityView,
    PreparedAttention,
};
use fellm_plugin_abi::c_abi::{
    CapabilityRegistryVtable, PLUGIN_MAX_FEATURES, PLUGIN_MAX_META, PluginCapabilityRegistration,
};
use fellm_plugin_abi::capability::{
    CapabilityKind, FeatureId, FeatureSet, NegotiationError, NegotiationReport, PreparedProviderId,
    ProviderDescriptor, ProviderSelection, ProviderVersion, negotiate_attention_and_kv_policy,
    negotiate_provider,
};
use fellm_plugin_abi::sequence_state::{FullRetentionPolicy, SequenceStatePolicy};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;

/// One registered capability provider (descriptor always present).
#[derive(Clone)]
pub struct RegisteredProvider {
    /// Descriptor for CLI / negotiation.
    pub descriptor: ProviderDescriptor,
    /// Stable prepared id assigned at registration (hot-path key).
    pub id: PreparedProviderId,
    /// Origin plugin path if dynamically loaded.
    pub source: Option<String>,
}

/// Host-side multi-capability registry.
pub struct CapabilityRegistry {
    by_name: BTreeMap<String, RegisteredProvider>,
    by_capability: BTreeMap<CapabilityKind, Vec<String>>,
    next_id: u64,
    /// Rust-side attention providers (static + first-party).
    attention: BTreeMap<String, Arc<dyn AttentionProvider>>,
    /// Rust-side sequence-state policies.
    kv_policies: BTreeMap<String, Arc<dyn SequenceStatePolicy>>,
    /// Prepared attention plans keyed by provider id + path kind raw.
    prepared_attention: BTreeMap<(u64, u8), PreparedAttention>,
    /// Selected prepared ids after negotiation.
    selected: BTreeMap<CapabilityKind, PreparedProviderId>,
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityRegistry {
    /// Empty registry with built-in first-party providers installed.
    #[must_use]
    pub fn new() -> Self {
        let mut reg = Self {
            by_name: BTreeMap::new(),
            by_capability: BTreeMap::new(),
            next_id: 1,
            attention: BTreeMap::new(),
            kv_policies: BTreeMap::new(),
            prepared_attention: BTreeMap::new(),
            selected: BTreeMap::new(),
        };
        reg.install_builtins();
        reg
    }

    fn install_builtins(&mut self) {
        let full = Arc::new(FullRetentionPolicy::new());
        let _ = self.register_sequence_policy(full);

        let host_attn = Arc::new(crate::builtin_attention::HostTiledAttentionProvider::new());
        let _ = self.register_attention_provider(host_attn);

        let cuda_attn = Arc::new(crate::builtin_attention::CudaAttentionProvider::new());
        let _ = self.register_attention_provider(cuda_attn);
    }

    fn alloc_id(&mut self) -> PreparedProviderId {
        let id = PreparedProviderId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }

    /// Register a descriptor-only provider (dynamic plugins without Rust trait object).
    pub fn register_descriptor(
        &mut self,
        descriptor: ProviderDescriptor,
        source: Option<String>,
    ) -> Result<PreparedProviderId> {
        let name = descriptor.name.clone();
        if name.is_empty() {
            return Err(FellmError::other("provider name is empty"));
        }
        if self.by_name.contains_key(&name) {
            return Err(FellmError::other(format!("duplicate provider '{name}'")));
        }
        let id = self.alloc_id();
        let cap = descriptor.capability;
        self.by_capability
            .entry(cap)
            .or_default()
            .push(name.clone());
        self.by_name.insert(
            name,
            RegisteredProvider {
                descriptor,
                id,
                source,
            },
        );
        Ok(id)
    }

    /// Register a Rust attention provider.
    pub fn register_attention_provider(
        &mut self,
        provider: Arc<dyn AttentionProvider>,
    ) -> Result<PreparedProviderId> {
        let desc = provider.descriptor().clone();
        let name = desc.name.clone();
        let id = self.register_descriptor(desc, Some("builtin".into()))?;
        self.attention.insert(name, provider);
        Ok(id)
    }

    /// Register a Rust sequence-state policy.
    pub fn register_sequence_policy(
        &mut self,
        provider: Arc<dyn SequenceStatePolicy>,
    ) -> Result<PreparedProviderId> {
        let desc = provider.descriptor().clone();
        let name = desc.name.clone();
        let id = self.register_descriptor(desc, Some("builtin".into()))?;
        self.kv_policies.insert(name, provider);
        Ok(id)
    }

    /// Attach a Rust trait object for a name that already has a descriptor
    /// (e.g. after dynamic C registration of a first-party plugin).
    pub fn inject_sequence_policy(
        &mut self,
        name: &str,
        provider: Arc<dyn SequenceStatePolicy>,
    ) -> Result<()> {
        if !self.by_name.contains_key(name) {
            return Err(FellmError::other(format!(
                "cannot inject policy for unknown provider '{name}'"
            )));
        }
        self.kv_policies.insert(name.to_owned(), provider);
        Ok(())
    }

    /// Register from C ABI record.
    pub fn register_c(&mut self, reg: &PluginCapabilityRegistration) -> c_int {
        let name = c_name_to_string(&reg.name);
        if name.is_empty() {
            return -1;
        }
        // Idempotent: first-party plugins may also be linked as rlib.
        if self.by_name.contains_key(&name) {
            return 0;
        }
        let capability = match reg.capability {
            1 => CapabilityKind::Architecture,
            2 => CapabilityKind::Backend,
            3 => CapabilityKind::Kernel,
            4 => CapabilityKind::Attention,
            5 => CapabilityKind::SequenceStatePolicy,
            6 => CapabilityKind::Sampler,
            7 => CapabilityKind::GraphTransform,
            _ => return -2,
        };
        let mut provides = FeatureSet::new();
        for i in 0..reg.n_provides as usize {
            if i >= PLUGIN_MAX_FEATURES {
                break;
            }
            provides.insert(FeatureId(reg.provides[i]));
        }
        let mut requires = FeatureSet::new();
        for i in 0..reg.n_requires as usize {
            if i >= PLUGIN_MAX_FEATURES {
                break;
            }
            requires.insert(FeatureId(reg.requires[i]));
        }
        let mut meta = BTreeMap::new();
        for i in 0..reg.n_meta as usize {
            if i >= PLUGIN_MAX_META {
                break;
            }
            let k = c_buf_to_string(&reg.meta_keys[i]);
            let v = c_buf_to_string(&reg.meta_values[i]);
            if !k.is_empty() {
                meta.insert(k, v);
            }
        }
        let summary = c_buf_to_string(&reg.summary);
        let mut descriptor = ProviderDescriptor::new(
            name,
            capability,
            ProviderVersion {
                major: reg.version_major,
                minor: reg.version_minor,
                patch: reg.version_patch,
            },
            summary,
        )
        .with_provides(provides)
        .with_requires(requires)
        .with_priority(reg.priority);
        descriptor.metadata = meta;
        match self.register_descriptor(descriptor, None) {
            Ok(_) => 0,
            Err(_) => -3,
        }
    }

    /// List all registered providers.
    #[must_use]
    pub fn list(&self) -> Vec<&RegisteredProvider> {
        self.by_name.values().collect()
    }

    /// List providers for one capability.
    #[must_use]
    pub fn list_capability(&self, cap: CapabilityKind) -> Vec<&RegisteredProvider> {
        self.by_capability
            .get(&cap)
            .into_iter()
            .flatten()
            .filter_map(|n| self.by_name.get(n))
            .collect()
    }

    /// Lookup by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RegisteredProvider> {
        self.by_name.get(name)
    }

    /// Prepared id for a provider name.
    #[must_use]
    pub fn id_of(&self, name: &str) -> Option<PreparedProviderId> {
        self.by_name.get(name).map(|p| p.id)
    }

    /// Name for a prepared id.
    #[must_use]
    pub fn name_of(&self, id: PreparedProviderId) -> Option<&str> {
        self.by_name
            .values()
            .find(|p| p.id == id)
            .map(|p| p.descriptor.name.as_str())
    }

    /// Sequence policy trait object by name.
    #[must_use]
    pub fn sequence_policy(&self, name: &str) -> Option<Arc<dyn SequenceStatePolicy>> {
        self.kv_policies.get(name).cloned()
    }

    /// Attention provider by name.
    #[must_use]
    pub fn attention_provider(&self, name: &str) -> Option<Arc<dyn AttentionProvider>> {
        self.attention.get(name).cloned()
    }

    /// Selected prepared id for a capability after [`Self::prepare`].
    #[must_use]
    pub fn selected(&self, cap: CapabilityKind) -> Option<PreparedProviderId> {
        self.selected.get(&cap).copied()
    }

    /// Prepared attention plan for (provider_id, path_kind).
    #[must_use]
    pub fn prepared_attention(
        &self,
        provider: PreparedProviderId,
        path: u8,
    ) -> Option<&PreparedAttention> {
        self.prepared_attention.get(&(provider.0, path))
    }

    /// C registration vtable.
    #[must_use]
    pub fn vtable(&mut self) -> CapabilityRegistryVtable {
        CapabilityRegistryVtable {
            registry: std::ptr::from_mut::<CapabilityRegistry>(self).cast::<c_void>(),
            register_capability: capability_register,
        }
    }

    /// Resolve providers for the selection, negotiate, and prepare attention plans.
    pub fn prepare(
        &mut self,
        selection: &ProviderSelection,
        device_features: &FeatureSet,
        device: &DeviceCapabilityView,
        workloads: &[AttentionWorkload],
        n_layers: u32,
    ) -> Result<NegotiationReport> {
        let mut notes = Vec::new();
        let mut selected_names: BTreeMap<CapabilityKind, String> = BTreeMap::new();

        // --- Attention ---
        let attn_name = self.resolve_name(
            CapabilityKind::Attention,
            selection.attention.as_deref(),
            device_features,
            |name| {
                let Some(p) = self.attention.get(name) else {
                    // Descriptor-only: accept if features negotiate.
                    return self
                        .by_name
                        .get(name)
                        .map(|r| r.descriptor.priority)
                        .unwrap_or(i32::MIN);
                };
                if workloads.is_empty() {
                    return p.descriptor().priority;
                }
                workloads
                    .iter()
                    .map(|w| p.applicability(w, device))
                    .max()
                    .unwrap_or(i32::MIN)
            },
            &mut notes,
        )?;
        let attn_reg = self
            .by_name
            .get(&attn_name)
            .ok_or_else(|| FellmError::other(format!("attention provider '{attn_name}' missing")))?
            .clone();
        negotiate_provider(&attn_reg.descriptor, device_features)
            .map_err(|e| FellmError::other(e.to_string()))?;
        if let Some(p) = self.attention.get(&attn_name) {
            p.validate_config(&selection.config)
                .map_err(|e| FellmError::other(format!("attention config: {e}")))?;
        }
        selected_names.insert(CapabilityKind::Attention, attn_name.clone());
        self.selected.insert(CapabilityKind::Attention, attn_reg.id);

        // --- KV / sequence policy ---
        let kv_name = self.resolve_name(
            CapabilityKind::SequenceStatePolicy,
            selection.kv_policy.as_deref(),
            device_features,
            |name| {
                self.by_name
                    .get(name)
                    .map(|r| r.descriptor.priority)
                    .unwrap_or(i32::MIN)
            },
            &mut notes,
        )?;
        let kv_reg = self
            .by_name
            .get(&kv_name)
            .ok_or_else(|| FellmError::other(format!("kv policy '{kv_name}' missing")))?
            .clone();
        // Policy requires are against runtime+attention composition, not only device.
        if let Some(p) = self.kv_policies.get(&kv_name) {
            p.validate_config(&selection.config)
                .map_err(|e| FellmError::other(format!("kv-policy config: {e}")))?;
        }
        negotiate_attention_and_kv_policy(&attn_reg.descriptor, &kv_reg.descriptor)
            .map_err(|e: NegotiationError| FellmError::other(e.to_string()))?;
        selected_names.insert(CapabilityKind::SequenceStatePolicy, kv_name.clone());
        self.selected
            .insert(CapabilityKind::SequenceStatePolicy, kv_reg.id);

        // Prepare attention plans once.
        if let Some(provider) = self.attention.get(&attn_name).cloned() {
            let ctx = AttentionPrepareContext {
                workloads,
                device,
                config: &selection.config,
                n_layers,
            };
            let plans = provider
                .prepare(&ctx)
                .map_err(|e| FellmError::other(format!("attention prepare: {e}")))?;
            for mut plan in plans {
                plan.provider = attn_reg.id;
                self.prepared_attention
                    .insert((attn_reg.id.0, plan.path as u8), plan);
            }
            notes.push(format!(
                "prepared attention provider '{}' id={}",
                attn_name, attn_reg.id.0
            ));
        } else {
            notes.push(format!(
                "attention provider '{attn_name}' is descriptor-only; kernel path uses plugin ops"
            ));
        }

        notes.push(format!("selected kv policy '{kv_name}' id={}", kv_reg.id.0));

        Ok(NegotiationReport {
            selected: selected_names,
            notes,
        })
    }

    fn resolve_name(
        &self,
        cap: CapabilityKind,
        explicit: Option<&str>,
        device_features: &FeatureSet,
        score: impl Fn(&str) -> i32,
        notes: &mut Vec<String>,
    ) -> Result<String> {
        if let Some(name) = explicit {
            let reg = self.by_name.get(name).ok_or_else(|| {
                FellmError::other(format!(
                    "explicit {cap} provider '{name}' not found; use `fellm plugins list`"
                ))
            })?;
            if reg.descriptor.capability != cap {
                return Err(FellmError::other(format!(
                    "provider '{name}' is {} , not {cap}",
                    reg.descriptor.capability
                )));
            }
            // Explicit: fail if requires not met (no silent substitute).
            negotiate_provider(&reg.descriptor, device_features)
                .map_err(|e| FellmError::other(e.to_string()))?;
            if score(name) == i32::MIN && self.attention.contains_key(name) {
                return Err(FellmError::other(format!(
                    "explicit attention provider '{name}' does not support the active model/device"
                )));
            }
            notes.push(format!("explicit {cap} provider '{name}'"));
            return Ok(name.to_owned());
        }

        let candidates = self.list_capability(cap);
        let mut best: Option<(i32, String)> = None;
        for c in candidates {
            if negotiate_provider(&c.descriptor, device_features).is_err() {
                continue;
            }
            let s = score(&c.descriptor.name);
            if s == i32::MIN {
                continue;
            }
            match &best {
                None => best = Some((s, c.descriptor.name.clone())),
                Some((bs, _)) if s > *bs => best = Some((s, c.descriptor.name.clone())),
                _ => {}
            }
        }
        let (s, name) = best.ok_or_else(|| {
            FellmError::other(format!(
                "no applicable {cap} provider for the active model/device"
            ))
        })?;
        notes.push(format!("auto-selected {cap} provider '{name}' (score={s})"));
        Ok(name)
    }
}

unsafe extern "C" fn capability_register(
    registry: *mut c_void,
    registration: *const PluginCapabilityRegistration,
) -> c_int {
    if registry.is_null() || registration.is_null() {
        return -1;
    }
    let reg = unsafe { &mut *registry.cast::<CapabilityRegistry>() };
    let rec = unsafe { &*registration };
    reg.register_c(rec)
}

fn c_name_to_string(buf: &[std::ffi::c_char; fellm_plugin_abi::PLUGIN_NAME_MAX]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn c_buf_to_string(buf: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
