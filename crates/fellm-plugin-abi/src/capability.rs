//! Typed plugin capabilities and preparation-time negotiation.
//!
//! The core understands **contracts**, not product or algorithm names.
//! Providers declare what they supply and require; the host matches them
//! once during model/plan preparation into prepared handles.

use std::collections::BTreeMap;
use std::fmt;

/// High-level capability a plugin may expose.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CapabilityKind {
    /// Model family graph construction and generation driver.
    Architecture = 1,
    /// Device compute backend.
    Backend = 2,
    /// Kernel implementations for semantic ops.
    Kernel = 3,
    /// Attention implementations selected for Attention ops.
    Attention = 4,
    /// Sequence / KV retention and compression policy.
    SequenceStatePolicy = 5,
    /// Token sampling strategies.
    Sampler = 6,
    /// Graph rewrite / fusion transforms.
    GraphTransform = 7,
    /// Draft/proposal generation for runtime-owned speculative verification.
    Speculator = 8,
}

impl CapabilityKind {
    /// Stable snake_case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Architecture => "architecture",
            Self::Backend => "backend",
            Self::Kernel => "kernel",
            Self::Attention => "attention",
            Self::SequenceStatePolicy => "sequence_state_policy",
            Self::Sampler => "sampler",
            Self::GraphTransform => "graph_transform",
            Self::Speculator => "speculator",
        }
    }

    /// Parse a capability name (case-insensitive, underscores or hyphens).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "architecture" | "arch" => Some(Self::Architecture),
            "backend" | "compute" => Some(Self::Backend),
            "kernel" | "kernels" => Some(Self::Kernel),
            "attention" | "attn" => Some(Self::Attention),
            "sequence_state_policy" | "kv_policy" | "kv" | "sequence_state" => {
                Some(Self::SequenceStatePolicy)
            }
            "sampler" | "sampling" => Some(Self::Sampler),
            "graph_transform" | "graph" => Some(Self::GraphTransform),
            "speculator" | "speculative" | "draft" => Some(Self::Speculator),
            _ => None,
        }
    }
}

impl fmt::Display for CapabilityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Extensible feature flags used during capability negotiation.
///
/// Named constants cover common contracts. Plugins may also attach free-form
/// feature strings in [`ProviderDescriptor::features`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FeatureId(pub u32);

impl FeatureId {
    // ---- Attention / KV view contracts ----
    /// Causal attention mask support.
    pub const ATTN_CAUSAL: Self = Self(1);
    /// Bidirectional / non-causal attention.
    pub const ATTN_BIDIRECTIONAL: Self = Self(2);
    /// Multi-head attention (n_heads == n_kv_heads).
    pub const ATTN_MHA: Self = Self(3);
    /// Multi-query attention (n_kv_heads == 1).
    pub const ATTN_MQA: Self = Self(4);
    /// Grouped-query attention.
    pub const ATTN_GQA: Self = Self(5);
    /// Contiguous dense KV storage.
    pub const ATTN_CONTIGUOUS_KV: Self = Self(6);
    /// Paged block-table KV storage.
    pub const ATTN_PAGED_KV: Self = Self(7);
    /// Prefill (multi-token query) path.
    pub const ATTN_PREFILL: Self = Self(8);
    /// Decode (single-token query) path.
    pub const ATTN_DECODE: Self = Self(9);
    /// Short batched decode.
    pub const ATTN_BATCHED_DECODE: Self = Self(10);
    /// FP16 activations / KV.
    pub const ATTN_FP16: Self = Self(11);
    /// BF16 activations / KV.
    pub const ATTN_BF16: Self = Self(12);
    /// Sliding-window attention mask.
    pub const ATTN_SLIDING_WINDOW: Self = Self(13);
    /// Consume position-indirect / non-dense retained KV views.
    pub const ATTN_INDIRECT_POSITIONS: Self = Self(14);
    /// Per-head (or per-KV-head) retention views.
    pub const ATTN_PER_HEAD_KV_VIEWS: Self = Self(15);
    /// Hardware async pipeline / producer-consumer style execution available.
    pub const HW_ASYNC_PIPELINE: Self = Self(16);
    /// Tensor-memory / bulk async copy class features.
    pub const HW_TMA_CLASS: Self = Self(17);
    /// Warp-group matrix-multiply class features (Hopper+).
    pub const HW_WGMMA_CLASS: Self = Self(18);
    /// Ampere/Ada-class tensor cores.
    pub const HW_AMPERE_ADA: Self = Self(19);
    /// Hopper-class features.
    pub const HW_HOPPER: Self = Self(20);
    /// Blackwell-class features (reserved for future providers).
    pub const HW_BLACKWELL: Self = Self(21);

    // ---- Sequence-state / KV policy contracts ----
    /// Policy may remap physical storage of retained entries.
    pub const KV_MUTABLE_REMAP: Self = Self(40);
    /// Policy may free physical blocks after compaction.
    pub const KV_PHYSICAL_RECLAIM: Self = Self(41);
    /// Policy needs original logical positions preserved after compact.
    pub const KV_LOGICAL_POSITIONS: Self = Self(42);
    /// Policy may retain different positions per head/layer.
    pub const KV_PER_HEAD_RETENTION: Self = Self(43);
    /// Policy may run device-resident scoring/compaction.
    pub const KV_DEVICE_COMPACTION: Self = Self(44);
    /// Policy distinguishes immutable shared prefix from private suffix.
    pub const KV_PREFIX_PRIVATE_SPLIT: Self = Self(45);
    /// Policy can operate without live post-RoPE attention scores.
    pub const KV_PRE_ROPE_SCORING: Self = Self(46);

    /// Human-readable name for built-in features.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.0 {
            1 => "attn.causal",
            2 => "attn.bidirectional",
            3 => "attn.mha",
            4 => "attn.mqa",
            5 => "attn.gqa",
            6 => "attn.contiguous_kv",
            7 => "attn.paged_kv",
            8 => "attn.prefill",
            9 => "attn.decode",
            10 => "attn.batched_decode",
            11 => "attn.fp16",
            12 => "attn.bf16",
            13 => "attn.sliding_window",
            14 => "attn.indirect_positions",
            15 => "attn.per_head_kv_views",
            16 => "hw.async_pipeline",
            17 => "hw.tma_class",
            18 => "hw.wgmma_class",
            19 => "hw.ampere_ada",
            20 => "hw.hopper",
            21 => "hw.blackwell",
            40 => "kv.mutable_remap",
            41 => "kv.physical_reclaim",
            42 => "kv.logical_positions",
            43 => "kv.per_head_retention",
            44 => "kv.device_compaction",
            45 => "kv.prefix_private_split",
            46 => "kv.pre_rope_scoring",
            _ => "feature.custom",
        }
    }
}

/// Sorted, deduplicated feature set used for negotiation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeatureSet {
    ids: Vec<FeatureId>,
}

impl FeatureSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from an iterator of feature ids.
    #[must_use]
    pub fn from_ids(ids: impl IntoIterator<Item = FeatureId>) -> Self {
        let mut s = Self::new();
        for id in ids {
            s.insert(id);
        }
        s
    }

    /// Insert a feature.
    pub fn insert(&mut self, id: FeatureId) {
        if let Err(i) = self.ids.binary_search_by_key(&id.0, |f| f.0) {
            self.ids.insert(i, id);
        }
    }

    /// True if `id` is present.
    #[must_use]
    pub fn contains(&self, id: FeatureId) -> bool {
        self.ids.binary_search_by_key(&id.0, |f| f.0).is_ok()
    }

    /// True if every feature in `required` is present.
    #[must_use]
    pub fn contains_all(&self, required: &FeatureSet) -> bool {
        required.ids.iter().all(|f| self.contains(*f))
    }

    /// Features in `required` that are missing from `self`.
    #[must_use]
    pub fn missing_from(&self, required: &FeatureSet) -> FeatureSet {
        FeatureSet::from_ids(required.ids.iter().copied().filter(|f| !self.contains(*f)))
    }

    /// Iterate feature ids.
    pub fn iter(&self) -> impl Iterator<Item = FeatureId> + '_ {
        self.ids.iter().copied()
    }

    /// Number of features.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Union.
    #[must_use]
    pub fn union(&self, other: &FeatureSet) -> FeatureSet {
        let mut out = self.clone();
        for f in other.iter() {
            out.insert(f);
        }
        out
    }
}

impl fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for id in &self.ids {
            if !first {
                f.write_str(",")?;
            }
            first = false;
            f.write_str(id.name())?;
        }
        Ok(())
    }
}

/// Semantic version triple for a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderVersion {
    /// Major.
    pub major: u16,
    /// Minor.
    pub minor: u16,
    /// Patch.
    pub patch: u16,
}

impl fmt::Display for ProviderVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Static description of a registered provider/capability implementation.
#[derive(Debug, Clone)]
pub struct ProviderDescriptor {
    /// Stable dynamic identity (e.g. `attention.cuda_tiled`, `kv.full`).
    pub name: String,
    /// Capability this provider implements.
    pub capability: CapabilityKind,
    /// Provider version.
    pub version: ProviderVersion,
    /// Human-readable one-line summary.
    pub summary: String,
    /// Features this provider **supplies**.
    pub provides: FeatureSet,
    /// Features this provider **requires** from peers / device.
    pub requires: FeatureSet,
    /// Higher values win automatic selection when multiple candidates apply.
    pub priority: i32,
    /// Free-form metadata (config schema hints, author, etc.).
    pub metadata: BTreeMap<String, String>,
}

impl ProviderDescriptor {
    /// Construct a descriptor with empty metadata.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        capability: CapabilityKind,
        version: ProviderVersion,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            capability,
            version,
            summary: summary.into(),
            provides: FeatureSet::new(),
            requires: FeatureSet::new(),
            priority: 0,
            metadata: BTreeMap::new(),
        }
    }

    /// Builder: set provides.
    #[must_use]
    pub fn with_provides(mut self, provides: FeatureSet) -> Self {
        self.provides = provides;
        self
    }

    /// Builder: set requires.
    #[must_use]
    pub fn with_requires(mut self, requires: FeatureSet) -> Self {
        self.requires = requires;
        self
    }

    /// Builder: set priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: insert metadata key.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Result of preparation-time capability negotiation.
#[derive(Debug, Clone)]
pub struct NegotiationReport {
    /// Selected provider names by capability.
    pub selected: BTreeMap<CapabilityKind, String>,
    /// Diagnostic notes (auto-selection reasons, etc.).
    pub notes: Vec<String>,
}

/// Why negotiation failed.
#[derive(Debug, Clone)]
pub struct NegotiationError {
    /// Human-readable message.
    pub message: String,
    /// Capability involved, if any.
    pub capability: Option<CapabilityKind>,
    /// Provider name involved, if any.
    pub provider: Option<String>,
    /// Missing features when applicable.
    pub missing: FeatureSet,
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for NegotiationError {}

/// Negotiate that `provider` can run against `available` device/peer features.
pub fn negotiate_provider(
    provider: &ProviderDescriptor,
    available: &FeatureSet,
) -> Result<(), NegotiationError> {
    if available.contains_all(&provider.requires) {
        return Ok(());
    }
    let missing = available.missing_from(&provider.requires);
    Err(NegotiationError {
        message: format!(
            "provider '{}' ({}) requires features not available: {} (have: {})",
            provider.name, provider.capability, missing, available
        ),
        capability: Some(provider.capability),
        provider: Some(provider.name.clone()),
        missing,
    })
}

/// Verify that an attention provider can compose with a KV/sequence policy.
pub fn negotiate_attention_and_kv_policy(
    attention: &ProviderDescriptor,
    kv_policy: &ProviderDescriptor,
) -> Result<(), NegotiationError> {
    // Policy requirements that the attention provider must supply.
    let needed = FeatureSet::from_ids(
        kv_policy
            .requires
            .iter()
            .filter(|f| {
                matches!(
                    f.0,
                    14 | 15 | 40 | 41 | 42 | 43 | 44 | 45 | 46 // attn indirect/per-head + kv contracts
                )
            })
            .map(|f| {
                // Map kv requirements onto attention provides where applicable.
                match f.0 {
                    40 | 41 | 42 | 44 | 45 | 46 => {
                        // These are runtime/allocator contracts, not attention kernels.
                        // Attention only needs indirect positions / per-head views when
                        // the policy retains non-dense sets.
                        None
                    }
                    43 => Some(FeatureId::ATTN_PER_HEAD_KV_VIEWS),
                    _ => Some(f),
                }
            })
            .flatten()
            .chain(
                // If policy provides non-dense retention, attention must accept indirect positions.
                if kv_policy.provides.contains(FeatureId::KV_LOGICAL_POSITIONS)
                    || kv_policy
                        .provides
                        .contains(FeatureId::KV_PER_HEAD_RETENTION)
                    || kv_policy.provides.contains(FeatureId::KV_MUTABLE_REMAP)
                {
                    Some(FeatureId::ATTN_INDIRECT_POSITIONS)
                } else {
                    None
                },
            ),
    );

    if attention.provides.contains_all(&needed) {
        return Ok(());
    }
    let missing = attention.provides.missing_from(&needed);
    Err(NegotiationError {
        message: format!(
            "attention provider '{}' cannot compose with sequence-state policy '{}': missing {}",
            attention.name, kv_policy.name, missing
        ),
        capability: Some(CapabilityKind::Attention),
        provider: Some(attention.name.clone()),
        missing,
    })
}

/// Generic plugin configuration: `plugin_name.key` → value strings.
///
/// Validated by each provider before inference begins.
#[derive(Debug, Clone, Default)]
pub struct PluginConfig {
    /// Flat key/value map. Keys may be `provider.key` or bare `key` scoped at apply time.
    pub entries: BTreeMap<String, String>,
}

impl PluginConfig {
    /// Empty config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse CLI-style `key=value` pairs (and optional `provider.key=value`).
    pub fn from_pairs(pairs: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Self, String> {
        let mut cfg = Self::new();
        for pair in pairs {
            let s = pair.as_ref();
            let (k, v) = s
                .split_once('=')
                .ok_or_else(|| format!("plugin config must be key=value, got '{s}'"))?;
            let k = k.trim();
            if k.is_empty() {
                return Err(format!("empty config key in '{s}'"));
            }
            cfg.entries.insert(k.to_owned(), v.trim().to_owned());
        }
        Ok(cfg)
    }

    /// Entries scoped to a provider name (`provider.key` or bare keys when `include_bare`).
    #[must_use]
    pub fn for_provider(&self, provider: &str, include_bare: bool) -> BTreeMap<String, String> {
        let prefix = format!("{provider}.");
        let mut out = BTreeMap::new();
        for (k, v) in &self.entries {
            if let Some(rest) = k.strip_prefix(&prefix) {
                out.insert(rest.to_owned(), v.clone());
            } else if include_bare && !k.contains('.') {
                out.insert(k.clone(), v.clone());
            }
        }
        out
    }

    /// Get a raw entry.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }
}

/// User selection of providers by capability name.
#[derive(Debug, Clone, Default)]
pub struct ProviderSelection {
    /// Explicit backend provider name, if any.
    pub backend: Option<String>,
    /// Explicit attention provider name, if any.
    pub attention: Option<String>,
    /// Explicit sequence-state / KV policy provider name, if any.
    pub kv_policy: Option<String>,
    /// Explicit sampler provider name, if any.
    pub sampler: Option<String>,
    /// Explicit speculative proposal provider, if any.
    pub speculator: Option<String>,
    /// Plugin-specific configuration.
    pub config: PluginConfig,
}

impl ProviderSelection {
    /// Empty (all automatic).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin a provider for a capability; fails if the capability is unknown.
    pub fn pin(&mut self, capability: CapabilityKind, name: impl Into<String>) {
        let name = name.into();
        match capability {
            CapabilityKind::Backend => self.backend = Some(name),
            CapabilityKind::Attention => self.attention = Some(name),
            CapabilityKind::SequenceStatePolicy => self.kv_policy = Some(name),
            CapabilityKind::Sampler => self.sampler = Some(name),
            CapabilityKind::Speculator => self.speculator = Some(name),
            _ => {
                // Architecture / kernel / graph are selected by other paths today.
            }
        }
    }

    /// Name pinned for a capability, if any.
    #[must_use]
    pub fn get(&self, capability: CapabilityKind) -> Option<&str> {
        match capability {
            CapabilityKind::Backend => self.backend.as_deref(),
            CapabilityKind::Attention => self.attention.as_deref(),
            CapabilityKind::SequenceStatePolicy => self.kv_policy.as_deref(),
            CapabilityKind::Sampler => self.sampler.as_deref(),
            CapabilityKind::Speculator => self.speculator.as_deref(),
            _ => None,
        }
    }
}

/// Opaque prepared provider handle used on the hot path.
///
/// Assigned once at preparation; steady-state code uses this integer (or a
/// function pointer stored beside it), never the provider name string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PreparedProviderId(pub u64);

impl PreparedProviderId {
    /// Invalid / unset id.
    pub const NONE: Self = Self(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_set_contains_all() {
        let have = FeatureSet::from_ids([
            FeatureId::ATTN_PAGED_KV,
            FeatureId::ATTN_DECODE,
            FeatureId::ATTN_INDIRECT_POSITIONS,
        ]);
        let need = FeatureSet::from_ids([FeatureId::ATTN_PAGED_KV, FeatureId::ATTN_DECODE]);
        assert!(have.contains_all(&need));
        assert!(!have.contains_all(&FeatureSet::from_ids([FeatureId::ATTN_PREFILL])));
    }

    #[test]
    fn parses_speculator_capability() {
        assert_eq!(
            CapabilityKind::parse("speculative"),
            Some(CapabilityKind::Speculator)
        );
        assert_eq!(CapabilityKind::Speculator.name(), "speculator");
    }

    #[test]
    fn negotiate_missing_features() {
        let provider = ProviderDescriptor::new(
            "attention.test",
            CapabilityKind::Attention,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "test",
        )
        .with_requires(FeatureSet::from_ids([FeatureId::HW_HOPPER]));
        let available = FeatureSet::from_ids([FeatureId::HW_AMPERE_ADA]);
        let err = negotiate_provider(&provider, &available).unwrap_err();
        assert!(err.message.contains("attention.test"));
        assert!(err.missing.contains(FeatureId::HW_HOPPER));
    }

    #[test]
    fn attention_kv_composition_requires_indirect() {
        let attn = ProviderDescriptor::new(
            "attention.dense_only",
            CapabilityKind::Attention,
            ProviderVersion::default(),
            "dense",
        )
        .with_provides(FeatureSet::from_ids([FeatureId::ATTN_PAGED_KV]));
        let kv = ProviderDescriptor::new(
            "kv.compress",
            CapabilityKind::SequenceStatePolicy,
            ProviderVersion::default(),
            "compress",
        )
        .with_provides(FeatureSet::from_ids([
            FeatureId::KV_MUTABLE_REMAP,
            FeatureId::KV_LOGICAL_POSITIONS,
        ]));
        let err = negotiate_attention_and_kv_policy(&attn, &kv).unwrap_err();
        assert!(err.missing.contains(FeatureId::ATTN_INDIRECT_POSITIONS));
    }

    #[test]
    fn plugin_config_parse_and_scope() {
        let cfg = PluginConfig::from_pairs([
            "budget=2048",
            "kv.triattention.warmup=128",
            "kv.triattention.budget=1024",
        ])
        .unwrap();
        let scoped = cfg.for_provider("kv.triattention", false);
        assert_eq!(scoped.get("budget").map(String::as_str), Some("1024"));
        assert_eq!(scoped.get("warmup").map(String::as_str), Some("128"));
        // Provider-scoped keys win over bare keys when both are present.
        let bare = cfg.for_provider("kv.triattention", true);
        assert_eq!(bare.get("budget").map(String::as_str), Some("1024"));
        assert_eq!(bare.get("warmup").map(String::as_str), Some("128"));
    }
}
