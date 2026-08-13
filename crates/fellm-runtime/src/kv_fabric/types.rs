//! Logical KV identity, encoding, groups, tiers, and configuration.

use std::fmt;

/// Logical page identity — sequences own these, never raw physical slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvPageId(pub u64);

impl fmt::Display for KvPageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "page:{}", self.0)
    }
}

/// Content-addressed shared object identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedKvId(pub u64);

/// Internal physical storage slot (fabric-private; not sequence-facing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalSlot(pub u32);

/// Element encoding for a KV page/segment. Architecture must not assume FP16 only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvEncoding {
    #[default]
    Fp16,
    Bf16,
    Fp8,
    Int8,
    Int4,
    Custom(u32),
}

impl KvEncoding {
    #[must_use]
    pub fn elem_bytes(self) -> usize {
        match self {
            Self::Fp16 | Self::Bf16 => 2,
            Self::Fp8 | Self::Int8 => 1,
            Self::Int4 => 1, // packed; two values per byte at higher levels
            Self::Custom(_) => 2,
        }
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fp16" | "f16" => Some(Self::Fp16),
            "bf16" => Some(Self::Bf16),
            "fp8" | "f8" => Some(Self::Fp8),
            "int8" | "i8" => Some(Self::Int8),
            "int4" | "i4" => Some(Self::Int4),
            _ => None,
        }
    }
}

/// Memory residency tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvTier {
    #[default]
    Device,
    HostPinned,
    Host,
    /// Modeled for future NVMe/disk; not fully implemented.
    Disk,
    NotResident,
}

impl KvTier {
    #[must_use]
    pub fn is_compute_ready(self) -> bool {
        matches!(self, Self::Device | Self::HostPinned | Self::Host)
    }
}

/// Where a page currently lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvLocation {
    Resident(KvTier),
    Migrating {
        from: KvTier,
        to: KvTier,
    },
    #[default]
    NotResident,
}

impl KvLocation {
    #[must_use]
    pub fn tier(self) -> KvTier {
        match self {
            Self::Resident(t) => t,
            Self::Migrating { to, .. } => to,
            Self::NotResident => KvTier::NotResident,
        }
    }
}

/// Attention / state group kind — heterogeneous model state support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvGroupKind {
    #[default]
    FullAttention,
    SlidingWindow,
    Local,
    CrossAttention,
    Recurrent,
    Custom(u32),
}

/// Hierarchical page size class. Token granularity need not be fixed for a
/// sequence's entire lifetime (prefix superpages, standard pages, active micro).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvPageClass {
    /// Active tail — smaller pages for fine-grained growth.
    Micro,
    /// Normal regions.
    #[default]
    Standard,
    /// Large immutable prefix regions (coalesced).
    Super,
}

impl KvPageClass {
    /// Token capacity for this class (default geometry; groups may override).
    #[must_use]
    pub fn tokens(self) -> usize {
        match self {
            Self::Micro => 4,
            Self::Standard => 16,
            Self::Super => 64,
        }
    }
}

/// Default standard page size used by the block-table addressing path and
/// attention kernels (must match kernel expectations).
pub const STANDARD_PAGE_TOKENS: usize = 16;

/// Exact (lossless fabric) vs Elastic (lossy policies allowed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvMode {
    /// Backend/device/architecture-driven defaults within the fabric only.
    #[default]
    Auto,
    /// Lossless: paging, VMM, sharing, CoW, tiering, prefetch, eviction, recompute.
    Exact,
    /// Exact plus quantization / sparse retention / compression policies.
    Elastic,
}

impl KvMode {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "exact" => Some(Self::Exact),
            "elastic" => Some(Self::Elastic),
            _ => None,
        }
    }

    /// Resolved concrete mode when Auto is selected.
    #[must_use]
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => Self::Exact,
            other => other,
        }
    }
}

/// Backend physical addressing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvAddressing {
    /// Explicit logical→physical block table (CPU and universal fallback).
    #[default]
    BlockTable,
    /// CUDA virtual memory management — contiguous virtual KV space.
    VirtualMemory,
}

impl KvAddressing {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "block" | "blocktable" | "block_table" => Some(Self::BlockTable),
            "vmm" | "virtual" | "virtualmemory" | "virtual_memory" => Some(Self::VirtualMemory),
            _ => None,
        }
    }
}

/// Per-segment encoding selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvEncodingPolicy {
    /// Always use the configured default encoding.
    #[default]
    Uniform,
    /// Elastic: hotter segments higher precision, colder lower.
    TemperatureTiered,
}

/// Named residency scoring policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResidencyPolicyKind {
    /// Value/cost scoring (primary).
    #[default]
    ValueCost,
    /// Fallback for tests only — not primary production policy.
    Lru,
}

/// How a page is owned for mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PageOwnership {
    /// Immutable shared (prefix / content-addressed store).
    SharedImmutable,
    /// Request-private mutable.
    #[default]
    Private,
    /// CoW private view of previously shared data.
    ForkedPrivate,
}

/// Descriptor for one KV/state group within a model.
#[derive(Debug, Clone)]
pub struct KvGroupDesc {
    pub kind: KvGroupKind,
    pub layer_start: u32,
    pub layer_count: u32,
    pub page_class: KvPageClass,
    pub encoding: KvEncoding,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Optional sliding window size.
    pub window: Option<u32>,
}

impl KvGroupDesc {
    #[must_use]
    pub fn full_attention(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        encoding: KvEncoding,
    ) -> Self {
        Self {
            kind: KvGroupKind::FullAttention,
            layer_start: 0,
            layer_count: n_layers.max(1) as u32,
            page_class: KvPageClass::Standard,
            encoding,
            n_kv_heads: n_kv_heads.max(1),
            head_dim: head_dim.max(1),
            window: None,
        }
    }

    #[must_use]
    pub fn tokens_stride(&self) -> usize {
        self.n_kv_heads.saturating_mul(self.head_dim)
    }

    #[must_use]
    pub fn page_tokens(&self) -> usize {
        self.page_class.tokens()
    }

    /// Bytes for one physical page (K|V, all tokens in page, one layer).
    #[must_use]
    pub fn page_bytes(&self) -> usize {
        let tokens = self.page_tokens();
        let stride = self.tokens_stride();
        let raw = 2usize
            .saturating_mul(tokens)
            .saturating_mul(stride)
            .saturating_mul(self.encoding.elem_bytes());
        (raw + 63) & !63
    }
}

/// Contiguous logical region of KV (prefix, body, active tail, …).
#[derive(Debug, Clone)]
pub struct KvSegment {
    pub group: u32,
    pub encoding: KvEncoding,
    pub page_class: KvPageClass,
    pub token_start: u32,
    pub token_len: u32,
    pub pages: Vec<KvPageId>,
    pub ownership: PageOwnership,
}

/// Per-layer logical→page map used only by the addressing backend.
/// Sequence-facing code holds [`KvSequence`]; physical slots stay inside fabric.
#[derive(Debug, Clone, Default)]
pub struct LayerPageMap {
    /// Fabric page ids for successive logical page slots of one layer.
    pages: Vec<KvPageId>,
}

impl LayerPageMap {
    #[must_use]
    pub fn new() -> Self {
        Self { pages: Vec::new() }
    }

    #[must_use]
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    #[must_use]
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }

    #[must_use]
    pub fn page(&self, logical: usize) -> KvPageId {
        self.pages[logical]
    }

    pub fn page_mut(&mut self, logical: usize) -> &mut KvPageId {
        &mut self.pages[logical]
    }

    pub fn push_page(&mut self, id: KvPageId) {
        self.pages.push(id);
    }

    pub fn set_pages(&mut self, pages: Vec<KvPageId>) {
        self.pages = pages;
    }

    pub fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Sequence-owned logical KV state. Does **not** own physical memory.
#[derive(Debug, Clone)]
pub struct KvSequence {
    layers: Vec<LayerPageMap>,
    /// Dense storage length (attention-visible token count).
    pub len_tokens: usize,
    /// Absolute generation cursor (RoPE / sampling).
    pub absolute_pos: usize,
    /// Original absolute positions when compressed (empty = identity).
    pub original_positions: Vec<u32>,
    /// Immutable shared prefix length in tokens.
    pub shared_prefix_len: usize,
    pub max_seq: usize,
    /// True when pages are not compute-resident (migrated out).
    pub non_resident: bool,
    /// Segments describing hierarchical regions (optional metadata).
    pub segments: Vec<KvSegment>,
    /// Group index this sequence primarily uses (usually 0 = full attention).
    pub primary_group: u32,
}

impl KvSequence {
    #[must_use]
    pub fn new(n_layers: usize, max_seq: usize) -> Self {
        Self {
            layers: (0..n_layers.max(1)).map(|_| LayerPageMap::new()).collect(),
            len_tokens: 0,
            absolute_pos: 0,
            original_positions: Vec::new(),
            shared_prefix_len: 0,
            max_seq,
            non_resident: false,
            segments: Vec::new(),
            primary_group: 0,
        }
    }

    #[must_use]
    pub fn kv_write_index(&self) -> usize {
        self.len_tokens
    }

    #[must_use]
    pub fn is_compressed(&self) -> bool {
        !self.original_positions.is_empty()
    }

    #[must_use]
    pub fn original_pos(&self, dense_i: usize) -> u32 {
        self.original_positions
            .get(dense_i)
            .copied()
            .unwrap_or(dense_i as u32)
    }

    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn layer_map(&self, layer: usize) -> &LayerPageMap {
        &self.layers[layer]
    }

    pub fn layer_map_mut(&mut self, layer: usize) -> &mut LayerPageMap {
        &mut self.layers[layer]
    }

    pub fn clear_maps(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.len_tokens = 0;
        self.absolute_pos = 0;
        self.original_positions.clear();
        self.shared_prefix_len = 0;
        self.non_resident = false;
        self.segments.clear();
    }

    /// Locate dense token → (logical page index, slot). Page size is group geometry.
    #[must_use]
    pub fn locate_token(&self, token_pos: usize, page_tokens: usize) -> (usize, usize) {
        let pt = page_tokens.max(1);
        (token_pos / pt, token_pos % pt)
    }

    /// Flatten logical page ids for addressing: `[layer0_p0, layer0_p1, ..., layer1_p0, ...]`.
    #[must_use]
    pub fn flatten_page_ids(&self) -> Vec<KvPageId> {
        let n = self.layers.first().map_or(0, LayerPageMap::num_pages);
        let mut out = Vec::with_capacity(n * self.layers.len());
        for layer in &self.layers {
            debug_assert_eq!(layer.num_pages(), n);
            out.extend_from_slice(layer.pages());
        }
        out
    }
}

/// Semantic identity for content-addressed shared KV.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedKvKey {
    pub model_fingerprint: u64,
    pub adapter_id: u64,
    pub arch_config: u64,
    pub group: u32,
    pub encoding: KvEncoding,
    pub rope_config: u64,
    /// Exact token chunk (one page of tokens).
    pub tokens: Vec<u32>,
}

impl SharedKvKey {
    #[must_use]
    pub fn hash64(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let enc = match self.encoding {
            KvEncoding::Fp16 => 1u64,
            KvEncoding::Bf16 => 2,
            KvEncoding::Fp8 => 3,
            KvEncoding::Int8 => 4,
            KvEncoding::Int4 => 5,
            KvEncoding::Custom(c) => 100 + u64::from(c),
        };
        for x in [
            self.model_fingerprint,
            self.adapter_id,
            self.arch_config,
            u64::from(self.group),
            enc,
            self.rope_config,
        ] {
            h ^= x;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        for &t in &self.tokens {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
}

/// Fully resolved memory plan for fabric startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvExecutionMemory {
    Host,
    Accelerator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMemoryPlan {
    pub weights_bytes: u64,
    pub activation_bytes: u64,
    pub page_bytes: usize,
    pub execution_memory: KvExecutionMemory,
    pub execution_pages: usize,
    pub execution_bytes: u64,
    pub overflow_host_pages: usize,
    pub overflow_host_bytes: u64,
    pub remaining_reserve_bytes: Option<u64>,
}

/// Runtime fabric configuration (structured; not a legacy dual-stack toggle).
#[derive(Debug, Clone)]
pub struct KvFabricConfig {
    pub mode: KvMode,
    /// Explicit device-side KV budget. `None` → automatic from device memory.
    pub device_budget: Option<u64>,
    /// Host / host-pinned secondary tier budget.
    pub host_budget: Option<u64>,
    pub addressing: KvAddressing,
    pub prefix_sharing: bool,
    pub prefetch: bool,
    pub encoding_policy: KvEncodingPolicy,
    pub residency_policy: ResidencyPolicyKind,
    /// Default storage encoding for Exact path.
    pub default_encoding: KvEncoding,
    /// Fraction of available memory usable in auto device budget mode.
    pub memory_fraction: f64,
    pub safety_reserve_bytes: u64,
    pub runtime_reserve_bytes: u64,
}

impl Default for KvFabricConfig {
    fn default() -> Self {
        Self {
            mode: KvMode::Auto,
            device_budget: None,
            host_budget: Some(0),
            addressing: KvAddressing::BlockTable,
            prefix_sharing: true,
            prefetch: true,
            encoding_policy: KvEncodingPolicy::Uniform,
            residency_policy: ResidencyPolicyKind::ValueCost,
            default_encoding: KvEncoding::Fp16,
            memory_fraction: 0.25,
            safety_reserve_bytes: 2 * 1024 * 1024 * 1024,
            runtime_reserve_bytes: 512 * 1024 * 1024,
        }
    }
}

impl KvFabricConfig {
    /// Legacy-compatible field mapping used while CLI migrates names.
    #[must_use]
    pub fn from_legacy_bytes(
        budget_bytes: Option<u64>,
        memory_fraction: f64,
        safety_reserve_bytes: u64,
        host_bytes: u64,
    ) -> Self {
        Self {
            device_budget: budget_bytes,
            host_budget: Some(host_bytes),
            memory_fraction,
            safety_reserve_bytes,
            ..Self::default()
        }
    }
}

/// Snapshot of fabric metrics for diagnostics / scheduler.
#[derive(Debug, Clone, Default)]
pub struct FabricMetrics {
    pub device_resident_pages: usize,
    pub host_resident_pages: usize,
    pub device_resident_bytes: u64,
    pub host_resident_bytes: u64,
    pub migrations: u64,
    pub migration_bytes: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub prefix_hit_tokens: u64,
    pub prefix_miss_tokens: u64,
    pub shared_kv_bytes: u64,
    pub shared_objects: usize,
    pub evictions: u64,
    pub recompute_decisions: u64,
    pub allocation_failures: u64,
    pub cow_forks: u64,
    pub free_device_pages: usize,
    pub total_device_pages: usize,
}

/// Signals exposed to the scheduling policy for memory-aware decisions.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidencySignals {
    pub resident_pages: usize,
    pub non_resident_pages: usize,
    pub resident_bytes: u64,
    pub non_resident_bytes: u64,
    pub pages_needed_next: usize,
    pub prefix_hit_tokens: usize,
    pub estimated_transfer_bytes: u64,
    pub estimated_compute_cost: f64,
    pub memory_pressure: f64,
    pub expected_output_growth: usize,
    pub priority: i32,
    pub latency_target_ms: Option<u32>,
}
