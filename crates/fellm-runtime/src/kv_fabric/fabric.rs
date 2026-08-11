//! KV Fabric: residency, allocation, migration, sharing, and sequence lifecycle.

use super::addressing::{AddressTranslator, AddressingView, translator_for};
use super::mapping::PageMap;
use super::policy::{PageScoreInput, ResidencyPolicy, policy_from_kind};
use super::share::{PrefixCacheStats, SharedKvStore, ensure_cow};
use super::storage::PageArena;
use super::types::{
    FabricMetrics, KvAddressing, KvEncoding, KvFabricConfig, KvGroupDesc, KvLocation, KvMemoryPlan,
    KvMode, KvPageId, KvSequence, KvTier, PhysicalSlot, ResidencySignals, STANDARD_PAGE_TOKENS,
};
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::DeviceMemoryInfo;
use half::f16;

// Re-export stats type used by callers.
pub use super::share::PrefixCacheStats as SharedPrefixStats;

impl KvMemoryPlan {
    pub fn resolve(
        config: &KvFabricConfig,
        memory: Option<DeviceMemoryInfo>,
        weights_bytes: u64,
        activation_bytes: u64,
        page_bytes: usize,
        minimum_pages: usize,
    ) -> Result<Self> {
        if page_bytes == 0 {
            return Err(FellmError::other("KV page size is zero"));
        }
        if !config.memory_fraction.is_finite()
            || config.memory_fraction <= 0.0
            || config.memory_fraction > 1.0
        {
            return Err(FellmError::other("KV memory fraction must be in (0, 1]"));
        }
        let fraction = config.memory_fraction;
        let automatic = memory.map(|info| {
            let usable = info
                .available_bytes
                .saturating_sub(weights_bytes)
                .saturating_sub(activation_bytes)
                .saturating_sub(config.runtime_reserve_bytes)
                .saturating_sub(config.safety_reserve_bytes);
            (usable as f64 * fraction) as u64
        });
        let requested = config.device_budget.or(automatic).ok_or_else(|| {
            FellmError::other(
                "automatic KV budget requires backend memory information; set an explicit byte budget",
            )
        })?;
        let device_pages = usize::try_from(requested / page_bytes as u64)
            .unwrap_or(usize::MAX)
            .max(minimum_pages.max(1));
        let kv_bytes = (device_pages as u64).saturating_mul(page_bytes as u64);
        let host_budget = config.host_budget.unwrap_or(0);
        let host_pages = usize::try_from(host_budget / page_bytes as u64).unwrap_or(usize::MAX);
        let host_bytes = (host_pages as u64).saturating_mul(page_bytes as u64);
        let remaining_reserve_bytes = memory.map(|info| {
            info.available_bytes
                .saturating_sub(weights_bytes)
                .saturating_sub(activation_bytes)
                .saturating_sub(config.runtime_reserve_bytes)
                .saturating_sub(kv_bytes)
        });
        Ok(Self {
            weights_bytes,
            activation_bytes,
            page_bytes,
            device_pages,
            kv_bytes,
            host_pages,
            host_bytes,
            remaining_reserve_bytes,
        })
    }
}

/// Central fabric: logical identity, physical storage, residency, sharing.
pub struct KvFabric {
    pub config: KvFabricConfig,
    pub groups: Vec<KvGroupDesc>,
    arena: PageArena,
    map: PageMap,
    share: SharedKvStore,
    policy: Box<dyn ResidencyPolicy>,
    addressing: Box<dyn AddressTranslator>,
    clock: u64,
    metrics: FabricMetrics,
    mode: KvMode,
}

impl KvFabric {
    pub fn new(
        config: KvFabricConfig,
        n_pages: usize,
        groups: Vec<KvGroupDesc>,
        host_pages: usize,
        model_fingerprint: u64,
    ) -> Result<Self> {
        let primary = groups
            .first()
            .cloned()
            .unwrap_or_else(|| KvGroupDesc::full_attention(1, 1, 1, config.default_encoding));
        let page_tokens = primary.page_tokens().max(STANDARD_PAGE_TOKENS);
        // Kernels currently expect standard 16-token pages for full attention.
        let page_tokens = if primary.kind == super::types::KvGroupKind::FullAttention {
            STANDARD_PAGE_TOKENS
        } else {
            page_tokens
        };
        let encoding = config.default_encoding;
        let arena = PageArena::new(
            n_pages,
            primary.layer_count as usize,
            primary.n_kv_heads,
            primary.head_dim,
            page_tokens,
            encoding,
            host_pages,
        )?;
        let share = SharedKvStore::new(page_tokens, model_fingerprint, encoding);
        let policy = policy_from_kind(config.residency_policy);
        let addressing = translator_for(config.addressing);
        let mode = config.mode.resolve();
        let map = PageMap::new();
        let mut metrics = FabricMetrics::default();
        metrics.total_device_pages = n_pages;
        metrics.free_device_pages = n_pages;
        Ok(Self {
            config,
            groups,
            arena,
            map,
            share,
            policy,
            addressing,
            clock: 0,
            metrics,
            mode,
        })
    }

    /// Convenience constructor matching common full-attention models.
    pub fn new_full_attention(
        config: KvFabricConfig,
        n_pages: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        host_pages: usize,
    ) -> Result<Self> {
        let group =
            KvGroupDesc::full_attention(n_layers, n_kv_heads, head_dim, config.default_encoding);
        Self::new(config, n_pages, vec![group], host_pages, 0)
    }

    #[must_use]
    pub fn mode(&self) -> KvMode {
        self.mode
    }

    #[must_use]
    pub fn addressing_strategy(&self) -> KvAddressing {
        self.addressing.strategy()
    }

    #[must_use]
    pub fn page_tokens(&self) -> usize {
        self.arena.page_tokens()
    }

    #[must_use]
    pub fn page_bytes(&self) -> usize {
        self.arena.page_bytes()
    }

    #[must_use]
    pub fn n_pages(&self) -> usize {
        self.arena.n_pages()
    }

    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.arena.n_layers()
    }

    #[must_use]
    pub fn tokens_stride(&self) -> usize {
        self.arena.tokens_stride()
    }

    #[must_use]
    pub fn free_count(&self) -> usize {
        self.arena.free_count()
    }

    #[must_use]
    pub fn encoding(&self) -> KvEncoding {
        self.arena.encoding()
    }

    pub fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    #[must_use]
    pub fn metrics(&self) -> FabricMetrics {
        let mut m = self.metrics.clone();
        m.free_device_pages = self.arena.free_count();
        m.total_device_pages = self.arena.n_pages();
        m.device_resident_pages = self.arena.allocated_count();
        m.device_resident_bytes =
            (m.device_resident_pages as u64).saturating_mul(self.arena.page_bytes() as u64);
        m.host_resident_pages = self
            .arena
            .n_pages()
            .saturating_sub(self.arena.free_count())
            .saturating_sub(
                self.arena
                    .allocated_slots()
                    .filter(|s| !self.arena.is_on_host(*s))
                    .count(),
            );
        let ps = self.share.stats();
        m.prefix_hits = ps.hits;
        m.prefix_misses = ps.misses;
        m.prefix_hit_tokens = ps.hit_tokens;
        m.prefix_miss_tokens = ps.miss_tokens;
        m.shared_kv_bytes = self.share.shared_bytes(self.arena.page_bytes());
        m.shared_objects = self.share.object_count();
        m.evictions = ps.evictions;
        m
    }

    #[must_use]
    pub fn prefix_stats(&self) -> PrefixCacheStats {
        self.share.stats()
    }

    #[must_use]
    pub fn memory_pressure(&self) -> f64 {
        let total = self.arena.n_pages().max(1) as f64;
        let used = self.arena.allocated_count() as f64;
        (used / total).clamp(0.0, 1.0)
    }

    /// Allocate an empty sequence address space.
    #[must_use]
    pub fn new_sequence(&self, max_seq: usize) -> KvSequence {
        KvSequence::new(self.arena.n_layers(), max_seq)
    }

    /// Host-tier free slots remaining (0 if no host tier).
    #[must_use]
    pub fn host_free_count(&self) -> usize {
        self.arena.host_free_count()
    }

    /// Host key used for content-addressed host-tier stashes of a logical page.
    #[must_use]
    pub fn host_key_for(page: KvPageId) -> u32 {
        (page.0 & 0xffff_ffff) as u32
    }

    /// Release all logical pages owned by a sequence, including host-tier stashes
    /// left after migrate_out (so preempted cancel/finish cannot leak host capacity).
    pub fn release_sequence(&mut self, seq: &mut KvSequence) {
        for layer in 0..seq.n_layers() {
            for &pid in seq.layer_map(layer).pages() {
                if let Some(slot) = self.map.resolve(pid) {
                    self.arena.dec_ref(slot);
                    // Keep map entry if still shared by prefix store; only unbind
                    // when physical refcount hit zero (dec_ref already freed slot).
                    if self
                        .map
                        .resolve(pid)
                        .is_some_and(|s| self.arena.refcount(s) == 0)
                    {
                        self.map.unbind(pid);
                    }
                } else {
                    // Unbound: may still hold an exclusive host-tier stash from migrate.
                    let host_key = Self::host_key_for(pid);
                    if self.arena.drop_host_stash(host_key) {
                        self.map.set_location(pid, KvLocation::NotResident);
                    }
                }
            }
        }
        seq.clear_maps();
    }

    /// Ensure dense position `pos` is writable for every layer (alloc / CoW).
    pub fn ensure_writable(&mut self, seq: &mut KvSequence, pos: usize) -> Result<()> {
        if pos >= seq.max_seq {
            return Err(FellmError::other(format!(
                "kv fabric: position {pos} >= max_seq {}",
                seq.max_seq
            )));
        }
        let pt = self.arena.page_tokens();
        let logical = pos / pt;
        for layer in 0..seq.n_layers() {
            while seq.layer_map(layer).num_pages() <= logical {
                let slot = self.alloc_or_reclaim()?;
                let pid = self.map.alloc_id();
                self.arena.inc_ref(slot);
                self.map
                    .bind(pid, slot, KvLocation::Resident(KvTier::Device));
                seq.layer_map_mut(layer).push_page(pid);
            }
            let pid = seq.layer_map(layer).page(logical);
            // CoW if shared — forks a new logical page id for this sequence only.
            let slot = self
                .map
                .resolve(pid)
                .ok_or_else(|| FellmError::other("kv fabric: page not bound"))?;
            if self.arena.refcount(slot) > 1 || self.arena.meta(slot).immutable {
                let cow = match ensure_cow(&mut self.arena, &mut self.map, pid, self.clock) {
                    Ok(c) => c,
                    Err(e) => {
                        let need = 1;
                        if self.evict_shared_until(need) == 0 {
                            self.metrics.allocation_failures =
                                self.metrics.allocation_failures.saturating_add(1);
                            return Err(e);
                        }
                        ensure_cow(&mut self.arena, &mut self.map, pid, self.clock)?
                    }
                };
                if cow.forked {
                    *seq.layer_map_mut(layer).page_mut(logical) = cow.page;
                    self.metrics.cow_forks = self.metrics.cow_forks.saturating_add(1);
                }
            } else {
                self.arena.touch(slot, self.clock);
            }
        }
        if pos + 1 > seq.len_tokens {
            seq.len_tokens = pos + 1;
        }
        Ok(())
    }

    fn alloc_or_reclaim(&mut self) -> Result<PhysicalSlot> {
        if let Some(slot) = self.arena.alloc_page(self.clock) {
            return Ok(slot);
        }
        // Value/cost eviction of shared idle objects first.
        let _ = self.evict_shared_until(1);
        if let Some(slot) = self.arena.alloc_page(self.clock) {
            return Ok(slot);
        }
        // Try demoting private pages to host tier to free device capacity —
        // for host-primary arena, migration stores a copy; true device free
        // requires separate device pool. Record decision.
        self.metrics.allocation_failures = self.metrics.allocation_failures.saturating_add(1);
        Err(FellmError::other("kv fabric: out of physical pages"))
    }

    pub fn evict_shared_until(&mut self, target_free: usize) -> usize {
        let n = self
            .share
            .evict_until(&mut self.arena, &self.map, target_free);
        if n > 0 {
            self.metrics.evictions = self.metrics.evictions.saturating_add(n as u64);
        }
        n
    }

    /// Attach content-addressed prefix; returns matched token count.
    pub fn attach_prefix(&mut self, tokens: &[u32], seq: &mut KvSequence) -> usize {
        if !self.config.prefix_sharing {
            return 0;
        }
        let matched = self
            .share
            .attach_match(&mut self.arena, &self.map, seq, tokens);
        if matched > 0 {
            seq.shared_prefix_len = matched;
            seq.len_tokens = seq.len_tokens.max(matched);
        }
        matched
    }

    pub fn insert_prefix(&mut self, tokens: &[u32], seq: &KvSequence) {
        if !self.config.prefix_sharing {
            return;
        }
        self.share
            .insert_prompt(&mut self.arena, &self.map, tokens, seq);
    }

    /// Compact sequence to retained absolute positions (Exact elastic retention host path).
    pub fn compact_sequence_to_positions(
        &mut self,
        seq: &mut KvSequence,
        retain_positions: &[u32],
        page_tokens: usize,
    ) -> usize {
        let pt = page_tokens.max(1);
        let prefix = seq.shared_prefix_len as u32;
        let mut keep: Vec<u32> = (0..prefix).collect();
        for &p in retain_positions {
            if p >= prefix {
                keep.push(p);
            }
        }
        keep.sort_unstable();
        keep.dedup();
        let prior_orig = seq.original_positions.clone();
        let was_compressed = !prior_orig.is_empty();
        let mut src_dense: Vec<usize> = Vec::with_capacity(keep.len());
        if was_compressed {
            for &abs in &keep {
                if let Some(i) = prior_orig.iter().position(|&p| p == abs) {
                    src_dense.push(i);
                }
            }
        } else {
            for &abs in &keep {
                if (abs as usize) < seq.len_tokens {
                    src_dense.push(abs as usize);
                }
            }
        }
        if src_dense.is_empty() {
            return 0;
        }

        let n_new = src_dense.len();
        let n_new_pages = n_new.div_ceil(pt).max(1);
        let mut reclaimed = 0usize;
        let orig: Vec<u32> = src_dense
            .iter()
            .map(|&i| {
                if was_compressed {
                    prior_orig.get(i).copied().unwrap_or(i as u32)
                } else {
                    i as u32
                }
            })
            .collect();

        for layer in 0..seq.n_layers() {
            let old_pages: Vec<KvPageId> = seq.layer_map(layer).pages().to_vec();
            let mut new_pages = Vec::with_capacity(n_new_pages);
            for _ in 0..n_new_pages {
                let slot = match self.arena.alloc_page(self.clock) {
                    Some(s) => s,
                    None => {
                        for &np in &new_pages {
                            if let Some(s) = self.map.resolve(np) {
                                self.arena.dec_ref(s);
                            }
                            self.map.unbind(np);
                        }
                        return reclaimed;
                    }
                };
                let pid = self.map.alloc_id();
                self.arena.inc_ref(slot);
                self.map
                    .bind(pid, slot, KvLocation::Resident(KvTier::Device));
                new_pages.push(pid);
            }
            for (new_i, &old_i) in src_dense.iter().enumerate() {
                let old_logical = old_i / pt;
                let old_slot = old_i % pt;
                let Some(&old_pid) = old_pages.get(old_logical) else {
                    continue;
                };
                let Some(old_phys) = self.map.resolve(old_pid) else {
                    continue;
                };
                let new_logical = new_i / pt;
                let new_slot_i = new_i % pt;
                let new_pid = new_pages[new_logical];
                let Some(new_phys) = self.map.resolve(new_pid) else {
                    continue;
                };
                let k_src: Vec<f16> = self.arena.k_row(old_phys, old_slot).to_vec();
                let v_src: Vec<f16> = self.arena.v_row(old_phys, old_slot).to_vec();
                self.arena
                    .k_row_mut(new_phys, new_slot_i)
                    .copy_from_slice(&k_src);
                self.arena
                    .v_row_mut(new_phys, new_slot_i)
                    .copy_from_slice(&v_src);
            }
            for &pid in &old_pages {
                if let Some(slot) = self.map.resolve(pid) {
                    let before = self.arena.free_count();
                    self.arena.dec_ref(slot);
                    if self.arena.free_count() > before {
                        reclaimed += 1;
                    }
                    if self.arena.refcount(slot) == 0 {
                        self.map.unbind(pid);
                    }
                }
            }
            seq.layer_map_mut(layer).set_pages(new_pages);
        }

        seq.original_positions = orig;
        seq.len_tokens = n_new;
        reclaimed
    }

    /// Migrate sequence pages to host tier and free device slots (hard preempt).
    ///
    /// Two-phase and atomic w.r.t. sequence maps: either every exclusive page is
    /// stashed then unbound, or on host-tier exhaustion all stashes created by
    /// this call are rolled back and the sequence is left fully device-resident
    /// (`non_resident` stays false).
    ///
    /// Host keys are derived from logical [`KvPageId`] so restore rebinds fresh
    /// physical slots without leaking identity into sequence-facing APIs.
    pub fn migrate_out(&mut self, seq: &mut KvSequence) -> Result<u64> {
        // Phase 1: plan exclusive device-resident pages (no mutation yet).
        let mut plan: Vec<(KvPageId, PhysicalSlot, u32)> = Vec::new();
        for layer in 0..seq.n_layers() {
            for &pid in seq.layer_map(layer).pages() {
                let Some(slot) = self.map.resolve(pid) else {
                    continue;
                };
                // Skip shared immutable pages (prefix) — only demote exclusive.
                if self.arena.meta(slot).immutable || self.arena.refcount(slot) > 1 {
                    continue;
                }
                plan.push((pid, slot, Self::host_key_for(pid)));
            }
        }
        if plan.is_empty() {
            // Nothing exclusive to move; still mark non-resident so scheduler
            // will soft-skip until shared pages are handled by admission.
            seq.non_resident = true;
            return Ok(0);
        }

        // Phase 2: stash all pages first. On any failure, drop stashes from this call.
        let mut bytes = 0u64;
        let mut stashed_keys: Vec<u32> = Vec::with_capacity(plan.len());
        for &(_pid, slot, host_key) in &plan {
            let already = self.arena.host_contains(host_key);
            match self.arena.stash_to_host(host_key, slot) {
                Ok(b) => {
                    if !already {
                        stashed_keys.push(host_key);
                    }
                    bytes = bytes.saturating_add(b);
                }
                Err(e) => {
                    for key in stashed_keys.drain(..) {
                        let _ = self.arena.drop_host_stash(key);
                    }
                    // Sequence maps untouched; still fully device-resident.
                    debug_assert!(!seq.non_resident);
                    return Err(e);
                }
            }
        }

        // Phase 3: commit — free device slots and unbind only after full stash success.
        for &(pid, slot, _host_key) in &plan {
            debug_assert!(
                self.map.resolve(pid) == Some(slot),
                "migrate_out commit: page must still be bound"
            );
            self.arena.dec_ref(slot);
            self.map.unbind(pid);
            self.map
                .set_location(pid, KvLocation::Resident(KvTier::Host));
        }
        seq.non_resident = true;
        self.metrics.migrations = self.metrics.migrations.saturating_add(1);
        self.metrics.migration_bytes = self.metrics.migration_bytes.saturating_add(bytes);
        Ok(bytes)
    }

    /// Restore sequence pages to device/compute tier (alloc + host restore).
    pub fn migrate_in(&mut self, seq: &mut KvSequence) -> Result<u64> {
        let mut bytes = 0u64;
        for layer in 0..seq.n_layers() {
            let n = seq.layer_map(layer).num_pages();
            for logical in 0..n {
                let pid = seq.layer_map(layer).page(logical);
                if self.map.resolve(pid).is_some() {
                    // Still device-resident (shared prefix pages).
                    continue;
                }
                let host_key = Self::host_key_for(pid);
                if !self.arena.host_contains(host_key) {
                    // Page was shared and never stashed — must still be mapped.
                    return Err(FellmError::other(
                        "migrate_in: page neither resident nor in host tier",
                    ));
                }
                let new_slot = self.alloc_or_reclaim()?;
                bytes += self.arena.restore_from_host_key(host_key, new_slot)?;
                self.arena.inc_ref(new_slot);
                self.map
                    .bind(pid, new_slot, KvLocation::Resident(KvTier::Device));
            }
        }
        seq.non_resident = false;
        self.metrics.migrations = self.metrics.migrations.saturating_add(1);
        self.metrics.migration_bytes = self.metrics.migration_bytes.saturating_add(bytes);
        Ok(bytes)
    }

    /// Prefetch: ensure pages for upcoming positions are device-resident.
    pub fn prefetch(&mut self, seq: &mut KvSequence, positions: &[usize]) -> Result<u64> {
        if !self.config.prefetch || !seq.non_resident {
            return Ok(0);
        }
        let _ = positions;
        self.migrate_in(seq)
    }

    /// Admission check: can we free/allocate enough pages for `need_pages`?
    #[must_use]
    pub fn can_admit(&self, need_pages: usize) -> bool {
        self.arena.free_count() >= need_pages
    }

    /// Estimate pages needed for a sequence to advance `extra_tokens`.
    #[must_use]
    pub fn pages_needed_for(&self, seq: &KvSequence, extra_tokens: usize) -> usize {
        let pt = self.arena.page_tokens().max(1);
        let end = seq.len_tokens.saturating_add(extra_tokens);
        let need_logical = end.div_ceil(pt);
        let have = seq.layer_map(0).num_pages();
        need_logical.saturating_sub(have) * seq.n_layers()
    }

    /// Build residency signals for the scheduler.
    #[must_use]
    pub fn residency_signals(
        &self,
        seq: &KvSequence,
        extra_tokens: usize,
        priority: i32,
        prefix_hit_tokens: usize,
    ) -> ResidencySignals {
        let page_bytes = self.arena.page_bytes() as u64;
        let mut resident = 0usize;
        let mut non_res = 0usize;
        for layer in 0..seq.n_layers() {
            for &pid in seq.layer_map(layer).pages() {
                match self.map.location(pid) {
                    KvLocation::Resident(KvTier::Device)
                    | KvLocation::Resident(KvTier::HostPinned)
                    | KvLocation::Resident(KvTier::Host)
                        if !seq.non_resident =>
                    {
                        resident += 1;
                    }
                    _ => non_res += 1,
                }
            }
        }
        if seq.non_resident {
            non_res = seq.layer_map(0).num_pages().saturating_mul(seq.n_layers());
            resident = 0;
        }
        let need = self.pages_needed_for(seq, extra_tokens);
        ResidencySignals {
            resident_pages: resident,
            non_resident_pages: non_res,
            resident_bytes: (resident as u64) * page_bytes,
            non_resident_bytes: (non_res as u64) * page_bytes,
            pages_needed_next: need,
            prefix_hit_tokens,
            estimated_transfer_bytes: (non_res as u64) * page_bytes,
            estimated_compute_cost: extra_tokens as f64,
            memory_pressure: self.memory_pressure(),
            expected_output_growth: extra_tokens,
            priority,
            latency_target_ms: None,
        }
    }

    /// Score physical pages for eviction (value/cost). Returns lowest-value first.
    pub fn rank_eviction_candidates(&self, limit: usize) -> Vec<(PhysicalSlot, f64)> {
        let pressure = self.memory_pressure();
        let mut scores: Vec<_> = self
            .arena
            .allocated_slots()
            .filter(|s| !self.arena.meta(*s).immutable && self.arena.refcount(*s) == 1)
            .map(|slot| {
                let score = self.policy.score(&PageScoreInput {
                    slot,
                    meta: self.arena.meta(slot),
                    clock: self.clock,
                    memory_pressure: pressure,
                    request_importance: 1.0,
                });
                (slot, score)
            })
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(limit);
        scores
    }

    /// Aggregate keep-value for a sequence under the active residency policy.
    ///
    /// Higher = more valuable to keep resident (prefer not to preempt).
    /// Used by the scheduler as the primary preemption ranking (not plain LRU).
    #[must_use]
    pub fn sequence_keep_value(&self, seq: &KvSequence, priority: i32, last_used: u64) -> f64 {
        let pressure = self.memory_pressure();
        let importance = (priority as f32).max(0.0) + 1.0;
        let mut total = 0.0f64;
        let mut n = 0usize;
        let mut exclusive = 0usize;
        for layer in 0..seq.n_layers() {
            for &pid in seq.layer_map(layer).pages() {
                let Some(slot) = self.map.resolve(pid) else {
                    continue;
                };
                let meta = self.arena.meta(slot);
                let score = self.policy.score(&PageScoreInput {
                    slot,
                    meta,
                    clock: self.clock.max(last_used),
                    memory_pressure: pressure,
                    request_importance: importance,
                });
                total += score;
                n += 1;
                if !meta.immutable && meta.refcount <= 1 {
                    exclusive += 1;
                }
            }
        }
        if n == 0 || exclusive == 0 {
            // Nothing useful to demote — never prefer as victim.
            return f64::MAX;
        }
        total / n as f64
    }

    /// Materialize addressing view for a batch of sequences (one row each).
    pub fn materialize_addressing(&self, sequences: &[&KvSequence]) -> AddressingView {
        let n_layers = self.arena.n_layers();
        let n_logical = sequences
            .first()
            .map(|s| s.layer_map(0).num_pages().max(1))
            .unwrap_or(1);
        let mut page_ids_per_row = Vec::with_capacity(sequences.len());
        for seq in sequences {
            page_ids_per_row.push(seq.flatten_page_ids());
        }
        self.addressing
            .materialize(&self.map, &page_ids_per_row, n_layers, n_logical)
    }

    /// Resolve logical page → physical slot for diagnostics / row access.
    #[must_use]
    pub fn resolve(&self, page: KvPageId) -> Option<PhysicalSlot> {
        self.map.resolve(page)
    }

    /// Locate dense token storage. Errors if the logical page is unbound
    /// (e.g. mid-migration or released) — never silently aliases physical slot 0.
    pub fn locate(
        &self,
        seq: &KvSequence,
        layer: usize,
        token_pos: usize,
    ) -> Result<(PhysicalSlot, usize)> {
        let pt = self.arena.page_tokens();
        let (logical, slot) = seq.locate_token(token_pos, pt);
        if layer >= seq.n_layers() || logical >= seq.layer_map(layer).num_pages() {
            return Err(FellmError::other(format!(
                "locate: token {token_pos} layer {layer} out of range"
            )));
        }
        let pid = seq.layer_map(layer).page(logical);
        let phys = self.map.resolve(pid).ok_or_else(|| {
            FellmError::other(format!(
                "locate: logical page {pid} unbound (non-resident or released)"
            ))
        })?;
        Ok((phys, slot))
    }

    pub fn k_row(&self, slot: PhysicalSlot, row: usize) -> &[f16] {
        self.arena.k_row(slot, row)
    }

    pub fn k_row_mut(&mut self, slot: PhysicalSlot, row: usize) -> &mut [f16] {
        self.arena.k_row_mut(slot, row)
    }

    pub fn v_row(&self, slot: PhysicalSlot, row: usize) -> &[f16] {
        self.arena.v_row(slot, row)
    }

    pub fn v_row_mut(&mut self, slot: PhysicalSlot, row: usize) -> &mut [f16] {
        self.arena.v_row_mut(slot, row)
    }

    #[must_use]
    pub fn arena_bytes(&self) -> &[u8] {
        self.arena.arena_bytes()
    }

    pub fn arena_bytes_mut(&mut self) -> &mut [u8] {
        self.arena.arena_bytes_mut()
    }

    #[must_use]
    pub fn arena_ptr_mut(&mut self) -> (*mut u8, usize) {
        self.arena.arena_ptr_mut()
    }

    /// Flatten physical block table for one sequence (kernel ABI).
    ///
    /// Layout: `[layer0_logical0..L, layer1_logical0..L, …]` matching
    /// `PagedKvSnapshot::physical_for` indexing.
    ///
    /// # Panics
    /// Debug builds panic if any logical page is unbound (never silently
    /// substitute physical slot 0 — that would alias every layer onto block 0
    /// and produce garbage CUDA attention).
    #[must_use]
    pub fn physical_block_table(&self, seq: &KvSequence) -> Vec<u32> {
        let pages = seq.flatten_page_ids();
        let mut out = Vec::with_capacity(pages.len());
        for pid in pages {
            let slot = self.map.resolve(pid).unwrap_or_else(|| {
                panic!("physical_block_table: unbound logical page {pid} (refusing slot-0 alias)");
            });
            debug_assert!(
                (slot.0 as usize) < self.arena.n_pages(),
                "physical id {} out of arena ({} pages)",
                slot.0,
                self.arena.n_pages()
            );
            out.push(slot.0);
        }
        out
    }

    /// Physical ids for one layer (kernel / policy view dense_block_table).
    #[must_use]
    pub fn physical_block_table_layer(&self, seq: &KvSequence, layer: usize) -> Vec<u32> {
        seq.layer_map(layer)
            .pages()
            .iter()
            .map(|&pid| {
                self.map
                    .resolve(pid)
                    .unwrap_or_else(|| {
                        panic!(
                            "physical_block_table_layer: unbound logical page {pid} layer {layer}"
                        );
                    })
                    .0
            })
            .collect()
    }

    /// Refcount of the physical slot behind a logical page (tests / diagnostics).
    #[must_use]
    pub fn page_refcount(&self, page: KvPageId) -> u32 {
        self.map
            .resolve(page)
            .map(|s| self.arena.refcount(s))
            .unwrap_or(0)
    }

    /// Select encoding for a segment under the active mode/policy.
    ///
    /// Exact mode always returns the default encoding. Elastic + TemperatureTiered
    /// may demote colder segments (hooks for quantization paths; storage may still
    /// materialize as default until kernels support mixed encodings).
    #[must_use]
    pub fn select_encoding_for_segment(&self, age_tokens: u32, is_active_tail: bool) -> KvEncoding {
        let mode = self.mode;
        if mode != KvMode::Elastic {
            return self.config.default_encoding;
        }
        match self.config.encoding_policy {
            super::types::KvEncodingPolicy::Uniform => self.config.default_encoding,
            super::types::KvEncodingPolicy::TemperatureTiered => {
                if is_active_tail {
                    self.config.default_encoding // FP16/BF16 active tail
                } else if age_tokens < 256 {
                    KvEncoding::Fp8
                } else if age_tokens < 2048 {
                    KvEncoding::Int8
                } else {
                    KvEncoding::Int4
                }
            }
        }
    }

    /// Invariant checks (debug / tests).
    pub fn assert_invariants(&self, sequences: &[&KvSequence]) {
        for seq in sequences {
            for layer in 0..seq.n_layers() {
                for &pid in seq.layer_map(layer).pages() {
                    let slot = self.map.resolve(pid).expect("no dangling page references");
                    assert!(
                        self.arena.refcount(slot) >= 1,
                        "refcount must stay valid for mapped pages"
                    );
                }
            }
        }
        // Accounting: free + allocated ≈ total
        assert_eq!(
            self.arena.free_count() + self.arena.allocated_count(),
            self.arena.n_pages(),
            "memory accounting must match allocations"
        );
    }
}
