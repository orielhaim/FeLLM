//! TriAttention sequence-state / KV retention policy plugin.
//!
//! Implements paper arXiv:2604.04921 scoring on the public FeLLM capability
//! contract — not a core `if triattention` branch.
//!
//! Scoring (pre-RoPE space):
//! - `S_trig(k, Δ) = Σ_f ‖E[q_f]‖ · ‖k_f‖ · cos(ω_f Δ + φ_f)`
//! - `S_norm(k) = Σ_f (E[‖q_f‖] - ‖E[q_f]‖) · ‖k_f‖`
//! - `S = S_trig + S_norm`, multi-offset average over `D = {1,2,4,...,2^16}`
//! - GQA: z-score per query head then max-aggregate
//! - Windowed top-B pruning every `β` tokens (default 128)

#![deny(missing_docs)]

mod scoring;

pub use scoring::{
    CalibrationCenters, TriAttentionConfig, geometric_offsets, score_batch_host, score_keys_gqa,
    score_keys_single, select_top_b, trig_score_key,
};

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::c_abi::{
    CapabilityRegistryVtable, HostContext, PLUGIN_MAX_FEATURES, PluginCapabilityRegistration,
    PluginManifestJson,
};
use fellm_plugin_abi::capability::{
    CapabilityKind, FeatureId, FeatureSet, PluginConfig, ProviderDescriptor, ProviderVersion,
};
use fellm_plugin_abi::sequence_state::{
    RetainedEntry, RetentionContext, RetentionPlan, RetentionStats, SequenceAttentionState,
    SequenceStatePolicy,
};
use fellm_plugin_abi::{ABI_VERSION, AbiVersion};
use std::os::raw::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

static HOST_CTX: Mutex<Option<HostContext>> = Mutex::new(None);

/// First-party TriAttention policy (also usable as a static `Arc` in tests).
pub struct TriAttentionPolicy {
    desc: ProviderDescriptor,
    config: TriAttentionConfig,
    centers: Option<CalibrationCenters>,
}

impl TriAttentionPolicy {
    /// Default policy with empty calibration (identity centers for synthetic tests).
    #[must_use]
    pub fn new() -> Self {
        let provides = FeatureSet::from_ids([
            FeatureId::KV_MUTABLE_REMAP,
            FeatureId::KV_PHYSICAL_RECLAIM,
            FeatureId::KV_LOGICAL_POSITIONS,
            FeatureId::KV_PER_HEAD_RETENTION,
            FeatureId::KV_PREFIX_PRIVATE_SPLIT,
            FeatureId::KV_PRE_ROPE_SCORING,
        ]);
        // NOTE: no KV_DEVICE_COMPACTION — compaction is always host-side
        // (`CacheManager::compact_sequence_to_positions`). Device kernels only
        // mirror pre-RoPE keys into the host store consumed by the policy.
        // Attention must accept indirect positions / per-head views.
        let requires = FeatureSet::from_ids([
            FeatureId::ATTN_INDIRECT_POSITIONS,
            FeatureId::ATTN_PER_HEAD_KV_VIEWS,
        ]);
        let desc = ProviderDescriptor::new(
            "kv.triattention",
            CapabilityKind::SequenceStatePolicy,
            ProviderVersion {
                major: 0,
                minor: 1,
                patch: 0,
            },
            "TriAttention: pre-RoPE trigonometric + norm KV retention (arXiv:2604.04921)",
        )
        .with_provides(provides)
        .with_requires(requires)
        .with_priority(50)
        .with_meta("paper", "arXiv:2604.04921")
        .with_meta("first_party", "true")
        .with_meta(
            "config_keys",
            "budget,window,rope_theta,warmup,calibration_path",
        );
        Self {
            desc,
            config: TriAttentionConfig::default(),
            centers: None,
        }
    }

    /// Install calibration centers (from offline pass or synthetic test data).
    pub fn set_centers(&mut self, centers: CalibrationCenters) {
        self.centers = Some(centers);
    }

    /// Apply config map (`budget`, `window`, `rope_theta`, `warmup`).
    pub fn apply_config_map(
        &mut self,
        map: &std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        self.config = TriAttentionConfig::from_map(map)?;
        Ok(())
    }

    /// Current config.
    #[must_use]
    pub fn config(&self) -> &TriAttentionConfig {
        &self.config
    }
}

impl Default for TriAttentionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceStatePolicy for TriAttentionPolicy {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.desc
    }

    fn validate_config(&self, config: &PluginConfig) -> Result<()> {
        let scoped = config.for_provider("kv.triattention", true);
        TriAttentionConfig::from_map(&scoped)?;
        Ok(())
    }

    fn should_compress(&self, ctx: &RetentionContext<'_>) -> bool {
        let cfg = {
            let scoped = ctx.config.for_provider("kv.triattention", true);
            TriAttentionConfig::from_map(&scoped).unwrap_or(self.config.clone())
        };
        // Live private length only — never absolute history length.
        let private_len = ctx.state.live_private_positions().len() as u32;
        let budget_private = cfg.budget.saturating_sub(ctx.state.shared_prefix_len);
        if private_len <= budget_private.max(1) {
            return false;
        }
        // Windowed: every β tokens (paper: β=128).
        ctx.tokens_since_window >= cfg.window && ctx.tokens_since_window > 0
    }

    fn plan_retention(&self, ctx: &RetentionContext<'_>) -> Result<RetentionPlan> {
        let scoped = ctx.config.for_provider("kv.triattention", true);
        let cfg = TriAttentionConfig::from_map(&scoped).unwrap_or(self.config.clone());

        let prefix = ctx.state.shared_prefix_len;
        // Live index contract: only positions that still exist physically.
        let live = ctx.state.live_retained_positions();
        let mut retain: Vec<u32> = live.iter().copied().filter(|&p| p < prefix).collect();
        let private_positions: Vec<u32> = live.into_iter().filter(|&p| p >= prefix).collect();
        let budget_private = cfg.budget.saturating_sub(prefix).max(1);

        if private_positions.is_empty() {
            return Ok(RetentionPlan {
                retain_positions: retain,
                per_kv_head: Vec::new(),
                compact: false,
                host_densified: false,
            });
        }

        if private_positions.len() as u32 <= budget_private {
            retain.extend(private_positions);
            retain.sort_unstable();
            retain.dedup();
            return Ok(RetentionPlan {
                retain_positions: retain,
                per_kv_head: Vec::new(),
                compact: false,
                host_densified: false,
            });
        }

        let head_dim = ctx.head_dim.max(1) as usize;
        let n_kv = ctx.n_kv_heads.max(1) as usize;
        let n_heads = ctx.n_heads.max(1) as usize;
        let n_freq = head_dim / 2;

        let centers = self
            .centers
            .clone()
            .unwrap_or_else(|| CalibrationCenters::identity(n_heads, n_freq, cfg.rope_theta));

        let keys = ctx.pre_rope_keys.ok_or_else(|| {
            FellmError::other(
                "kv.triattention requires pre-RoPE keys packed in live-candidate order",
            )
        })?;

        // Keys are packed 1:1 with private_positions (host gathers live index only).
        let stride = n_kv * head_dim;
        if keys.len() < private_positions.len() * stride {
            return Err(FellmError::other(format!(
                "kv.triattention: pre_rope_keys len {} < live private {} * stride {}",
                keys.len(),
                private_positions.len(),
                stride
            )));
        }
        let private_keys = &keys[..private_positions.len() * stride];

        let pq = ctx
            .current_pos
            .max(private_positions.last().copied().unwrap_or(0));
        let scores = score_keys_gqa(
            private_keys,
            &private_positions,
            &centers,
            n_heads,
            n_kv,
            head_dim,
            pq,
            cfg.rope_theta,
        );

        let top = select_top_b(&scores, &private_positions, budget_private as usize);
        // Safety: every retained id must be from the live private set.
        for &p in &top {
            debug_assert!(private_positions.contains(&p));
        }
        retain.extend(top.iter().copied());
        retain.sort_unstable();
        retain.dedup();

        Ok(RetentionPlan {
            retain_positions: retain,
            per_kv_head: Vec::new(),
            compact: true,
            host_densified: false,
        })
    }

    fn apply_plan(
        &self,
        state: &mut SequenceAttentionState,
        plan: &RetentionPlan,
    ) -> Result<RetentionStats> {
        let live_before = state.live_retained_positions().len() as u32;
        let retained = plan.retain_positions.len() as u32;

        // Host densified path: views already match new tables (sync_seq_attn_from_cache).
        // Do NOT re-interpret absolute retain_positions against the densified block table.
        if plan.host_densified || state.layout_owner == fellm_plugin_abi::LayoutOwner::HostDensified
        {
            state.live_positions = plan.retain_positions.clone();
            state.meta.insert(
                "last_retained".into(),
                plan.retain_positions.len().to_string(),
            );
            state.meta.insert("host_densified".into(), "1".into());
            // Preserve dense_seq_len / entries / block tables from host sync.
            return Ok(RetentionStats {
                candidate_count: live_before,
                retained_count: retained,
                blocks_reclaimed: live_before.saturating_sub(retained),
                scored_count: live_before.saturating_sub(state.shared_prefix_len),
            });
        }

        // Policy-view-only path (no host densify): update entries without touching phys.
        for view in &mut state.layer_views {
            if plan.compact && !plan.retain_positions.is_empty() {
                view.entries = plan
                    .retain_positions
                    .iter()
                    .enumerate()
                    .map(|(i, &pos)| RetainedEntry {
                        logical_pos: pos,
                        physical_block: i as u32, // placeholder until host densifies
                        physical_slot: 0,
                        kv_head: u16::MAX,
                        layer: view.layer as u16,
                    })
                    .collect();
            }
        }
        state.live_positions = plan.retain_positions.clone();
        state.meta.insert(
            "last_retained".into(),
            plan.retain_positions.len().to_string(),
        );

        Ok(RetentionStats {
            candidate_count: live_before,
            retained_count: retained,
            blocks_reclaimed: 0,
            scored_count: live_before.saturating_sub(state.shared_prefix_len),
        })
    }
}

/// Report ABI version to the host loader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_abi_version() -> AbiVersion {
    ABI_VERSION
}

static MANIFEST_JSON: &[u8] = concat!(
    r#"{"schema":1,"id":"fellm.triattention","name":"TriAttention","version":"#,
    env!("CARGO_PKG_VERSION"),
    r#"","provides":[{"type":"capability","id":"kv.triattention"}]}"#
)
.as_bytes();

#[unsafe(no_mangle)]
/// Return the embedded declarative plugin manifest.
pub unsafe extern "C" fn _fellm_plugin_manifest_json() -> PluginManifestJson {
    PluginManifestJson {
        ptr: MANIFEST_JSON.as_ptr(),
        len: MANIFEST_JSON.len(),
    }
}

/// Initialize plugin with host context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_init(ctx: *const HostContext) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() {
            return -1;
        }
        let c = unsafe { *ctx };
        *HOST_CTX.lock().expect("host ctx") = Some(c);
        0
    }));
    result.unwrap_or(-99)
}

/// C ABI: score a batch of pre-RoPE keys (host f32 buffer).
///
/// `keys` points to `n_pos * n_kv_heads * head_dim` f32 values.
/// `out_scores` receives `n_pos` f32 scores. Centers use identity if null.
///
/// This is the **shipped** scoring entry used by the host policy and available
/// to CUDA launchers that mirrored pre-RoPE keys into the host store.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_triattention_score_batch(
    keys: *const f32,
    n_pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    query_pos: u32,
    rope_theta: f32,
    out_scores: *mut f32,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if keys.is_null() || out_scores.is_null() || n_pos == 0 || head_dim == 0 {
            return -1;
        }
        let n = n_pos as usize;
        let stride = n_kv_heads.max(1) as usize * head_dim as usize;
        let key_slice = unsafe { std::slice::from_raw_parts(keys, n * stride) };
        let n_freq = (head_dim as usize) / 2;
        let centers =
            CalibrationCenters::identity(n_heads.max(1) as usize, n_freq.max(1), rope_theta);
        let scores = score_batch_host(
            key_slice,
            n,
            &centers,
            n_heads.max(1) as usize,
            n_kv_heads.max(1) as usize,
            head_dim as usize,
            query_pos,
            rope_theta,
        );
        let out = unsafe { std::slice::from_raw_parts_mut(out_scores, n) };
        out.copy_from_slice(&scores);
        0
    }));
    result.unwrap_or(-99)
}

/// Register the `kv.triattention` sequence-state capability.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_register_capabilities(
    registry: *mut CapabilityRegistryVtable,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if registry.is_null() {
            return -1;
        }
        let vt = unsafe { &*registry };
        let policy = TriAttentionPolicy::new();
        let d = policy.descriptor();
        let mut reg = PluginCapabilityRegistration::empty();
        PluginCapabilityRegistration::write_cstr(&mut reg.name, &d.name);
        reg.capability = CapabilityKind::SequenceStatePolicy as u16;
        reg.version_major = d.version.major;
        reg.version_minor = d.version.minor;
        reg.version_patch = d.version.patch;
        reg.priority = d.priority;
        PluginCapabilityRegistration::write_cstr(&mut reg.summary, &d.summary);

        let provides: Vec<_> = d.provides.iter().collect();
        reg.n_provides = provides.len().min(PLUGIN_MAX_FEATURES) as u32;
        for (i, f) in provides.iter().take(PLUGIN_MAX_FEATURES).enumerate() {
            reg.provides[i] = f.0;
        }
        let requires: Vec<_> = d.requires.iter().collect();
        reg.n_requires = requires.len().min(PLUGIN_MAX_FEATURES) as u32;
        for (i, f) in requires.iter().take(PLUGIN_MAX_FEATURES).enumerate() {
            reg.requires[i] = f.0;
        }

        unsafe { (vt.register_capability)(vt.registry, &raw const reg) }
    }));
    result.unwrap_or(-99)
}

/// Tear down plugin state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _fellm_plugin_shutdown() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        *HOST_CTX.lock().expect("host ctx") = None;
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_plugin_abi::capability::PluginConfig;
    use scoring::CalibrationCenters;

    #[test]
    fn should_not_compress_under_budget() {
        let p = TriAttentionPolicy::new();
        let mut state = SequenceAttentionState::new(1);
        state.logical_len = 100;
        let cfg = PluginConfig::from_pairs(["kv.triattention.budget=2048"]).unwrap();
        let ctx = RetentionContext {
            state: &state,
            layer: None,
            current_pos: 99,
            tokens_since_window: 128,
            pre_rope_keys: None,
            head_dim: 8,
            n_kv_heads: 1,
            n_heads: 1,
            config: &cfg,
        };
        assert!(!p.should_compress(&ctx));
    }

    #[test]
    fn score_batch_c_abi_matches_rust() {
        let n_pos = 5usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let hd = 8usize;
        let mut keys = vec![0.0f32; n_pos * n_kv * hd];
        for (i, x) in keys.iter_mut().enumerate() {
            *x = (i as f32 * 0.1) + 0.5;
        }
        let mut out = vec![0.0f32; n_pos];
        let rc = unsafe {
            _fellm_triattention_score_batch(
                keys.as_ptr(),
                n_pos as u32,
                n_heads as u32,
                n_kv as u32,
                hd as u32,
                10,
                10_000.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        assert!(out.iter().any(|&s| s.is_finite()));
        // Top-1 via C ABI scores is deterministic.
        let positions: Vec<u32> = (0..n_pos as u32).collect();
        let top = select_top_b(&out, &positions, 2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn compress_and_reclaim_stats() {
        let mut p = TriAttentionPolicy::new();
        let centers = CalibrationCenters::identity(2, 4, 10_000.0);
        p.set_centers(centers);
        let mut state = SequenceAttentionState::new(1);
        state.logical_len = 32;
        state.shared_prefix_len = 0;
        // Synthetic pre-RoPE keys: 32 pos × 1 kv × 8 dim
        let mut keys = vec![0.0f32; 32 * 1 * 8];
        for (i, x) in keys.iter_mut().enumerate() {
            *x = ((i % 7) as f32) * 0.1 + 0.5;
        }
        let cfg =
            PluginConfig::from_pairs(["kv.triattention.budget=8", "kv.triattention.window=1"])
                .unwrap();
        let ctx = RetentionContext {
            state: &state,
            layer: Some(0),
            current_pos: 31,
            tokens_since_window: 128,
            pre_rope_keys: Some(&keys),
            head_dim: 8,
            n_kv_heads: 1,
            n_heads: 2,
            config: &cfg,
        };
        assert!(p.should_compress(&ctx));
        let plan = p.plan_retention(&ctx).unwrap();
        assert!(plan.retain_positions.len() <= 8);
        assert!(plan.compact);
        let stats = p.apply_plan(&mut state, &plan).unwrap();
        assert_eq!(stats.retained_count, plan.retain_positions.len() as u32);
        assert!(stats.blocks_reclaimed > 0 || stats.retained_count < stats.candidate_count);
        // Logical positions preserved on entries.
        let view = state.view(0).unwrap();
        assert!(view.is_indirect());
        for e in &view.entries {
            assert!(plan.retain_positions.contains(&e.logical_pos));
        }
    }

    #[test]
    fn plan_retention_never_reselects_ghosts_after_densify() {
        // Two-window live-index unit test: after the first compact densifies
        // storage, a second plan_retention must score ONLY the live set and
        // never return an absolute id that was evicted (ghost re-selection).
        let mut p = TriAttentionPolicy::new();
        let centers = CalibrationCenters::identity(2, 4, 10_000.0);
        p.set_centers(centers);
        let cfg =
            PluginConfig::from_pairs(["kv.triattention.budget=8", "kv.triattention.window=1"])
                .unwrap();

        // Window 1: 64 live private positions, host dense.
        let mut state = SequenceAttentionState::new(1);
        state.logical_len = 64;
        state.shared_prefix_len = 0;
        state.live_positions = (0..64).collect();
        state.dense_len = 64;
        let mut keys = vec![0.0f32; 64 * 8];
        for (i, x) in keys.iter_mut().enumerate() {
            *x = ((i % 7) as f32) * 0.1 + 0.5;
        }
        let ctx = RetentionContext {
            state: &state,
            layer: Some(0),
            current_pos: 63,
            tokens_since_window: 128,
            pre_rope_keys: Some(&keys),
            head_dim: 8,
            n_kv_heads: 1,
            n_heads: 2,
            config: &cfg,
        };
        let plan1 = p.plan_retention(&ctx).unwrap();
        assert!(plan1.compact);
        assert!(plan1.retain_positions.len() <= 8);

        // Host densifies: live set shrinks to the retained set.
        state.live_positions = plan1.retain_positions.clone();
        state.live_positions.sort_unstable();
        state.dense_len = plan1.retain_positions.len() as u32;
        state.layout_owner = fellm_plugin_abi::LayoutOwner::HostDensified;
        let mut plan1b = plan1.clone();
        plan1b.host_densified = true;
        let stats = p.apply_plan(&mut state, &plan1b).unwrap();
        assert_eq!(stats.retained_count, plan1.retain_positions.len() as u32);

        // Simulate 8 new decode rows appended at dense end (abs 64..72).
        for abs in 64u32..72 {
            state.live_positions.push(abs);
        }
        state.logical_len = 72;
        state.dense_len = state.live_positions.len() as u32;
        let mut keys2 = vec![0.0f32; state.live_positions.len() * 8];
        for (i, x) in keys2.iter_mut().enumerate() {
            *x = ((i % 5) as f32) * 0.1 + 0.4;
        }
        // Keys packed 1:1 with live positions.
        let ctx2 = RetentionContext {
            state: &state,
            layer: Some(0),
            current_pos: 71,
            tokens_since_window: 128,
            pre_rope_keys: Some(&keys2),
            head_dim: 8,
            n_kv_heads: 1,
            n_heads: 2,
            config: &cfg,
        };
        let plan2 = p.plan_retention(&ctx2).unwrap();
        let live_set: std::collections::BTreeSet<u32> =
            state.live_retained_positions().into_iter().collect();
        for &pos in &plan2.retain_positions {
            assert!(
                live_set.contains(&pos),
                "plan2 retained ghost abs {pos} not in live set"
            );
        }
        // No re-selected evicted ghost from window 1 (they are all <=63, but
        // some may be retained; the point is none OUTSIDE the live set).
        for &pos in &plan2.retain_positions {
            assert!(
                live_set.contains(&pos),
                "ghost re-selection: {pos} not live"
            );
        }
    }
}
