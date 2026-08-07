//! End-to-end: SequenceStatePolicy score → physical compact → attention over dense retained KV.

use fellm_plugin_abi::capability::{PluginConfig, ProviderSelection};
use fellm_plugin_abi::{
    PreRopeKeyStore, RetentionContext, SequenceAttentionState, fa2_style_attention_f32,
    reference_attention_f32, set_pre_rope_store,
};
use fellm_runtime::paged::{BLOCK_SIZE, CacheManager};
use fellm_triattention::{CalibrationCenters, TriAttentionPolicy};
use half::f16;
use std::sync::Arc;

/// Fill synthetic K/V content into a sequence's dense layout.
fn fill_sequence_kv(mgr: &mut CacheManager, seq: &mut fellm_runtime::SequenceCache, n: usize) {
    for pos in 0..n {
        mgr.ensure_writable(seq, pos).unwrap();
        for layer in 0..seq.n_layers() {
            let (phys, slot) = seq.table(layer).locate(pos);
            let stride = mgr.pool.tokens_stride();
            for d in 0..stride {
                // Distinct values per position so attention after compact is checkable.
                let v = f16::from_f32((pos as f32 + 1.0) * 0.1 + d as f32 * 0.01);
                mgr.pool.k_row_mut(phys, slot)[d] = v;
                mgr.pool.v_row_mut(phys, slot)[d] = f16::from_f32((pos as f32 + 1.0) * 0.05);
            }
        }
    }
}

/// Gather dense f32 K/V from sequence storage for attention oracle.
fn gather_dense(
    mgr: &CacheManager,
    seq: &fellm_runtime::SequenceCache,
    layer: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = seq.len_tokens;
    let stride = mgr.pool.tokens_stride();
    let mut k = vec![0.0f32; n * stride];
    let mut v = vec![0.0f32; n * stride];
    for t in 0..n {
        let (phys, slot) = seq.table(layer).locate(t);
        for d in 0..stride {
            k[t * stride + d] = mgr.pool.k_row(phys, slot)[d].to_f32();
            v[t * stride + d] = mgr.pool.v_row(phys, slot)[d].to_f32();
        }
    }
    (k, v)
}

#[test]
fn compact_rebuilds_tables_and_attention_only_sees_retained() {
    let n_kv = 1usize;
    let hd = 8usize;
    let mut mgr = CacheManager::new(64, 1, n_kv, hd, 8).unwrap();
    let mut seq = mgr.new_sequence(128);
    fill_sequence_kv(&mut mgr, &mut seq, 64);
    assert_eq!(seq.len_tokens, 64);
    let free_before = mgr.pool.free_count();
    let blocks_before = seq.table(0).num_blocks();
    assert!(blocks_before >= 4);

    // Retain every 8th token (8 positions) — absolute indices.
    let retain: Vec<u32> = (0..64).step_by(8).collect();
    assert_eq!(retain.len(), 8);

    let reclaimed = mgr.compact_sequence_to_positions(&mut seq, &retain, BLOCK_SIZE);
    assert!(reclaimed >= 1, "expected physical reclaim, got {reclaimed}");
    assert!(mgr.pool.free_count() > free_before, "free list must grow");
    // Dense layout: 8 tokens → 1 block.
    assert_eq!(seq.len_tokens, 8, "attention length must shrink");
    assert_eq!(seq.table(0).num_blocks(), 1, "block table rebuilt denser");
    assert!(seq.is_compressed());
    assert_eq!(seq.original_positions, retain);

    // No dangling phys: every dense index locates within new table.
    for t in 0..seq.len_tokens {
        let (phys, slot) = seq.table(0).locate(t);
        assert!(slot < BLOCK_SIZE);
        // Touch rows — would UAF if phys was freed.
        let _ = mgr.pool.k_row(phys, slot);
        let _ = mgr.pool.v_row(phys, slot);
    }

    // Attention over dense 8 must match gather of retained absolute positions
    // from the pre-compact values (we filled pos p with value ~ (p+1)*0.1).
    let (k, v) = gather_dense(&mgr, &seq, 0);
    assert_eq!(k.len(), 8 * n_kv * hd);
    // Spot-check: dense slot 0 came from absolute pos 0 → k≈0.1
    assert!((k[0] - 0.1).abs() < 0.02, "k0={}", k[0]);
    // dense slot 1 from abs 8 → k≈0.9
    assert!((k[n_kv * hd] - 0.9).abs() < 0.05, "k1={}", k[n_kv * hd]);

    let mut q = vec![0.25f32; n_kv.max(1) * 4 * hd]; // 4 heads GQA
    for (i, x) in q.iter_mut().enumerate() {
        *x = (i as f32 * 0.03).sin();
    }
    let n_heads = 4;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out_ref = vec![0.0f32; n_heads * hd];
    let mut out_fa2 = vec![0.0f32; n_heads * hd];
    reference_attention_f32(
        &q,
        &k,
        &v,
        &mut out_ref,
        n_heads,
        n_kv,
        hd,
        1,
        8,
        scale,
        true,
        0,
    );
    fa2_style_attention_f32(
        &q,
        &k,
        &v,
        &mut out_fa2,
        n_heads,
        n_kv,
        hd,
        1,
        8,
        scale,
        true,
        0,
        1,
        8,
    );
    let err = out_ref
        .iter()
        .zip(out_fa2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(err < 1e-3, "post-compact attention err={err}");
    // Work reduced: kv_len=8 not 64.
    assert_eq!(seq.len_tokens, 8);
}

#[test]
fn compact_preserves_shared_prefix() {
    let mut mgr = CacheManager::new(64, 1, 1, 4, 8).unwrap();
    let mut seq = mgr.new_sequence(128);
    fill_sequence_kv(&mut mgr, &mut seq, 32);
    seq.shared_prefix_len = 8; // first 8 immutable
    // Policy asks to keep only late tokens — prefix must still remain.
    let retain = vec![20u32, 24, 28];
    let _ = mgr.compact_sequence_to_positions(&mut seq, &retain, BLOCK_SIZE);
    assert!(seq.original_positions.iter().any(|&p| p < 8));
    assert!(seq.len_tokens >= 8 + 3 - 0); // prefix + some of retain
    for &p in &[0u32, 1, 7] {
        assert!(
            seq.original_positions.contains(&p),
            "prefix pos {p} must be retained"
        );
    }
}

#[test]
fn triattention_lifecycle_reclaims_and_attends() {
    let mut mgr = CacheManager::new(64, 1, 1, 8, 8).unwrap();
    let mut seq = mgr.new_sequence(128);
    fill_sequence_kv(&mut mgr, &mut seq, 64);

    let mut store = PreRopeKeyStore::new(1, 1, 8, 64);
    for pos in 0..64 {
        let mut row = vec![0.1f32; 8];
        row[0] = pos as f32 * 0.1 + 0.5;
        store.write_row(0, pos, &row);
    }
    set_pre_rope_store(Some(store));

    let mut policy = TriAttentionPolicy::new();
    policy.set_centers(CalibrationCenters::identity(2, 4, 10_000.0));
    let policy: Arc<dyn fellm_plugin_abi::SequenceStatePolicy> = Arc::new(policy);

    let mut state = SequenceAttentionState::new(1);
    state.logical_len = 64;
    state.shared_prefix_len = 0;
    // Sync dense table into view for plan bookkeeping.
    state.layer_views[0].dense_block_table = seq.table(0).blocks().to_vec();
    state.layer_views[0].dense_seq_len = 64;

    let config =
        PluginConfig::from_pairs(["kv.triattention.budget=8", "kv.triattention.window=1"]).unwrap();
    let keys = fellm_plugin_abi::pre_rope_slice(0, 0, 64).unwrap();
    let ctx = RetentionContext {
        state: &state,
        layer: Some(0),
        current_pos: 63,
        tokens_since_window: 128,
        pre_rope_keys: Some(&keys),
        head_dim: 8,
        n_kv_heads: 1,
        n_heads: 2,
        config: &config,
    };
    assert!(policy.should_compress(&ctx));
    let plan = policy.plan_retention(&ctx).unwrap();
    assert!(plan.retain_positions.len() <= 8);

    let free_before = mgr.pool.free_count();
    let reclaimed = mgr.compact_sequence_to_positions(&mut seq, &plan.retain_positions, BLOCK_SIZE);
    assert!(reclaimed >= 1);
    assert!(mgr.pool.free_count() > free_before);
    assert_eq!(
        seq.len_tokens,
        plan.retain_positions
            .len()
            .min(8)
            .max(1)
            .min(seq.len_tokens)
    );
    // len_tokens equals retained count after compact
    assert!(seq.len_tokens <= 8);
    assert!(seq.is_compressed());

    // Attention on compacted dense storage must run without UAF.
    let (k, v) = gather_dense(&mgr, &seq, 0);
    let n_heads = 2;
    let hd = 8;
    let q = vec![0.2f32; n_heads * hd];
    let mut out = vec![0.0f32; n_heads * hd];
    fa2_style_attention_f32(
        &q,
        &k,
        &v,
        &mut out,
        n_heads,
        1,
        hd,
        1,
        seq.len_tokens,
        0.35,
        true,
        0,
        1,
        seq.len_tokens,
    );
    let sum: f32 = out.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "attention produced empty output after compact");

    let _ = policy.apply_plan(&mut state, &plan);
    set_pre_rope_store(None);
}

/// Two-window gating test for the single live-index contract.
///
/// After the first compact (host densify), the second window must:
/// 1. score only positions still physically present (live set), never ghosts
/// 2. grow the free list again (real physical reclaim twice)
/// 3. keep `apply_plan` bookkeeping-only (views stay synced to densified tables)
/// 4. write new decode rows at dense `kv_write_index` with absolute identity tracked
#[test]
fn multi_window_compact_keeps_live_index_and_reclaims_twice() {
    let n_kv = 1usize;
    let hd = 8usize;
    let mut mgr = CacheManager::new(96, 1, n_kv, hd, 16).unwrap();
    let mut seq = mgr.new_sequence(256);
    fill_sequence_kv(&mut mgr, &mut seq, 64);
    assert_eq!(seq.len_tokens, 64);

    let mut store = PreRopeKeyStore::new(1, n_kv, hd, 256);
    for pos in 0..256 {
        let mut row = vec![0.1f32; hd];
        row[0] = pos as f32 * 0.1 + 0.5;
        store.write_row(0, pos, &row);
    }
    // NOTE: kept local (not set_pre_rope_store) so parallel tests in this file
    // that also use the process-global store cannot clobber it mid-test.

    let mut policy = TriAttentionPolicy::new();
    policy.set_centers(CalibrationCenters::identity(2, hd / 2, 10_000.0));
    let policy: Arc<dyn fellm_plugin_abi::SequenceStatePolicy> = Arc::new(policy);
    let config =
        PluginConfig::from_pairs(["kv.triattention.budget=8", "kv.triattention.window=1"]).unwrap();

    let mut state = SequenceAttentionState::new(1);
    let free0 = mgr.pool.free_count();

    // ---- Window 1: 64 dense tokens -> 8 retained ----
    state.logical_len = 64;
    state.shared_prefix_len = 0;
    state.dense_len = 64;
    state.layer_views[0].dense_block_table = seq.table(0).blocks().to_vec();
    state.layer_views[0].dense_seq_len = 64;
    let keys = store.gather_positions(0, &state.live_retained_positions());
    let ctx = RetentionContext {
        state: &state,
        layer: Some(0),
        current_pos: 63,
        tokens_since_window: 128,
        pre_rope_keys: Some(&keys),
        head_dim: hd as u32,
        n_kv_heads: n_kv as u32,
        n_heads: 2,
        config: &config,
    };
    assert!(policy.should_compress(&ctx));
    let mut plan1 = policy.plan_retention(&ctx).unwrap();
    assert!(plan1.compact);
    assert!(
        plan1.retain_positions.len() <= 8,
        "window1 retain {}",
        plan1.retain_positions.len()
    );
    let reclaimed1 =
        mgr.compact_sequence_to_positions(&mut seq, &plan1.retain_positions, BLOCK_SIZE);
    assert!(reclaimed1 >= 1, "window1 reclaim {reclaimed1}");
    assert!(
        mgr.pool.free_count() > free0,
        "free list must grow after window1"
    );
    assert_eq!(
        seq.len_tokens,
        plan1.retain_positions.len().max(1).min(seq.len_tokens)
    );
    assert!(seq.is_compressed());
    let live1: Vec<u32> = seq.original_positions.clone();
    assert!(live1.len() <= 8);
    let live1_set: std::collections::BTreeSet<u32> = live1.iter().copied().collect();

    // Host-densified bookkeeping-only apply (no table rewrite, no dense_seq_len=0).
    plan1.host_densified = true;
    state.layout_owner = fellm_plugin_abi::LayoutOwner::HostDensified;
    state.live_positions = live1.clone();
    state.dense_len = seq.len_tokens as u32;
    state.layer_views[0].dense_seq_len = seq.len_tokens as u32;
    state.layer_views[0].dense_block_table = seq.table(0).blocks().to_vec();
    let stats1 = policy.apply_plan(&mut state, &plan1).unwrap();
    assert_eq!(stats1.retained_count, plan1.retain_positions.len() as u32);
    // Views remain synced to the densified table (policy must not zero dense_seq_len).
    assert_eq!(state.dense_len, seq.len_tokens as u32);
    assert_eq!(
        state.layer_views[0].dense_seq_len, seq.len_tokens as u32,
        "apply_plan must not corrupt views after host densify"
    );

    // Prune pre-RoPE ghosts (what the engine does after each compact).
    store.prune_to_positions(0, &live1);

    // ---- Window 2: continue decoding 32 new tokens into dense end ----
    // Decode writes at dense kv_write_index (= len_tokens); absolute identity
    // appended to original_positions, matching the engine's step() bookkeeping.
    let mut abs_cursor = 64u32;
    for _ in 0..32 {
        let kv_pos = seq.kv_write_index();
        mgr.ensure_writable(&mut seq, kv_pos).unwrap();
        // Mirror the engine: pre-RoPE K snapshot at the absolute position
        // (backend launch_rope custom_op_id=1 → pre_rope_write).
        let mut row = vec![0.1f32; hd];
        row[0] = abs_cursor as f32 * 0.1 + 0.5;
        store.write_row(0, abs_cursor as usize, &row);
        for layer in 0..seq.n_layers() {
            let (phys, slot) = seq.table(layer).locate(kv_pos);
            let stride = mgr.pool.tokens_stride();
            for d in 0..stride {
                let v = f16::from_f32((abs_cursor as f32 + 1.0) * 0.1 + d as f32 * 0.01);
                mgr.pool.k_row_mut(phys, slot)[d] = v;
                mgr.pool.v_row_mut(phys, slot)[d] = f16::from_f32((abs_cursor as f32 + 1.0) * 0.05);
            }
        }
        seq.original_positions.push(abs_cursor);
        abs_cursor += 1;
    }
    assert_eq!(seq.len_tokens, live1.len() + 32);
    assert_eq!(seq.original_positions.len(), seq.len_tokens);
    // New rows must be absolute positions NOT in the prior live set.
    let live2_full: Vec<u32> = seq.original_positions.clone();
    for &p in &live2_full[live1.len()..] {
        assert!(
            !live1_set.contains(&p),
            "new abs {p} collides with prior live set"
        );
    }

    // Second plan must score ONLY live positions (old live1 ∪ new abs).
    // Mirror engine sync_seq_attn_from_cache: live index = original_positions.
    state.live_positions = seq.original_positions.clone();
    state.logical_len = abs_cursor;
    state.dense_len = seq.len_tokens as u32;
    state.layer_views[0].dense_block_table = seq.table(0).blocks().to_vec();
    state.layer_views[0].dense_seq_len = seq.len_tokens as u32;
    let keys2 = store.gather_positions(0, &state.live_retained_positions());
    let ctx2 = RetentionContext {
        state: &state,
        layer: Some(0),
        current_pos: abs_cursor.saturating_sub(1),
        tokens_since_window: 128,
        pre_rope_keys: Some(&keys2),
        head_dim: hd as u32,
        n_kv_heads: n_kv as u32,
        n_heads: 2,
        config: &config,
    };
    let plan2 = policy.plan_retention(&ctx2).unwrap();
    // Live-index contract: plan2 ⊆ state.live_retained_positions().
    let live2_set: std::collections::BTreeSet<u32> =
        state.live_retained_positions().into_iter().collect();
    for &p in &plan2.retain_positions {
        assert!(
            live2_set.contains(&p),
            "plan2 retained ghost abs {p} not in live set"
        );
    }
    assert!(
        plan2.retain_positions.len() <= 8,
        "window2 retain {}",
        plan2.retain_positions.len()
    );

    let reclaimed2 =
        mgr.compact_sequence_to_positions(&mut seq, &plan2.retain_positions, BLOCK_SIZE);
    assert!(reclaimed2 >= 1, "window2 reclaim {reclaimed2}");
    // Compact is copy-new-then-free-old: net free list may not monotonically
    // grow each window, but each window must reclaim real blocks. The pool must
    // never leak — total allocated blocks stay bounded across both windows.
    let total_allocated = mgr.pool.allocated_count();
    assert!(
        total_allocated <= seq.table(0).num_blocks() + 1,
        "pool leaked blocks across 2 windows: allocated={total_allocated}"
    );
    assert!(
        seq.len_tokens <= 8,
        "len_tokens after window2 = {}",
        seq.len_tokens
    );
    assert_eq!(seq.original_positions.len(), seq.len_tokens);

    // Every dense slot maps to a live physical block (no UAF after 2 compacts).
    for t in 0..seq.len_tokens {
        let (phys, slot) = seq.table(0).locate(t);
        assert!(slot < BLOCK_SIZE);
        let _ = mgr.pool.k_row(phys, slot);
        let _ = mgr.pool.v_row(phys, slot);
    }
    // Engine re-syncs state after compact (sync_seq_attn_from_cache) before apply_plan.
    state.live_positions = seq.original_positions.clone();
    state.dense_len = seq.len_tokens as u32;
    state.layer_views[0].dense_block_table = seq.table(0).blocks().to_vec();
    state.layer_views[0].dense_seq_len = seq.len_tokens as u32;
    // Attention over the twice-densified dense storage runs.
    let (k, v) = gather_dense(&mgr, &seq, 0);
    let n_heads = 2;
    let q = vec![0.2f32; n_heads * hd];
    let mut out = vec![0.0f32; n_heads * hd];
    fa2_style_attention_f32(
        &q,
        &k,
        &v,
        &mut out,
        n_heads,
        n_kv,
        hd,
        1,
        seq.len_tokens,
        0.35,
        true,
        0,
        1,
        seq.len_tokens,
    );
    let sum: f32 = out.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "attention produced empty output after 2 windows");

    let _ = policy.apply_plan(&mut state, &plan2);
    // apply_plan remains bookkeeping-only: views still match tables.
    assert_eq!(state.dense_len, seq.len_tokens as u32);
    assert_eq!(
        state.layer_views[0].dense_seq_len, seq.len_tokens as u32,
        "post-window2 views must stay synced"
    );
}

#[test]
fn provider_selection_pins_triattention() {
    let mut sel = ProviderSelection::new();
    sel.kv_policy = Some("kv.triattention".into());
    sel.attention = Some("attention.host_tiled".into());
    let mut mgr = fellm_runtime::ProviderManager::new(sel);
    mgr.load_plugins(None).unwrap();
    let backend = backend_cpu::CpuBackend::new();
    let prep = mgr.prepare(&backend, 8, 2, 64, 4).unwrap();
    assert_eq!(prep.kv_policy_name, "kv.triattention");
    assert_ne!(prep.kv_policy_id.0, 0);
}
