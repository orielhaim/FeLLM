//! KV Fabric: logical, shareable, movable, policy-driven KV memory subsystem.
//!
//! Layers:
//! 1. **Logical identity** — sequences own [`KvPageId`] / segments / groups
//! 2. **Mapping** — logical pages resolve to physical slots
//! 3. **Residency** — Device / HostPinned / Host / Disk / NotResident
//! 4. **Sharing** — content-addressed store with refcount + CoW
//! 5. **Addressing** — BlockTable vs VirtualMemory backends
//! 6. **Policies** — Exact/Elastic, encoding, value/cost residency

pub mod addressing;
pub mod dummy;
pub mod fabric;
pub mod mapping;
pub mod policy;
pub mod share;
pub mod storage;
pub mod types;

pub use dummy::DummyKvBuffers;
pub use fabric::KvFabric;
pub use share::PrefixCacheStats;
pub use types::*;

/// Tokens per standard page (kernel-compatible default geometry).
pub const BLOCK_SIZE: usize = STANDARD_PAGE_TOKENS;

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn fabric(pages: usize, layers: usize) -> KvFabric {
        KvFabric::new_full_attention(
            KvFabricConfig {
                device_budget: Some((pages * 4096) as u64),
                host_budget: Some(pages as u64 * 4096),
                ..KvFabricConfig::default()
            },
            pages,
            layers,
            1,
            4,
            pages,
        )
        .unwrap()
    }

    #[test]
    fn provisional_suffix_commit_and_rollback_reclaim_pages() {
        let mut fabric = fabric(16, 2);
        let mut sequence = fabric.new_sequence(128);
        for position in 0..16 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        let baseline_free = fabric.free_count();

        let mut rollback = fabric.begin_transaction(&sequence);
        for position in 16..48 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        assert!(fabric.free_count() < baseline_free);
        fabric
            .rollback_transaction(&mut sequence, &mut rollback)
            .unwrap();
        assert_eq!(sequence.len_tokens, 16);
        assert_eq!(fabric.free_count(), baseline_free);
        assert!(
            fabric
                .rollback_transaction(&mut sequence, &mut rollback)
                .is_err()
        );

        let mut commit = fabric.begin_transaction(&sequence);
        for position in 16..48 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        fabric
            .commit_transaction(&mut sequence, &mut commit, 16)
            .unwrap();
        assert_eq!(sequence.len_tokens, 32);
        assert_eq!(fabric.free_count(), baseline_free - 2);
    }

    #[test]
    fn invalid_commit_remains_rollbackable_and_all_prefix_lengths_reuse_pages() {
        let mut fabric = fabric(16, 2);
        let mut sequence = fabric.new_sequence(128);
        for position in 0..16 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        sequence.absolute_pos = 16;
        let baseline_free = fabric.free_count();

        let mut invalid = fabric.begin_transaction(&sequence);
        for position in 16..24 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        assert!(
            fabric
                .commit_transaction(&mut sequence, &mut invalid, 9)
                .is_err()
        );
        fabric
            .rollback_transaction(&mut sequence, &mut invalid)
            .unwrap();
        assert_eq!((sequence.len_tokens, sequence.absolute_pos), (16, 16));
        assert_eq!(fabric.free_count(), baseline_free);

        for accepted in [0, 4, 8] {
            let mut transaction = fabric.begin_transaction(&sequence);
            for position in 16..24 {
                fabric.ensure_writable(&mut sequence, position).unwrap();
            }
            if accepted == 0 {
                fabric
                    .rollback_transaction(&mut sequence, &mut transaction)
                    .unwrap();
            } else {
                fabric
                    .commit_transaction(&mut sequence, &mut transaction, accepted)
                    .unwrap();
            }
            assert_eq!(sequence.len_tokens, 16 + accepted);
            fabric.truncate_sequence(&mut sequence, 16).unwrap();
            sequence.absolute_pos = 16;
            assert_eq!(fabric.free_count(), baseline_free);
        }
    }

    #[test]
    fn commit_restores_absolute_cursor_to_accepted_prefix() {
        let mut fabric = fabric(16, 2);
        let mut sequence = fabric.new_sequence(128);
        for position in 0..8 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        sequence.absolute_pos = 8;
        let mut transaction = fabric.begin_transaction(&sequence);
        for position in 8..12 {
            fabric.ensure_writable(&mut sequence, position).unwrap();
        }
        sequence.absolute_pos = 12;
        fabric
            .commit_transaction(&mut sequence, &mut transaction, 2)
            .unwrap();
        assert_eq!((sequence.len_tokens, sequence.absolute_pos), (10, 10));
    }

    #[test]
    fn ragged_block_tables_are_rectangular_without_aliasing_live_pages() {
        let mut fabric = fabric(32, 2);
        let mut short = fabric.new_sequence(128);
        let mut long = fabric.new_sequence(128);
        fabric.ensure_writable(&mut short, 0).unwrap();
        fabric.ensure_writable(&mut long, 32).unwrap();

        let short_table = fabric.physical_block_table_padded(&short, 3).unwrap();
        let long_table = fabric.physical_block_table_padded(&long, 3).unwrap();
        assert_eq!(short_table.len(), 6);
        assert_eq!(long_table.len(), 6);
        for layer in 0..2 {
            let short_layer = &short_table[layer * 3..layer * 3 + 3];
            assert_eq!(short_layer, &[short_layer[0]; 3]);
            let expected = fabric.physical_block_table_layer(&long, layer);
            assert_eq!(&long_table[layer * 3..layer * 3 + 3], expected);
        }
    }

    #[test]
    fn logical_pages_not_physical_identity() {
        let mut fab = fabric(16, 1);
        let mut seq = fab.new_sequence(64);
        fab.ensure_writable(&mut seq, 0).unwrap();
        let pid = seq.layer_map(0).page(0);
        let slot = fab.resolve(pid).unwrap();
        // Sequence holds logical identity; fabric maps it to a physical slot.
        assert_eq!(pid.0, 0);
        assert!(fab.resolve(pid).is_some());
        let _ = slot;
        fab.assert_invariants(&[&seq]);
    }

    #[test]
    fn cow_preserves_shared_immutable() {
        let mut fab = fabric(32, 1);
        let tokens: Vec<u32> = (0..32).collect();
        let mut source = fab.new_sequence(64);
        for p in 0..32 {
            fab.ensure_writable(&mut source, p).unwrap();
            let (phys, slot) = fab.locate(&source, 0, p).unwrap();
            fab.k_row_mut(phys, slot)[0] = f16::from_f32(p as f32);
        }
        fab.insert_prefix(&tokens, &source);
        let source_page0 = source.layer_map(0).page(0);
        let source_slot_before = fab.resolve(source_page0).unwrap();
        let source_val = fab.k_row(source_slot_before, 0)[0].to_f32();

        let mut target = fab.new_sequence(64);
        let matched = fab.attach_prefix(&tokens, &mut target);
        assert_eq!(matched, 32);
        assert_eq!(target.layer_map(0).page(0), source_page0);

        // Writing to target must CoW into a new logical page + physical slot.
        fab.ensure_writable(&mut target, 0).unwrap();
        let target_page0 = target.layer_map(0).page(0);
        assert_ne!(target_page0, source_page0, "CoW must fork logical identity");
        let target_slot = fab.resolve(target_page0).unwrap();
        fab.k_row_mut(target_slot, 0)[0] = f16::from_f32(999.0);

        let source_slot_after = fab.resolve(source_page0).unwrap();
        assert_eq!(source_slot_before, source_slot_after);
        assert!((fab.k_row(source_slot_after, 0)[0].to_f32() - source_val).abs() < 0.01);
        assert!((fab.k_row(target_slot, 0)[0].to_f32() - 999.0).abs() < 0.01);
        fab.assert_invariants(&[&source, &target]);
    }

    #[test]
    fn migration_preserves_contents() {
        let mut fab = fabric(16, 1);
        let mut seq = fab.new_sequence(32);
        fab.ensure_writable(&mut seq, 0).unwrap();
        let (phys, slot) = fab.locate(&seq, 0, 0).unwrap();
        fab.k_row_mut(phys, slot)[0] = f16::from_f32(42.0);
        fab.migrate_out(&mut seq).unwrap();
        assert!(seq.non_resident);
        fab.migrate_in(&mut seq).unwrap();
        assert!(!seq.non_resident);
        let (phys2, slot2) = fab.locate(&seq, 0, 0).unwrap();
        assert!((fab.k_row(phys2, slot2)[0].to_f32() - 42.0).abs() < 0.01);
        fab.assert_invariants(&[&seq]);
    }

    #[test]
    fn compact_reclaims_and_keeps_live_index() {
        let mut fab = fabric(64, 1);
        let mut seq = fab.new_sequence(128);
        for pos in 0..64 {
            fab.ensure_writable(&mut seq, pos).unwrap();
            let (phys, slot) = fab.locate(&seq, 0, pos).unwrap();
            fab.k_row_mut(phys, slot)[0] = f16::from_f32((pos as f32 + 1.0) * 0.1);
        }
        let free_before = fab.free_count();
        let retain: Vec<u32> = (0..64).step_by(8).collect();
        let reclaimed = fab.compact_sequence_to_positions(&mut seq, &retain, BLOCK_SIZE);
        assert!(reclaimed >= 1);
        assert!(fab.free_count() > free_before);
        assert_eq!(seq.len_tokens, 8);
        assert!(seq.is_compressed());
        assert_eq!(seq.original_positions, retain);
        fab.assert_invariants(&[&seq]);
    }

    #[test]
    fn encoding_not_hardcoded_invariant() {
        let cfg = KvFabricConfig {
            default_encoding: KvEncoding::Bf16,
            device_budget: Some(1 << 20),
            ..KvFabricConfig::default()
        };
        let fab = KvFabric::new_full_attention(cfg, 8, 1, 1, 4, 0).unwrap();
        assert_eq!(fab.encoding(), KvEncoding::Bf16);
        assert_eq!(KvEncoding::Fp8.elem_bytes(), 1);
        assert_eq!(KvPageClass::Super.tokens(), 64);
        assert_eq!(KvPageClass::Micro.tokens(), 4);
    }

    #[test]
    fn memory_plan_resolves_pages() {
        let config = KvFabricConfig {
            device_budget: Some(10_000),
            host_budget: Some(4_096),
            ..KvFabricConfig::default()
        };
        let plan = KvMemoryPlan::resolve(
            &config,
            None,
            20,
            30,
            1024,
            2,
            KvExecutionMemory::Accelerator,
        )
        .unwrap();
        assert_eq!(plan.execution_pages, 9);
        assert_eq!(plan.execution_bytes, 9_216);
        assert_eq!(plan.overflow_host_pages, 4);

        let host_plan =
            KvMemoryPlan::resolve(&config, None, 20, 30, 1024, 2, KvExecutionMemory::Host).unwrap();
        assert_eq!(host_plan.execution_pages, 4);
        assert_eq!(host_plan.overflow_host_pages, 0);
    }

    #[test]
    fn prefix_match_stats() {
        let mut fab = fabric(32, 1);
        let tokens: Vec<u32> = (0..20).collect();
        let mut source = fab.new_sequence(64);
        for p in 0..20 {
            fab.ensure_writable(&mut source, p).unwrap();
        }
        fab.insert_prefix(&tokens, &source);
        let mut target = fab.new_sequence(64);
        assert_eq!(fab.attach_prefix(&tokens, &mut target), 16);
        let stats = fab.prefix_stats();
        assert_eq!(stats.hit_tokens, 16);
        assert_eq!(stats.miss_tokens, 4);
    }

    #[test]
    fn residency_signals_for_scheduler() {
        let mut fab = fabric(16, 2);
        let mut seq = fab.new_sequence(64);
        fab.ensure_writable(&mut seq, 0).unwrap();
        let sig = fab.residency_signals(&seq, 8, 1, 0);
        assert!(sig.resident_pages >= 2);
        assert_eq!(sig.priority, 1);
        assert!(sig.memory_pressure >= 0.0);
    }

    #[test]
    fn value_cost_policy_is_primary() {
        let fab = fabric(8, 1);
        assert_eq!(fab.config.residency_policy, ResidencyPolicyKind::ValueCost);
        let ranks = fab.rank_eviction_candidates(4);
        assert!(ranks.len() <= 4);
    }

    #[test]
    fn elastic_encoding_policy_tiers_by_age() {
        let cfg = KvFabricConfig {
            mode: KvMode::Elastic,
            encoding_policy: KvEncodingPolicy::TemperatureTiered,
            default_encoding: KvEncoding::Fp16,
            device_budget: Some(1 << 20),
            ..KvFabricConfig::default()
        };
        let fab = KvFabric::new_full_attention(cfg, 8, 1, 1, 4, 0).unwrap();
        assert_eq!(fab.select_encoding_for_segment(0, true), KvEncoding::Fp16);
        assert_eq!(fab.select_encoding_for_segment(100, false), KvEncoding::Fp8);
        assert_eq!(
            fab.select_encoding_for_segment(1000, false),
            KvEncoding::Int8
        );
        assert_eq!(
            fab.select_encoding_for_segment(4096, false),
            KvEncoding::Int4
        );
    }

    #[test]
    fn migrate_out_rolls_back_on_host_exhaustion() {
        // Host tier sized for exactly 1 page; sequence needs 2 pages → second stash fails.
        let mut fab = KvFabric::new_full_attention(
            KvFabricConfig {
                device_budget: Some(8 * 1024 * 1024),
                host_budget: Some(1), // will resolve to 0 host pages via plan, force via constructor
                ..KvFabricConfig::default()
            },
            16,
            1,
            1,
            4,
            1, // only 1 host page
        )
        .unwrap();
        assert_eq!(fab.host_free_count(), 1);
        let mut seq = fab.new_sequence(64);
        // 32 tokens → 2 standard pages
        for p in 0..32 {
            fab.ensure_writable(&mut seq, p).unwrap();
            let (phys, slot) = fab.locate(&seq, 0, p).unwrap();
            fab.k_row_mut(phys, slot)[0] = f16::from_f32(p as f32 + 1.0);
        }
        assert_eq!(seq.layer_map(0).num_pages(), 2);
        let free_before = fab.free_count();
        let host_before = fab.host_free_count();
        let err = fab.migrate_out(&mut seq);
        assert!(
            err.is_err(),
            "must fail when host tier cannot hold all pages"
        );
        assert!(
            !seq.non_resident,
            "failed migrate must not mark sequence non-resident"
        );
        // Both logical pages still device-bound.
        assert!(fab.resolve(seq.layer_map(0).page(0)).is_some());
        assert!(fab.resolve(seq.layer_map(0).page(1)).is_some());
        // locate must still work for all tokens (no silent slot-0 alias).
        for p in 0..32 {
            let (phys, slot) = fab.locate(&seq, 0, p).unwrap();
            assert!((fab.k_row(phys, slot)[0].to_f32() - (p as f32 + 1.0)).abs() < 0.01);
        }
        assert_eq!(fab.free_count(), free_before, "device free list unchanged");
        assert_eq!(
            fab.host_free_count(),
            host_before,
            "host stashes rolled back"
        );
        fab.assert_invariants(&[&seq]);
    }

    #[test]
    fn locate_errors_on_unbound_page() {
        let mut fab = fabric(16, 1);
        let mut seq = fab.new_sequence(64);
        for p in 0..16 {
            fab.ensure_writable(&mut seq, p).unwrap();
        }
        fab.migrate_out(&mut seq).unwrap();
        assert!(seq.non_resident);
        let err = fab.locate(&seq, 0, 0);
        assert!(err.is_err(), "unbound page must not resolve to physical 0");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("unbound") || msg.contains("non-resident"),
            "msg={msg}"
        );
    }

    #[test]
    fn release_after_migrate_frees_host_tier() {
        let mut fab = fabric(16, 1);
        let mut seq = fab.new_sequence(64);
        for p in 0..16 {
            fab.ensure_writable(&mut seq, p).unwrap();
        }
        let host_full = fab.host_free_count();
        fab.migrate_out(&mut seq).unwrap();
        assert!(fab.host_free_count() < host_full);
        fab.release_sequence(&mut seq);
        assert_eq!(
            fab.host_free_count(),
            host_full,
            "release must drop host stashes after migrate"
        );
    }

    #[test]
    fn sequence_keep_value_ranks_low_priority_cheaper() {
        let mut fab = fabric(32, 1);
        let mut hot = fab.new_sequence(64);
        let mut cold = fab.new_sequence(64);
        for p in 0..16 {
            fab.ensure_writable(&mut hot, p).unwrap();
            fab.ensure_writable(&mut cold, p).unwrap();
        }
        // Touch hot pages so access_count / last_used rise.
        for _ in 0..5 {
            fab.tick();
            let _ = fab.ensure_writable(&mut hot, 0);
        }
        let clock = fab.tick();
        let hot_keep = fab.sequence_keep_value(&hot, 10, clock);
        let cold_keep = fab.sequence_keep_value(&cold, 0, 0);
        assert!(
            hot_keep > cold_keep,
            "hot/high-priority keep={hot_keep} cold={cold_keep}"
        );
    }

    #[test]
    fn fabric_metrics_track_share_and_cow() {
        let mut fab = fabric(32, 1);
        let tokens: Vec<u32> = (0..16).collect();
        let mut source = fab.new_sequence(64);
        for p in 0..16 {
            fab.ensure_writable(&mut source, p).unwrap();
        }
        fab.insert_prefix(&tokens, &source);
        let mut target = fab.new_sequence(64);
        assert_eq!(fab.attach_prefix(&tokens, &mut target), 16);
        fab.ensure_writable(&mut target, 0).unwrap();
        let m = fab.metrics();
        assert!(m.cow_forks >= 1, "cow_forks={}", m.cow_forks);
        assert!(m.shared_objects >= 1, "shared_objects={}", m.shared_objects);
        assert!(m.prefix_hits >= 1, "prefix_hits={}", m.prefix_hits);
        assert_eq!(m.total_device_pages, fab.n_pages());
        assert_eq!(
            m.free_device_pages + m.device_resident_pages,
            m.total_device_pages
        );
    }
}
