//! Fabric lifecycle: compact, share/CoW, migration, admission signals.

use fellm_runtime::{BLOCK_SIZE, KvFabric, KvFabricConfig, KvSequence};
use half::f16;

fn make_fabric(pages: usize, n_kv: usize, hd: usize, host: usize) -> KvFabric {
    KvFabric::new_full_attention(
        KvFabricConfig {
            device_budget: Some((pages as u64) * 64 * 1024),
            host_budget: Some((host as u64) * 64 * 1024),
            ..KvFabricConfig::default()
        },
        pages,
        1,
        n_kv,
        hd,
        host,
    )
    .unwrap()
}

fn fill_sequence_kv(fab: &mut KvFabric, seq: &mut KvSequence, n: usize) {
    for pos in 0..n {
        fab.ensure_writable(seq, pos).unwrap();
        for layer in 0..seq.n_layers() {
            let (phys, slot) = fab.locate(seq, layer, pos).unwrap();
            let stride = fab.tokens_stride();
            for d in 0..stride {
                let v = f16::from_f32((pos as f32 + 1.0) * 0.1 + d as f32 * 0.01);
                fab.k_row_mut(phys, slot)[d] = v;
                fab.v_row_mut(phys, slot)[d] = f16::from_f32((pos as f32 + 1.0) * 0.05);
            }
        }
    }
}

fn gather_dense(fab: &KvFabric, seq: &KvSequence, layer: usize) -> (Vec<f32>, Vec<f32>) {
    let n = seq.len_tokens;
    let stride = fab.tokens_stride();
    let mut k = vec![0.0f32; n * stride];
    let mut v = vec![0.0f32; n * stride];
    for t in 0..n {
        let (phys, slot) = fab.locate(seq, layer, t).unwrap();
        for d in 0..stride {
            k[t * stride + d] = fab.k_row(phys, slot)[d].to_f32();
            v[t * stride + d] = fab.v_row(phys, slot)[d].to_f32();
        }
    }
    (k, v)
}

#[test]
fn compact_rebuilds_maps_and_preserves_retained_values() {
    let n_kv = 1usize;
    let hd = 8usize;
    let mut fab = make_fabric(64, n_kv, hd, 8);
    let mut seq = fab.new_sequence(128);
    fill_sequence_kv(&mut fab, &mut seq, 64);
    assert_eq!(seq.len_tokens, 64);
    let free_before = fab.free_count();
    assert!(seq.layer_map(0).num_pages() >= 4);

    let retain: Vec<u32> = (0..64).step_by(8).collect();
    let reclaimed = fab.compact_sequence_to_positions(&mut seq, &retain, BLOCK_SIZE);
    assert!(reclaimed >= 1, "expected physical reclaim, got {reclaimed}");
    assert!(fab.free_count() > free_before, "free list must grow");
    assert_eq!(seq.len_tokens, 8);
    assert_eq!(seq.layer_map(0).num_pages(), 1);
    assert!(seq.is_compressed());
    assert_eq!(seq.original_positions, retain);

    for t in 0..seq.len_tokens {
        let (phys, slot) = fab.locate(&seq, 0, t).unwrap();
        assert!(slot < BLOCK_SIZE);
        let _ = fab.k_row(phys, slot);
    }

    let (k, _v) = gather_dense(&fab, &seq, 0);
    assert_eq!(k.len(), 8 * n_kv * hd);
    assert!((k[0] - 0.1).abs() < 0.02, "k0={}", k[0]);
    assert!((k[n_kv * hd] - 0.9).abs() < 0.05, "k1={}", k[n_kv * hd]);
    fab.assert_invariants(&[&seq]);
}

#[test]
fn shared_prefix_cow_and_release() {
    let mut fab = make_fabric(64, 1, 4, 8);
    let tokens: Vec<u32> = (0..32).collect();
    let mut source = fab.new_sequence(64);
    fill_sequence_kv(&mut fab, &mut source, 32);
    fab.insert_prefix(&tokens, &source);
    let first = source.layer_map(0).page(0);
    assert!(fab.page_refcount(first) >= 2);
    fab.release_sequence(&mut source);
    assert_eq!(fab.page_refcount(first), 1);
    let before = fab.free_count();
    assert!(fab.evict_shared_until(before + 1) > 0);
    assert!(fab.free_count() > before);
}

#[test]
fn fabric_migration_roundtrip_preserves_contents() {
    let mut fab = make_fabric(32, 1, 4, 16);
    let mut seq = fab.new_sequence(64);
    fill_sequence_kv(&mut fab, &mut seq, 16);
    let (phys, slot) = fab.locate(&seq, 0, 3).unwrap();
    let expected = fab.k_row(phys, slot)[0].to_f32();
    let free_before = fab.free_count();
    fab.migrate_out(&mut seq).unwrap();
    assert!(seq.non_resident);
    assert!(
        fab.free_count() > free_before,
        "hard migrate must free device pages"
    );
    fab.migrate_in(&mut seq).unwrap();
    assert!(!seq.non_resident);
    let (phys2, slot2) = fab.locate(&seq, 0, 3).unwrap();
    assert!((fab.k_row(phys2, slot2)[0].to_f32() - expected).abs() < 0.02);
    fab.assert_invariants(&[&seq]);
}

#[test]
fn admission_signals_and_can_admit() {
    let mut fab = make_fabric(16, 2, 4, 8);
    let mut seq = fab.new_sequence(64);
    fill_sequence_kv(&mut fab, &mut seq, 8);
    let sig = fab.residency_signals(&seq, 16, 3, 0);
    assert!(sig.resident_pages >= 1);
    assert_eq!(sig.priority, 3);
    assert!(fab.can_admit(1));
    let need = fab.n_pages() + 1;
    assert!(!fab.can_admit(need));
    assert!(sig.pages_needed_next >= 1 || sig.resident_pages >= 1);
}

#[test]
fn logical_identity_independent_of_physical_slot() {
    let mut fab = make_fabric(16, 1, 4, 0);
    let mut seq = fab.new_sequence(32);
    fab.ensure_writable(&mut seq, 0).unwrap();
    let pid = seq.layer_map(0).page(0);
    let slot_a = fab.resolve(pid).unwrap();
    // CoW path: bump ref by insert+attach so next write forks physical slot.
    let tokens = vec![1u32; 16];
    // Fill full page so prefix can index it.
    for p in 1..16 {
        fab.ensure_writable(&mut seq, p).unwrap();
    }
    fab.insert_prefix(&tokens, &seq);
    let mut other = fab.new_sequence(32);
    assert_eq!(fab.attach_prefix(&tokens, &mut other), 16);
    fab.ensure_writable(&mut other, 0).unwrap();
    let slot_b = fab.resolve(other.layer_map(0).page(0)).unwrap();
    // After CoW, other has a different physical slot; logical page id is its own.
    assert_ne!(slot_a, slot_b);
    fab.assert_invariants(&[&seq, &other]);
}
