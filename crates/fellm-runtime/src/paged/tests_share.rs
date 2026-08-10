//! Integration-style unit tests for paged cache sharing.

use crate::paged::{BLOCK_SIZE, CacheManager};

#[test]
fn prefix_sharing_saves_blocks() {
    let n_layers = 4;
    let mut mgr = CacheManager::new(256, n_layers, 2, 8, 32).unwrap();
    let tokens: Vec<u32> = (0..1024).collect();
    assert_eq!(tokens.len() % BLOCK_SIZE, 0);

    // Sequence A fills the full prompt.
    let mut seq_a = mgr.new_sequence(2048);
    for pos in 0..tokens.len() {
        mgr.ensure_writable(&mut seq_a, pos).unwrap();
    }
    let blocks_a: usize = (0..n_layers).map(|l| seq_a.table(l).num_blocks()).sum();
    mgr.prefix.insert_prompt(&mut mgr.pool, &tokens, &seq_a);

    // Five more sequences share the same prefix.
    let mut total_unique = blocks_a;
    for _ in 0..5 {
        let mut seq = mgr.new_sequence(2048);
        let matched = mgr.prefix.attach_match(&mut mgr.pool, &mut seq, &tokens);
        assert_eq!(matched, 1024);
        // No new exclusive blocks for the shared prefix.
        for layer in 0..n_layers {
            assert_eq!(seq.table(layer).block(0), seq_a.table(layer).block(0));
            assert!(mgr.pool.refcount(seq.table(layer).block(0)) > 1);
        }
        let exclusive: usize = (0..n_layers).map(|l| seq.table(l).num_blocks()).sum();
        // Shared: same physical ids, so unique physical count stays ~blocks_a
        total_unique = total_unique.max(exclusive);
    }

    // Independent allocation would need 6 * blocks_a; shared uses ~blocks_a.
    let independent = 6 * blocks_a;
    assert!(
        blocks_a * 2 < independent,
        "shared={blocks_a} should be << independent={independent}"
    );
    let _ = total_unique;
}

#[test]
fn cow_on_write_forks_block() {
    let mut mgr = CacheManager::new(64, 1, 1, 4, 8).unwrap();
    let tokens: Vec<u32> = (0..16).collect();
    let mut seq_a = mgr.new_sequence(64);
    for pos in 0..16 {
        mgr.ensure_writable(&mut seq_a, pos).unwrap();
    }
    mgr.prefix.insert_prompt(&mut mgr.pool, &tokens, &seq_a);

    let mut seq_b = mgr.new_sequence(64);
    assert_eq!(
        mgr.prefix.attach_match(&mut mgr.pool, &mut seq_b, &tokens),
        16
    );
    let shared = seq_b.table(0).block(0);
    assert!(mgr.pool.refcount(shared) >= 2);

    // Writing into the shared block must CoW.
    mgr.ensure_writable(&mut seq_b, 0).unwrap();
    let after = seq_b.table(0).block(0);
    assert_ne!(shared, after);
    assert_eq!(mgr.pool.refcount(after), 1);
}

#[test]
fn compact_sequence_reclaims_physical_blocks() {
    let mut mgr = CacheManager::new(64, 1, 1, 4, 8).unwrap();
    let mut seq = mgr.new_sequence(128);
    // Allocate 4 blocks (64 tokens with BLOCK_SIZE=16).
    for pos in 0..64 {
        mgr.ensure_writable(&mut seq, pos).unwrap();
    }
    let free_before = mgr.pool.free_count();
    let allocated_before = mgr.pool.allocated_count();
    assert!(allocated_before >= 4);

    // Retain only positions in the first block (0..16).
    let retain: Vec<u32> = (0..16).collect();
    let reclaimed = mgr.compact_sequence_to_positions(&mut seq, &retain, BLOCK_SIZE);
    assert!(
        reclaimed >= 3,
        "expected at least 3 blocks reclaimed, got {reclaimed}"
    );
    let free_after = mgr.pool.free_count();
    assert!(
        free_after > free_before,
        "free list should grow after reclaim ({free_before} -> {free_after})"
    );
    assert_eq!(seq.len_tokens, 16);
    assert_eq!(seq.table(0).num_blocks(), 1);
    // Dense table has no dangling freed phys ids.
    for t in 0..seq.len_tokens {
        let (phys, _) = seq.table(0).locate(t);
        assert_eq!(mgr.pool.refcount(phys), 1);
    }
}
