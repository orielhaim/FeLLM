//! Radix prefix cache keyed by token-id chunks of [`BLOCK_SIZE`].

use super::pool::{BLOCK_SIZE, PhysicalPool};
use super::table::SequenceCache;
use std::collections::HashMap;

/// Node in the prefix radix tree.
#[derive(Default)]
struct PrefixNode {
    /// Child edges keyed by the 16-token chunk hash (or full chunk key).
    children: HashMap<u64, Box<PrefixNode>>,
    /// Physical block id per attention layer for this chunk (if terminal for the chunk).
    physical: Option<Vec<u32>>,
}

/// Radix tree over prompt token chunks for KV block sharing.
pub struct PrefixTree {
    root: PrefixNode,
}

impl PrefixTree {
    /// Empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: PrefixNode::default(),
        }
    }

    /// Hash one chunk of exactly [`BLOCK_SIZE`] tokens (or fewer for a partial last chunk — not inserted).
    fn chunk_key(tokens: &[u32]) -> u64 {
        // FNV-1a 64
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Match the longest prefix of `tokens` that exists in the tree.
    ///
    /// Returns `(matched_token_count, per_layer physical block lists for matched full chunks)`.
    pub fn match_prefix(&self, tokens: &[u32]) -> (usize, Vec<Vec<u32>>) {
        let n_full = tokens.len() / BLOCK_SIZE;
        if n_full == 0 {
            return (0, Vec::new());
        }
        let mut node = &self.root;
        let mut matched_blocks: Vec<Vec<u32>> = Vec::new();
        let mut matched_chunks = 0usize;
        for i in 0..n_full {
            let chunk = &tokens[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE];
            let key = Self::chunk_key(chunk);
            match node.children.get(&key) {
                Some(child) => {
                    if let Some(phys) = &child.physical {
                        matched_blocks.push(phys.clone());
                        matched_chunks += 1;
                        node = child;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        // Transpose: matched_blocks[chunk][layer] → per_layer[layer][chunk]
        if matched_chunks == 0 {
            return (0, Vec::new());
        }
        let n_layers = matched_blocks[0].len();
        let mut per_layer = vec![Vec::with_capacity(matched_chunks); n_layers];
        for chunk_phys in &matched_blocks {
            for (layer, &id) in chunk_phys.iter().enumerate() {
                per_layer[layer].push(id);
            }
        }
        (matched_chunks * BLOCK_SIZE, per_layer)
    }

    /// Attach matched prefix blocks to `seq`, bumping refcounts.
    pub fn attach_match(
        &self,
        pool: &mut PhysicalPool,
        seq: &mut SequenceCache,
        tokens: &[u32],
    ) -> usize {
        let (matched_tokens, per_layer) = self.match_prefix(tokens);
        if matched_tokens == 0 {
            return 0;
        }
        for (layer, blocks) in per_layer.into_iter().enumerate() {
            for &id in &blocks {
                pool.inc_ref(id);
                pool.touch(id, 0);
            }
            seq.table_mut(layer).set_blocks(blocks);
        }
        seq.len_tokens = matched_tokens;
        matched_tokens
    }

    /// Insert full chunks from a completed prompt into the tree.
    ///
    /// `seq` must already hold physical blocks for those tokens.
    pub fn insert_prompt(&mut self, tokens: &[u32], seq: &SequenceCache) {
        let n_full = tokens.len() / BLOCK_SIZE;
        if n_full == 0 || seq.n_layers() == 0 {
            return;
        }
        let mut node = &mut self.root;
        for i in 0..n_full {
            let chunk = &tokens[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE];
            let key = Self::chunk_key(chunk);
            let child = node
                .children
                .entry(key)
                .or_insert_with(|| Box::new(PrefixNode::default()));
            if child.physical.is_none() {
                let mut phys = Vec::with_capacity(seq.n_layers());
                for layer in 0..seq.n_layers() {
                    phys.push(seq.table(layer).block(i));
                }
                child.physical = Some(phys);
            }
            node = child;
        }
    }
}

impl Default for PrefixTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::paged::CacheManager;

    #[test]
    fn match_and_share() {
        let mut mgr = CacheManager::new(64, 2, 1, 4, 8).unwrap();
        let tokens: Vec<u32> = (0..32).collect();

        // Build seq A with two full blocks.
        let mut seq_a = mgr.new_sequence(128);
        for pos in 0..32 {
            mgr.ensure_writable(&mut seq_a, pos).unwrap();
        }
        mgr.prefix.insert_prompt(&tokens, &seq_a);

        let mut seq_b = mgr.new_sequence(128);
        let matched = mgr.prefix.attach_match(&mut mgr.pool, &mut seq_b, &tokens);
        assert_eq!(matched, 32);
        assert_eq!(seq_b.table(0).block(0), seq_a.table(0).block(0));
        assert!(mgr.pool.refcount(seq_a.table(0).block(0)) >= 2);
    }
}
