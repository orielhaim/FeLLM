//! Value-aware radix prefix cache keyed by full [`BLOCK_SIZE`] token chunks.

use super::pool::{BLOCK_SIZE, PhysicalPool};
use super::table::SequenceCache;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_tokens: u64,
    pub miss_tokens: u64,
    pub cached_tokens: usize,
    pub occupied_blocks: usize,
    pub occupied_bytes: usize,
    pub evictions: u64,
    pub evicted_tokens: u64,
    pub tokens_saved: u64,
}

#[derive(Default)]
struct PrefixNode {
    children: HashMap<u64, Box<PrefixNode>>,
    /// Exact edge value prevents an FNV collision from becoming a false hit.
    token_chunk: Vec<u32>,
    /// One physical KV block per attention layer. The cache owns one ref each.
    physical: Option<Vec<u32>>,
    token_len: usize,
    access_count: u64,
    last_access: u64,
}

pub struct PrefixTree {
    root: PrefixNode,
    clock: u64,
    stats: PrefixCacheStats,
}

impl PrefixTree {
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: PrefixNode::default(),
            clock: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    fn chunk_key(tokens: &[u32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &token in tokens {
            h ^= u64::from(token);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Match and account for the longest full-block prefix.
    fn match_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<Vec<u32>>) {
        self.clock = self.clock.wrapping_add(1);
        let n_full = tokens.len() / BLOCK_SIZE;
        let mut node = &mut self.root;
        let mut chunks = Vec::new();
        for chunk in tokens.chunks_exact(BLOCK_SIZE).take(n_full) {
            let Some(child) = node.children.get_mut(&Self::chunk_key(chunk)) else {
                break;
            };
            if child.token_chunk != chunk {
                break;
            }
            let Some(physical) = &child.physical else {
                break;
            };
            child.access_count = child.access_count.saturating_add(1);
            child.last_access = self.clock;
            chunks.push(physical.clone());
            node = child;
        }
        let matched = chunks.len() * BLOCK_SIZE;
        if matched == 0 {
            self.stats.misses = self.stats.misses.saturating_add(1);
        } else {
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.stats.tokens_saved = self.stats.tokens_saved.saturating_add(matched as u64);
        }
        self.stats.hit_tokens = self.stats.hit_tokens.saturating_add(matched as u64);
        self.stats.miss_tokens = self
            .stats
            .miss_tokens
            .saturating_add(tokens.len().saturating_sub(matched) as u64);
        if chunks.is_empty() {
            return (0, Vec::new());
        }
        let mut per_layer = vec![Vec::with_capacity(chunks.len()); chunks[0].len()];
        for physical in chunks {
            for (layer, id) in physical.into_iter().enumerate() {
                per_layer[layer].push(id);
            }
        }
        (matched, per_layer)
    }

    pub fn attach_match(
        &mut self,
        pool: &mut PhysicalPool,
        seq: &mut SequenceCache,
        tokens: &[u32],
    ) -> usize {
        let (matched, per_layer) = self.match_prefix(tokens);
        for (layer, blocks) in per_layer.into_iter().enumerate() {
            for &id in &blocks {
                pool.inc_ref(id);
                pool.touch(id, self.clock);
            }
            seq.table_mut(layer).set_blocks(blocks);
        }
        seq.len_tokens = matched;
        matched
    }

    /// Insert full prompt chunks and acquire one cache-owned reference per block.
    pub fn insert_prompt(&mut self, pool: &mut PhysicalPool, tokens: &[u32], seq: &SequenceCache) {
        if seq.n_layers() == 0 {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        let mut node = &mut self.root;
        for (index, chunk) in tokens.chunks_exact(BLOCK_SIZE).enumerate() {
            let key = Self::chunk_key(chunk);
            let child = node.children.entry(key).or_insert_with(|| {
                Box::new(PrefixNode {
                    token_chunk: chunk.to_vec(),
                    token_len: (index + 1) * BLOCK_SIZE,
                    ..PrefixNode::default()
                })
            });
            if child.token_chunk != chunk {
                break;
            }
            if child.physical.is_none() {
                let mut physical = Vec::with_capacity(seq.n_layers());
                for layer in 0..seq.n_layers() {
                    let Some(&id) = seq.table(layer).blocks().get(index) else {
                        return;
                    };
                    pool.inc_ref(id);
                    physical.push(id);
                }
                self.stats.cached_tokens = self.stats.cached_tokens.saturating_add(BLOCK_SIZE);
                self.stats.occupied_blocks =
                    self.stats.occupied_blocks.saturating_add(physical.len());
                self.stats.occupied_bytes = self
                    .stats
                    .occupied_blocks
                    .saturating_mul(pool.block_bytes());
                child.physical = Some(physical);
            }
            child.last_access = self.clock;
            node = child;
        }
    }

    /// Reclaim least valuable unreferenced leaf entries until `target_free` is met.
    pub fn evict_until(&mut self, pool: &mut PhysicalPool, target_free: usize) -> usize {
        let mut evicted = 0;
        while pool.free_count() < target_free {
            let mut candidates = Vec::new();
            collect_evictable_leaves(
                &self.root,
                pool,
                &mut Vec::new(),
                &mut candidates,
                self.clock,
            );
            let Some((path, _score)) = candidates.into_iter().min_by(|a, b| a.1.total_cmp(&b.1))
            else {
                break;
            };
            let Some(node) = remove_path(&mut self.root, &path) else {
                break;
            };
            let blocks = node.physical.unwrap_or_default();
            for id in &blocks {
                pool.dec_ref(*id);
            }
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            self.stats.evicted_tokens = self.stats.evicted_tokens.saturating_add(BLOCK_SIZE as u64);
            self.stats.cached_tokens = self.stats.cached_tokens.saturating_sub(BLOCK_SIZE);
            self.stats.occupied_blocks = self.stats.occupied_blocks.saturating_sub(blocks.len());
            self.stats.occupied_bytes = self
                .stats
                .occupied_blocks
                .saturating_mul(pool.block_bytes());
            evicted += blocks.len();
        }
        evicted
    }

    #[must_use]
    pub fn stats(&self) -> PrefixCacheStats {
        self.stats
    }
}

fn collect_evictable_leaves(
    node: &PrefixNode,
    pool: &PhysicalPool,
    path: &mut Vec<u64>,
    out: &mut Vec<(Vec<u64>, f64)>,
    clock: u64,
) {
    for (&key, child) in &node.children {
        path.push(key);
        if child.children.is_empty()
            && child
                .physical
                .as_ref()
                .is_some_and(|blocks| blocks.iter().all(|&id| pool.refcount(id) == 1))
        {
            let age = clock.saturating_sub(child.last_access).saturating_add(1) as f64;
            let recompute = child.token_len.max(BLOCK_SIZE) as f64;
            let frequency = child.access_count.saturating_add(1) as f64;
            let cost = child.physical.as_ref().map_or(1, Vec::len) as f64;
            out.push((path.clone(), frequency * recompute / (cost * age)));
        }
        collect_evictable_leaves(child, pool, path, out, clock);
        path.pop();
    }
}

fn remove_path(root: &mut PrefixNode, path: &[u64]) -> Option<Box<PrefixNode>> {
    let (&last, parents) = path.split_last()?;
    let mut node = root;
    for key in parents {
        node = node.children.get_mut(key)?;
    }
    node.children.remove(&last)
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
    fn cache_owns_blocks_after_source_release_and_evicts_when_idle() {
        let mut manager = CacheManager::new(8, 1, 1, 4, 0).unwrap();
        let tokens: Vec<u32> = (0..32).collect();
        let mut source = manager.new_sequence(64);
        for position in 0..32 {
            manager.ensure_writable(&mut source, position).unwrap();
        }
        manager
            .prefix
            .insert_prompt(&mut manager.pool, &tokens, &source);
        let first = source.table(0).block(0);
        assert_eq!(manager.pool.refcount(first), 2);
        manager.release_sequence(&mut source);
        assert_eq!(manager.pool.refcount(first), 1);
        let before = manager.pool.free_count();
        assert!(manager.prefix.evict_until(&mut manager.pool, before + 1) > 0);
        assert!(manager.pool.free_count() > before);
    }

    #[test]
    fn match_tracks_hit_and_miss_tokens() {
        let mut manager = CacheManager::new(8, 1, 1, 4, 0).unwrap();
        let tokens: Vec<u32> = (0..20).collect();
        let mut source = manager.new_sequence(64);
        for position in 0..20 {
            manager.ensure_writable(&mut source, position).unwrap();
        }
        manager
            .prefix
            .insert_prompt(&mut manager.pool, &tokens, &source);
        let mut target = manager.new_sequence(64);
        assert_eq!(
            manager
                .prefix
                .attach_match(&mut manager.pool, &mut target, &tokens),
            16
        );
        let stats = manager.prefix.stats();
        assert_eq!(stats.hit_tokens, 16);
        assert_eq!(stats.miss_tokens, 4);
    }
}
