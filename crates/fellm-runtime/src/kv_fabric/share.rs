//! Content-addressed shared KV store with refcounting and CoW support.
//!
//! A radix/trie indexes shared objects for longest-prefix lookup; ownership
//! of page data remains with the fabric storage + mapping layer.

use super::mapping::PageMap;
use super::storage::PageArena;
use super::types::{
    KvEncoding, KvPageId, KvSequence, PageOwnership, PhysicalSlot, STANDARD_PAGE_TOKENS,
    SharedKvId, SharedKvKey,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_tokens: u64,
    pub miss_tokens: u64,
    pub cached_tokens: usize,
    pub occupied_pages: usize,
    pub occupied_bytes: usize,
    pub evictions: u64,
    pub evicted_tokens: u64,
    pub tokens_saved: u64,
}

struct SharedObject {
    #[allow(dead_code)]
    id: SharedKvId,
    key: SharedKvKey,
    /// One logical page id per attention layer for this token chunk.
    pages: Vec<KvPageId>,
    token_len: usize,
    access_count: u64,
    last_access: u64,
    /// Store-owned references (independent of sequence refs).
    #[allow(dead_code)]
    store_refcount: u32,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u64, Box<TrieNode>>,
    token_chunk: Vec<u32>,
    object: Option<SharedKvId>,
}

/// Content-addressed shared store + radix index for prefix lookup.
pub struct SharedKvStore {
    objects: HashMap<SharedKvId, SharedObject>,
    by_hash: HashMap<u64, Vec<SharedKvId>>,
    root: TrieNode,
    next_id: u64,
    clock: u64,
    stats: PrefixCacheStats,
    page_tokens: usize,
    model_fingerprint: u64,
    encoding: KvEncoding,
}

impl SharedKvStore {
    #[must_use]
    pub fn new(page_tokens: usize, model_fingerprint: u64, encoding: KvEncoding) -> Self {
        Self {
            objects: HashMap::new(),
            by_hash: HashMap::new(),
            root: TrieNode::default(),
            next_id: 1,
            clock: 0,
            stats: PrefixCacheStats::default(),
            page_tokens: page_tokens.max(1),
            model_fingerprint,
            encoding,
        }
    }

    #[must_use]
    pub fn stats(&self) -> PrefixCacheStats {
        self.stats.clone()
    }

    #[must_use]
    pub fn shared_bytes(&self, page_bytes: usize) -> u64 {
        (self.stats.occupied_pages as u64).saturating_mul(page_bytes as u64)
    }

    #[must_use]
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    fn chunk_key(tokens: &[u32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &token in tokens {
            h ^= u64::from(token);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    fn match_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<Vec<KvPageId>>) {
        self.clock = self.clock.wrapping_add(1);
        let n_full = tokens.len() / self.page_tokens;
        let mut node = &mut self.root;
        let mut chunks = Vec::new();
        for chunk in tokens.chunks_exact(self.page_tokens).take(n_full) {
            let Some(child) = node.children.get_mut(&Self::chunk_key(chunk)) else {
                break;
            };
            if child.token_chunk != chunk {
                break;
            }
            let Some(oid) = child.object else {
                break;
            };
            let Some(obj) = self.objects.get_mut(&oid) else {
                break;
            };
            obj.access_count = obj.access_count.saturating_add(1);
            obj.last_access = self.clock;
            chunks.push(obj.pages.clone());
            node = child;
        }
        let matched = chunks.len() * self.page_tokens;
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
        let n_layers = chunks[0].len();
        let mut per_layer = vec![Vec::with_capacity(chunks.len()); n_layers];
        for pages in chunks {
            for (layer, id) in pages.into_iter().enumerate() {
                per_layer[layer].push(id);
            }
        }
        (matched, per_layer)
    }

    /// Attach longest content-addressed prefix onto `seq`. Returns matched tokens.
    pub fn attach_match(
        &mut self,
        arena: &mut PageArena,
        map: &PageMap,
        seq: &mut KvSequence,
        tokens: &[u32],
    ) -> usize {
        let (matched, per_layer) = self.match_prefix(tokens);
        for (layer, pages) in per_layer.into_iter().enumerate() {
            for &pid in &pages {
                if let Some(slot) = map.resolve(pid) {
                    arena.inc_ref(slot);
                    arena.touch(slot, self.clock);
                    arena.meta_mut(slot).immutable = true;
                }
            }
            seq.layer_map_mut(layer).set_pages(pages);
        }
        seq.len_tokens = matched;
        matched
    }

    /// Insert completed prompt chunks into the shared store.
    pub fn insert_prompt(
        &mut self,
        arena: &mut PageArena,
        map: &PageMap,
        tokens: &[u32],
        seq: &KvSequence,
    ) {
        if seq.n_layers() == 0 {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        let mut node = &mut self.root;
        for (index, chunk) in tokens.chunks_exact(self.page_tokens).enumerate() {
            let key = Self::chunk_key(chunk);
            let child = node.children.entry(key).or_insert_with(|| {
                Box::new(TrieNode {
                    token_chunk: chunk.to_vec(),
                    ..TrieNode::default()
                })
            });
            if child.token_chunk != chunk {
                break;
            }
            if child.object.is_none() {
                let mut pages = Vec::with_capacity(seq.n_layers());
                for layer in 0..seq.n_layers() {
                    let Some(&pid) = seq.layer_map(layer).pages().get(index) else {
                        return;
                    };
                    if let Some(slot) = map.resolve(pid) {
                        arena.inc_ref(slot);
                        arena.meta_mut(slot).immutable = true;
                    }
                    pages.push(pid);
                }
                let oid = SharedKvId(self.next_id);
                self.next_id = self.next_id.wrapping_add(1);
                let sk = SharedKvKey {
                    model_fingerprint: self.model_fingerprint,
                    adapter_id: 0,
                    arch_config: 0,
                    group: 0,
                    encoding: self.encoding,
                    rope_config: 0,
                    tokens: chunk.to_vec(),
                };
                let h = sk.hash64();
                self.by_hash.entry(h).or_default().push(oid);
                self.objects.insert(
                    oid,
                    SharedObject {
                        id: oid,
                        key: sk,
                        pages: pages.clone(),
                        token_len: (index + 1) * self.page_tokens,
                        access_count: 1,
                        last_access: self.clock,
                        store_refcount: 1,
                    },
                );
                child.object = Some(oid);
                self.stats.cached_tokens =
                    self.stats.cached_tokens.saturating_add(self.page_tokens);
                self.stats.occupied_pages = self.stats.occupied_pages.saturating_add(pages.len());
                self.stats.occupied_bytes =
                    self.stats.occupied_pages.saturating_mul(arena.page_bytes());
            }
            if let Some(oid) = child.object
                && let Some(obj) = self.objects.get_mut(&oid)
            {
                obj.last_access = self.clock;
            }
            node = child;
        }
    }

    /// Evict least-valuable unreferenced shared objects until free pages ≥ target.
    pub fn evict_until(
        &mut self,
        arena: &mut PageArena,
        map: &PageMap,
        target_free: usize,
    ) -> usize {
        let mut evicted = 0;
        while arena.free_count() < target_free {
            let mut candidates = Vec::new();
            collect_evictable(
                &self.root,
                &self.objects,
                arena,
                map,
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
            let Some(oid) = node.object else {
                break;
            };
            let Some(obj) = self.objects.remove(&oid) else {
                break;
            };
            let h = obj.key.hash64();
            if let Some(list) = self.by_hash.get_mut(&h) {
                list.retain(|x| *x != oid);
            }
            for pid in &obj.pages {
                if let Some(slot) = map.resolve(*pid) {
                    arena.dec_ref(slot);
                }
            }
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            self.stats.evicted_tokens = self
                .stats
                .evicted_tokens
                .saturating_add(self.page_tokens as u64);
            self.stats.cached_tokens = self.stats.cached_tokens.saturating_sub(self.page_tokens);
            self.stats.occupied_pages = self.stats.occupied_pages.saturating_sub(obj.pages.len());
            self.stats.occupied_bytes =
                self.stats.occupied_pages.saturating_mul(arena.page_bytes());
            evicted += obj.pages.len();
        }
        evicted
    }
}

fn collect_evictable(
    node: &TrieNode,
    objects: &HashMap<SharedKvId, SharedObject>,
    arena: &PageArena,
    map: &PageMap,
    path: &mut Vec<u64>,
    out: &mut Vec<(Vec<u64>, f64)>,
    clock: u64,
) {
    for (&key, child) in &node.children {
        path.push(key);
        if child.children.is_empty()
            && let Some(oid) = child.object
            && let Some(obj) = objects.get(&oid)
            && obj
                .pages
                .iter()
                .all(|pid| map.resolve(*pid).is_some_and(|s| arena.refcount(s) == 1))
        {
            let age = clock.saturating_sub(obj.last_access).saturating_add(1) as f64;
            let recompute = obj.token_len.max(STANDARD_PAGE_TOKENS) as f64;
            let frequency = obj.access_count.saturating_add(1) as f64;
            let cost = obj.pages.len().max(1) as f64;
            out.push((path.clone(), frequency * recompute / (cost * age)));
        }
        collect_evictable(child, objects, arena, map, path, out, clock);
        path.pop();
    }
}

fn remove_path(root: &mut TrieNode, path: &[u64]) -> Option<Box<TrieNode>> {
    let (&last, parents) = path.split_last()?;
    let mut node = root;
    for key in parents {
        node = node.children.get_mut(key)?;
    }
    node.children.remove(&last)
}

/// Result of making a logical page writable.
#[derive(Debug, Clone, Copy)]
pub struct CowResult {
    /// Physical slot ready for mutation.
    pub slot: PhysicalSlot,
    /// Logical page the caller must store in its layer map.
    /// Differs from the input page when a shared page was forked.
    pub page: KvPageId,
    pub forked: bool,
}

/// Ensure a page is writable: CoW if shared (refcount > 1 or immutable).
///
/// When forking, a **new** logical [`KvPageId`] is allocated so other sequences
/// that still hold the original shared id keep the immutable mapping.
pub fn ensure_cow(
    arena: &mut PageArena,
    map: &mut PageMap,
    page: KvPageId,
    clock: u64,
) -> Result<CowResult, fellm_core::error::FellmError> {
    use fellm_core::error::FellmError;
    let slot = map
        .resolve(page)
        .ok_or_else(|| FellmError::other("ensure_cow: unbound page"))?;
    let needs_cow = arena.refcount(slot) > 1 || arena.meta(slot).immutable;
    if !needs_cow {
        arena.touch(slot, clock);
        return Ok(CowResult {
            slot,
            page,
            forked: false,
        });
    }
    let new_slot = arena
        .alloc_page(clock)
        .ok_or_else(|| FellmError::other("ensure_cow: alloc failed"))?;
    arena.copy_page(slot, new_slot);
    arena.inc_ref(new_slot);
    // Drop this sequence's share of the old page; shared store / others remain.
    arena.dec_ref(slot);
    arena.meta_mut(new_slot).immutable = false;
    let new_page = map.alloc_id();
    map.bind(
        new_page,
        new_slot,
        super::types::KvLocation::Resident(super::types::KvTier::Device),
    );
    Ok(CowResult {
        slot: new_slot,
        page: new_page,
        forked: true,
    })
}

/// Ownership class after attach/fork.
#[must_use]
pub fn ownership_for_prefix(shared_prefix: bool) -> PageOwnership {
    if shared_prefix {
        PageOwnership::SharedImmutable
    } else {
        PageOwnership::Private
    }
}
