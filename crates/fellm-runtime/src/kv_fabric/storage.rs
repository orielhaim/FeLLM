//! Physical page arenas across residency tiers.
//!
//! Physical slots are fabric-internal. Sequence APIs never see them as identity.

use super::types::{
    KvEncoding, KvGroupDesc, KvLocation, KvTier, PhysicalSlot, STANDARD_PAGE_TOKENS,
};
use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use half::f16;

/// Metadata for one physical page slot.
#[derive(Debug, Clone)]
pub struct PageMeta {
    pub refcount: u32,
    pub last_used: u64,
    pub location: KvLocation,
    pub encoding: KvEncoding,
    pub page_tokens: usize,
    /// Access / value signals for residency scoring.
    pub access_count: u64,
    pub share_count: u32,
    pub recompute_cost: f32,
    pub immutable: bool,
}

impl Default for PageMeta {
    fn default() -> Self {
        Self {
            refcount: 0,
            last_used: 0,
            location: KvLocation::NotResident,
            encoding: KvEncoding::Fp16,
            page_tokens: STANDARD_PAGE_TOKENS,
            access_count: 0,
            share_count: 0,
            recompute_cost: 1.0,
            immutable: false,
        }
    }
}

/// Host (and primary) physical page arena.
///
/// Device tier may mirror this arena (CUDA VRAM pool) without changing logical ids.
pub struct PageArena {
    arena: AlignedBuffer,
    meta: Vec<PageMeta>,
    free: Vec<u32>,
    n_pages: usize,
    n_layers: usize,
    tokens_stride: usize,
    page_tokens: usize,
    page_elems: usize,
    page_bytes: usize,
    encoding: KvEncoding,
    /// Secondary host tier for migrated pages (Host / HostPinned modeling).
    host_tier: Option<HostTierArena>,
}

struct HostTierArena {
    arena: AlignedBuffer,
    free_slots: Vec<u32>,
    /// physical_slot → host tier slot
    map: std::collections::HashMap<u32, u32>,
    page_bytes: usize,
    #[allow(dead_code)]
    tier: KvTier,
}

impl PageArena {
    #[must_use]
    pub fn page_bytes_for(group: &KvGroupDesc) -> usize {
        group.page_bytes()
    }

    #[must_use]
    pub fn page_bytes_for_dims(
        n_kv_heads: usize,
        head_dim: usize,
        page_tokens: usize,
        encoding: KvEncoding,
    ) -> usize {
        let stride = n_kv_heads.max(1).saturating_mul(head_dim.max(1));
        let raw = 2usize
            .saturating_mul(page_tokens.max(1))
            .saturating_mul(stride)
            .saturating_mul(encoding.elem_bytes());
        (raw + 63) & !63
    }

    pub fn new(
        n_pages: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        page_tokens: usize,
        encoding: KvEncoding,
        host_pages: usize,
    ) -> Result<Self> {
        if n_pages == 0 {
            return Err(FellmError::other("PageArena: n_pages must be > 0"));
        }
        let page_tokens = page_tokens.max(1);
        let tokens_stride = n_kv_heads.max(1) * head_dim.max(1);
        let page_elems = 2 * page_tokens * tokens_stride;
        let page_bytes = Self::page_bytes_for_dims(n_kv_heads, head_dim, page_tokens, encoding);
        let total_bytes = n_pages
            .checked_mul(page_bytes)
            .ok_or_else(|| FellmError::other("PageArena: size overflow"))?;
        let arena = AlignedBuffer::new_zeroed(total_bytes, 64);
        let mut free = Vec::with_capacity(n_pages);
        for i in (0..n_pages).rev() {
            free.push(i as u32);
        }
        let host_tier = if host_pages > 0 {
            let total = host_pages
                .checked_mul(page_bytes)
                .ok_or_else(|| FellmError::other("PageArena: host tier overflow"))?;
            let mut free_slots = Vec::with_capacity(host_pages);
            for i in (0..host_pages).rev() {
                free_slots.push(i as u32);
            }
            Some(HostTierArena {
                arena: AlignedBuffer::new_zeroed(total, 64),
                free_slots,
                map: std::collections::HashMap::new(),
                page_bytes,
                tier: KvTier::Host,
            })
        } else {
            None
        };
        Ok(Self {
            arena,
            meta: vec![PageMeta::default(); n_pages],
            free,
            n_pages,
            n_layers,
            tokens_stride,
            page_tokens,
            page_elems,
            page_bytes,
            encoding,
            host_tier,
        })
    }

    #[must_use]
    pub fn n_pages(&self) -> usize {
        self.n_pages
    }

    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    #[must_use]
    pub fn tokens_stride(&self) -> usize {
        self.tokens_stride
    }

    #[must_use]
    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    #[must_use]
    pub fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    #[must_use]
    pub fn page_elems(&self) -> usize {
        self.page_elems
    }

    #[must_use]
    pub fn encoding(&self) -> KvEncoding {
        self.encoding
    }

    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.n_pages.saturating_sub(self.free.len())
    }

    #[must_use]
    pub fn host_free_count(&self) -> usize {
        self.host_tier.as_ref().map_or(0, |h| h.free_slots.len())
    }

    pub fn alloc_page(&mut self, clock: u64) -> Option<PhysicalSlot> {
        let id = self.free.pop()?;
        self.meta[id as usize] = PageMeta {
            refcount: 0,
            last_used: clock,
            location: KvLocation::Resident(KvTier::Device),
            encoding: self.encoding,
            page_tokens: self.page_tokens,
            access_count: 1,
            share_count: 0,
            recompute_cost: self.page_tokens as f32,
            immutable: false,
        };
        self.page_bytes_mut(PhysicalSlot(id)).fill(0);
        Some(PhysicalSlot(id))
    }

    pub fn inc_ref(&mut self, slot: PhysicalSlot) {
        let m = &mut self.meta[slot.0 as usize];
        m.refcount = m.refcount.saturating_add(1);
        m.share_count = m.refcount;
    }

    pub fn dec_ref(&mut self, slot: PhysicalSlot) {
        let idx = slot.0 as usize;
        let m = &mut self.meta[idx];
        m.refcount = m.refcount.saturating_sub(1);
        m.share_count = m.refcount;
        if m.refcount == 0 {
            // Host-tier entries are keyed by logical page id (not physical slot)
            // and are released explicitly by restore_from_host_key / share eviction.
            m.location = KvLocation::NotResident;
            self.free.push(slot.0);
        }
    }

    #[must_use]
    pub fn refcount(&self, slot: PhysicalSlot) -> u32 {
        self.meta[slot.0 as usize].refcount
    }

    #[must_use]
    pub fn meta(&self, slot: PhysicalSlot) -> &PageMeta {
        &self.meta[slot.0 as usize]
    }

    pub fn meta_mut(&mut self, slot: PhysicalSlot) -> &mut PageMeta {
        &mut self.meta[slot.0 as usize]
    }

    pub fn touch(&mut self, slot: PhysicalSlot, clock: u64) {
        let m = &mut self.meta[slot.0 as usize];
        m.last_used = clock;
        m.access_count = m.access_count.saturating_add(1);
    }

    #[must_use]
    pub fn block_byte_offset(&self, slot: PhysicalSlot) -> usize {
        (slot.0 as usize) * self.page_bytes
    }

    fn page_bytes_mut(&mut self, slot: PhysicalSlot) -> &mut [u8] {
        let off = self.block_byte_offset(slot);
        let end = off + self.page_bytes;
        &mut self.arena.as_mut_slice()[off..end]
    }

    fn payload_bytes(&self) -> usize {
        self.page_elems * self.encoding.elem_bytes()
    }

    fn page_f16(&self, slot: PhysicalSlot) -> &[f16] {
        debug_assert_eq!(self.encoding.elem_bytes(), 2);
        let off = self.block_byte_offset(slot);
        let bytes = &self.arena.as_slice()[off..off + self.payload_bytes()];
        bytemuck::cast_slice(bytes)
    }

    fn page_f16_mut(&mut self, slot: PhysicalSlot) -> &mut [f16] {
        let off = self.block_byte_offset(slot);
        let n = self.payload_bytes();
        let bytes = &mut self.arena.as_mut_slice()[off..off + n];
        bytemuck::cast_slice_mut(bytes)
    }

    pub fn copy_page(&mut self, src: PhysicalSlot, dst: PhysicalSlot) {
        debug_assert_ne!(src.0, dst.0);
        let src_off = self.block_byte_offset(src);
        let dst_off = self.block_byte_offset(dst);
        let n = self.page_bytes;
        let arena = self.arena.as_mut_slice();
        unsafe {
            let sp = arena.as_ptr().add(src_off);
            let dp = arena.as_mut_ptr().add(dst_off);
            core::ptr::copy_nonoverlapping(sp, dp, n);
        }
        self.meta[dst.0 as usize].encoding = self.meta[src.0 as usize].encoding;
        self.meta[dst.0 as usize].page_tokens = self.meta[src.0 as usize].page_tokens;
    }

    pub fn k_row(&self, slot: PhysicalSlot, row: usize) -> &[f16] {
        debug_assert!(row < self.page_tokens);
        let page = self.page_f16(slot);
        let base = row * self.tokens_stride;
        &page[base..base + self.tokens_stride]
    }

    pub fn k_row_mut(&mut self, slot: PhysicalSlot, row: usize) -> &mut [f16] {
        debug_assert!(row < self.page_tokens);
        let stride = self.tokens_stride;
        let page = self.page_f16_mut(slot);
        let base = row * stride;
        &mut page[base..base + stride]
    }

    pub fn v_row(&self, slot: PhysicalSlot, row: usize) -> &[f16] {
        debug_assert!(row < self.page_tokens);
        let page = self.page_f16(slot);
        let base = self.page_tokens * self.tokens_stride + row * self.tokens_stride;
        &page[base..base + self.tokens_stride]
    }

    pub fn v_row_mut(&mut self, slot: PhysicalSlot, row: usize) -> &mut [f16] {
        debug_assert!(row < self.page_tokens);
        let stride = self.tokens_stride;
        let pt = self.page_tokens;
        let page = self.page_f16_mut(slot);
        let base = pt * stride + row * stride;
        &mut page[base..base + stride]
    }

    #[must_use]
    pub fn arena_bytes(&self) -> &[u8] {
        self.arena.as_slice()
    }

    pub fn arena_bytes_mut(&mut self) -> &mut [u8] {
        self.arena.as_mut_slice()
    }

    #[must_use]
    pub fn arena_ptr_mut(&mut self) -> (*mut u8, usize) {
        let len = self.arena.len();
        (self.arena.as_mut_slice().as_mut_ptr(), len)
    }

    pub fn read_page_bytes(&self, slot: PhysicalSlot, dst: &mut [u8]) {
        let off = self.block_byte_offset(slot);
        dst.copy_from_slice(&self.arena.as_slice()[off..off + self.page_bytes]);
    }

    pub fn write_page_bytes(&mut self, slot: PhysicalSlot, src: &[u8]) {
        let off = self.block_byte_offset(slot);
        self.arena.as_mut_slice()[off..off + self.page_bytes].copy_from_slice(src);
    }

    /// Copy page bytes into host tier under `host_key` (typically logical page id low bits
    /// or prior physical id). Does not free the device slot.
    pub fn stash_to_host(&mut self, host_key: u32, src: PhysicalSlot) -> Result<u64> {
        let host = self
            .host_tier
            .as_mut()
            .ok_or_else(|| FellmError::other("PageArena: no host tier configured"))?;
        if host.map.contains_key(&host_key) {
            return Ok(0);
        }
        let hs = host
            .free_slots
            .pop()
            .ok_or_else(|| FellmError::other("PageArena: host tier exhausted"))?;
        let off = (hs as usize) * host.page_bytes;
        let src_off = (src.0 as usize) * self.page_bytes;
        host.arena.as_mut_slice()[off..off + host.page_bytes]
            .copy_from_slice(&self.arena.as_slice()[src_off..src_off + self.page_bytes]);
        host.map.insert(host_key, hs);
        Ok(self.page_bytes as u64)
    }

    /// Restore stashed host bytes into `dst` device slot and free the host slot.
    pub fn restore_from_host_key(&mut self, host_key: u32, dst: PhysicalSlot) -> Result<u64> {
        let host = self
            .host_tier
            .as_mut()
            .ok_or_else(|| FellmError::other("PageArena: no host tier configured"))?;
        let hs = host
            .map
            .remove(&host_key)
            .ok_or_else(|| FellmError::other("PageArena: page not in host tier"))?;
        let off = (hs as usize) * host.page_bytes;
        let dst_off = (dst.0 as usize) * self.page_bytes;
        self.arena.as_mut_slice()[dst_off..dst_off + self.page_bytes]
            .copy_from_slice(&host.arena.as_slice()[off..off + host.page_bytes]);
        host.free_slots.push(hs);
        self.meta[dst.0 as usize].location = KvLocation::Resident(KvTier::Device);
        Ok(self.page_bytes as u64)
    }

    #[must_use]
    pub fn host_contains(&self, host_key: u32) -> bool {
        self.host_tier
            .as_ref()
            .is_some_and(|h| h.map.contains_key(&host_key))
    }

    /// Drop a host-tier stash without restoring to device (release / rollback).
    pub fn drop_host_stash(&mut self, host_key: u32) -> bool {
        let Some(host) = self.host_tier.as_mut() else {
            return false;
        };
        let Some(hs) = host.map.remove(&host_key) else {
            return false;
        };
        host.free_slots.push(hs);
        true
    }

    #[must_use]
    pub fn is_on_host(&self, slot: PhysicalSlot) -> bool {
        matches!(
            self.meta[slot.0 as usize].location,
            KvLocation::Resident(KvTier::Host | KvTier::HostPinned)
        )
    }

    /// All allocated slots for residency scoring.
    pub fn allocated_slots(&self) -> impl Iterator<Item = PhysicalSlot> + '_ {
        self.meta.iter().enumerate().filter_map(|(i, m)| {
            if m.refcount > 0 {
                Some(PhysicalSlot(i as u32))
            } else {
                None
            }
        })
    }
}
