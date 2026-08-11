//! Logical page identity ↔ physical slot mapping.
//!
//! Sequences hold [`KvPageId`]; only the fabric resolves them to storage.

use super::types::{KvLocation, KvPageId, PhysicalSlot};
use std::collections::HashMap;

/// Global logical→physical map with refcounting at the logical layer.
#[derive(Debug, Default)]
pub struct PageMap {
    next_id: u64,
    /// logical page → physical slot
    to_phys: HashMap<KvPageId, PhysicalSlot>,
    /// reverse lookup for reclaim bookkeeping
    to_logical: HashMap<PhysicalSlot, KvPageId>,
    locations: HashMap<KvPageId, KvLocation>,
}

impl PageMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_id(&mut self) -> KvPageId {
        let id = KvPageId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    pub fn bind(&mut self, page: KvPageId, slot: PhysicalSlot, location: KvLocation) {
        if let Some(old) = self.to_phys.insert(page, slot) {
            self.to_logical.remove(&old);
        }
        self.to_logical.insert(slot, page);
        self.locations.insert(page, location);
    }

    pub fn unbind(&mut self, page: KvPageId) -> Option<PhysicalSlot> {
        self.locations.remove(&page);
        let slot = self.to_phys.remove(&page)?;
        self.to_logical.remove(&slot);
        Some(slot)
    }

    #[must_use]
    pub fn resolve(&self, page: KvPageId) -> Option<PhysicalSlot> {
        self.to_phys.get(&page).copied()
    }

    #[must_use]
    pub fn logical_of(&self, slot: PhysicalSlot) -> Option<KvPageId> {
        self.to_logical.get(&slot).copied()
    }

    #[must_use]
    pub fn location(&self, page: KvPageId) -> KvLocation {
        self.locations
            .get(&page)
            .copied()
            .unwrap_or(KvLocation::NotResident)
    }

    pub fn set_location(&mut self, page: KvPageId, location: KvLocation) {
        self.locations.insert(page, location);
    }

    pub fn rebind_slot(&mut self, page: KvPageId, new_slot: PhysicalSlot) {
        if let Some(old) = self.to_phys.insert(page, new_slot) {
            self.to_logical.remove(&old);
        }
        self.to_logical.insert(new_slot, page);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.to_phys.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.to_phys.is_empty()
    }
}
