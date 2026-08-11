//! Backend addressing strategies: block-table vs CUDA virtual memory.
//!
//! Logical KV is independent of which strategy materializes physical layout
//! for kernels. Attention plugins still consume a block table view when the
//! strategy is [`KvAddressing::BlockTable`].

use super::mapping::PageMap;
use super::types::{KvAddressing, KvPageId, PhysicalSlot};

/// Resolved addressing view for one inference step.
#[derive(Debug, Clone)]
pub struct AddressingView {
    pub strategy: KvAddressing,
    /// Flattened physical slots for block-table kernels:
    /// `[row][layer][logical] → phys`.
    pub physical_table: Vec<u32>,
    pub n_logical_pages: usize,
    pub n_layers: usize,
    pub batch_rows: usize,
    /// When VirtualMemory is active and mapped, base virtual pointer (host-side
    /// placeholder until CUDA VMM binds real device VAs).
    pub virtual_base: Option<u64>,
}

/// Address translation backend.
pub trait AddressTranslator: Send {
    fn strategy(&self) -> KvAddressing;

    /// Build a kernel-facing view from logical page ids.
    fn materialize(
        &self,
        map: &PageMap,
        page_ids_per_row: &[Vec<KvPageId>],
        n_layers: usize,
        n_logical: usize,
    ) -> AddressingView;
}

/// Classic paged block table (CPU + universal fallback).
#[derive(Debug, Default)]
pub struct BlockTableAddressing;

impl AddressTranslator for BlockTableAddressing {
    fn strategy(&self) -> KvAddressing {
        KvAddressing::BlockTable
    }

    fn materialize(
        &self,
        map: &PageMap,
        page_ids_per_row: &[Vec<KvPageId>],
        n_layers: usize,
        n_logical: usize,
    ) -> AddressingView {
        let batch_rows = page_ids_per_row.len().max(1);
        let mut physical_table = Vec::with_capacity(batch_rows * n_layers * n_logical.max(1));
        for row_pages in page_ids_per_row {
            // row_pages is flattened layer-major: n_layers * n_logical
            for &pid in row_pages {
                let phys = map.resolve(pid).map(|s| s.0).unwrap_or(0);
                physical_table.push(phys);
            }
            // Pad if empty
            if row_pages.is_empty() {
                physical_table.extend(std::iter::repeat_n(0u32, n_layers * n_logical.max(1)));
            }
        }
        AddressingView {
            strategy: KvAddressing::BlockTable,
            physical_table,
            n_logical_pages: n_logical.max(1),
            n_layers,
            batch_rows,
            virtual_base: None,
        }
    }
}

/// CUDA Virtual Memory Management addressing.
///
/// Maps physical GPU pages into a large contiguous virtual KV address space so
/// attention kernels need not walk fragmented block tables. When VMM is
/// unavailable, materialize falls back to block-table physical ids while still
/// advertising the VirtualMemory strategy for metrics/config.
#[derive(Debug, Default)]
pub struct VirtualMemoryAddressing {
    /// Optional reserved virtual base (device VA), set when CUDA VMM binds.
    pub virtual_base: Option<u64>,
    pub enabled: bool,
}

impl VirtualMemoryAddressing {
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            virtual_base: None,
            enabled,
        }
    }

    /// Record a device virtual base after successful CUDA VMM reserve/map.
    pub fn bind_virtual_base(&mut self, base: u64) {
        self.virtual_base = Some(base);
        self.enabled = true;
    }
}

impl AddressTranslator for VirtualMemoryAddressing {
    fn strategy(&self) -> KvAddressing {
        KvAddressing::VirtualMemory
    }

    fn materialize(
        &self,
        map: &PageMap,
        page_ids_per_row: &[Vec<KvPageId>],
        n_layers: usize,
        n_logical: usize,
    ) -> AddressingView {
        // Until kernels consume pure VA, still emit physical table for compatibility
        // with existing attention paths — but tag strategy as VirtualMemory and
        // expose virtual_base for progressive migration.
        let mut view = BlockTableAddressing.materialize(map, page_ids_per_row, n_layers, n_logical);
        view.strategy = KvAddressing::VirtualMemory;
        view.virtual_base = self.virtual_base;
        view
    }
}

#[must_use]
pub fn translator_for(strategy: KvAddressing) -> Box<dyn AddressTranslator> {
    match strategy {
        KvAddressing::BlockTable => Box::new(BlockTableAddressing),
        KvAddressing::VirtualMemory => Box::new(VirtualMemoryAddressing::new(true)),
    }
}

/// Resolve one logical page to physical slot (fabric-internal helper).
#[must_use]
pub fn resolve_slot(map: &PageMap, page: KvPageId) -> Option<PhysicalSlot> {
    map.resolve(page)
}
