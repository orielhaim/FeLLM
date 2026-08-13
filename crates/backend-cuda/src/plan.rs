//! Device-native CUDA execution-plan storage and static arena planning.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::{
    AllocationId, ArenaSlot, DevicePtr, DeviceTensor, PhysicalPlan, PlanTensorDesc, StorageClass,
};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr as _};

/// Immutable tensor payload supplied to the CUDA weight fabric.
pub struct WeightBlob<'a> {
    /// Stable logical identity, independent of the GGUF mmap address.
    pub id: fellm_memory::WeightId,
    /// Physical-plan tensor identity.
    pub tensor: fellm_plugin_abi::PlanTensorId,
    /// Raw GGUF tensor payload.
    pub bytes: &'a [u8],
    /// Required device alignment.
    pub alignment: usize,
}

/// CUDA-resident replica set owned by the Weight Fabric.
///
/// The initial provider uses one packed permanent allocation. The identity map is deliberately
/// separate so bounded streaming providers can replace residency without changing graph identity.
pub struct CudaWeightFabric {
    #[cfg(feature = "cuda")]
    storage: CudaSlice<u8>,
    offsets: HashMap<fellm_memory::WeightId, (usize, usize)>,
    tensors: HashMap<fellm_plugin_abi::PlanTensorId, fellm_memory::WeightId>,
    byte_len: usize,
}

impl CudaWeightFabric {
    /// Pack and upload constants once. GGUF remains a disk format, not a hot-path layout contract.
    pub fn materialize(device: &crate::CudaDeviceState, blobs: &[WeightBlob<'_>]) -> Result<Self> {
        let mut offsets = HashMap::with_capacity(blobs.len());
        let mut tensors = HashMap::with_capacity(blobs.len());
        let mut cursor = 0usize;
        for blob in blobs {
            tensors.insert(blob.tensor, blob.id);
            if offsets.contains_key(&blob.id) {
                continue;
            }
            cursor = align_up(cursor, blob.alignment.max(128))?;
            offsets.insert(blob.id, (cursor, blob.bytes.len()));
            cursor = cursor
                .checked_add(blob.bytes.len())
                .ok_or_else(|| FellmError::other("CUDA weight replica size overflow"))?;
        }
        let byte_len = align_up(cursor, 256)?;
        #[cfg(feature = "cuda")]
        {
            let mut packed = vec![0u8; byte_len.max(1)];
            for blob in blobs {
                let (offset, len) = offsets[&blob.id];
                debug_assert_eq!(len, blob.bytes.len());
                packed[offset..offset + len].copy_from_slice(blob.bytes);
            }
            let mut storage = device
                .stream()
                .alloc_zeros::<u8>(packed.len())
                .map_err(|e| FellmError::other(format!("allocate CUDA weight replica: {e}")))?;
            device
                .stream()
                .memcpy_htod(&packed, &mut storage)
                .map_err(|e| FellmError::other(format!("upload CUDA weight replica: {e}")))?;
            Ok(Self {
                storage,
                offsets,
                tensors,
                byte_len,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, byte_len, tensors);
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Total packed bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Resolve a constant to its stable device address and byte length.
    pub fn resolve(
        &self,
        tensor: fellm_plugin_abi::PlanTensorId,
    ) -> Result<(fellm_plugin_abi::DevicePtr, usize)> {
        let id = self
            .tensors
            .get(&tensor)
            .ok_or_else(|| FellmError::other("model tensor has no logical WeightId"))?;
        let (offset, len) = self
            .offsets
            .get(id)
            .copied()
            .ok_or_else(|| FellmError::other("model tensor absent from CUDA image"))?;
        #[cfg(feature = "cuda")]
        {
            let (base, _sync) = self.storage.device_ptr(self.storage.stream());
            Ok((
                fellm_plugin_abi::DevicePtr(base as u64 + offset as u64),
                len,
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (offset, len);
            Err(FellmError::other("cuda feature disabled"))
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeRegion {
    offset: usize,
    size: usize,
    available_after: u32,
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let alignment = alignment.max(1);
    if !alignment.is_power_of_two() {
        return Err(FellmError::other(format!(
            "device arena alignment {alignment} is not a power of two"
        )));
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or_else(|| FellmError::other("device arena alignment overflow"))
}

/// Plan transient tensors into one deterministic arena using lifetime reuse.
pub fn plan_static_arena(tensors: &[PlanTensorDesc]) -> Result<PhysicalPlan> {
    let mut transient: Vec<_> = tensors
        .iter()
        .filter(|t| t.storage == StorageClass::Transient)
        .collect();
    transient.sort_unstable_by_key(|t| (t.first_use, std::cmp::Reverse(t.byte_len()), t.id));

    let mut slots = Vec::with_capacity(transient.len());
    let mut regions: Vec<FreeRegion> = Vec::new();
    let mut high_water = 0usize;
    for tensor in transient {
        if tensor.last_use < tensor.first_use {
            return Err(FellmError::other(format!(
                "tensor {:?} has an inverted lifetime",
                tensor.id
            )));
        }
        let size = tensor
            .byte_len()
            .ok_or_else(|| FellmError::other("tensor byte size overflow"))?;
        let alignment = tensor.alignment.max(16);

        let candidate = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.available_after < tensor.first_use)
            .filter_map(|(index, region)| {
                let offset = align_up(region.offset, alignment).ok()?;
                let end = offset.checked_add(size)?;
                (end <= region.offset + region.size).then_some((index, offset, region.size))
            })
            .min_by_key(|(_, _, region_size)| *region_size);

        let (offset, region_index) = if let Some((index, offset, _)) = candidate {
            (offset, Some(index))
        } else {
            (align_up(high_water, alignment)?, None)
        };
        let end = offset
            .checked_add(size)
            .ok_or_else(|| FellmError::other("device arena size overflow"))?;
        high_water = high_water.max(end);
        if let Some(index) = region_index {
            regions[index].available_after = tensor.last_use;
        } else {
            regions.push(FreeRegion {
                offset,
                size,
                available_after: tensor.last_use,
            });
        }
        slots.push(ArenaSlot {
            tensor: tensor.id,
            offset,
            size,
            alignment,
            first_use: tensor.first_use,
            last_use: tensor.last_use,
        });
    }
    Ok(PhysicalPlan {
        arena: slots,
        arena_bytes: align_up(high_water, 256)?,
        operations: Vec::new(),
    })
}

/// One allocation containing every transient decode activation and scratch value.
pub struct CudaStaticArena {
    #[cfg(feature = "cuda")]
    storage: CudaSlice<u8>,
    allocation: AllocationId,
    byte_len: usize,
}

impl CudaStaticArena {
    /// Allocate a zeroed arena once when the decode plan is created.
    pub fn new(device: &crate::CudaDeviceState, byte_len: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let storage = device
                .stream()
                .alloc_zeros::<u8>(byte_len.max(1))
                .map_err(|e| FellmError::other(format!("allocate static CUDA arena: {e}")))?;
            Ok(Self {
                storage,
                allocation: AllocationId(0),
                byte_len,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, byte_len);
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Arena size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Resolve every planned slot to a direct device tensor descriptor.
    pub fn resolve(
        &self,
        plan: &PhysicalPlan,
        tensors: &[PlanTensorDesc],
    ) -> Result<HashMap<fellm_plugin_abi::PlanTensorId, DeviceTensor>> {
        let descriptions: HashMap<_, _> = tensors.iter().map(|t| (t.id, t)).collect();
        #[cfg(feature = "cuda")]
        let base = {
            let (ptr, _sync) = self.storage.device_ptr(self.storage.stream());
            ptr as u64
        };
        #[cfg(not(feature = "cuda"))]
        let base = 0u64;
        plan.arena
            .iter()
            .map(|slot| {
                let desc = descriptions
                    .get(&slot.tensor)
                    .ok_or_else(|| FellmError::other("arena slot has no tensor descriptor"))?;
                if desc.dtype.is_quantized() {
                    return Err(FellmError::other(
                        "quantized values are model allocations, not strided activation arena tensors",
                    ));
                }
                let mut stride = desc.dtype.bytes_per_block() as u64;
                let mut strides = vec![0; desc.shape.len()];
                for (index, dimension) in desc.shape.iter().enumerate().rev() {
                    strides[index] = stride;
                    stride = stride
                        .checked_mul(*dimension)
                        .ok_or_else(|| FellmError::other("device tensor stride overflow"))?;
                }
                Ok((
                    slot.tensor,
                    DeviceTensor {
                        ptr: DevicePtr(base + slot.offset as u64),
                        allocation: self.allocation,
                        dtype: desc.dtype,
                        shape: desc.shape.clone(),
                        strides,
                        offset_bytes: slot.offset,
                    },
                ))
            })
            .collect()
    }
}

/// Long-lived state shared by every replay of one decode plan.
pub struct DecodeDeviceState {
    /// Static activation/scratch allocation.
    pub arena: CudaStaticArena,
    /// Device-resident per-step controls.
    #[cfg(feature = "cuda")]
    pub params: CudaSlice<u8>,
    /// Device-resident next-token result.
    #[cfg(feature = "cuda")]
    pub next_token: CudaSlice<u32>,
    /// Fixed-capacity per-request page table used by every replay.
    #[cfg(feature = "cuda")]
    pub page_table: CudaSlice<u32>,
    page_table_capacity: usize,
    page_table_stride: usize,
    page_table_upload: Vec<u32>,
    page_table_shadow: Vec<u32>,
}

impl DecodeDeviceState {
    /// Allocate all steady-state decode storage. No allocation is needed per token.
    pub fn new(
        device: &crate::CudaDeviceState,
        arena_bytes: usize,
        page_table_capacity: usize,
        page_table_layers: usize,
    ) -> Result<Self> {
        let arena = CudaStaticArena::new(device, arena_bytes)?;
        #[cfg(feature = "cuda")]
        {
            let params = device
                .stream()
                .alloc_zeros::<u8>(core::mem::size_of::<fellm_plugin_abi::DeviceStepParams>())
                .map_err(|e| FellmError::other(format!("allocate DeviceStepParams: {e}")))?;
            let next_token = device
                .stream()
                .alloc_zeros::<u32>(1)
                .map_err(|e| FellmError::other(format!("allocate next-token result: {e}")))?;
            let page_table = device
                .stream()
                .alloc_zeros::<u32>(page_table_capacity.max(1))
                .map_err(|e| FellmError::other(format!("allocate device page table: {e}")))?;
            Ok(Self {
                arena,
                params,
                next_token,
                page_table,
                page_table_capacity,
                page_table_stride: page_table_capacity / page_table_layers.max(1),
                page_table_upload: vec![u32::MAX; page_table_capacity],
                page_table_shadow: Vec::new(),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            Ok(Self {
                arena,
                page_table_capacity,
                page_table_stride: page_table_capacity / page_table_layers.max(1),
                page_table_upload: vec![u32::MAX; page_table_capacity],
                page_table_shadow: Vec::new(),
            })
        }
    }

    /// Upload the page table only when allocation topology changes.
    pub fn sync_page_table(
        &mut self,
        device: &crate::CudaDeviceState,
        table: &[u32],
        n_layers: usize,
        n_logical: usize,
    ) -> Result<bool> {
        if table.len() > self.page_table_capacity {
            return Err(FellmError::other(format!(
                "device page table capacity exceeded: {} > {}",
                table.len(),
                self.page_table_capacity
            )));
        }
        if self.page_table_shadow == table {
            return Ok(false);
        }
        if table.len() != n_layers.saturating_mul(n_logical) || n_logical > self.page_table_stride {
            return Err(FellmError::other("invalid compact page-table dimensions"));
        }
        self.page_table_upload.fill(u32::MAX);
        for layer in 0..n_layers {
            let source = &table[layer * n_logical..(layer + 1) * n_logical];
            let start = layer * self.page_table_stride;
            self.page_table_upload[start..start + n_logical].copy_from_slice(source);
        }
        #[cfg(feature = "cuda")]
        {
            let mut target = self
                .page_table
                .try_slice_mut(..self.page_table_upload.len())
                .ok_or_else(|| FellmError::other("invalid device page-table range"))?;
            device
                .stream()
                .memcpy_htod(&self.page_table_upload, &mut target)
                .map_err(|error| FellmError::other(format!("upload device page table: {error}")))?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        self.page_table_shadow.clear();
        self.page_table_shadow.extend_from_slice(table);
        Ok(true)
    }

    /// Stable device page-table address and current logical length.
    #[must_use]
    pub fn page_table_ptr(&self) -> (*mut u32, usize) {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::DevicePtr;
            let (pointer, _guard) = self.page_table.device_ptr(self.page_table.stream());
            (pointer as *mut u32, self.page_table_capacity)
        }
        #[cfg(not(feature = "cuda"))]
        {
            (std::ptr::null_mut(), 0)
        }
    }

    /// Fixed logical-page stride between adjacent layers.
    #[must_use]
    pub fn page_table_stride(&self) -> usize {
        self.page_table_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_core::dtype::DType;
    use fellm_plugin_abi::{PlanTensorId, StorageClass};

    fn tensor(id: u32, bytes: u64, first: u32, last: u32) -> PlanTensorDesc {
        PlanTensorDesc {
            id: PlanTensorId(id),
            dtype: DType::U8,
            shape: vec![bytes],
            alignment: 64,
            storage: StorageClass::Transient,
            first_use: first,
            last_use: last,
        }
    }

    #[test]
    fn reuses_non_overlapping_lifetimes() {
        let plan = plan_static_arena(&[
            tensor(0, 1024, 0, 3),
            tensor(1, 1024, 4, 7),
            tensor(2, 512, 2, 5),
        ])
        .unwrap();
        let a = plan.arena.iter().find(|s| s.tensor.0 == 0).unwrap();
        let b = plan.arena.iter().find(|s| s.tensor.0 == 1).unwrap();
        assert_eq!(a.offset, b.offset);
        assert_eq!(plan.arena_bytes, 1536);
    }

    #[test]
    fn overlapping_lifetimes_never_alias() {
        let plan = plan_static_arena(&[tensor(0, 256, 0, 2), tensor(1, 256, 2, 4)]).unwrap();
        assert_ne!(plan.arena[0].offset, plan.arena[1].offset);
    }
}
