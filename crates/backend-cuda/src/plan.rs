//! Device-native CUDA execution-plan storage and static arena planning.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::{
    AllocationId, ArenaSlot, DevicePtr, DeviceTensor, PhysicalPlan, PlanTensorDesc, StorageClass,
};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr as _};

/// Immutable tensor payload supplied to the CUDA model-image packer.
pub struct ModelBlob<'a> {
    /// Physical-plan tensor identity.
    pub tensor: fellm_plugin_abi::PlanTensorId,
    /// Raw GGUF tensor payload.
    pub bytes: &'a [u8],
    /// Required device alignment.
    pub alignment: usize,
}

/// One contiguous, model-lifetime GPU allocation containing every weight.
pub struct ModelImage {
    #[cfg(feature = "cuda")]
    storage: CudaSlice<u8>,
    offsets: HashMap<fellm_plugin_abi::PlanTensorId, (usize, usize)>,
    byte_len: usize,
}

impl ModelImage {
    /// Pack and upload constants once. GGUF remains a disk format, not a hot-path layout contract.
    pub fn upload(device: &crate::CudaDeviceState, blobs: &[ModelBlob<'_>]) -> Result<Self> {
        let mut offsets = HashMap::with_capacity(blobs.len());
        let mut cursor = 0usize;
        for blob in blobs {
            cursor = align_up(cursor, blob.alignment.max(128))?;
            offsets.insert(blob.tensor, (cursor, blob.bytes.len()));
            cursor = cursor
                .checked_add(blob.bytes.len())
                .ok_or_else(|| FellmError::other("CUDA model image size overflow"))?;
        }
        let byte_len = align_up(cursor, 256)?;
        #[cfg(feature = "cuda")]
        {
            let mut packed = vec![0u8; byte_len.max(1)];
            for blob in blobs {
                let (offset, len) = offsets[&blob.tensor];
                packed[offset..offset + len].copy_from_slice(blob.bytes);
            }
            let mut storage = device
                .stream()
                .alloc_zeros::<u8>(packed.len())
                .map_err(|e| FellmError::other(format!("allocate CUDA model image: {e}")))?;
            device
                .stream()
                .memcpy_htod(&packed, &mut storage)
                .map_err(|e| FellmError::other(format!("upload CUDA model image: {e}")))?;
            Ok(Self {
                storage,
                offsets,
                byte_len,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, byte_len);
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
        let (offset, len) = self
            .offsets
            .get(&tensor)
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
}

impl DecodeDeviceState {
    /// Allocate all steady-state decode storage. No allocation is needed per token.
    pub fn new(device: &crate::CudaDeviceState, arena_bytes: usize) -> Result<Self> {
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
            Ok(Self {
                arena,
                params,
                next_token,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            Ok(Self { arena })
        }
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
