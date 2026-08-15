use crate::{ExecutionGroup, StorageExtent, StorageObject, StorageObjectId, StorageObjectMember};
use crate::{WeightDescriptor, WeightId};
use fellm_core::error::{FellmError, Result};
use std::collections::HashMap;

/// Compact O(1) logical-weight to physical-object lookup.
#[derive(Debug, Clone)]
pub struct StorageObjectIndex {
    objects: Box<[StorageObject]>,
    weight_to_object: HashMap<WeightId, StorageObjectId>,
}

impl StorageObjectIndex {
    /// Build graph-ordered physical objects, merging only weights consumed by the same execution
    /// group. This avoids the cyclic-LRU pathology and never performs runtime name scans.
    pub fn from_execution_groups(
        weights: &[WeightDescriptor],
        groups: &[ExecutionGroup],
        max_gap: u64,
        max_object_bytes: u64,
    ) -> Result<Self> {
        let by_id = weights
            .iter()
            .map(|weight| (weight.id, weight))
            .collect::<HashMap<_, _>>();
        let mut objects = Vec::new();
        let mut weight_to_object = HashMap::with_capacity(weights.len());

        for group in groups {
            let mut members = group
                .weights
                .iter()
                .filter_map(|id| by_id.get(id).copied())
                .collect::<Vec<_>>();
            members.sort_by_key(|weight| weight.home.offset);
            let mut run: Vec<&WeightDescriptor> = Vec::new();
            for weight in members {
                let can_merge = run.last().is_none_or(|last| {
                    let start = run.first().expect("non-empty run").home.offset;
                    let last_end = last.home.offset.saturating_add(last.home.len);
                    let new_end = weight.home.offset.saturating_add(weight.home.len);
                    last.home.provider == weight.home.provider
                        && last.home.path == weight.home.path
                        && weight.home.offset <= last_end.saturating_add(max_gap)
                        && new_end.saturating_sub(start) <= max_object_bytes
                });
                if !can_merge {
                    Self::finish_object(&mut objects, &mut weight_to_object, &mut run)?;
                }
                run.push(weight);
            }
            Self::finish_object(&mut objects, &mut weight_to_object, &mut run)?;
        }

        Ok(Self {
            objects: objects.into_boxed_slice(),
            weight_to_object,
        })
    }

    fn finish_object(
        objects: &mut Vec<StorageObject>,
        index: &mut HashMap<WeightId, StorageObjectId>,
        run: &mut Vec<&WeightDescriptor>,
    ) -> Result<()> {
        if run.is_empty() {
            return Ok(());
        }
        let first = run[0];
        const DIRECT_ALIGNMENT: u64 = 4096;
        let useful_start = first.home.offset;
        let useful_end = run
            .iter()
            .map(|weight| weight.home.offset.saturating_add(weight.home.len))
            .max()
            .unwrap_or(useful_start);
        let start = useful_start / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
        let aligned_end = useful_end.div_ceil(DIRECT_ALIGNMENT) * DIRECT_ALIGNMENT;
        let file_len = std::fs::metadata(&first.home.path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        if useful_end > file_len && file_len > 0 {
            return Err(FellmError::other(format!(
                "corrupt/truncated GGUF shard:\ntensor {} requires bytes {useful_start}..{useful_end},\nbut shard {} is only {file_len} bytes",
                run.last().map(|weight| weight.name.as_str()).unwrap_or("?"),
                first.home.path.display()
            )));
        }
        let id = StorageObjectId(objects.len() as u64);
        let mut useful_bytes = 0u64;
        let names: Vec<&str> = run.iter().map(|weight| weight.name.as_str()).collect();
        let raw_offsets: Vec<u64> = run.iter().map(|weight| weight.home.offset).collect();
        let byte_sizes: Vec<u64> = run.iter().map(|weight| weight.byte_len).collect();
        let tensor_ends: Vec<u64> = run
            .iter()
            .map(|weight| weight.home.offset.saturating_add(weight.home.len))
            .collect();
        let members = run
            .drain(..)
            .map(|weight| {
                useful_bytes = useful_bytes.saturating_add(weight.byte_len);
                index.insert(weight.id, id);
                StorageObjectMember {
                    weight: weight.id,
                    offset: weight.home.offset.saturating_sub(start),
                    len: weight.byte_len,
                }
            })
            .collect();
        if useful_end < start {
            return Err(FellmError::other("storage object extent overflow"));
        }
        tracing::trace!(
            object_id = id.0,
            tensors = ?names,
            tensor_raw_offsets = ?raw_offsets,
            tensor_byte_sizes = ?byte_sizes,
            tensor_ends = ?tensor_ends,
            coalesced_logical_start = useful_start,
            coalesced_logical_end = useful_end,
            aligned_physical_start = start,
            aligned_physical_end = aligned_end,
            shard_payload_base = start,
            resolved_file_start = start,
            resolved_file_end = useful_end,
            actual_file_size = file_len,
            alignment_padding = aligned_end.saturating_sub(useful_end),
            "storage object extent"
        );
        if aligned_end > file_len && file_len > 0 {
            tracing::debug!(
                object_id = id.0,
                aligned_physical_end = aligned_end,
                logical_end = useful_end,
                file_size = file_len,
                padding = aligned_end.saturating_sub(useful_end),
                "clipping Direct I/O alignment padding that would extend past EOF"
            );
        }
        objects.push(StorageObject {
            id,
            extent: StorageExtent {
                provider: first.home.provider.clone(),
                path: first.home.path.clone(),
                offset: start,
                len: useful_end - start,
                alignment: DIRECT_ALIGNMENT,
            },
            members,
            useful_bytes,
        });
        Ok(())
    }

    #[must_use]
    pub fn object_for_weight(&self, weight: WeightId) -> Option<&StorageObject> {
        let id = *self.weight_to_object.get(&weight)?;
        self.objects.get(id.0 as usize)
    }

    #[must_use]
    pub fn objects(&self) -> &[StorageObject] {
        &self.objects
    }

    #[must_use]
    pub fn metadata_bytes(&self) -> usize {
        self.objects.len() * std::mem::size_of::<StorageObject>()
            + self.weight_to_object.capacity()
                * (std::mem::size_of::<WeightId>() + std::mem::size_of::<StorageObjectId>())
            + self
                .objects
                .iter()
                .map(|object| {
                    object.members.capacity() * std::mem::size_of::<StorageObjectMember>()
                })
                .sum::<usize>()
    }

    #[must_use]
    pub fn physical_bytes(&self) -> u64 {
        self.objects.iter().map(|object| object.extent.len).sum()
    }

    #[must_use]
    pub fn useful_bytes(&self) -> u64 {
        self.objects.iter().map(|object| object.useful_bytes).sum()
    }

    /// Distinct physical objects referenced by any one execution group.
    #[must_use]
    pub fn max_objects_per_group(&self, groups: &[ExecutionGroup]) -> usize {
        groups
            .iter()
            .map(|group| {
                group
                    .weights
                    .iter()
                    .filter_map(|id| self.weight_to_object.get(id).copied())
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            })
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn weight(id: u64, offset: u64, len: u64) -> WeightDescriptor {
        WeightDescriptor {
            id: WeightId(id),
            name: format!("w{id}"),
            home: StorageExtent {
                provider: "gguf".into(),
                path: PathBuf::from("model.gguf"),
                offset,
                len,
                alignment: 32,
            },
            byte_len: len,
            replicas: Vec::new(),
        }
    }

    #[test]
    fn builds_graph_local_objects_and_constant_time_lookup() {
        let weights = [
            weight(1, 4096, 1024),
            weight(2, 6144, 1024),
            weight(3, 16384, 512),
        ];
        let groups = [ExecutionGroup {
            id: 0,
            weights: vec![WeightId(2), WeightId(1), WeightId(3)],
            byte_len: 2560,
            first_op: 0,
            last_op: 2,
            reuse_count: 3,
            cpu_compute_time: None,
        }];
        let index =
            StorageObjectIndex::from_execution_groups(&weights, &groups, 2048, 8192).unwrap();
        assert_eq!(index.objects().len(), 2);
        let object = index.object_for_weight(WeightId(2)).unwrap();
        assert_eq!(object.members.len(), 2);
        assert_eq!(object.extent.offset, 4096);
        assert_eq!(object.extent.len, 3072);
        assert_eq!(object.extent.alignment, 4096);
        assert_eq!(object.useful_bytes, 2048);
        assert!(index.metadata_bytes() < 2048);
    }

    #[test]
    fn logical_extent_does_not_include_eof_alignment_padding() {
        let weights = [weight(1, 4096, 1000)];
        let groups = [ExecutionGroup {
            id: 0,
            weights: vec![WeightId(1)],
            byte_len: 1000,
            first_op: 0,
            last_op: 0,
            reuse_count: 1,
            cpu_compute_time: None,
        }];
        let index =
            StorageObjectIndex::from_execution_groups(&weights, &groups, 2048, 8192).unwrap();
        let object = index.object_for_weight(WeightId(1)).unwrap();
        assert_eq!(object.extent.offset, 4096);
        assert_eq!(object.extent.len, 1000);
        assert_eq!(object.members[0].offset, 0);
        assert_eq!(object.members[0].len, 1000);
    }
}
