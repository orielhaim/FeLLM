//! Top-level GGUF file loader.

use crate::meta::{MetaMap, read_value};
use crate::reader::Reader;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::Storage;
use fellm_core::tensor::Tensor;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

/// Per-tensor descriptor from the GGUF header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Name.
    pub name: String,
    /// Element type.
    pub dtype: DType,
    /// Shape (row-major, GGUF stores in reverse — we normalize).
    pub shape: Shape,
    /// Offset in bytes relative to the start of `tensor_data`.
    pub relative_offset: u64,
}

/// A loaded GGUF file.
///
/// The file is memory-mapped; tensor accesses are zero-copy views.
pub struct GgufFile {
    mmap: Arc<Mmap>,
    /// Absolute offset within the file where tensor payloads start.
    tensor_data_offset: u64,
    /// Alignment (bytes) required for tensor payloads.
    alignment: u64,
    /// Metadata KV.
    pub metadata: MetaMap,
    /// Ordered list of tensors (as declared).
    pub tensor_infos: Vec<TensorInfo>,
    /// Lookup by name.
    by_name: BTreeMap<String, usize>,
}

impl GgufFile {
    /// Open a GGUF file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: the file remains open for the lifetime of the mapping,
        // and we treat the bytes as immutable throughout.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(Arc::new(mmap))
    }

    /// Construct from an existing memory map.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let bytes: &[u8] = &mmap;
        let mut r = Reader::new(bytes);

        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            return Err(FellmError::BadGgufMagic(magic));
        }
        let version = r.u32()?;
        if version != 3 && version != 2 {
            return Err(FellmError::UnsupportedGgufVersion(version));
        }
        let tensor_count = r.u64()?;
        let meta_count = r.u64()?;

        // Metadata KV
        let mut metadata = MetaMap::new();
        for _ in 0..meta_count {
            let key = r.gguf_string()?;
            let val = read_value(&mut r)?;
            metadata.insert(key, val);
        }

        // Tensor infos
        let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = r.gguf_string()?;
            let n_dims = r.u32()? as usize;
            if n_dims > 4 {
                return Err(FellmError::parse(format!(
                    "GGUF tensor {name} has {n_dims} dims (> 4)"
                )));
            }
            let mut dims_reversed = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims_reversed.push(r.u64()?);
            }
            // GGUF stores dimensions in reverse (fortran-like). Reverse to
            // get row-major.
            dims_reversed.reverse();
            let shape = Shape::new(&dims_reversed)?;
            let ggml_type = r.u32()?;
            let dtype = DType::from_ggml_code(ggml_type)?;
            let relative_offset = r.u64()?;
            tensor_infos.push(TensorInfo {
                name,
                dtype,
                shape,
                relative_offset,
            });
        }

        // Alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                crate::meta::MetaValue::U32(x) => Some(u64::from(*x)),
                crate::meta::MetaValue::U64(x) => Some(*x),
                _ => None,
            })
            .unwrap_or(32);

        // Align current cursor upward to `alignment`.
        let unaligned = r.pos() as u64;
        let tensor_data_offset = align_up(unaligned, alignment);

        // Build name lookup.
        let mut by_name = BTreeMap::new();
        for (i, ti) in tensor_infos.iter().enumerate() {
            by_name.insert(ti.name.clone(), i);
        }

        tracing::debug!(
            gguf_version = version,
            tensor_count,
            meta_count,
            alignment,
            tensor_data_offset,
            "opened GGUF file"
        );

        Ok(Self {
            mmap,
            tensor_data_offset,
            alignment,
            metadata,
            tensor_infos,
            by_name,
        })
    }

    /// Get a shared handle to the underlying mmap.
    #[must_use]
    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }

    /// Absolute byte offset where tensor data starts.
    #[must_use]
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Alignment.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Look up a tensor by name and return a zero-copy `Tensor`.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let idx = self
            .by_name
            .get(name)
            .ok_or_else(|| FellmError::TensorNotFound(name.into()))?;
        self.tensor_at(*idx)
    }

    /// Look up a tensor by index.
    pub fn tensor_at(&self, idx: usize) -> Result<Tensor> {
        let ti = self
            .tensor_infos
            .get(idx)
            .ok_or_else(|| FellmError::parse(format!("tensor index {idx} out of range")))?;
        let byte_size = ti.dtype.byte_size(ti.shape.num_elements());
        let absolute = self.tensor_data_offset + ti.relative_offset;
        let layout = Layout {
            dtype: ti.dtype,
            shape: ti.shape.clone(),
            strides: ti.shape.row_major_strides(),
            offset_bytes: 0,
        };
        let storage = Storage::Mmap {
            mmap: Arc::clone(&self.mmap),
            offset: absolute as usize,
            len: byte_size,
        };
        Ok(Tensor::from_storage(layout, Arc::new(storage)))
    }

    /// True if a tensor with this name exists.
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Iterate over all tensor infos.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensor_infos.iter()
    }
}

fn align_up(x: u64, align: u64) -> u64 {
    if align <= 1 {
        return x;
    }
    (x + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 1), 100);
    }
}
