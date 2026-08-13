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
use std::io::Read;
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
    mmap: Option<Arc<Mmap>>,
    source_path: Option<std::path::PathBuf>,
    metadata_only: bool,
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
        #[cfg(target_os = "linux")]
        if std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .is_ok_and(|release| release.to_ascii_lowercase().contains("microsoft"))
            && path.as_ref().starts_with("/mnt/")
        {
            tracing::warn!(
                path = %path.as_ref().display(),
                "WSL model is on a Windows-mounted filesystem; use the Linux filesystem for substantially faster model paging"
            );
        }
        let file = File::open(path.as_ref())?;
        // SAFETY: the file remains open for the lifetime of the mapping,
        // and we treat the bytes as immutable throughout.
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(target_os = "linux")]
        {
            // Dense CPU decode cyclically scans a very large, immutable mapping. Ask Linux to
            // collapse page-table coverage where supported; this is only a hint and preserves
            // the bounded, file-backed storage model when file THP is unavailable.
            let _ = mmap.advise(memmap2::Advice::HugePage);
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }
        let mut gguf = Self::from_mmap(Arc::new(mmap))?;
        gguf.source_path = Some(path.as_ref().to_path_buf());
        Ok(gguf)
    }

    /// Open only the bounded GGUF metadata/index. Weight payloads remain file extents and no
    /// model-wide virtual mapping is created.
    pub fn open_storage_native<P: AsRef<Path>>(path: P) -> Result<Self> {
        const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
        let mut file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len();
        let header_len = file_len.min(MAX_HEADER_BYTES) as usize;
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)?;
        let mut gguf = Self::from_bytes(&header, None)?;
        if gguf.tensor_data_offset > MAX_HEADER_BYTES {
            return Err(FellmError::other(format!(
                "GGUF metadata/index exceeds bounded {} MiB header reader",
                MAX_HEADER_BYTES >> 20
            )));
        }
        gguf.source_path = Some(path.as_ref().to_path_buf());
        gguf.metadata_only = true;
        tracing::info!(
            metadata_bytes = gguf.tensor_data_offset,
            "opened storage-native GGUF index without mapping weight payloads"
        );
        Ok(gguf)
    }

    /// Construct from an existing memory map.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        Self::from_bytes(&mmap, Some(Arc::clone(&mmap)))
    }

    fn from_bytes(bytes: &[u8], mmap: Option<Arc<Mmap>>) -> Result<Self> {
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
            source_path: None,
            metadata_only: false,
            tensor_data_offset,
            alignment,
            metadata,
            tensor_infos,
            by_name,
        })
    }

    /// Get a shared handle to the underlying mmap.
    #[must_use]
    pub fn mmap(&self) -> Option<&Arc<Mmap>> {
        self.mmap.as_ref()
    }

    /// Original backing-store path, when opened from a file.
    #[must_use]
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
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
        let storage =
            if let Some(mmap) = &self.mmap {
                Storage::Mmap {
                    mmap: Arc::clone(mmap),
                    offset: absolute as usize,
                    len: byte_size,
                }
            } else {
                Storage::FileExtent {
                    path: Arc::new(self.source_path.clone().ok_or_else(|| {
                        FellmError::other("storage-native GGUF has no source path")
                    })?),
                    offset: absolute,
                    len: byte_size,
                }
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
