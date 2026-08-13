//! The `Tensor` type: metadata + storage.

use crate::dtype::DType;
use crate::error::{FellmError, Result};
use crate::shape::{Layout, Shape};
use crate::storage::{AlignedBuffer, Storage};
use std::sync::Arc;

/// A tensor is a `Layout` plus a `Storage`.
///
/// Cloning is cheap: the storage is `Arc`-shared.
#[derive(Clone, Debug)]
pub struct Tensor {
    layout: Layout,
    storage: Arc<Storage>,
}

impl Tensor {
    /// Construct from parts.
    pub fn from_storage(layout: Layout, storage: Arc<Storage>) -> Self {
        Self { layout, storage }
    }

    /// Allocate a new zero-initialized tensor of the given dtype and shape.
    ///
    /// Uses 64-byte alignment.
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        let layout = Layout::contiguous(dtype, shape);
        let bytes = layout.byte_size();
        let buf = AlignedBuffer::new_zeroed(bytes, 64);
        let storage = Arc::new(Storage::Owned(Arc::new(buf)));
        Self { layout, storage }
    }

    /// Layout accessor.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Dtype accessor.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.layout.dtype
    }

    /// Shape accessor.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.layout.shape
    }

    /// Storage accessor.
    #[must_use]
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// Read-only raw bytes covering the tensor's payload.
    ///
    /// Respects `layout.offset_bytes`.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        let all = self.storage.as_bytes();
        let start = self.layout.offset_bytes;
        let end = start + self.layout.byte_size();
        &all[start..end]
    }

    /// Stable payload extent in the original mmap backing store, when applicable.
    #[must_use]
    pub fn mmap_extent(&self) -> Option<(usize, usize)> {
        self.storage.mmap_extent().map(|(base, _)| {
            (
                base.saturating_add(self.layout.offset_bytes),
                self.layout.byte_size(),
            )
        })
    }

    /// Stable payload extent in a source file when the model was opened metadata-only.
    #[must_use]
    pub fn file_extent(&self) -> Option<(&std::path::PathBuf, u64, usize)> {
        self.storage.file_extent().map(|(path, base, _)| {
            (
                path,
                base.saturating_add(self.layout.offset_bytes as u64),
                self.layout.byte_size(),
            )
        })
    }

    /// Interpret the payload as `&[T]` if the dtype matches size-wise and the
    /// tensor is contiguous.
    ///
    /// # Errors
    /// If the tensor is quantized or non-contiguous.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        if self.layout.dtype.is_quantized() {
            return Err(FellmError::UnsupportedDType(self.layout.dtype));
        }
        if !self.layout.is_contiguous() {
            return Err(FellmError::other("as_slice requires contiguous layout"));
        }
        let bytes = self.as_bytes();
        bytemuck::try_cast_slice(bytes)
            .map_err(|e| FellmError::other(format!("bytemuck cast failed: {e:?}")))
    }

    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> Result<&mut [T]> {
        if self.layout.dtype.is_quantized() {
            return Err(FellmError::UnsupportedDType(self.layout.dtype));
        }
        if !self.layout.is_contiguous() {
            return Err(FellmError::other("as_mut_slice requires contiguous layout"));
        }
        let offset = self.layout.offset_bytes;
        let byte_len = self.layout.byte_size();
        let storage = Arc::get_mut(&mut self.storage)
            .ok_or_else(|| FellmError::other("as_mut_slice requires uniquely owned storage Arc"))?;
        let Storage::Owned(buf) = storage else {
            return Err(FellmError::other("as_mut_slice requires Storage::Owned"));
        };
        let buf = Arc::get_mut(buf).ok_or_else(|| {
            FellmError::other("as_mut_slice requires uniquely owned AlignedBuffer")
        })?;
        let bytes = &mut buf.as_mut_slice()[offset..offset + byte_len];
        bytemuck::try_cast_slice_mut(bytes)
            .map_err(|e| FellmError::other(format!("bytemuck cast failed: {e:?}")))
    }

    /// Create a zero-copy view of one row from a contiguous rank-2 tensor.
    pub fn row(&self, index: usize) -> Result<Self> {
        let dims = self.layout.shape.dims();
        if dims.len() != 2 || !self.layout.is_contiguous() {
            return Err(FellmError::other("row requires a contiguous rank-2 tensor"));
        }
        let rows = dims[0] as usize;
        let columns = dims[1] as usize;
        if index >= rows {
            return Err(FellmError::other(format!(
                "row index {index} out of bounds for {rows} rows"
            )));
        }
        let mut layout = Layout::contiguous(self.layout.dtype, Shape::new(&[dims[1]])?);
        layout.offset_bytes =
            self.layout.offset_bytes + index * self.layout.dtype.byte_size(columns);
        Ok(Self::from_storage(layout, self.storage.clone()))
    }
}
