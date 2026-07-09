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
}
