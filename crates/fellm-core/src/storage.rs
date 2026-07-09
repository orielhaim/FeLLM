//! Backing storage for tensors: mmap, owned heap, or slice-of-parent view.

use memmap2::Mmap;
use std::alloc::{Layout as StdLayout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::Arc;

/// A tensor's backing bytes.
#[derive(Clone)]
pub enum Storage {
    /// A slice into a memory-mapped file (e.g. GGUF weights).
    Mmap {
        /// Shared handle keeping the mapping alive.
        mmap: Arc<Mmap>,
        /// Offset in bytes from the start of the mapping.
        offset: usize,
        /// Length in bytes.
        len: usize,
    },
    /// Owned, aligned heap buffer.
    Owned(Arc<AlignedBuffer>),
    /// A borrowed view into another storage.
    View {
        /// Parent storage.
        parent: Arc<Storage>,
        /// Offset in bytes relative to the parent's start-of-data.
        offset: usize,
        /// Length in bytes.
        len: usize,
    },
}

impl Storage {
    /// Get a read-only view of the raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Mmap { mmap, offset, len } => &mmap[*offset..*offset + *len],
            Self::Owned(buf) => buf.as_slice(),
            Self::View {
                parent,
                offset,
                len,
            } => {
                let base = parent.as_bytes();
                &base[*offset..*offset + *len]
            }
        }
    }

    /// Byte length.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.as_bytes().len()
    }
}

impl core::fmt::Debug for Storage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Mmap { len, .. } => f.debug_struct("Mmap").field("len", len).finish(),
            Self::Owned(b) => f.debug_struct("Owned").field("len", &b.len()).finish(),
            Self::View { len, .. } => f.debug_struct("View").field("len", len).finish(),
        }
    }
}

/// An aligned heap buffer.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

// SAFETY: The pointer is uniquely owned by this buffer and only exposed
// through &self / &mut self methods.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate `len` bytes with the given alignment (defaults to 64).
    ///
    /// The buffer is zero-initialized.
    ///
    /// # Panics
    /// If allocation fails or `align` is invalid.
    #[must_use]
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "align must be power of two");
        let align = align.max(1);
        let layout = StdLayout::from_size_align(len.max(1), align).expect("valid layout");
        // SAFETY: layout has non-zero size (we forced it above).
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).expect("allocation failed");
        // SAFETY: freshly allocated, size = len.
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0, len);
        }
        Self { ptr, len, align }
    }

    /// Read-only byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for `len` bytes, aligned, alive for lifetime of self.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable byte slice.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for `len` bytes, aligned, alive for lifetime of self,
        // and we hold &mut self.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment.
    #[must_use]
    pub fn align(&self) -> usize {
        self.align
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated with the same layout in `new_zeroed`.
        let layout =
            StdLayout::from_size_align(self.len.max(1), self.align).expect("valid layout on drop");
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffer_zeroed() {
        let b = AlignedBuffer::new_zeroed(128, 64);
        assert_eq!(b.len(), 128);
        assert!(b.as_slice().iter().all(|&x| x == 0));
        assert_eq!(b.ptr.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn aligned_buffer_mut() {
        let mut b = AlignedBuffer::new_zeroed(16, 16);
        b.as_mut_slice()[0] = 42;
        assert_eq!(b.as_slice()[0], 42);
    }
}
