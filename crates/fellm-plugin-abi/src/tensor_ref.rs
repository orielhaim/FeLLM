//! FFI-safe tensor references.
//!
//! These are pure descriptors: dtype + shape + strides + raw pointer + len.
//! No `Arc`, no `Vec`, no destructors.

use fellm_core::dtype::DType;

/// Maximum rank exposed at the FFI boundary.
pub const ABI_MAX_RANK: usize = 5;

/// Read-only tensor descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorRef {
    /// Stable logical identity. Zero means an ephemeral activation/input.
    pub logical_id: u64,
    /// dtype as ggml code (see [`DType::from_ggml_code`]).
    pub dtype: u32,
    /// Number of dimensions.
    pub rank: u32,
    /// Dimensions, in row-major order. Positions >= `rank` are 0.
    pub dims: [u64; ABI_MAX_RANK],
    /// Strides in *elements*.
    pub strides: [u64; ABI_MAX_RANK],
    /// Raw pointer to first byte.
    pub data: *const u8,
    /// Total byte length starting at `data`.
    pub byte_len: u64,
}

// SAFETY: TensorRef only holds a pointer that the caller owns; sending it
// across threads is safe as long as the caller ensures the backing storage
// is `Send`. Every actual sender in FeLLM does.
unsafe impl Send for TensorRef {}
unsafe impl Sync for TensorRef {}

/// Mutable tensor descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorMut {
    /// dtype as ggml code.
    pub dtype: u32,
    /// Rank.
    pub rank: u32,
    /// Dimensions.
    pub dims: [u64; ABI_MAX_RANK],
    /// Strides in elements.
    pub strides: [u64; ABI_MAX_RANK],
    /// Mutable pointer to first byte.
    pub data: *mut u8,
    /// Byte length.
    pub byte_len: u64,
}

// SAFETY: Same rationale as TensorRef.
unsafe impl Send for TensorMut {}
unsafe impl Sync for TensorMut {}

impl TensorRef {
    /// Build a TensorRef from raw parts.
    ///
    /// # Safety
    /// `data` must be valid for `byte_len` bytes for the lifetime of use.
    pub unsafe fn from_raw(
        dtype: DType,
        dims: &[u64],
        strides: &[u64],
        data: *const u8,
        byte_len: usize,
    ) -> Self {
        let mut d = [0u64; ABI_MAX_RANK];
        let mut s = [0u64; ABI_MAX_RANK];
        for (i, &v) in dims.iter().enumerate() {
            d[i] = v;
        }
        for (i, &v) in strides.iter().enumerate() {
            s[i] = v;
        }
        Self {
            logical_id: 0,
            dtype: dtype as u32,
            rank: dims.len() as u32,
            dims: d,
            strides: s,
            data,
            byte_len: byte_len as u64,
        }
    }

    /// Attach a stable logical identity to an immutable tensor view.
    #[must_use]
    pub const fn with_logical_id(mut self, logical_id: u64) -> Self {
        self.logical_id = logical_id;
        self
    }

    /// The dtype.
    pub fn dtype(&self) -> Option<DType> {
        DType::from_ggml_code(self.dtype).ok()
    }

    /// Dimensions as a slice.
    pub fn dims_slice(&self) -> &[u64] {
        &self.dims[..self.rank as usize]
    }

    /// Strides as a slice.
    pub fn strides_slice(&self) -> &[u64] {
        &self.strides[..self.rank as usize]
    }

    /// View as `&[u8]`.
    ///
    /// # Safety
    /// The caller must uphold the lifetime and validity invariants.
    pub unsafe fn as_bytes(&self) -> &[u8] {
        // SAFETY: caller guarantees data+byte_len is a valid readable slice.
        unsafe { core::slice::from_raw_parts(self.data, self.byte_len as usize) }
    }
}

impl TensorMut {
    /// Build a TensorMut from raw parts.
    ///
    /// # Safety
    /// `data` must be valid for `byte_len` bytes for exclusive writes.
    pub unsafe fn from_raw(
        dtype: DType,
        dims: &[u64],
        strides: &[u64],
        data: *mut u8,
        byte_len: usize,
    ) -> Self {
        let mut d = [0u64; ABI_MAX_RANK];
        let mut s = [0u64; ABI_MAX_RANK];
        for (i, &v) in dims.iter().enumerate() {
            d[i] = v;
        }
        for (i, &v) in strides.iter().enumerate() {
            s[i] = v;
        }
        Self {
            dtype: dtype as u32,
            rank: dims.len() as u32,
            dims: d,
            strides: s,
            data,
            byte_len: byte_len as u64,
        }
    }

    /// The dtype.
    pub fn dtype(&self) -> Option<DType> {
        DType::from_ggml_code(self.dtype).ok()
    }

    /// Dimensions as a slice.
    pub fn dims_slice(&self) -> &[u64] {
        &self.dims[..self.rank as usize]
    }

    /// Strides as a slice.
    pub fn strides_slice(&self) -> &[u64] {
        &self.strides[..self.rank as usize]
    }

    /// View as `&mut [u8]`.
    ///
    /// # Safety
    /// The caller must uphold aliasing/lifetime invariants.
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: caller guarantees exclusive access to data+byte_len.
        unsafe { core::slice::from_raw_parts_mut(self.data, self.byte_len as usize) }
    }
}
