//! Shapes and strides. Fixed-capacity to avoid heap allocation per tensor.

use crate::dtype::DType;
use crate::error::{FellmError, Result};
use smallvec::SmallVec;

/// Maximum tensor rank supported. GGUF caps at 4; we allow one more for
/// internal scratch (e.g. batched attention).
pub const MAX_RANK: usize = 5;

/// A shape: up to [`MAX_RANK`] dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(SmallVec<[u64; MAX_RANK]>);

impl Shape {
    /// Construct a shape from a slice.
    pub fn new(dims: &[u64]) -> Result<Self> {
        if dims.len() > MAX_RANK {
            return Err(FellmError::RankTooHigh(dims.len()));
        }
        Ok(Self(dims.iter().copied().collect()))
    }

    /// Scalar shape.
    #[must_use]
    pub fn scalar() -> Self {
        Self(SmallVec::new())
    }

    /// Rank (number of dimensions).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        if self.0.is_empty() {
            1
        } else {
            self.0.iter().product::<u64>() as usize
        }
    }

    /// Dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.0
    }

    /// The i-th dimension.
    ///
    /// # Panics
    /// If `i >= rank()`.
    #[must_use]
    pub fn dim(&self, i: usize) -> u64 {
        self.0[i]
    }

    /// Compute default row-major (C-order) strides in elements.
    #[must_use]
    pub fn row_major_strides(&self) -> Strides {
        let mut strides = SmallVec::<[u64; MAX_RANK]>::new();
        strides.resize(self.0.len(), 1);
        let n = self.0.len();
        for i in (0..n.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        Strides(strides)
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Strides in *elements* (not bytes).
///
/// For quantized dtypes an "element" is an abstraction; the [`Layout`] carries
/// the block-view metadata needed to interpret raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides(SmallVec<[u64; MAX_RANK]>);

impl Strides {
    /// Construct from a slice.
    pub fn new(strides: &[u64]) -> Result<Self> {
        if strides.len() > MAX_RANK {
            return Err(FellmError::RankTooHigh(strides.len()));
        }
        Ok(Self(strides.iter().copied().collect()))
    }

    /// As a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }
}

/// How the raw bytes of a tensor should be interpreted.
///
/// `Dense` covers all non-quantized dtypes and quantized dtypes that are
/// stored contiguously in canonical block order. `Blocked` records the
/// block-view metadata explicitly for interoperability with dequant kernels.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Element type.
    pub dtype: DType,
    /// Shape in logical elements.
    pub shape: Shape,
    /// Strides in *elements* — only meaningful for non-quantized dtypes or
    /// when the tensor is a contiguous block-major layout for a quantized dtype.
    pub strides: Strides,
    /// Byte offset from the start of the underlying storage.
    pub offset_bytes: usize,
}

impl Layout {
    /// Build a contiguous row-major layout.
    pub fn contiguous(dtype: DType, shape: Shape) -> Self {
        let strides = shape.row_major_strides();
        Self {
            dtype,
            shape,
            strides,
            offset_bytes: 0,
        }
    }

    /// True if the tensor is contiguous row-major (dense element view).
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        let expected = self.shape.row_major_strides();
        expected == self.strides
    }

    /// Total byte size of the payload.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.dtype.byte_size(self.shape.num_elements())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_row_major() {
        let s = Shape::new(&[2, 3, 4]).unwrap();
        assert_eq!(s.row_major_strides().as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn num_elements() {
        assert_eq!(Shape::new(&[2, 3, 4]).unwrap().num_elements(), 24);
        assert_eq!(Shape::scalar().num_elements(), 1);
    }

    #[test]
    fn rank_limit() {
        assert!(Shape::new(&[1, 2, 3, 4, 5, 6]).is_err());
    }
}
