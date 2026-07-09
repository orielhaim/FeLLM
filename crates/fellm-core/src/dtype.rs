//! Element types. Mirrors GGML's `ggml_type` plus standard non-quantized types.

use crate::error::{FellmError, Result};

/// The element type of a tensor.
///
/// Values are stable and match GGUF's `ggml_type` where applicable.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    // Standard floating point
    /// IEEE 754 single-precision.
    F32 = 0,
    /// IEEE 754 half-precision.
    F16 = 1,

    // GGUF k-quants and legacy quants (values match ggml_type)
    /// GGUF Q4_0.
    Q4_0 = 2,
    /// GGUF Q4_1.
    Q4_1 = 3,
    /// GGUF Q5_0.
    Q5_0 = 6,
    /// GGUF Q5_1.
    Q5_1 = 7,
    /// GGUF Q8_0.
    Q8_0 = 8,
    /// GGUF Q8_1.
    Q8_1 = 9,
    /// GGUF Q2_K.
    Q2K = 10,
    /// GGUF Q3_K.
    Q3K = 11,
    /// GGUF Q4_K.
    Q4K = 12,
    /// GGUF Q5_K.
    Q5K = 13,
    /// GGUF Q6_K.
    Q6K = 14,
    /// GGUF Q8_K.
    Q8K = 15,

    // Integer / bool
    /// Signed 8-bit.
    I8 = 24,
    /// Signed 16-bit.
    I16 = 25,
    /// Signed 32-bit.
    I32 = 26,
    /// Signed 64-bit.
    I64 = 27,
    /// Unsigned 8-bit.
    U8 = 28,
    /// Unsigned 16-bit.
    U16 = 29,
    /// Unsigned 32-bit.
    U32 = 30,
    /// Unsigned 64-bit.
    U64 = 31,
    /// Bool as 1 byte.
    Bool = 32,

    // Extra float
    /// bfloat16.
    BF16 = 33,
}

impl DType {
    /// Convert from the raw `ggml_type` code as it appears in GGUF files.
    pub fn from_ggml_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::U8,
            29 => Self::U16,
            30 => Self::U32,
            31 => Self::U64,
            32 => Self::Bool,
            33 => Self::BF16,
            other => return Err(FellmError::UnknownDType(other)),
        })
    }

    /// Number of elements packed in one block.
    ///
    /// For non-quantized types this is always 1.
    #[must_use]
    pub const fn elements_per_block(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 1,
        }
    }

    /// Number of bytes in one block.
    #[must_use]
    pub const fn bytes_per_block(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I64 | Self::U64 => 8,
            // Legacy 4-bit quants: 2 bytes scale (fp16) + 16 bytes weights = 18
            Self::Q4_0 => 18,
            // Q4_1: 2 bytes scale + 2 bytes min + 16 bytes weights = 20
            Self::Q4_1 => 20,
            // Q5_0: 2 bytes scale + 4 bytes qh + 16 bytes weights = 22
            Self::Q5_0 => 22,
            // Q5_1: 2 + 2 + 4 + 16 = 24
            Self::Q5_1 => 24,
            // Q8_0: 2 bytes scale + 32 bytes weights = 34
            Self::Q8_0 => 34,
            // Q8_1: 4 (d as f32) + 4 (s as f32) + 32 = but actually 2+2+32=36. Use GGML canonical: 36.
            Self::Q8_1 => 36,
            // K-quants: super-block of 256 weights
            // Q2_K: 16 (scales as u8) + 64 (weights, 2bpw) + 2 (d) + 2 (dmin) = 84
            Self::Q2K => 84,
            // Q3_K: 32 (hmask) + 64 (weights, 2bpw) + 12 (scales) + 2 (d) = 110
            Self::Q3K => 110,
            // Q4_K: 2 (d) + 2 (dmin) + 12 (scales) + 128 (weights, 4bpw) = 144
            Self::Q4K => 144,
            // Q5_K: 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (weights) = 176
            Self::Q5K => 176,
            // Q6_K: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210
            Self::Q6K => 210,
            // Q8_K: 4 (d as f32) + 256 (weights) + 32 (bsums as i16 pairs -> 16*2 = 32) = 292
            Self::Q8K => 292,
        }
    }

    /// True if this type is a GGUF k-quant / legacy quant format.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q8K
        )
    }

    /// True if this is a floating point type (dense).
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Compute the number of bytes needed to store `n_elements` of this dtype.
    #[must_use]
    pub const fn byte_size(self, n_elements: usize) -> usize {
        let epb = self.elements_per_block();
        let bpb = self.bytes_per_block();
        // Round up in blocks
        n_elements.div_ceil(epb) * bpb
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q8K => "q8_k",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_block_size_matches_ggml() {
        assert_eq!(DType::Q4K.bytes_per_block(), 144);
        assert_eq!(DType::Q4K.elements_per_block(), 256);
    }

    #[test]
    fn q6k_block_size_matches_ggml() {
        assert_eq!(DType::Q6K.bytes_per_block(), 210);
        assert_eq!(DType::Q6K.elements_per_block(), 256);
    }

    #[test]
    fn f32_byte_size() {
        assert_eq!(DType::F32.byte_size(10), 40);
    }

    #[test]
    fn q4k_byte_size_rounds_up() {
        // 500 elements => ceil(500/256) = 2 blocks => 288 bytes
        assert_eq!(DType::Q4K.byte_size(500), 288);
    }

    #[test]
    fn from_ggml_code_roundtrip() {
        assert_eq!(DType::from_ggml_code(12).unwrap(), DType::Q4K);
        assert_eq!(DType::from_ggml_code(14).unwrap(), DType::Q6K);
        assert!(DType::from_ggml_code(9999).is_err());
    }
}
