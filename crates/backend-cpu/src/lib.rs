//! Pure-Rust CPU backend.
//!
//! Uses `faer` (which internally uses `pulp` for SIMD) for f32 matmul,
//! hand-rolled dequant + fused-matmul kernels for quantized weight paths,
//! and small `wide`-based kernels for norm/rope/silu/softmax.

#![deny(missing_docs)]

/// CPU backend implementing the FeLLM plugin ABI.
pub mod backend;
/// GGUF k-quant / legacy quant dequantization.
pub mod dequant;
/// Hand-rolled CPU kernels (matmul, attention, RoPE, sampling, …).
pub mod kernels;

pub use backend::CpuBackend;
