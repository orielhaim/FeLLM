//! Pure-Rust CPU backend.
//!
//! Uses `faer` (which internally uses `pulp` for SIMD) for f32 matmul,
//! hand-rolled dequant + fused-matmul kernels for quantized weight paths,
//! and small `wide`-based kernels for norm/rope/silu/softmax.

#![deny(missing_docs)]

pub mod backend;
pub mod dequant;
pub mod kernels;

pub use backend::CpuBackend;
