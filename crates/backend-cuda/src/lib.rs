//! CUDA compute backend (stable Rust + cudarc).
//!
//! Hot kernels (`Q4_K` GEMV, `PagedAttention`) are loaded from dynamic plugins
//! built with cuda-oxide under WSL2. This crate owns device context, VRAM
//! arenas, pinned swap, and CUDA graph capture.

#![deny(missing_docs)]

mod backend;
mod device;
mod pinned_swap;
mod plan;
mod vram_pool;

pub use backend::CudaBackend;
pub use device::CudaDeviceState;
pub use pinned_swap::PinnedSwapArena;
pub use plan::{CudaStaticArena, DecodeDeviceState, ModelBlob, ModelImage, plan_static_arena};
pub use vram_pool::DeviceKvArena;
