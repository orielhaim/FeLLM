//! Stable FFI contract between core, backends, and architecture plugins.
//!
//! Phase 1 does not `dlopen` anything — everything is statically linked —
//! but the traits and data types are the ones that a future dynamic loader
//! would consume. The core is coded strictly against these traits so that
//! Phase 2 can lift architectures/backends out of the binary without churning
//! the core.

#![deny(missing_docs)]

pub mod op;
pub mod tensor_ref;
pub mod traits;

pub use op::{OpAttrs, OpKind};
pub use tensor_ref::{TensorMut, TensorRef};
pub use traits::{Architecture, Backend, BackendCaps, KernelHandle};

/// A stream handle. On CPU this is always 0; on GPU backends this wraps
/// the vendor stream/queue pointer as a `u64`.
pub type StreamHandle = u64;

/// Semantic version.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiVersion {
    /// Major.
    pub major: u16,
    /// Minor.
    pub minor: u16,
    /// Patch.
    pub patch: u16,
}

/// The ABI version this crate advertises. Bump on breaking changes.
pub const ABI_VERSION: AbiVersion = AbiVersion {
    major: 0,
    minor: 1,
    patch: 0,
};
