//! Foundational vocabulary types for FeLLM: dtypes, shapes, tensors, errors.
//!
//! This crate has no dependencies on `faer`, `petgraph`, `abi_stable`, or any
//! backend. Every other FeLLM crate builds on top of it.

#![deny(missing_docs)]

pub mod dtype;
pub mod error;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use dtype::DType;
pub use error::{FellmError, Result};
pub use shape::{Layout, Shape, Strides};
pub use storage::{AlignedBuffer, Storage};
pub use tensor::Tensor;
