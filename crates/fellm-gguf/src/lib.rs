//! Zero-copy GGUF v3 reader.
//!
//! Only little-endian files are supported (the near-universal case in the wild).

#![deny(missing_docs)]

pub mod file;
pub mod meta;
pub mod reader;

pub use file::{GgufFile, TensorInfo};
pub use meta::{MetaMap, MetaValue};
