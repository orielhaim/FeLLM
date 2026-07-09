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
