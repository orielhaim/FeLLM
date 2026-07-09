//! Central error type for the entire FeLLM stack.

use thiserror::Error;

/// The one error type used by every FeLLM crate.
#[derive(Debug, Error)]
pub enum FellmError {
    /// I/O error from the OS.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Unknown GGML dtype code.
    #[error("unknown ggml dtype code: {0}")]
    UnknownDType(u32),

    /// A dtype was encountered that this build does not implement.
    #[error("unsupported dtype for operation: {0}")]
    UnsupportedDType(crate::dtype::DType),

    /// Shape mismatch between two tensors.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<u64>,
        /// Actual shape.
        got: Vec<u64>,
    },

    /// A rank exceeded [`crate::shape::MAX_RANK`].
    #[error("rank {0} exceeds maximum ({max})", max = crate::shape::MAX_RANK)]
    RankTooHigh(usize),

    /// The GGUF magic bytes did not match.
    #[error("bad GGUF magic: {0:#010x}")]
    BadGgufMagic(u32),

    /// The GGUF version is not supported.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedGgufVersion(u32),

    /// A named tensor was not found in the model.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    /// A metadata key was not found.
    #[error("metadata key not found: {0}")]
    MetadataKeyNotFound(String),

    /// A metadata value did not have the expected type.
    #[error("metadata type mismatch for {key}: expected {expected}, got {got}")]
    MetadataTypeMismatch {
        /// The key.
        key: String,
        /// Expected type name.
        expected: &'static str,
        /// Actual type name.
        got: &'static str,
    },

    /// A tokenizer format we don't handle.
    #[error("unsupported tokenizer model: {0}")]
    UnsupportedTokenizer(String),

    /// Failed to tokenize / detokenize.
    #[error("tokenization error: {0}")]
    Tokenization(String),

    /// A model architecture that no registered plugin claims.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// The compute graph is invalid.
    #[error("invalid graph: {0}")]
    InvalidGraph(String),

    /// No kernel is registered for a required operation.
    #[error("no kernel for op {op} with dtypes {dtypes}")]
    NoKernel {
        /// The op kind.
        op: String,
        /// Comma-joined dtypes of inputs.
        dtypes: String,
    },

    /// UTF-8 decoding failed while reading a GGUF string.
    #[error("invalid UTF-8 in GGUF string")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// A parse-time invariant was violated.
    #[error("parse error: {0}")]
    Parse(String),

    /// Something failed for a reason not otherwise classified.
    #[error("{0}")]
    Other(String),
}

/// Alias for `Result<T, FellmError>`.
pub type Result<T> = core::result::Result<T, FellmError>;

impl FellmError {
    /// Construct a generic error with a message.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Construct a parse error with a message.
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
}
