//! Tokenizers driven by GGUF metadata.
//!
//! Supports two GGUF `tokenizer.ggml.model` values:
//!   - `"gpt2"` / `"llama"` / `"llama3"` — byte-level BPE
//!   - `"llama"` (SentencePiece unigram — detected via presence of `scores`)

#![deny(missing_docs)]

pub mod bpe;
pub mod template;
pub mod unicode_bytes;

use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;

/// A token id.
pub type TokenId = u32;

/// A tokenizer capable of encode/decode.
pub trait Tokenizer: Send + Sync {
    /// Encode text to a sequence of token ids.
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>>;

    /// Decode a single token id to bytes (may not be valid UTF-8 in isolation).
    fn decode_token(&self, id: TokenId) -> Result<Vec<u8>>;

    /// Decode a sequence to a String.
    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decode_token(id)?);
        }
        String::from_utf8(bytes)
            .map_err(|e| FellmError::Tokenization(format!("decode: {e}")))
    }

    /// BOS token id, if any.
    fn bos(&self) -> Option<TokenId>;

    /// EOS token id, if any.
    fn eos(&self) -> Option<TokenId>;

    /// The vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Chat template string (Jinja source), if any.
    fn chat_template(&self) -> Option<&str>;
}

/// Load a tokenizer from a GGUF file.
pub fn load(gguf: &GgufFile) -> Result<Box<dyn Tokenizer>> {
    let model = gguf.metadata.get_string("tokenizer.ggml.model")?;
    match model {
        "gpt2" | "llama" | "llama3" => {
            let tok = bpe::BpeTokenizer::from_gguf(gguf)?;
            Ok(Box::new(tok))
        }
        other => Err(FellmError::UnsupportedTokenizer(other.into())),
    }
}
