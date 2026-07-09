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
use std::collections::BTreeMap;

pub use template::{Message, TemplateContext, Value, render as render_chat_template};

/// A token id.
pub type TokenId = u32;

/// A tokenizer capable of encode/decode.
pub trait Tokenizer: Send + Sync {
    /// Encode text to a sequence of token ids.
    ///
    /// When `add_special` is true, a BOS token is prepended if the model
    /// defines one (and the text does not already start with it).
    /// Special / control tokens embedded in `text` (e.g. `<|eot_id|>`) are
    /// always parsed as atomic tokens when present in the vocabulary.
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>>;

    /// Decode a single token id to bytes (may not be valid UTF-8 in isolation).
    fn decode_token(&self, id: TokenId) -> Result<Vec<u8>>;

    /// Decode a sequence to a String.
    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decode_token(id)?);
        }
        String::from_utf8(bytes).map_err(|e| FellmError::Tokenization(format!("decode: {e}")))
    }

    /// BOS token id, if any.
    fn bos(&self) -> Option<TokenId>;

    /// EOS token id, if any.
    fn eos(&self) -> Option<TokenId>;

    /// The vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Chat template string (Jinja source), if any.
    fn chat_template(&self) -> Option<&str>;

    /// Surface form of the BOS token (for template `bos_token` variable).
    fn bos_str(&self) -> Option<&str> {
        None
    }

    /// Surface form of the EOS token (for template `eos_token` variable).
    fn eos_str(&self) -> Option<&str> {
        None
    }

    /// Apply the model's chat template to `messages` and return the prompt text.
    ///
    /// Returns `None` if the model has no chat template.
    fn apply_chat_template(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<Option<String>> {
        let Some(tmpl) = self.chat_template() else {
            return Ok(None);
        };
        let mut vars = BTreeMap::new();
        if let Some(s) = self.bos_str() {
            vars.insert("bos_token".into(), Value::String(s.to_string()));
        }
        if let Some(s) = self.eos_str() {
            vars.insert("eos_token".into(), Value::String(s.to_string()));
        }
        let ctx = TemplateContext {
            messages: messages.to_vec(),
            add_generation_prompt,
            vars,
        };
        Ok(Some(render_chat_template(tmpl, &ctx)?))
    }
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
