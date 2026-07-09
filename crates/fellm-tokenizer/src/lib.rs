//! Tokenizers driven by GGUF metadata.
//!
//! Encoding uses the HuggingFace [`tokenizers`] crate (byte-level BPE).
//! Chat templates are rendered with [`minijinja`] (+ pycompat), matching
//! HuggingFace / TGI behaviour.

#![deny(missing_docs)]

mod chat;
mod gguf_hf;

use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;
use tokenizers::Tokenizer as HfTokenizer;

pub use chat::{AssistantOutput, Message, ToolCall, ToolDef, render_chat_template};

/// A token id.
pub type TokenId = u32;

/// A tokenizer capable of encode/decode and chat-template application.
pub trait Tokenizer: Send + Sync {
    /// Encode text to token ids.
    ///
    /// When `add_special` is true, a BOS token is prepended if the model
    /// defines one and the text does not already start with it.
    /// Special / control tokens embedded in `text` are parsed as atomic tokens.
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>>;

    /// Decode a single token id to bytes (may not be valid UTF-8 alone).
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

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Chat template string (Jinja source), if any.
    fn chat_template(&self) -> Option<&str>;

    /// Surface form of the BOS token (for template `bos_token`).
    fn bos_str(&self) -> Option<&str> {
        None
    }

    /// Surface form of the EOS token (for template `eos_token`).
    fn eos_str(&self) -> Option<&str> {
        None
    }

    /// Apply the model's chat template (no tools).
    fn apply_chat_template(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<Option<String>> {
        self.apply_chat_template_with_tools(messages, &[], add_generation_prompt)
    }

    /// Apply the GGUF chat template with an optional tool list.
    fn apply_chat_template_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        add_generation_prompt: bool,
    ) -> Result<Option<String>> {
        let Some(tmpl) = self.chat_template() else {
            return Ok(None);
        };
        Ok(Some(render_chat_template(
            tmpl,
            messages,
            tools,
            add_generation_prompt,
            self.bos_str(),
            self.eos_str(),
        )?))
    }
}

/// GGUF-backed tokenizer wrapping HuggingFace `tokenizers` + MiniJinja.
pub struct GgufTokenizer {
    inner: HfTokenizer,
    /// Token surface forms by id (from GGUF), for single-token decode.
    tokens: Vec<String>,
    /// GGML token types (control → empty decode).
    token_types: Vec<i32>,
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    bos_str: Option<String>,
    eos_str: Option<String>,
    chat_template: Option<String>,
}

impl GgufTokenizer {
    /// Build from a GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens = gguf
            .metadata
            .get_string_array("tokenizer.ggml.tokens")?
            .to_vec();
        let token_types = gguf
            .metadata
            .get_i32_array("tokenizer.ggml.token_type")
            .ok()
            .map(<[i32]>::to_vec)
            .unwrap_or_else(|| vec![1; tokens.len()]);

        let bos = gguf.metadata.get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos = gguf.metadata.get_u32("tokenizer.ggml.eos_token_id").ok();
        let bos_str = bos.and_then(|id| tokens.get(id as usize).cloned());
        let eos_str = eos.and_then(|id| tokens.get(id as usize).cloned());
        let chat_template = gguf
            .metadata
            .get_string("tokenizer.chat_template")
            .ok()
            .map(String::from);

        let inner = gguf_hf::build_hf_tokenizer(gguf)?;

        Ok(Self {
            inner,
            tokens,
            token_types,
            bos,
            eos,
            bos_str,
            eos_str,
            chat_template,
        })
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        // `add_special_tokens=false`: we manage BOS ourselves; specials in the
        // text are still matched via AddedToken registration.
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| FellmError::Tokenization(format!("encode: {e}")))?;
        let mut ids: Vec<TokenId> = encoding.get_ids().to_vec();

        if add_special
            && let Some(b) = self.bos
        {
            let already = self
                .bos_str
                .as_ref()
                .is_some_and(|s| text.starts_with(s.as_str()))
                || ids.first().copied() == Some(b);
            if !already {
                ids.insert(0, b);
            }
        }
        Ok(ids)
    }

    fn decode_token(&self, id: TokenId) -> Result<Vec<u8>> {
        let ttype = *self.token_types.get(id as usize).unwrap_or(&1);
        // Control / unknown → empty (llama.cpp-style streaming decode).
        if ttype == 2 || ttype == 3 {
            return Ok(Vec::new());
        }

        let decoded = self
            .inner
            .decode(&[id], false)
            .map_err(|e| FellmError::Tokenization(format!("decode_token: {e}")))?;
        Ok(decoded.into_bytes())
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| FellmError::Tokenization(format!("decode: {e}")))
    }

    fn bos(&self) -> Option<TokenId> {
        self.bos
    }

    fn eos(&self) -> Option<TokenId> {
        self.eos
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    fn bos_str(&self) -> Option<&str> {
        self.bos_str.as_deref()
    }

    fn eos_str(&self) -> Option<&str> {
        self.eos_str.as_deref()
    }
}

/// Load a tokenizer from a GGUF file.
pub fn load(gguf: &GgufFile) -> Result<Box<dyn Tokenizer>> {
    Ok(Box::new(GgufTokenizer::from_gguf(gguf)?))
}
