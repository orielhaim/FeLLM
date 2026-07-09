//! Byte-level BPE tokenizer (GPT-2 / Llama-3 style).

use crate::unicode_bytes::{bytes_to_unicode, unicode_to_bytes};
use crate::{TokenId, Tokenizer};
use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;
use std::collections::HashMap;

/// A byte-level BPE tokenizer read from GGUF metadata.
pub struct BpeTokenizer {
    /// Vocabulary: index -> token string (as encoded via byte->unicode).
    tokens: Vec<String>,
    /// Reverse: token string -> index.
    token_to_id: HashMap<String, TokenId>,
    /// Merge priorities: (left, right) -> rank (lower = merged first).
    merges: HashMap<(String, String), u32>,
    /// Special token ids.
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    /// Unicode codepoint -> byte inverse map.
    unicode_to_bytes: HashMap<u32, u8>,
    /// Byte -> unicode codepoint forward map.
    byte_to_unicode: [u32; 256],
    /// Token types (0=normal, 1=unknown, 2=control, 3=user_defined, 4=unused, 5=byte).
    token_types: Vec<i32>,
    chat_template: Option<String>,
    /// Optional pre-tokenization regex (only used for a "GPT-2 like" fast path).
    /// We use a simple whitespace/punct split.
    /// Cached bos/eos strings for quick prepend/append if desired.
    #[allow(dead_code)]
    bos_str: Option<String>,
}

impl BpeTokenizer {
    /// Construct from a GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens = gguf.metadata.get_string_array("tokenizer.ggml.tokens")?.to_vec();
        let token_types = gguf
            .metadata
            .get_i32_array("tokenizer.ggml.token_type")
            .ok()
            .map(<[i32]>::to_vec)
            .unwrap_or_else(|| vec![0i32; tokens.len()]);

        let merges_raw = gguf
            .metadata
            .get_string_array("tokenizer.ggml.merges")
            .unwrap_or(&[]);
        let mut merges = HashMap::with_capacity(merges_raw.len());
        for (rank, m) in merges_raw.iter().enumerate() {
            // A merge string is "left right".
            let mut it = m.splitn(2, ' ');
            let l = it
                .next()
                .ok_or_else(|| FellmError::Tokenization(format!("bad merge: {m}")))?;
            let r = it
                .next()
                .ok_or_else(|| FellmError::Tokenization(format!("bad merge: {m}")))?;
            merges.insert((l.to_string(), r.to_string()), rank as u32);
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as TokenId);
        }

        let bos = gguf.metadata.get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos = gguf.metadata.get_u32("tokenizer.ggml.eos_token_id").ok();

        let chat_template = gguf.metadata.get_string("tokenizer.chat_template").ok().map(String::from);

        let bos_str = bos.and_then(|id| tokens.get(id as usize).cloned());

        Ok(Self {
            tokens,
            token_to_id,
            merges,
            bos,
            eos,
            unicode_to_bytes: unicode_to_bytes(),
            byte_to_unicode: bytes_to_unicode(),
            token_types,
            chat_template,
            bos_str,
        })
    }

    /// Encode a single "word" (pre-tokenized chunk) using BPE greedy merges.
    fn bpe_word(&self, chars: Vec<String>) -> Vec<String> {
        if chars.len() <= 1 {
            return chars;
        }
        let mut word = chars;
        loop {
            // Find lowest-rank adjacent pair.
            let mut best_rank = u32::MAX;
            let mut best_idx: Option<usize> = None;
            for i in 0..word.len() - 1 {
                if let Some(&rank) = self.merges.get(&(word[i].clone(), word[i + 1].clone()))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_idx = Some(i);
                }
            }
            let Some(i) = best_idx else { break };
            let merged = format!("{}{}", word[i], word[i + 1]);
            word.splice(i..i + 2, [merged]);
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    /// Map each input byte to its unicode-char representation as a single-char String.
    fn bytes_to_chars(&self, bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::with_capacity(bytes.len());
        for &b in bytes {
            let cp = self.byte_to_unicode[b as usize];
            let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
            out.push(c.to_string());
        }
        out
    }

    /// Simple pre-tokenizer: split on whitespace transitions, keep leading spaces
    /// attached to the following word (like GPT-2's regex approximation).
    fn pre_tokenize(text: &str) -> Vec<String> {
        // We split into chunks that start with an optional leading space, then
        // runs of non-space characters. Newlines break chunks.
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == ' ' {
                // Start a new chunk with the space attached to the next non-space run.
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                cur.push(c);
                // Consume the next word if present.
                while let Some(&nc) = chars.peek() {
                    if nc == ' ' || nc == '\n' || nc == '\t' {
                        break;
                    }
                    cur.push(nc);
                    chars.next();
                }
                out.push(std::mem::take(&mut cur));
            } else if c == '\n' || c == '\t' {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(c.to_string());
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let mut ids = Vec::new();
        if add_special && let Some(b) = self.bos {
            ids.push(b);
        }

        for chunk in Self::pre_tokenize(text) {
            let chars = self.bytes_to_chars(chunk.as_bytes());
            let merged = self.bpe_word(chars);
            for piece in merged {
                if let Some(&id) = self.token_to_id.get(&piece) {
                    ids.push(id);
                } else {
                    // Fall back to per-byte tokens (byte tokens are token_type=5).
                    for b in piece.bytes() {
                        let cp = self.byte_to_unicode[b as usize];
                        let s = char::from_u32(cp).unwrap_or('\u{FFFD}').to_string();
                        if let Some(&id) = self.token_to_id.get(&s) {
                            ids.push(id);
                        } else {
                            return Err(FellmError::Tokenization(format!(
                                "no token for byte-char {s:?}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    fn decode_token(&self, id: TokenId) -> Result<Vec<u8>> {
        let s = self
            .tokens
            .get(id as usize)
            .ok_or_else(|| FellmError::Tokenization(format!("id {id} out of range")))?;
        // For control/special tokens, emit nothing (per llama.cpp convention we
        // could emit the raw string; keep this minimal).
        let ttype = *self.token_types.get(id as usize).unwrap_or(&0);
        if ttype == 2 {
            // control token — omit
            return Ok(Vec::new());
        }
        // Reverse the byte->unicode mapping char by char.
        let mut out = Vec::with_capacity(s.len());
        for c in s.chars() {
            if let Some(&b) = self.unicode_to_bytes.get(&(c as u32)) {
                out.push(b);
            } else {
                // Fall back to the char's UTF-8 encoding directly.
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        Ok(out)
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
}
