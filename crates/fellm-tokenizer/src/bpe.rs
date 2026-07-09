//! Byte-level BPE tokenizer (GPT-2 / Llama-3 style).

use crate::unicode_bytes::{bytes_to_unicode, unicode_to_bytes};
use crate::{TokenId, Tokenizer};
use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;
use std::collections::HashMap;

/// GGML token_type values.
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
// Some GGUF writers use 0 for normal; treat 0 as normal too.
#[allow(dead_code)]
const TOKEN_TYPE_BYTE: i32 = 6;

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
    /// Token types (0/1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte).
    token_types: Vec<i32>,
    /// Special / added tokens to partition out of raw text before BPE.
    /// Sorted longest-first so longer matches win (llama.cpp `tokenizer_st_partition`).
    special_tokens: Vec<(String, TokenId)>,
    chat_template: Option<String>,
    bos_str: Option<String>,
}

impl BpeTokenizer {
    /// Construct from a GGUF file.
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
            .unwrap_or_else(|| vec![TOKEN_TYPE_NORMAL; tokens.len()]);

        let merges_raw = gguf
            .metadata
            .get_string_array("tokenizer.ggml.merges")
            .unwrap_or(&[]);
        let mut merges = HashMap::with_capacity(merges_raw.len());
        for (rank, m) in merges_raw.iter().enumerate() {
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

        let chat_template = gguf
            .metadata
            .get_string("tokenizer.chat_template")
            .ok()
            .map(String::from);

        let bos_str = bos.and_then(|id| tokens.get(id as usize).cloned());

        let mut special_tokens: Vec<(String, TokenId)> = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let ttype = *token_types.get(i).unwrap_or(&0);
            let is_special_type = matches!(
                ttype,
                TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
            );
            let looks_special = t.starts_with("<|") && t.ends_with("|>");
            if (is_special_type || looks_special) && !t.is_empty() {
                special_tokens.push((t.clone(), i as TokenId));
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));

        Ok(Self {
            tokens,
            token_to_id,
            merges,
            bos,
            eos,
            unicode_to_bytes: unicode_to_bytes(),
            byte_to_unicode: bytes_to_unicode(),
            token_types,
            special_tokens,
            chat_template,
            bos_str,
        })
    }

    fn bpe_word(&self, chars: Vec<String>) -> Vec<String> {
        if chars.len() <= 1 {
            return chars;
        }
        let mut word = chars;
        loop {
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
    /// attached to the following word (GPT-2 / Llama-BPE approximation).
    fn pre_tokenize(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == ' ' {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                cur.push(c);
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

    /// Partition `text` into ordinary spans and special-token ids.
    ///
    /// Mirrors llama.cpp `tokenizer_st_partition`: longest special match wins.
    fn partition_specials<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        if self.special_tokens.is_empty() || text.is_empty() {
            return vec![Fragment::Text(text)];
        }

        let mut out: Vec<Fragment<'a>> = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut best: Option<(usize, usize, TokenId)> = None; // (start, len, id)
            for (surface, id) in &self.special_tokens {
                if let Some(pos) = rest.find(surface.as_str()) {
                    let len = surface.len();
                    match best {
                        None => best = Some((pos, len, *id)),
                        Some((bpos, blen, _)) => {
                            // Prefer earlier match; on tie, longer (already sorted).
                            if pos < bpos || (pos == bpos && len > blen) {
                                best = Some((pos, len, *id));
                            }
                        }
                    }
                }
            }
            let Some((pos, len, id)) = best else {
                out.push(Fragment::Text(rest));
                break;
            };
            if pos > 0 {
                out.push(Fragment::Text(&rest[..pos]));
            }
            out.push(Fragment::Special(id));
            rest = &rest[pos + len..];
        }
        out
    }

    fn encode_text_chunk(&self, chunk: &str, ids: &mut Vec<TokenId>) -> Result<()> {
        for piece in Self::pre_tokenize(chunk) {
            let chars = self.bytes_to_chars(piece.as_bytes());
            let merged = self.bpe_word(chars);
            for piece in merged {
                if let Some(&id) = self.token_to_id.get(&piece) {
                    ids.push(id);
                } else {
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
        Ok(())
    }

    /// Encode with optional BOS and optional special-token parsing.
    pub fn encode_ex(
        &self,
        text: &str,
        add_bos: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>> {
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.bos {
            // Avoid duplicating BOS if the text already starts with it
            // (chat templates often prepend `bos_token` themselves).
            let already_has_bos = parse_special
                && self
                    .bos_str
                    .as_ref()
                    .is_some_and(|s| text.starts_with(s.as_str()));
            if !already_has_bos {
                ids.push(b);
            }
        }

        if parse_special {
            for frag in self.partition_specials(text) {
                match frag {
                    Fragment::Text(t) => self.encode_text_chunk(t, &mut ids)?,
                    Fragment::Special(id) => ids.push(id),
                }
            }
        } else {
            self.encode_text_chunk(text, &mut ids)?;
        }
        Ok(ids)
    }

    /// Surface string for a token id, if in range.
    pub fn token_str(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
}

enum Fragment<'a> {
    Text(&'a str),
    Special(TokenId),
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        // Default: parse specials so chat-formatted prompts work.
        self.encode_ex(text, add_special, true)
    }

    fn decode_token(&self, id: TokenId) -> Result<Vec<u8>> {
        let s = self
            .tokens
            .get(id as usize)
            .ok_or_else(|| FellmError::Tokenization(format!("id {id} out of range")))?;
        let ttype = *self.token_types.get(id as usize).unwrap_or(&0);
        // Control tokens emit nothing (llama.cpp default for specials).
        // GGML: 3 = control. Some writers use 2 for control-like.
        if ttype == TOKEN_TYPE_CONTROL || ttype == 2 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(s.len());
        for c in s.chars() {
            if let Some(&b) = self.unicode_to_bytes.get(&(c as u32)) {
                out.push(b);
            } else {
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

    fn bos_str(&self) -> Option<&str> {
        self.bos_str.as_deref()
    }

    fn eos_str(&self) -> Option<&str> {
        self.eos
            .and_then(|id| self.tokens.get(id as usize).map(String::as_str))
    }
}
