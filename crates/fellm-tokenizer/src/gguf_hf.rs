//! Build a HuggingFace `tokenizers` Tokenizer from GGUF `tokenizer.ggml.*` metadata.

use ahash::AHashMap;
use fellm_core::error::{FellmError, Result};
use fellm_gguf::GgufFile;
use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::{AddedToken, SplitDelimiterBehavior};

/// GGML token_type values.
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// Llama-3 / Llama-BPE pre-tokenizer regex (HF `LlamaTokenizerFast` / llama.cpp `llama-bpe`).
const LLAMA3_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Construct an HF BPE tokenizer from GGUF metadata.
pub fn build_hf_tokenizer(gguf: &GgufFile) -> Result<HfTokenizer> {
    let model = gguf.metadata.get_string("tokenizer.ggml.model")?;
    match model {
        "gpt2" | "llama" | "llama3" | "gemma4" => {}
        other => return Err(FellmError::UnsupportedTokenizer(other.into())),
    }

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
    let merges_raw = gguf
        .metadata
        .get_string_array("tokenizer.ggml.merges")
        .unwrap_or(&[]);

    let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }

    let mut merges = Vec::with_capacity(merges_raw.len());
    for m in merges_raw {
        let mut it = m.splitn(2, ' ');
        let l = it
            .next()
            .ok_or_else(|| FellmError::Tokenization(format!("bad merge: {m}")))?;
        let r = it
            .next()
            .ok_or_else(|| FellmError::Tokenization(format!("bad merge: {m}")))?;
        merges.push((l.to_string(), r.to_string()));
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| FellmError::Tokenization(format!("BPE build: {e}")))?;

    let mut tokenizer = HfTokenizer::new(bpe);

    let pre = gguf
        .metadata
        .get_string("tokenizer.ggml.pre")
        .unwrap_or("default");

    // Byte-level BPE: map UTF-8 bytes ↔ printable unicode, then BPE.
    // Llama-3 style uses an extra Split regex before ByteLevel (use_regex=false).
    // GPT-2 / LFM2 use ByteLevel's built-in GPT-2 regex.
    match pre {
        "llama-bpe" | "llama3" | "deepseek-llm" | "deepseek-coder" | "falcon" => {
            let split = Split::new(
                SplitPattern::String(LLAMA3_SPLIT_REGEX.into()),
                SplitDelimiterBehavior::Isolated,
                false, /* invert */
            )
            .map_err(|e| FellmError::Tokenization(format!("split pretokenizer: {e}")))?;
            let byte_level = ByteLevel::new(
                false, /* add_prefix_space */
                false, /* trim_offsets */
                false, /* use_regex — Split already did it */
            );
            tokenizer.with_pre_tokenizer(Some(PreSequence::new(vec![
                split.into(),
                byte_level.into(),
            ])));
        }
        _ => {
            // gpt2, lfm2, default, qwen2, …
            tokenizer.with_pre_tokenizer(Some(ByteLevel::new(
                false, /* add_prefix_space */
                false, /* trim_offsets */
                true,  /* use_regex */
            )));
        }
    }

    // Same ByteLevel type implements Decoder + PostProcessor.
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_post_processor(Some(ByteLevel::new(false, false, false)));

    // Register special / control tokens so they stay atomic in encode().
    let mut added = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let ttype = *token_types.get(i).unwrap_or(&1);
        let is_special_type = matches!(
            ttype,
            TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
        );
        let looks_special = t.starts_with("<|") && t.ends_with("|>");
        if (is_special_type || looks_special) && !t.is_empty() {
            added.push(AddedToken::from(t.clone(), true));
        }
    }
    if !added.is_empty() {
        let _n = tokenizer.add_special_tokens(added);
    }

    Ok(tokenizer)
}
