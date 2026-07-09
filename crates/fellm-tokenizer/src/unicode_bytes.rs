//! GPT-2/Llama-3 style byte-to-unicode mapping used for BPE display strings.
//!
//! BPE vocabs from HuggingFace encode raw bytes as printable unicode
//! characters via a fixed permutation. This function inverts that mapping
//! so we can recover the original bytes from a token's string form.

/// The classic byte-to-char mapping (see GPT-2 encoder.py).
///
/// Returns 256 entries: byte -> unicode codepoint (as u32).
#[must_use]
pub fn bytes_to_unicode() -> [u32; 256] {
    // "Printable" bytes remain as themselves. Others get remapped to a
    // codepoint >= 256 that IS printable.
    let mut out = [0u32; 256];
    let mut i = 0u32;
    for b in 0..=255u32 {
        let is_printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        if is_printable {
            out[b as usize] = b;
        } else {
            out[b as usize] = 256 + i;
            i += 1;
        }
    }
    out
}

/// The inverse mapping: from codepoint to byte, for the codepoints that
/// appear in the bytes_to_unicode() output.
#[must_use]
pub fn unicode_to_bytes() -> std::collections::HashMap<u32, u8> {
    let b2u = bytes_to_unicode();
    let mut out = std::collections::HashMap::with_capacity(256);
    for (b, cp) in b2u.iter().enumerate() {
        out.insert(*cp, b as u8);
    }
    out
}
