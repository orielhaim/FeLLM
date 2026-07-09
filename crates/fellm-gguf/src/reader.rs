//! Low-level byte reader.

use fellm_core::error::{FellmError, Result};

/// Cursor over `&[u8]` for little-endian primitive parsing.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Construct a new reader.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte offset.
    #[must_use]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set the position.
    pub fn set_pos(&mut self, p: usize) {
        self.pos = p;
    }

    /// Remaining bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(FellmError::parse(format!(
                "unexpected EOF at pos {} needing {n} bytes (len {})",
                self.pos,
                self.data.len()
            )));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Read a little-endian u8.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian i8.
    pub fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }

    /// Read a little-endian u16.
    pub fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    /// Read a little-endian i16.
    pub fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    /// Read a little-endian u32.
    pub fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Read a little-endian i32.
    pub fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    /// Read a little-endian u64.
    pub fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(u64::from_le_bytes(b))
    }

    /// Read a little-endian i64.
    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    /// Read a little-endian f32.
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    /// Read a little-endian f64.
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Read a GGUF string (u64 length + bytes) as a UTF-8 owned String.
    pub fn gguf_string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(FellmError::from)
    }

    /// Peek at the next `n` bytes without advancing.
    pub fn peek(&self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(FellmError::parse("peek past EOF"));
        }
        Ok(&self.data[self.pos..self.pos + n])
    }
}
