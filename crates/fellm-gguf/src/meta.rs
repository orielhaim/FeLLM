//! GGUF metadata: typed key-value store.

use crate::reader::Reader;
use fellm_core::error::{FellmError, Result};
use std::collections::BTreeMap;

/// GGUF metadata value types (matches `gguf_metadata_value_type`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaKind {
    /// u8
    U8 = 0,
    /// i8
    I8 = 1,
    /// u16
    U16 = 2,
    /// i16
    I16 = 3,
    /// u32
    U32 = 4,
    /// i32
    I32 = 5,
    /// f32
    F32 = 6,
    /// bool (1 byte)
    Bool = 7,
    /// string
    String = 8,
    /// array
    Array = 9,
    /// u64
    U64 = 10,
    /// i64
    I64 = 11,
    /// f64
    F64 = 12,
}

impl MetaKind {
    fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(FellmError::parse(format!("unknown meta kind {other}"))),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }
}

/// A typed metadata value.
#[derive(Debug, Clone)]
pub enum MetaValue {
    /// u8
    U8(u8),
    /// i8
    I8(i8),
    /// u16
    U16(u16),
    /// i16
    I16(i16),
    /// u32
    U32(u32),
    /// i32
    I32(i32),
    /// f32
    F32(f32),
    /// bool
    Bool(bool),
    /// string
    String(String),
    /// u64
    U64(u64),
    /// i64
    I64(i64),
    /// f64
    F64(f64),
    /// Array of homogeneous values.
    Array(MetaArray),
}

/// A metadata array, typed.
#[derive(Debug, Clone)]
pub enum MetaArray {
    /// u8 array.
    U8(Vec<u8>),
    /// i8 array.
    I8(Vec<i8>),
    /// u16 array.
    U16(Vec<u16>),
    /// i16 array.
    I16(Vec<i16>),
    /// u32 array.
    U32(Vec<u32>),
    /// i32 array.
    I32(Vec<i32>),
    /// f32 array.
    F32(Vec<f32>),
    /// bool array.
    Bool(Vec<bool>),
    /// String array.
    String(Vec<String>),
    /// u64 array.
    U64(Vec<u64>),
    /// i64 array.
    I64(Vec<i64>),
    /// f64 array.
    F64(Vec<f64>),
}

impl MetaArray {
    /// Length of the array.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MetaValue {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
            Self::Array(_) => "array",
        }
    }
}

/// Read one metadata value from `r`.
pub fn read_value(r: &mut Reader<'_>) -> Result<MetaValue> {
    let kind = MetaKind::from_code(r.u32()?)?;
    read_value_of_kind(r, kind)
}

fn read_value_of_kind(r: &mut Reader<'_>, kind: MetaKind) -> Result<MetaValue> {
    Ok(match kind {
        MetaKind::U8 => MetaValue::U8(r.u8()?),
        MetaKind::I8 => MetaValue::I8(r.i8()?),
        MetaKind::U16 => MetaValue::U16(r.u16()?),
        MetaKind::I16 => MetaValue::I16(r.i16()?),
        MetaKind::U32 => MetaValue::U32(r.u32()?),
        MetaKind::I32 => MetaValue::I32(r.i32()?),
        MetaKind::F32 => MetaValue::F32(r.f32()?),
        MetaKind::Bool => MetaValue::Bool(r.u8()? != 0),
        MetaKind::String => MetaValue::String(r.gguf_string()?),
        MetaKind::U64 => MetaValue::U64(r.u64()?),
        MetaKind::I64 => MetaValue::I64(r.i64()?),
        MetaKind::F64 => MetaValue::F64(r.f64()?),
        MetaKind::Array => MetaValue::Array(read_array(r)?),
    })
}

fn read_array(r: &mut Reader<'_>) -> Result<MetaArray> {
    let elem_kind = MetaKind::from_code(r.u32()?)?;
    let n = r.u64()? as usize;
    Ok(match elem_kind {
        MetaKind::U8 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.u8()?);
            }
            MetaArray::U8(v)
        }
        MetaKind::I8 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.i8()?);
            }
            MetaArray::I8(v)
        }
        MetaKind::U16 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.u16()?);
            }
            MetaArray::U16(v)
        }
        MetaKind::I16 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.i16()?);
            }
            MetaArray::I16(v)
        }
        MetaKind::U32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.u32()?);
            }
            MetaArray::U32(v)
        }
        MetaKind::I32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.i32()?);
            }
            MetaArray::I32(v)
        }
        MetaKind::F32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.f32()?);
            }
            MetaArray::F32(v)
        }
        MetaKind::Bool => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.u8()? != 0);
            }
            MetaArray::Bool(v)
        }
        MetaKind::String => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.gguf_string()?);
            }
            MetaArray::String(v)
        }
        MetaKind::U64 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.u64()?);
            }
            MetaArray::U64(v)
        }
        MetaKind::I64 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.i64()?);
            }
            MetaArray::I64(v)
        }
        MetaKind::F64 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.f64()?);
            }
            MetaArray::F64(v)
        }
        MetaKind::Array => {
            return Err(FellmError::parse("nested arrays are not permitted in GGUF"));
        }
    })
}

/// A metadata table indexed by string key.
#[derive(Debug, Clone, Default)]
pub struct MetaMap {
    inner: BTreeMap<String, MetaValue>,
}

impl MetaMap {
    /// Empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key/value.
    pub fn insert(&mut self, k: String, v: MetaValue) {
        self.inner.insert(k, v);
    }

    /// Get a raw value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.inner.get(key)
    }

    /// Iterate.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetaValue)> {
        self.inner.iter()
    }

    /// The architecture identifier, from `general.architecture`.
    pub fn arch(&self) -> Result<&str> {
        self.get_string("general.architecture")
    }

    /// Fetch a string.
    pub fn get_string(&self, key: &str) -> Result<&str> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::String(s) => Ok(s.as_str()),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "string",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch a u32 (accepts widening from u8/u16 too).
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::U32(v) => Ok(*v),
            MetaValue::U16(v) => Ok(u32::from(*v)),
            MetaValue::U8(v) => Ok(u32::from(*v)),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "u32",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch a u64 (accepts widening).
    pub fn get_u64(&self, key: &str) -> Result<u64> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::U64(v) => Ok(*v),
            MetaValue::U32(v) => Ok(u64::from(*v)),
            MetaValue::U16(v) => Ok(u64::from(*v)),
            MetaValue::U8(v) => Ok(u64::from(*v)),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "u64",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch an f32.
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::F32(v) => Ok(*v),
            MetaValue::F64(v) => Ok(*v as f32),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "f32",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch a bool.
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::Bool(v) => Ok(*v),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "bool",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch a string array.
    pub fn get_string_array(&self, key: &str) -> Result<&[String]> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::Array(MetaArray::String(v)) => Ok(v.as_slice()),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "array<string>",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch an f32 array.
    pub fn get_f32_array(&self, key: &str) -> Result<&[f32]> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::Array(MetaArray::F32(v)) => Ok(v.as_slice()),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "array<f32>",
                got: other.kind_name(),
            }),
        }
    }

    /// Fetch an i32 array.
    pub fn get_i32_array(&self, key: &str) -> Result<&[i32]> {
        match self
            .inner
            .get(key)
            .ok_or_else(|| FellmError::MetadataKeyNotFound(key.into()))?
        {
            MetaValue::Array(MetaArray::I32(v)) => Ok(v.as_slice()),
            other => Err(FellmError::MetadataTypeMismatch {
                key: key.into(),
                expected: "array<i32>",
                got: other.kind_name(),
            }),
        }
    }
}
