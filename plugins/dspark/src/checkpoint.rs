use crate::DsparkCheckpointConfig;
use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_core::shape::{Layout, Shape};
use fellm_core::storage::{AlignedBuffer, Storage};
use fellm_core::tensor::Tensor;
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
struct TensorEntry {
    dtype: DType,
    shape: Shape,
    offset: usize,
    len: usize,
}

/// Memory-mapped Hugging Face safetensors checkpoint released by DeepSpec.
///
/// Tensor storage remains file-backed and is exposed as ordinary FeLLM
/// tensors, allowing the normal backend and Memory Fabric preparation paths to
/// choose residency instead of eagerly copying the checkpoint into RAM.
#[derive(Debug)]
pub struct DsparkCheckpoint {
    directory: PathBuf,
    pub config: DsparkCheckpointConfig,
    mmap: Arc<Mmap>,
    tensors: HashMap<String, TensorEntry>,
}

impl DsparkCheckpoint {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        let config = DsparkCheckpointConfig::from_directory(&directory)?;
        let path = directory.join("model.safetensors");
        let file = File::open(&path).map_err(|error| {
            FellmError::other(format!(
                "failed to open DSpark checkpoint {}: {error}",
                path.display()
            ))
        })?;
        // SAFETY: the read-only mapping is kept alive by `Self` and no code in
        // FeLLM mutates model checkpoint files through this handle.
        let mmap = Arc::new(unsafe { MmapOptions::new().map(&file) }.map_err(|error| {
            FellmError::other(format!(
                "failed to map DSpark checkpoint {}: {error}",
                path.display()
            ))
        })?);
        if mmap.len() < 8 {
            return Err(FellmError::other("truncated DSpark safetensors header"));
        }
        let header_len = usize::try_from(u64::from_le_bytes(
            mmap[..8].try_into().expect("checked header prefix"),
        ))
        .map_err(|_| FellmError::other("DSpark safetensors header is too large"))?;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| FellmError::other("DSpark safetensors header overflow"))?;
        if data_start > mmap.len() {
            return Err(FellmError::other("truncated DSpark safetensors metadata"));
        }
        let headers: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_start]).map_err(|error| {
                FellmError::other(format!("invalid DSpark safetensors metadata: {error}"))
            })?;
        let mut tensors = HashMap::with_capacity(headers.len());
        for (name, value) in headers {
            if name == "__metadata__" {
                continue;
            }
            let header: TensorHeader = serde_json::from_value(value).map_err(|error| {
                FellmError::other(format!(
                    "invalid DSpark tensor metadata for {name}: {error}"
                ))
            })?;
            let dtype = match header.dtype.as_str() {
                "BF16" => DType::BF16,
                "F16" => DType::F16,
                "F32" => DType::F32,
                other => {
                    return Err(FellmError::other(format!(
                        "unsupported DSpark safetensors dtype {other} for {name}"
                    )));
                }
            };
            let shape = Shape::new(&header.shape)?;
            let relative_len = header.data_offsets[1]
                .checked_sub(header.data_offsets[0])
                .ok_or_else(|| FellmError::other(format!("invalid data offsets for {name}")))?;
            let expected_len = dtype.byte_size(shape.num_elements());
            if relative_len != expected_len {
                return Err(FellmError::other(format!(
                    "DSpark tensor {name} has {relative_len} bytes, expected {expected_len}"
                )));
            }
            let offset = data_start
                .checked_add(header.data_offsets[0])
                .ok_or_else(|| FellmError::other(format!("data offset overflow for {name}")))?;
            let end = offset
                .checked_add(relative_len)
                .ok_or_else(|| FellmError::other(format!("data extent overflow for {name}")))?;
            if end > mmap.len() {
                return Err(FellmError::other(format!(
                    "DSpark tensor {name} extends beyond checkpoint"
                )));
            }
            tensors.insert(
                name,
                TensorEntry {
                    dtype,
                    shape,
                    offset,
                    len: relative_len,
                },
            );
        }
        let checkpoint = Self {
            directory,
            config,
            mmap,
            tensors,
        };
        checkpoint.validate_released_layout()?;
        Ok(checkpoint)
    }

    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FellmError::other(format!("missing DSpark tensor {name}")))?;
        // CPU elementwise kernels consume learned vectors as f32. Promote the
        // tiny rank-1 parameters once at load time; large matrices remain
        // mmap-backed BF16/F16 and are handled by backend matmul/embedding.
        if entry.shape.rank() == 1 && entry.dtype != DType::F32 {
            let bytes = &self.mmap[entry.offset..entry.offset + entry.len];
            let values = dense_to_f32(entry.dtype, bytes, entry.shape.num_elements())?;
            let mut buffer = AlignedBuffer::new_zeroed(values.len() * 4, 64);
            buffer
                .as_mut_slice()
                .copy_from_slice(bytemuck::cast_slice(&values));
            return Ok(Tensor::from_storage(
                Layout::contiguous(DType::F32, entry.shape.clone()),
                Arc::new(Storage::Owned(Arc::new(buffer))),
            ));
        }
        let layout = Layout::contiguous(entry.dtype, entry.shape.clone());
        let storage = Arc::new(Storage::Mmap {
            mmap: self.mmap.clone(),
            offset: entry.offset,
            len: entry.len,
        });
        Ok(Tensor::from_storage(layout, storage))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        match tensor.dtype() {
            DType::F32 => Ok(tensor.as_slice::<f32>()?.to_vec()),
            DType::BF16 => Ok(tensor
                .as_slice::<u16>()?
                .iter()
                .map(|&bits| f32::from_bits(u32::from(bits) << 16))
                .collect()),
            DType::F16 => Ok(tensor
                .as_slice::<u16>()?
                .iter()
                .map(|&bits| half_to_f32(bits))
                .collect()),
            dtype => Err(FellmError::other(format!(
                "cannot materialize DSpark head tensor {name} from {dtype}"
            ))),
        }
    }

    pub fn tensor_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FellmError::other(format!("missing DSpark tensor {name}")))?;
        let dims = entry.shape.dims();
        if dims.len() != 2 || row >= dims[0] as usize {
            return Err(FellmError::other(format!(
                "invalid DSpark tensor row {row} for {name}"
            )));
        }
        let columns = dims[1] as usize;
        let row_bytes = entry.dtype.byte_size(columns);
        let start = entry.offset + row * row_bytes;
        dense_to_f32(entry.dtype, &self.mmap[start..start + row_bytes], columns)
    }

    fn expect_shape(&self, name: &str, expected: &[u64]) -> Result<()> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FellmError::other(format!("missing DSpark tensor {name}")))?;
        if entry.shape.dims() != expected {
            return Err(FellmError::other(format!(
                "DSpark tensor {name} has shape {:?}, expected {expected:?}",
                entry.shape.dims()
            )));
        }
        Ok(())
    }

    fn validate_released_layout(&self) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size as u64;
        let v = c.vocab_size as u64;
        let r = c.markov_rank as u64;
        let fused = (c.target_layer_ids.len() * c.hidden_size) as u64;
        self.expect_shape("embed_tokens.weight", &[v, h])?;
        self.expect_shape("fc.weight", &[h, fused])?;
        self.expect_shape("hidden_norm.weight", &[h])?;
        self.expect_shape("norm.weight", &[h])?;
        self.expect_shape("lm_head.weight", &[v, h])?;
        self.expect_shape("markov_head.markov_w1.weight", &[v, r])?;
        self.expect_shape("markov_head.markov_w2.weight", &[v, r])?;
        self.expect_shape("confidence_head.proj.weight", &[1, h + r])?;
        self.expect_shape("confidence_head.proj.bias", &[1])?;
        let q = (c.num_attention_heads * c.head_dim) as u64;
        let kv = (c.num_key_value_heads * c.head_dim) as u64;
        let ff = c.intermediate_size as u64;
        for layer in 0..c.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            self.expect_shape(&format!("{prefix}.input_layernorm.weight"), &[h])?;
            self.expect_shape(&format!("{prefix}.post_attention_layernorm.weight"), &[h])?;
            self.expect_shape(&format!("{prefix}.self_attn.q_proj.weight"), &[q, h])?;
            self.expect_shape(&format!("{prefix}.self_attn.k_proj.weight"), &[kv, h])?;
            self.expect_shape(&format!("{prefix}.self_attn.v_proj.weight"), &[kv, h])?;
            self.expect_shape(&format!("{prefix}.self_attn.o_proj.weight"), &[h, q])?;
            self.expect_shape(
                &format!("{prefix}.self_attn.q_norm.weight"),
                &[c.head_dim as u64],
            )?;
            self.expect_shape(
                &format!("{prefix}.self_attn.k_norm.weight"),
                &[c.head_dim as u64],
            )?;
            self.expect_shape(&format!("{prefix}.mlp.gate_proj.weight"), &[ff, h])?;
            self.expect_shape(&format!("{prefix}.mlp.up_proj.weight"), &[ff, h])?;
            self.expect_shape(&format!("{prefix}.mlp.down_proj.weight"), &[h, ff])?;
        }
        Ok(())
    }
}

fn dense_to_f32(dtype: DType, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    Ok(match dtype {
        DType::F32 => bytemuck::try_cast_slice::<u8, f32>(bytes)
            .map_err(|_| FellmError::other("unaligned f32 checkpoint tensor"))?
            .to_vec(),
        DType::BF16 => bytemuck::try_cast_slice::<u8, u16>(bytes)
            .map_err(|_| FellmError::other("unaligned bf16 checkpoint tensor"))?
            .iter()
            .take(elements)
            .map(|&bits| f32::from_bits(u32::from(bits) << 16))
            .collect(),
        DType::F16 => bytemuck::try_cast_slice::<u8, u16>(bytes)
            .map_err(|_| FellmError::other("unaligned f16 checkpoint tensor"))?
            .iter()
            .take(elements)
            .map(|&bits| half_to_f32(bits))
            .collect(),
        other => return Err(FellmError::UnsupportedDType(other)),
    })
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let converted = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = fraction;
            let mut shift = 0u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            sign | ((113 - shift) << 23) | ((mantissa & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(converted)
}
