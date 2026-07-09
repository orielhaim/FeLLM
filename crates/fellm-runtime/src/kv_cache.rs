//! Contiguous per-layer KV cache. Phase 1 shape:
//! `k[layer]: [max_seq, n_kv_heads, head_dim]`, same for v.

use fellm_core::error::Result;
use fellm_core::storage::AlignedBuffer;

/// One layer of the KV cache.
pub struct CacheLayer {
    /// K buffer, `max_seq * n_kv_heads * head_dim` f32.
    pub k: AlignedBuffer,
    /// V buffer, same shape.
    pub v: AlignedBuffer,
}

/// Contiguous KV cache across all layers.
pub struct KvCache {
    layers: Vec<CacheLayer>,
    /// Current filled length in tokens.
    pub len: usize,
    /// Per-token stride in f32 elements per layer (i.e. `n_kv_heads * head_dim`).
    pub tokens_stride: usize,
    /// Maximum sequence length.
    pub max_seq: usize,
    /// Number of layers.
    pub n_layers: usize,
}

impl KvCache {
    /// Allocate a fresh cache.
    pub fn new(
        n_layers: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let per_token = n_kv_heads * head_dim;
        let bytes_per_layer = max_seq * per_token * 4;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(CacheLayer {
                k: AlignedBuffer::new_zeroed(bytes_per_layer, 64),
                v: AlignedBuffer::new_zeroed(bytes_per_layer, 64),
            });
        }
        Ok(Self {
            layers,
            len: 0,
            tokens_stride: per_token,
            max_seq,
            n_layers,
        })
    }

    /// Get read/write access to a layer's K.
    pub fn k_mut(&mut self, layer: usize) -> &mut [f32] {
        let bytes = self.layers[layer].k.as_mut_slice();
        bytemuck::cast_slice_mut(bytes)
    }

    /// Get read/write access to a layer's V.
    pub fn v_mut(&mut self, layer: usize) -> &mut [f32] {
        let bytes = self.layers[layer].v.as_mut_slice();
        bytemuck::cast_slice_mut(bytes)
    }

    /// Read-only K.
    pub fn k(&self, layer: usize) -> &[f32] {
        let bytes = self.layers[layer].k.as_slice();
        bytemuck::cast_slice(bytes)
    }

    /// Read-only V.
    pub fn v(&self, layer: usize) -> &[f32] {
        let bytes = self.layers[layer].v.as_slice();
        bytemuck::cast_slice(bytes)
    }

    /// Append `k_row` and `v_row` (each of length `tokens_stride`) to layer.
    ///
    /// # Panics
    /// If the cache is already at `max_seq` capacity.
    pub fn append(&mut self, layer: usize, k_row: &[f32], v_row: &[f32], position: usize) {
        assert!(position < self.max_seq, "kv cache overflow");
        assert_eq!(k_row.len(), self.tokens_stride);
        assert_eq!(v_row.len(), self.tokens_stride);
        let off = position * self.tokens_stride;
        let end = off + self.tokens_stride;
        self.k_mut(layer)[off..end].copy_from_slice(k_row);
        self.v_mut(layer)[off..end].copy_from_slice(v_row);
    }

    /// Bump the logical length after all layers have been appended for the token.
    pub fn advance(&mut self) {
        self.len += 1;
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.len = 0;
    }
}
