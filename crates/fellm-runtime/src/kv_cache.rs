//! Contiguous per-layer KV cache (legacy Phase 1).
//!
//! Prefer [`crate::paged::CacheManager`] for new code. Kept for tests and
//! as dummy graph-binding buffers.

use fellm_core::error::Result;
use fellm_core::storage::AlignedBuffer;
use std::cell::RefCell;
use std::rc::Rc;

/// One layer of the KV cache.
pub struct CacheLayer {
    /// K buffer, `max_seq * n_kv_heads * head_dim` f32.
    pub k: Rc<RefCell<AlignedBuffer>>,
    /// V buffer, same shape.
    pub v: Rc<RefCell<AlignedBuffer>>,
}

/// Contiguous KV cache across all layers.
pub struct KvCache {
    /// Per-layer buffers.
    pub layers: Vec<CacheLayer>,
    /// Current filled length in tokens.
    pub len: usize,
    /// Per-token stride in f32 elements per layer.
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
        let bytes_per_layer = max_seq.max(1) * per_token.max(1) * 4;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(CacheLayer {
                k: Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes_per_layer, 64))),
                v: Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes_per_layer, 64))),
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

    /// Shared K buffer for a layer.
    pub fn k_buffer(&self, layer: usize) -> Rc<RefCell<AlignedBuffer>> {
        self.layers[layer].k.clone()
    }

    /// Shared V buffer for a layer.
    pub fn v_buffer(&self, layer: usize) -> Rc<RefCell<AlignedBuffer>> {
        self.layers[layer].v.clone()
    }

    /// Bump the logical length after a step.
    pub fn advance(&mut self) {
        self.len += 1;
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.len = 0;
    }
}
