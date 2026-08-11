//! Graph-binding dummy K/V buffers.
//!
//! Paged/fabric kernels ignore these when a KV context is installed; the graph
//! still needs stable buffer bindings for `k_in_*` / `v_in_*`.

use fellm_core::error::Result;
use fellm_core::storage::AlignedBuffer;
use std::cell::RefCell;
use std::rc::Rc;

pub struct DummyLayer {
    pub k: Rc<RefCell<AlignedBuffer>>,
    pub v: Rc<RefCell<AlignedBuffer>>,
}

/// Contiguous placeholder buffers for graph binding only.
pub struct DummyKvBuffers {
    pub layers: Vec<DummyLayer>,
    pub tokens_stride: usize,
    pub max_seq: usize,
    pub n_layers: usize,
}

impl DummyKvBuffers {
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
            layers.push(DummyLayer {
                k: Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes_per_layer, 64))),
                v: Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes_per_layer, 64))),
            });
        }
        Ok(Self {
            layers,
            tokens_stride: per_token,
            max_seq,
            n_layers,
        })
    }

    #[must_use]
    pub fn k_buffer(&self, layer: usize) -> Rc<RefCell<AlignedBuffer>> {
        self.layers[layer].k.clone()
    }

    #[must_use]
    pub fn v_buffer(&self, layer: usize) -> Rc<RefCell<AlignedBuffer>> {
        self.layers[layer].v.clone()
    }
}
