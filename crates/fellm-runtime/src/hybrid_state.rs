//! Fixed-size `ShortConv` state for hybrid models (non-paged).

use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use std::cell::RefCell;
use std::rc::Rc;

/// Recurrent `ShortConv` buffers for LFM2-style hybrids.
pub struct HybridConvState {
    /// `ShortConv` buffers, one per recurrent layer.
    pub conv: Vec<Rc<RefCell<AlignedBuffer>>>,
    /// Original block indices that are attention layers.
    pub attn_layer_ids: Vec<usize>,
    /// Original block indices that are recurrent `ShortConv` layers.
    pub conv_layer_ids: Vec<usize>,
    conv_bytes: usize,
}

impl HybridConvState {
    /// Allocate from per-layer KV-head counts (`0` means recurrent).
    pub fn new(layer_kv_heads: &[usize], n_embd: usize, shortconv_l_cache: usize) -> Result<Self> {
        let mut attn_layer_ids = Vec::new();
        let mut conv_layer_ids = Vec::new();
        let mut kv_heads = None;
        for (layer, &n_kv) in layer_kv_heads.iter().enumerate() {
            if n_kv == 0 {
                conv_layer_ids.push(layer);
            } else {
                if let Some(prev) = kv_heads {
                    if prev != n_kv {
                        return Err(FellmError::other(
                            "hybrid state currently requires uniform attention KV heads",
                        ));
                    }
                } else {
                    kv_heads = Some(n_kv);
                }
                attn_layer_ids.push(layer);
            }
        }

        let d_conv = shortconv_l_cache.saturating_sub(1);
        let conv_bytes = d_conv * n_embd * 4;
        let conv = (0..conv_layer_ids.len())
            .map(|_| Rc::new(RefCell::new(AlignedBuffer::new_zeroed(conv_bytes, 64))))
            .collect();

        Ok(Self {
            conv,
            attn_layer_ids,
            conv_layer_ids,
            conv_bytes,
        })
    }

    /// Zero recurrent state.
    pub fn reset(&mut self) {
        for buf in &self.conv {
            buf.borrow_mut().as_mut_slice().fill(0);
        }
    }

    /// Shared `ShortConv` buffer for a recurrent ordinal.
    pub fn conv_buffer(&self, conv_ord: usize) -> Rc<RefCell<AlignedBuffer>> {
        self.conv[conv_ord].clone()
    }

    /// Shape of each `ShortConv` state buffer in f32 elements.
    #[must_use]
    pub fn conv_elements(&self) -> usize {
        self.conv_bytes / 4
    }
}
