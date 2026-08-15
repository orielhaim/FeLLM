//! Transactional recurrent state for hybrid models (non-paged).

use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use std::cell::RefCell;
use std::rc::Rc;

/// Recurrent convolution and matrix buffers for hybrid sequence mixers.
pub struct HybridConvState {
    /// Causal convolution history, one buffer per recurrent layer.
    pub conv: Vec<Rc<RefCell<AlignedBuffer>>>,
    /// Gated DeltaNet matrices. Empty buffers correspond to ShortConv layers.
    pub ssm: Vec<Rc<RefCell<AlignedBuffer>>>,
    /// Original block indices that are attention layers.
    pub attn_layer_ids: Vec<usize>,
    /// Original block indices that are recurrent `ShortConv` layers.
    pub conv_layer_ids: Vec<usize>,
    conv_bytes: Vec<usize>,
    ssm_bytes: Vec<usize>,
}

impl Clone for HybridConvState {
    fn clone(&self) -> Self {
        let clone_buffers = |buffers: &[Rc<RefCell<AlignedBuffer>>], sizes: &[usize]| {
            buffers
                .iter()
                .zip(sizes)
                .map(|(source, &bytes)| {
                    let source = source.borrow();
                    let mut target = AlignedBuffer::new_zeroed(bytes, 64);
                    target.as_mut_slice().copy_from_slice(source.as_slice());
                    Rc::new(RefCell::new(target))
                })
                .collect()
        };
        Self {
            conv: clone_buffers(&self.conv, &self.conv_bytes),
            ssm: clone_buffers(&self.ssm, &self.ssm_bytes),
            attn_layer_ids: self.attn_layer_ids.clone(),
            conv_layer_ids: self.conv_layer_ids.clone(),
            conv_bytes: self.conv_bytes.clone(),
            ssm_bytes: self.ssm_bytes.clone(),
        }
    }
}

impl std::fmt::Debug for HybridConvState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HybridConvState")
            .field("recurrent_layers", &self.conv.len())
            .field("attention_layer_ids", &self.attn_layer_ids)
            .field("conv_layer_ids", &self.conv_layer_ids)
            .field("conv_bytes", &self.conv_bytes)
            .field("ssm_bytes", &self.ssm_bytes)
            .finish_non_exhaustive()
    }
}

impl HybridConvState {
    /// Allocate from per-layer KV-head counts (`0` means recurrent).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_kv_heads: &[usize],
        n_embd: usize,
        shortconv_l_cache: usize,
        gdn_conv_kernel: usize,
        gdn_inner_size: usize,
        gdn_key_heads: usize,
        gdn_value_heads: usize,
        gdn_state_size: usize,
    ) -> Result<Self> {
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

        let uses_gdn = gdn_state_size > 0;
        let conv_elements = if uses_gdn {
            gdn_conv_kernel.saturating_sub(1)
                * (gdn_inner_size + 2 * gdn_key_heads * gdn_state_size)
        } else {
            shortconv_l_cache.saturating_sub(1) * n_embd
        };
        let ssm_elements = if uses_gdn {
            gdn_value_heads * gdn_state_size * gdn_state_size
        } else {
            0
        };
        let conv_bytes = vec![conv_elements * 4; conv_layer_ids.len()];
        let ssm_bytes = vec![ssm_elements * 4; conv_layer_ids.len()];
        let conv = conv_bytes
            .iter()
            .map(|&bytes| Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes, 64))))
            .collect();
        let ssm = ssm_bytes
            .iter()
            .map(|&bytes| Rc::new(RefCell::new(AlignedBuffer::new_zeroed(bytes, 64))))
            .collect();

        Ok(Self {
            conv,
            ssm,
            attn_layer_ids,
            conv_layer_ids,
            conv_bytes,
            ssm_bytes,
        })
    }

    /// Zero recurrent state.
    pub fn reset(&mut self) {
        for buf in &self.conv {
            buf.borrow_mut().as_mut_slice().fill(0);
        }
        for buf in &self.ssm {
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
        self.conv_bytes.first().copied().unwrap_or(0) / 4
    }

    pub fn ssm_buffer(&self, conv_ord: usize) -> Rc<RefCell<AlignedBuffer>> {
        self.ssm[conv_ord].clone()
    }

    #[must_use]
    pub fn ssm_elements(&self) -> usize {
        self.ssm_bytes.first().copied().unwrap_or(0) / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_is_a_deep_transactional_snapshot() {
        let mut state = HybridConvState::new(&[0, 2], 4, 3, 0, 0, 0, 0, 0).unwrap();
        state.conv[0].borrow_mut().as_mut_slice()[0] = 7;
        let checkpoint = state.clone();
        state.conv[0].borrow_mut().as_mut_slice()[0] = 9;
        assert_eq!(checkpoint.conv[0].borrow().as_slice()[0], 7);
        checkpoint.conv[0].borrow_mut().as_mut_slice()[1] = 3;
        assert_eq!(state.conv[0].borrow().as_slice()[1], 0);
        state.reset();
    }
}
