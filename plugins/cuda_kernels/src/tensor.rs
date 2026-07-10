//! Host-side helpers for reading `TensorRef` / `TensorMut` in the plugin.

use fellm_core::dtype::DType;
use fellm_plugin_abi::{TensorMut, TensorRef};

#[inline]
pub fn f32_slice(t: &TensorRef) -> Result<&[f32], i32> {
    if t.dtype != DType::F32 as u32 {
        return Err(-10);
    }
    let n = (t.byte_len as usize) / 4;
    // SAFETY: host owns the buffer for the launch duration.
    Ok(unsafe { core::slice::from_raw_parts(t.data.cast::<f32>(), n) })
}

#[inline]
pub fn f32_slice_mut(t: &mut TensorMut) -> Result<&mut [f32], i32> {
    if t.dtype != DType::F32 as u32 {
        return Err(-11);
    }
    let n = (t.byte_len as usize) / 4;
    // SAFETY: host owns the buffer for the launch duration.
    Ok(unsafe { core::slice::from_raw_parts_mut(t.data.cast::<f32>(), n) })
}

#[inline]
pub fn bytes_slice(t: &TensorRef) -> &[u8] {
    // SAFETY: host owns the buffer for the launch duration.
    unsafe { core::slice::from_raw_parts(t.data, t.byte_len as usize) }
}

#[inline]
pub fn u32_slice(t: &TensorRef) -> Result<&[u32], i32> {
    if t.dtype != DType::U32 as u32 {
        return Err(-10);
    }
    let n = (t.byte_len as usize) / 4;
    // SAFETY: host owns the buffer for the launch duration.
    Ok(unsafe { core::slice::from_raw_parts(t.data.cast::<u32>(), n) })
}

#[inline]
pub fn dims(t: &TensorRef) -> &[u64] {
    &t.dims[..t.rank as usize]
}
