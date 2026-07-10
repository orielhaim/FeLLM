//! Persistent device buffers: Q4_K weight cache.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn weight_cache() -> &'static Mutex<HashMap<usize, DeviceBuffer<u8>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, DeviceBuffer<u8>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ensure `host` weights are resident in VRAM; return the cache key (host ptr).
pub fn ensure_weight(stream: &CudaStream, host: &[u8]) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    let mut guard = weight_cache().lock().map_err(|_| -30)?;
    if !guard.contains_key(&key) {
        let buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
        guard.insert(key, buf);
    }
    Ok(key)
}

/// Run `f` with a shared reference to the cached weight buffer.
pub fn with_weight<R>(key: usize, f: impl FnOnce(&DeviceBuffer<u8>) -> R) -> Result<R, i32> {
    let guard = weight_cache().lock().map_err(|_| -30)?;
    let buf = guard.get(&key).ok_or(-31)?;
    Ok(f(buf))
}

/// Wrap a host-owned device pointer without taking ownership.
///
/// # Safety
/// `ptr` must be a valid device allocation of `len` bytes on this device.
pub unsafe fn wrap_device_bytes(
    ptr: *mut u8,
    len: usize,
    ctx: Arc<CudaContext>,
) -> DeviceBuffer<u8> {
    unsafe { DeviceBuffer::<u8>::from_raw_parts(ptr as u64, len, ctx) }
}

pub fn release_wrap(buf: DeviceBuffer<u8>) {
    let _ = buf.into_raw_parts();
}
