//! Persistent device buffers: Q4_K weights, f32 activations, u32 block tables.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// --- Q4_K / byte weight cache ------------------------------------------------

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

// --- f32 activation cache ----------------------------------------------------

struct F32Entry {
    buf: DeviceBuffer<f32>,
    len: usize,
    device_valid: bool,
}

fn f32_cache() -> &'static Mutex<HashMap<usize, F32Entry>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, F32Entry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ensure `host` is resident on device. Reuses allocations; always H2D so
/// hybrid CPU ops (Add/Embedding) cannot leave stale VRAM.
pub fn ensure_f32(stream: &CudaStream, host: &[f32], _force_upload: bool) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    let len = host.len();
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    if let Some(e) = guard.get_mut(&key) {
        if e.len != len {
            e.buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
            e.len = len;
        } else {
            e.buf.copy_from_host(stream, host).map_err(|_| -3)?;
        }
        e.device_valid = true;
        return Ok(key);
    }
    let buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
    guard.insert(
        key,
        F32Entry {
            buf,
            len,
            device_valid: true,
        },
    );
    Ok(key)
}

/// Ensure a device buffer exists for an output slice (no H2D; marks invalid).
pub fn ensure_f32_out(stream: &CudaStream, host: &mut [f32]) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    let len = host.len();
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    if let Some(e) = guard.get_mut(&key) {
        if e.len != len {
            e.buf = DeviceBuffer::<f32>::zeroed(stream, len).map_err(|_| -3)?;
            e.len = len;
        }
        e.device_valid = false;
        return Ok(key);
    }
    let buf = DeviceBuffer::<f32>::zeroed(stream, len).map_err(|_| -3)?;
    guard.insert(
        key,
        F32Entry {
            buf,
            len,
            device_valid: false,
        },
    );
    Ok(key)
}

/// Mark the device buffer for `key` as matching host (after a successful kernel write).
pub fn mark_valid(key: usize) -> Result<(), i32> {
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.get_mut(&key).ok_or(-31)?;
    e.device_valid = true;
    Ok(())
}

/// Download device buffer to host (keeps host coherent for CPU fallback / sampling).
pub fn download_to(stream: &CudaStream, key: usize, host: &mut [f32]) -> Result<(), i32> {
    let guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.get(&key).ok_or(-31)?;
    if e.buf.len() != host.len() {
        return Err(-2);
    }
    e.buf.copy_to_host(stream, host).map_err(|_| -5)
}

/// Take ownership of an f32 cache entry (caller must [`put_f32`] later).
pub fn take_f32(key: usize) -> Result<(DeviceBuffer<f32>, bool), i32> {
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.remove(&key).ok_or(-31)?;
    Ok((e.buf, e.device_valid))
}

/// Return an f32 buffer previously taken with [`take_f32`].
pub fn put_f32(key: usize, buf: DeviceBuffer<f32>, device_valid: bool) -> Result<(), i32> {
    let len = buf.len();
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    guard.insert(
        key,
        F32Entry {
            buf,
            len,
            device_valid,
        },
    );
    Ok(())
}

// --- u32 block-table cache ---------------------------------------------------

struct U32Entry {
    buf: DeviceBuffer<u32>,
    len: usize,
}

fn u32_cache() -> &'static Mutex<HashMap<usize, U32Entry>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, U32Entry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ensure a u32 slice (e.g. block table) is resident; keyed by host ptr.
pub fn ensure_u32(stream: &CudaStream, host: &[u32]) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    let len = host.len();
    let mut guard = u32_cache().lock().map_err(|_| -30)?;
    if let Some(e) = guard.get_mut(&key) {
        if e.len == len {
            return Ok(key);
        }
        e.buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
        e.len = len;
        return Ok(key);
    }
    let buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
    guard.insert(key, U32Entry { buf, len });
    Ok(key)
}

/// Alias for [`ensure_u32`].
pub fn ensure_block_table(stream: &CudaStream, table: &[u32]) -> Result<usize, i32> {
    ensure_u32(stream, table)
}

pub fn take_u32(key: usize) -> Result<DeviceBuffer<u32>, i32> {
    let mut guard = u32_cache().lock().map_err(|_| -30)?;
    let e = guard.remove(&key).ok_or(-31)?;
    Ok(e.buf)
}

pub fn put_u32(key: usize, buf: DeviceBuffer<u32>) -> Result<(), i32> {
    let len = buf.len();
    let mut guard = u32_cache().lock().map_err(|_| -30)?;
    guard.insert(key, U32Entry { buf, len });
    Ok(())
}

// --- raw device wrap (paged KV arena) ----------------------------------------

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
