//! Persistent device buffers: Q4_K weights, f32 activations, u32 block tables.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};

// --- Q4_K / byte weight cache ------------------------------------------------

struct WeightEntry {
    buf: DeviceBuffer<u8>,
    stamp: u64,
}

struct WeightCache {
    entries: HashMap<usize, WeightEntry>,
    bytes: usize,
    clock: u64,
    limit: usize,
}

fn weight_cache_limit() -> usize {
    const DEFAULT: usize = 8 * 1024 * 1024 * 1024;
    env::var("FELLM_CUDA_WEIGHT_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT)
}

fn weight_cache() -> &'static Mutex<WeightCache> {
    static CACHE: OnceLock<Mutex<WeightCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(WeightCache {
            entries: HashMap::new(),
            bytes: 0,
            clock: 0,
            limit: weight_cache_limit(),
        })
    })
}

/// Ensure `host` weights are resident in VRAM; return the cache key (host ptr).
pub fn ensure_weight(stream: &CudaStream, host: &[u8]) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    let mut guard = weight_cache().lock().map_err(|_| -30)?;
    guard.clock = guard.clock.wrapping_add(1);
    let stamp = guard.clock;
    if let Some(entry) = guard.entries.get_mut(&key) {
        entry.stamp = stamp;
        return Ok(key);
    }
    while guard.bytes.saturating_add(host.len()) > guard.limit && !guard.entries.is_empty() {
        let victim = guard
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.stamp)
            .map(|(&victim, _)| victim)
            .ok_or(-31)?;
        if let Some(entry) = guard.entries.remove(&victim) {
            guard.bytes = guard.bytes.saturating_sub(entry.buf.len());
        }
    }
    let buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
    guard.bytes = guard.bytes.saturating_add(buf.len());
    guard.entries.insert(key, WeightEntry { buf, stamp });
    Ok(key)
}

/// Run `f` with a shared reference to the cached weight buffer.
pub fn with_weight<R>(key: usize, f: impl FnOnce(&DeviceBuffer<u8>) -> R) -> Result<R, i32> {
    let mut guard = weight_cache().lock().map_err(|_| -30)?;
    guard.clock = guard.clock.wrapping_add(1);
    let stamp = guard.clock;
    let entry = guard.entries.get_mut(&key).ok_or(-31)?;
    entry.stamp = stamp;
    Ok(f(&entry.buf))
}

// --- f32 activation cache ----------------------------------------------------

struct F32Entry {
    buf: DeviceBuffer<f32>,
    len: usize,
    device_valid: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferKey {
    ptr: usize,
    len: usize,
}

fn f32_cache() -> &'static Mutex<HashMap<BufferKey, F32Entry>> {
    static CACHE: OnceLock<Mutex<HashMap<BufferKey, F32Entry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ensure `host` is resident on device. Reuses allocations; skips H2D when the
/// cached buffer already matches `len` and is `device_valid` (unless `force_upload`).
pub fn ensure_f32(
    stream: &CudaStream,
    host: &[f32],
    force_upload: bool,
) -> Result<BufferKey, i32> {
    let len = host.len();
    let key = BufferKey {
        ptr: host.as_ptr() as usize,
        len,
    };
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    if let Some(e) = guard.get_mut(&key) {
        if e.len == len && !force_upload && e.device_valid {
            return Ok(key);
        }
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

/// Mark a host f32 buffer's device mirror stale (e.g. after a CPU fallback write).
///
/// Next [`ensure_f32`] will H2D. No-op if the pointer is not cached.
pub fn invalidate_f32(host_ptr: *const f32, len: usize) {
    if host_ptr.is_null() || len == 0 {
        return;
    }
    let key = BufferKey {
        ptr: host_ptr as usize,
        len,
    };
    let Ok(mut guard) = f32_cache().lock() else {
        return;
    };
    if let Some(e) = guard.get_mut(&key) {
        e.device_valid = false;
    }
}

/// Ensure a device buffer exists for an output slice (no H2D; marks invalid).
pub fn ensure_f32_out(stream: &CudaStream, host: &mut [f32]) -> Result<BufferKey, i32> {
    let len = host.len();
    let key = BufferKey {
        ptr: host.as_ptr() as usize,
        len,
    };
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
pub fn mark_valid(key: BufferKey) -> Result<(), i32> {
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.get_mut(&key).ok_or(-31)?;
    e.device_valid = true;
    Ok(())
}

/// Download device buffer to host (keeps host coherent for CPU fallback / sampling).
pub fn download_to(stream: &CudaStream, key: BufferKey, host: &mut [f32]) -> Result<(), i32> {
    let guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.get(&key).ok_or(-31)?;
    if e.buf.len() != host.len() {
        return Err(-2);
    }
    e.buf.copy_to_host(stream, host).map_err(|error| {
        eprintln!("cuda_kernels: D2H failed after kernel launch: {error}");
        -5
    })
}

/// Take ownership of an f32 cache entry (caller must [`put_f32`] later).
pub fn take_f32(key: BufferKey) -> Result<(DeviceBuffer<f32>, bool), i32> {
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.remove(&key).ok_or(-31)?;
    Ok((e.buf, e.device_valid))
}

/// Return an f32 buffer previously taken with [`take_f32`].
pub fn put_f32(key: BufferKey, buf: DeviceBuffer<f32>, device_valid: bool) -> Result<(), i32> {
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

// --- reusable graph-local scratch -------------------------------------------

fn f32_scratch() -> &'static Mutex<HashMap<usize, Vec<DeviceBuffer<f32>>>> {
    static POOL: OnceLock<Mutex<HashMap<usize, Vec<DeviceBuffer<f32>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn u32_scratch() -> &'static Mutex<HashMap<usize, Vec<DeviceBuffer<u32>>>> {
    static POOL: OnceLock<Mutex<HashMap<usize, Vec<DeviceBuffer<u32>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn i8_scratch() -> &'static Mutex<HashMap<usize, Vec<DeviceBuffer<i8>>>> {
    static POOL: OnceLock<Mutex<HashMap<usize, Vec<DeviceBuffer<i8>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn take_scratch_f32(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<f32>, i32> {
    if let Some(buffer) = f32_scratch()
        .lock()
        .map_err(|_| -30)?
        .entry(len)
        .or_default()
        .pop()
    {
        return Ok(buffer);
    }
    DeviceBuffer::<f32>::zeroed(stream, len).map_err(|_| -3)
}

pub fn put_scratch_f32(buffer: DeviceBuffer<f32>) -> Result<(), i32> {
    f32_scratch()
        .lock()
        .map_err(|_| -30)?
        .entry(buffer.len())
        .or_default()
        .push(buffer);
    Ok(())
}

pub fn take_scratch_u32(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<u32>, i32> {
    if let Some(buffer) = u32_scratch()
        .lock()
        .map_err(|_| -30)?
        .entry(len)
        .or_default()
        .pop()
    {
        return Ok(buffer);
    }
    DeviceBuffer::<u32>::zeroed(stream, len).map_err(|_| -3)
}

pub fn put_scratch_u32(buffer: DeviceBuffer<u32>) -> Result<(), i32> {
    u32_scratch()
        .lock()
        .map_err(|_| -30)?
        .entry(buffer.len())
        .or_default()
        .push(buffer);
    Ok(())
}

pub fn take_scratch_i8(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<i8>, i32> {
    if let Some(buffer) = i8_scratch()
        .lock()
        .map_err(|_| -30)?
        .entry(len)
        .or_default()
        .pop()
    {
        return Ok(buffer);
    }
    DeviceBuffer::<i8>::zeroed(stream, len).map_err(|_| -3)
}

pub fn put_scratch_i8(buffer: DeviceBuffer<i8>) -> Result<(), i32> {
    i8_scratch()
        .lock()
        .map_err(|_| -30)?
        .entry(buffer.len())
        .or_default()
        .push(buffer);
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

fn u32_shadows() -> &'static Mutex<HashMap<usize, Vec<u32>>> {
    static SHADOWS: OnceLock<Mutex<HashMap<usize, Vec<u32>>>> = OnceLock::new();
    SHADOWS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ensure a u32 slice (e.g. block table) is resident; keyed by host ptr.
///
/// Re-upload only when contents change. Attention and KV-write nodes repeatedly
/// present the same table within a token, and a redundant async copy here also
/// creates an implicit synchronization point on the CUDA stream.
pub fn ensure_u32(stream: &CudaStream, host: &[u32]) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    let len = host.len();
    let mut guard = u32_cache().lock().map_err(|_| -30)?;
    let mut shadows = u32_shadows().lock().map_err(|_| -30)?;
    if let Some(e) = guard.get_mut(&key) {
        if e.len != len {
            e.buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
            e.len = len;
            shadows.insert(key, host.to_vec());
        } else if shadows.get(&key).is_none_or(|shadow| shadow != host) {
            e.buf.copy_from_host(stream, host).map_err(|_| -3)?;
            shadows.insert(key, host.to_vec());
        }
        return Ok(key);
    }
    let buf = DeviceBuffer::from_host(stream, host).map_err(|_| -3)?;
    guard.insert(key, U32Entry { buf, len });
    shadows.insert(key, host.to_vec());
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
