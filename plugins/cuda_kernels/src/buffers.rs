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

fn external_weights() -> &'static Mutex<HashMap<(usize, usize), (u64, usize)>> {
    static EXTERNAL: OnceLock<Mutex<HashMap<(usize, usize), (u64, usize)>>> = OnceLock::new();
    EXTERNAL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn external_tensor(host_ptr: usize, len: usize) -> Result<Option<(u64, usize)>, i32> {
    Ok(external_weights()
        .lock()
        .map_err(|_| -30)?
        .get(&(host_ptr, len))
        .copied())
}

/// Bind a host GGUF view to a stable address inside the host-owned model image.
pub fn register_external_weight(
    host_ptr: *const u8,
    len: usize,
    device_ptr: u64,
) -> Result<(), i32> {
    if host_ptr.is_null() || len == 0 || device_ptr == 0 {
        return Err(-1);
    }
    external_weights()
        .lock()
        .map_err(|_| -30)?
        .insert((host_ptr as usize, len), (device_ptr, len));
    Ok(())
}

/// Ensure `host` weights are resident in VRAM; return the cache key (host ptr).
pub fn ensure_weight(stream: &CudaStream, host: &[u8]) -> Result<usize, i32> {
    let key = host.as_ptr() as usize;
    if external_weights()
        .lock()
        .map_err(|_| -30)?
        .contains_key(&(key, host.len()))
    {
        return Ok(key);
    }
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
    // Do not let the mutex guard temporary live across `f`. Multi-weight
    // macro-kernels nest `with_weight` calls and would otherwise self-deadlock.
    let external = {
        external_weights()
            .lock()
            .map_err(|_| -30)?
            .iter()
            .find_map(|(&(ptr, _), &binding)| (ptr == key).then_some(binding))
    };
    if let Some((ptr, len)) = external {
        let buffer =
            unsafe { DeviceBuffer::<u8>::from_raw_parts(ptr, len, crate::oxide_ctx().clone()) };
        let result = f(&buffer);
        let _ = buffer.into_raw_parts();
        return Ok(result);
    }
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

fn f32_versions() -> &'static Mutex<HashMap<BufferKey, u64>> {
    static VERSIONS: OnceLock<Mutex<HashMap<BufferKey, u64>>> = OnceLock::new();
    VERSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn bump_f32_version(key: BufferKey) -> Result<(), i32> {
    let mut versions = f32_versions().lock().map_err(|_| -30)?;
    let version = versions.entry(key).or_default();
    *version = version.wrapping_add(1).max(1);
    Ok(())
}

/// Content generation of a cached activation.
pub fn f32_version(key: BufferKey) -> Result<u64, i32> {
    Ok(*f32_versions()
        .lock()
        .map_err(|_| -30)?
        .entry(key)
        .or_insert(1))
}

/// Ensure `host` is resident on device. Reuses allocations; skips H2D when the
/// cached buffer already matches `len` and is `device_valid` (unless `force_upload`).
pub fn ensure_f32(stream: &CudaStream, host: &[f32], force_upload: bool) -> Result<BufferKey, i32> {
    let len = host.len();
    let key = BufferKey {
        ptr: host.as_ptr() as usize,
        len,
    };
    if let Some((_, bytes)) = external_tensor(key.ptr, len * core::mem::size_of::<f32>())? {
        if bytes != len * core::mem::size_of::<f32>() {
            return Err(-2);
        }
        if force_upload {
            let (ptr, _) =
                external_tensor(key.ptr, len * core::mem::size_of::<f32>())?.ok_or(-31)?;
            let mut buffer = unsafe {
                DeviceBuffer::<f32>::from_raw_parts(ptr, len, crate::oxide_ctx().clone())
            };
            buffer.copy_from_host(stream, host).map_err(|_| -3)?;
            let _ = buffer.into_raw_parts();
            bump_f32_version(key)?;
        }
        return Ok(key);
    }
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
        drop(guard);
        bump_f32_version(key)?;
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
    drop(guard);
    bump_f32_version(key)?;
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
    if let Some((_, bytes)) = external_tensor(key.ptr, len * core::mem::size_of::<f32>())? {
        return if bytes == len * core::mem::size_of::<f32>() {
            Ok(key)
        } else {
            Err(-2)
        };
    }
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
    if external_tensor(key.ptr, key.len * core::mem::size_of::<f32>())?.is_some() {
        return bump_f32_version(key);
    }
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.get_mut(&key).ok_or(-31)?;
    e.device_valid = true;
    drop(guard);
    bump_f32_version(key)?;
    Ok(())
}

// --- reusable Q8_1 activation transforms -----------------------------------

struct Q8ActivationEntry {
    quantized: DeviceBuffer<i8>,
    scales: DeviceBuffer<f32>,
    source_version: u64,
}

fn q8_activations() -> &'static Mutex<HashMap<BufferKey, Q8ActivationEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<BufferKey, Q8ActivationEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Quantize an activation at most once per content generation, then reuse it
/// across packed-projection siblings such as Q/K/V and gate/up.
pub fn with_q8_activation<R>(
    stream: &CudaStream,
    key: BufferKey,
    len: usize,
    quantize: impl FnOnce(&mut DeviceBuffer<i8>, &mut DeviceBuffer<f32>) -> Result<(), i32>,
    use_activation: impl FnOnce(&DeviceBuffer<i8>, &DeviceBuffer<f32>) -> R,
) -> Result<R, i32> {
    let version = f32_version(key)?;
    let mut cache = q8_activations().lock().map_err(|_| -30)?;
    if !cache.contains_key(&key) {
        cache.insert(
            key,
            Q8ActivationEntry {
                quantized: DeviceBuffer::<i8>::zeroed(stream, len).map_err(|_| -3)?,
                scales: DeviceBuffer::<f32>::zeroed(stream, len.div_ceil(32)).map_err(|_| -3)?,
                source_version: 0,
            },
        );
    }
    let entry = cache.get_mut(&key).ok_or(-31)?;
    if entry.quantized.len() != len {
        entry.quantized = DeviceBuffer::<i8>::zeroed(stream, len).map_err(|_| -3)?;
        entry.scales = DeviceBuffer::<f32>::zeroed(stream, len.div_ceil(32)).map_err(|_| -3)?;
        entry.source_version = 0;
    }
    if entry.source_version != version {
        quantize(&mut entry.quantized, &mut entry.scales)?;
        entry.source_version = version;
    }
    Ok(use_activation(&entry.quantized, &entry.scales))
}

/// Download device buffer to host (keeps host coherent for CPU fallback / sampling).
pub fn download_to(stream: &CudaStream, key: BufferKey, host: &mut [f32]) -> Result<(), i32> {
    if let Some((ptr, bytes)) = external_tensor(key.ptr, key.len * core::mem::size_of::<f32>())? {
        if bytes != host.len() * core::mem::size_of::<f32>() {
            return Err(-2);
        }
        let buffer = unsafe {
            DeviceBuffer::<f32>::from_raw_parts(ptr, host.len(), crate::oxide_ctx().clone())
        };
        let result = buffer.copy_to_host(stream, host).map_err(|error| {
            eprintln!("cuda_kernels: D2H failed after kernel launch: {error}");
            -5
        });
        let _ = buffer.into_raw_parts();
        return result;
    }
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
    if let Some((ptr, bytes)) = external_tensor(key.ptr, key.len * core::mem::size_of::<f32>())? {
        if bytes != key.len * core::mem::size_of::<f32>() {
            return Err(-2);
        }
        let buffer = unsafe {
            DeviceBuffer::<f32>::from_raw_parts(ptr, key.len, crate::oxide_ctx().clone())
        };
        return Ok((buffer, true));
    }
    let mut guard = f32_cache().lock().map_err(|_| -30)?;
    let e = guard.remove(&key).ok_or(-31)?;
    Ok((e.buf, e.device_valid))
}

/// Return an f32 buffer previously taken with [`take_f32`].
pub fn put_f32(key: BufferKey, buf: DeviceBuffer<f32>, device_valid: bool) -> Result<(), i32> {
    if external_tensor(key.ptr, key.len * core::mem::size_of::<f32>())?.is_some() {
        let _ = buf.into_raw_parts();
        if device_valid {
            bump_f32_version(key)?;
        }
        return Ok(());
    }
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
    let len = host.len();
    let key = host.as_ptr() as usize;
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
