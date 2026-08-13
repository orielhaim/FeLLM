use crate::StorageExtent;
use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use memmap2::Mmap;
use std::collections::VecDeque;
use std::fs::File;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Storage-side transfer source. Providers fill bounded caller-owned staging buffers.
pub trait TransferProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()>;
    fn advise_will_need(&self, _offset: u64, _len: u64) -> Result<()> {
        Ok(())
    }
}

/// mmap + OS page-cache provider; ideal when the machine can cache most working weights.
pub struct MmapProvider {
    map: Arc<Mmap>,
}

impl MmapProvider {
    pub fn open(file: &File) -> Result<Self> {
        // SAFETY: the provider holds the mapping for its entire use and never mutates the file.
        let map = unsafe { Mmap::map(file) }.map_err(FellmError::Io)?;
        Ok(Self { map: Arc::new(map) })
    }
}

impl TransferProvider for MmapProvider {
    fn name(&self) -> &'static str {
        "mmap-page-cache"
    }
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        let start =
            usize::try_from(offset).map_err(|_| FellmError::other("mmap offset overflow"))?;
        let end = start
            .checked_add(target.len())
            .ok_or_else(|| FellmError::other("mmap range overflow"))?;
        let source = self
            .map
            .get(start..end)
            .ok_or_else(|| FellmError::other("mmap read outside model"))?;
        target.copy_from_slice(source);
        Ok(())
    }
}

/// Explicit bounded file reads. Multiple provider instances can be used as an async read pool.
pub struct FileProvider {
    file: File,
}

/// Native unbuffered provider for sector-aligned storage objects.
#[cfg(any(target_os = "linux", windows))]
pub struct DirectFileProvider {
    file: File,
    #[cfg(windows)]
    file_len: u64,
    #[cfg(windows)]
    fallback: FileProvider,
}

#[cfg(target_os = "linux")]
impl DirectFileProvider {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        // Linux O_DIRECT. Callers must provide sector-aligned offsets, lengths, and buffers.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(0o40000)
            .open(path.as_ref())
            .map_err(FellmError::Io)?;
        Ok(Self { file })
    }
}

#[cfg(windows)]
impl DirectFileProvider {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
        const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED)
            .open(path.as_ref())
            .map_err(FellmError::Io)?;
        let file_len = file.metadata().map_err(FellmError::Io)?.len();
        Ok(Self {
            file,
            file_len,
            fallback: FileProvider::open(path.as_ref())?,
        })
    }
}

#[cfg(target_os = "linux")]
impl TransferProvider for DirectFileProvider {
    fn name(&self) -> &'static str {
        "linux-direct-io"
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        if !offset.is_multiple_of(4096)
            || !target.len().is_multiple_of(4096)
            || !(target.as_ptr() as usize).is_multiple_of(4096)
        {
            return Err(FellmError::other(
                "O_DIRECT read requires 4096-byte aligned offset, length, and staging buffer",
            ));
        }
        let mut read = 0usize;
        while read < target.len() {
            let count = self
                .file
                .read_at(&mut target[read..], offset.saturating_add(read as u64))
                .map_err(FellmError::Io)?;
            if count == 0 {
                if read > 0 {
                    target[read..].fill(0);
                    return Ok(());
                }
                return Err(FellmError::other("unexpected EOF in O_DIRECT read"));
            }
            read += count;
        }
        Ok(())
    }
}

#[cfg(windows)]
impl TransferProvider for DirectFileProvider {
    fn name(&self) -> &'static str {
        "windows-unbuffered-io"
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{
            CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, GetLastError, HANDLE,
        };
        use windows_sys::Win32::Storage::FileSystem::ReadFile;
        use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED, OVERLAPPED_0_0};
        use windows_sys::Win32::System::Threading::CreateEventW;
        if !offset.is_multiple_of(4096)
            || !target.len().is_multiple_of(4096)
            || !(target.as_ptr() as usize).is_multiple_of(4096)
        {
            return Err(FellmError::other(
                "Windows unbuffered read requires 4096-byte aligned offset, length, and staging buffer",
            ));
        }
        if offset.saturating_add(target.len() as u64) > self.file_len {
            return self.fallback.read_at(offset, target);
        }
        let mut read = 0usize;
        while read < target.len() {
            let position = offset.saturating_add(read as u64);
            let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
            if event.is_null() {
                return Err(FellmError::Io(std::io::Error::last_os_error()));
            }
            let mut overlapped = OVERLAPPED::default();
            overlapped.hEvent = event;
            overlapped.Anonymous.Anonymous = OVERLAPPED_0_0 {
                Offset: position as u32,
                OffsetHigh: (position >> 32) as u32,
            };
            let chunk = (target.len() - read).min(u32::MAX as usize) as u32;
            let handle = self.file.as_raw_handle() as HANDLE;
            let submitted = unsafe {
                ReadFile(
                    handle,
                    target[read..].as_mut_ptr(),
                    chunk,
                    std::ptr::null_mut(),
                    &mut overlapped,
                )
            };
            if submitted == 0 {
                let error = unsafe { GetLastError() };
                if error != ERROR_IO_PENDING {
                    unsafe { CloseHandle(event) };
                    if error == ERROR_HANDLE_EOF && read > 0 {
                        target[read..].fill(0);
                        return Ok(());
                    }
                    return Err(FellmError::Io(std::io::Error::from_raw_os_error(
                        error as i32,
                    )));
                }
            }
            let mut count = 0u32;
            let completed = unsafe { GetOverlappedResult(handle, &overlapped, &mut count, 1) };
            unsafe { CloseHandle(event) };
            if completed == 0 {
                return Err(FellmError::Io(std::io::Error::last_os_error()));
            }
            let count = count as usize;
            if count == 0 {
                if read > 0 {
                    target[read..].fill(0);
                    return Ok(());
                }
                return Err(FellmError::other("unexpected EOF in unbuffered read"));
            }
            read += count;
        }
        Ok(())
    }
}

impl FileProvider {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            file: File::open(path).map_err(FellmError::Io)?,
        })
    }
}

impl TransferProvider for FileProvider {
    fn name(&self) -> &'static str {
        "buffered-file"
    }
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        let mut read = 0usize;
        while read < target.len() {
            #[cfg(unix)]
            let count = {
                use std::os::unix::fs::FileExt;
                self.file
                    .read_at(&mut target[read..], offset.saturating_add(read as u64))
                    .map_err(FellmError::Io)?
            };
            #[cfg(windows)]
            let count = {
                use std::os::windows::fs::FileExt;
                self.file
                    .seek_read(&mut target[read..], offset.saturating_add(read as u64))
                    .map_err(FellmError::Io)?
            };
            #[cfg(not(any(unix, windows)))]
            let count = return Err(FellmError::other(
                "positional file reads are unsupported on this platform",
            ));
            if count == 0 {
                if read > 0 {
                    target[read..].fill(0);
                    return Ok(());
                }
                return Err(FellmError::other(
                    "unexpected EOF in buffered positional read",
                ));
            }
            read += count;
        }
        Ok(())
    }
}

/// Merge nearby extents into aligned large reads while preserving storage-provider boundaries.
pub fn coalesce_extents(
    extents: &[StorageExtent],
    max_gap: u64,
    max_read: u64,
) -> Vec<StorageExtent> {
    let mut sorted = extents.to_vec();
    sorted.sort_by(|a, b| (&a.provider, &a.path, a.offset).cmp(&(&b.provider, &b.path, b.offset)));
    let mut output: Vec<StorageExtent> = Vec::new();
    for extent in sorted {
        if let Some(last) = output.last_mut() {
            let last_end = last.offset.saturating_add(last.len);
            let new_end = extent.offset.saturating_add(extent.len);
            if last.provider == extent.provider
                && last.path == extent.path
                && extent.offset <= last_end.saturating_add(max_gap)
                && new_end.saturating_sub(last.offset) <= max_read
            {
                last.len = last.len.max(new_end.saturating_sub(last.offset));
                last.alignment = last.alignment.max(extent.alignment);
                continue;
            }
        }
        output.push(extent);
    }
    output
}

struct ReadCommand {
    extent: StorageExtent,
    result: SyncSender<Result<PrefetchedRead>>,
}

/// Completed read backed by one fixed-capacity staging allocation.
/// Dropping it returns the exact allocation to the bounded pool for reuse.
pub struct PrefetchedRead {
    extent: StorageExtent,
    buffer: Option<AlignedBuffer>,
    valid_len: usize,
    available: SyncSender<AlignedBuffer>,
}

impl PrefetchedRead {
    #[must_use]
    pub fn extent(&self) -> &StorageExtent {
        &self.extent
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        let Some(buffer) = &self.buffer else {
            return &[];
        };
        &buffer.as_slice()[..self.valid_len]
    }

    /// Stable staging address for the lifetime of this lease.
    #[must_use]
    pub fn staging_address(&self) -> *const u8 {
        self.buffer
            .as_ref()
            .map_or(std::ptr::null(), |buffer| buffer.as_slice().as_ptr())
    }
}

impl Drop for PrefetchedRead {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.available.send(buffer);
        }
    }
}

/// Explicit asynchronous storage reader with a hard bound on resident staging memory.
///
/// Buffers are allocated once and leased to worker reads. There is no per-weight allocation,
/// and backpressure naturally stops read-ahead when every staging slot is in flight.
pub struct BoundedTransferPool {
    submit: Option<SyncSender<ReadCommand>>,
    workers: Vec<JoinHandle<()>>,
    buffer_bytes: usize,
    metrics: Arc<TransferPoolMetricsState>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TransferPoolMetrics {
    pub physical_reads: u64,
    pub physical_bytes: u64,
    pub failed_reads: u64,
    pub latency_p50_nanos: u64,
    pub latency_p95_nanos: u64,
}

#[derive(Default)]
struct TransferPoolMetricsState {
    physical_reads: std::sync::atomic::AtomicU64,
    physical_bytes: std::sync::atomic::AtomicU64,
    failed_reads: std::sync::atomic::AtomicU64,
    latencies: Mutex<VecDeque<u64>>,
}

impl BoundedTransferPool {
    pub fn new(
        provider: Arc<dyn TransferProvider>,
        buffer_count: usize,
        buffer_bytes: usize,
    ) -> Result<Self> {
        if buffer_count == 0 || buffer_bytes == 0 {
            return Err(FellmError::other(
                "bounded transfer pool requires non-zero buffers",
            ));
        }
        let (available_tx, available_rx) = mpsc::sync_channel(buffer_count);
        for _ in 0..buffer_count {
            available_tx
                .send(AlignedBuffer::new_zeroed(buffer_bytes, 4096))
                .map_err(|_| FellmError::other("initialize transfer staging pool"))?;
        }
        let available_rx = Arc::new(Mutex::new(available_rx));
        let (submit, commands) = mpsc::sync_channel::<ReadCommand>(buffer_count);
        let commands = Arc::new(Mutex::new(commands));
        let metrics = Arc::new(TransferPoolMetricsState::default());
        let mut workers = Vec::with_capacity(buffer_count);
        for index in 0..buffer_count {
            let provider = Arc::clone(&provider);
            let commands = Arc::clone(&commands);
            let available_rx = Arc::clone(&available_rx);
            let available_tx = available_tx.clone();
            let metrics = Arc::clone(&metrics);
            workers.push(
                std::thread::Builder::new()
                    .name(format!("fellm-storage-prefetch-{index}"))
                    .spawn(move || {
                        worker_loop(provider, commands, available_rx, available_tx, metrics);
                    })
                    .map_err(|error| {
                        FellmError::other(format!("spawn storage prefetch worker: {error}"))
                    })?,
            );
        }
        Ok(Self {
            submit: Some(submit),
            workers,
            buffer_bytes,
            metrics,
        })
    }

    /// Schedule a read. Submission is bounded and may apply backpressure.
    pub fn prefetch(&self, extent: StorageExtent) -> Result<Receiver<Result<PrefetchedRead>>> {
        let len = usize::try_from(extent.len)
            .map_err(|_| FellmError::other("prefetch extent length overflow"))?;
        if len > self.buffer_bytes {
            return Err(FellmError::other(format!(
                "prefetch extent {} exceeds staging buffer {}",
                extent.len, self.buffer_bytes
            )));
        }
        let (result, receive) = mpsc::sync_channel(1);
        self.submit
            .as_ref()
            .ok_or_else(|| FellmError::other("transfer pool stopped"))?
            .send(ReadCommand { extent, result })
            .map_err(|_| FellmError::other("storage prefetch workers stopped"))?;
        Ok(receive)
    }

    /// Attempt speculative read-ahead without ever blocking the execution thread behind a full
    /// submission queue. Demand reads use [`Self::prefetch`] after reserving a ring slot.
    pub fn try_prefetch(
        &self,
        extent: StorageExtent,
    ) -> Result<Option<Receiver<Result<PrefetchedRead>>>> {
        let len = usize::try_from(extent.len)
            .map_err(|_| FellmError::other("prefetch extent length overflow"))?;
        if len > self.buffer_bytes {
            return Err(FellmError::other("prefetch extent exceeds staging buffer"));
        }
        let (result, receive) = mpsc::sync_channel(1);
        match self
            .submit
            .as_ref()
            .ok_or_else(|| FellmError::other("transfer pool stopped"))?
            .try_send(ReadCommand { extent, result })
        {
            Ok(()) => Ok(Some(receive)),
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(FellmError::other("storage prefetch workers stopped"))
            }
        }
    }

    #[must_use]
    pub const fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    #[must_use]
    pub fn metrics(&self) -> TransferPoolMetrics {
        use std::sync::atomic::Ordering;
        let mut latencies = self
            .metrics
            .latencies
            .lock()
            .expect("transfer latency metrics lock")
            .iter()
            .copied()
            .collect::<Vec<_>>();
        latencies.sort_unstable();
        let percentile = |percent: usize| {
            if latencies.is_empty() {
                0
            } else {
                latencies[(latencies.len() - 1) * percent / 100]
            }
        };
        TransferPoolMetrics {
            physical_reads: self.metrics.physical_reads.load(Ordering::Relaxed),
            physical_bytes: self.metrics.physical_bytes.load(Ordering::Relaxed),
            failed_reads: self.metrics.failed_reads.load(Ordering::Relaxed),
            latency_p50_nanos: percentile(50),
            latency_p95_nanos: percentile(95),
        }
    }
}

impl Drop for BoundedTransferPool {
    fn drop(&mut self) {
        self.submit.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(
    provider: Arc<dyn TransferProvider>,
    commands: Arc<Mutex<Receiver<ReadCommand>>>,
    available: Arc<Mutex<Receiver<AlignedBuffer>>>,
    available_tx: SyncSender<AlignedBuffer>,
    metrics: Arc<TransferPoolMetricsState>,
) {
    loop {
        let command = {
            let Ok(receiver) = commands.lock() else {
                return;
            };
            receiver.recv()
        };
        let Ok(command) = command else { return };
        let buffer = {
            let Ok(receiver) = available.lock() else {
                return;
            };
            receiver.recv()
        };
        let Ok(mut buffer) = buffer else { return };
        let len = command.extent.len as usize;
        let started = std::time::Instant::now();
        let read = provider.read_at(command.extent.offset, &mut buffer.as_mut_slice()[..len]);
        let elapsed = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        metrics
            .physical_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        metrics
            .physical_bytes
            .fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        if read.is_err() {
            metrics
                .failed_reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if let Ok(mut latencies) = metrics.latencies.lock() {
            if latencies.len() == 4096 {
                latencies.pop_front();
            }
            latencies.push_back(elapsed);
        }
        let result = match read {
            Ok(()) => Ok(PrefetchedRead {
                extent: command.extent,
                buffer: Some(buffer),
                valid_len: len,
                available: available_tx.clone(),
            }),
            Err(error) => {
                let _ = available_tx.send(buffer);
                Err(error)
            }
        };
        let _ = command.result.send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn coalesces_adjacent_weight_ranges() {
        let e = |offset, len| StorageExtent {
            provider: "gguf".into(),
            path: "m.gguf".into(),
            offset,
            len,
            alignment: 4096,
        };
        let ranges = coalesce_extents(&[e(0, 4096), e(8192, 4096), e(32768, 4096)], 4096, 16384);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].len, 12288);
    }

    struct PatternProvider;
    impl TransferProvider for PatternProvider {
        fn name(&self) -> &'static str {
            "pattern"
        }
        fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()> {
            for (index, byte) in target.iter_mut().enumerate() {
                *byte = offset.wrapping_add(index as u64) as u8;
            }
            Ok(())
        }
    }

    #[test]
    fn bounded_pool_reuses_stable_staging_allocations() {
        let pool = BoundedTransferPool::new(Arc::new(PatternProvider), 2, 4096).unwrap();
        let extent = |offset| StorageExtent {
            provider: "pattern".into(),
            path: "none".into(),
            offset,
            len: 32,
            alignment: 1,
        };
        let first = pool.prefetch(extent(7)).unwrap().recv().unwrap().unwrap();
        let first_address = first.staging_address();
        assert_eq!(first.bytes()[0], 7);
        drop(first);
        let mut observed = Vec::new();
        for offset in 0..3 {
            let read = pool
                .prefetch(extent(offset))
                .unwrap()
                .recv()
                .unwrap()
                .unwrap();
            observed.push(read.staging_address());
        }
        assert!(observed.contains(&first_address));
    }

    struct ConcurrencyProvider {
        active: AtomicUsize,
        maximum: AtomicUsize,
    }

    impl TransferProvider for ConcurrencyProvider {
        fn name(&self) -> &'static str {
            "concurrency-test"
        }

        fn read_at(&self, _offset: u64, _target: &mut [u8]) -> Result<()> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.maximum.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn bounded_pool_reaches_real_queue_depth() {
        let provider = Arc::new(ConcurrencyProvider {
            active: AtomicUsize::new(0),
            maximum: AtomicUsize::new(0),
        });
        let pool = BoundedTransferPool::new(provider.clone(), 4, 4096).unwrap();
        let receivers = (0..4)
            .map(|offset| {
                pool.prefetch(StorageExtent {
                    provider: "test".into(),
                    path: "none".into(),
                    offset,
                    len: 4096,
                    alignment: 4096,
                })
                .unwrap()
            })
            .collect::<Vec<_>>();
        for receiver in receivers {
            receiver.recv().unwrap().unwrap();
        }
        assert_eq!(provider.maximum.load(Ordering::SeqCst), 4);
    }
}
