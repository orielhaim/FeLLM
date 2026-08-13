use crate::StorageExtent;
use fellm_core::error::{FellmError, Result};
use fellm_core::storage::AlignedBuffer;
use memmap2::Mmap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
    file: Mutex<File>,
}

/// Linux O_DIRECT provider for aligned `.fellm-pack` extents.
#[cfg(target_os = "linux")]
pub struct DirectFileProvider {
    file: File,
}

#[cfg(target_os = "linux")]
impl DirectFileProvider {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        // Linux O_DIRECT. Callers must provide sector-aligned offsets, lengths, and buffers.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(0o40000)
            .open(path)
            .map_err(FellmError::Io)?;
        Ok(Self { file })
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
                return Err(FellmError::other("unexpected EOF in O_DIRECT read"));
            }
            read += count;
        }
        Ok(())
    }
}

impl FileProvider {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            file: Mutex::new(File::open(path).map_err(FellmError::Io)?),
        })
    }
}

impl TransferProvider for FileProvider {
    fn name(&self) -> &'static str {
        "buffered-file"
    }
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        let mut file = self
            .file
            .lock()
            .map_err(|_| FellmError::other("file provider lock poisoned"))?;
        file.seek(SeekFrom::Start(offset)).map_err(FellmError::Io)?;
        file.read_exact(target).map_err(FellmError::Io)
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
        let mut workers = Vec::with_capacity(buffer_count);
        for index in 0..buffer_count {
            let provider = Arc::clone(&provider);
            let commands = Arc::clone(&commands);
            let available_rx = Arc::clone(&available_rx);
            let available_tx = available_tx.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("fellm-storage-prefetch-{index}"))
                    .spawn(move || worker_loop(provider, commands, available_rx, available_tx))
                    .map_err(|error| {
                        FellmError::other(format!("spawn storage prefetch worker: {error}"))
                    })?,
            );
        }
        Ok(Self {
            submit: Some(submit),
            workers,
            buffer_bytes,
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

    #[must_use]
    pub const fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
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
        let read = provider.read_at(command.extent.offset, &mut buffer.as_mut_slice()[..len]);
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
}
