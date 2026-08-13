use crate::TransferCapabilities;
use fellm_core::error::{FellmError, Result};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StorageProviderRequest {
    #[default]
    Auto,
    PageCache,
    Mmap,
    Buffered,
    Direct,
    IoUring,
    Gds,
}

impl FromStr for StorageProviderRequest {
    type Err = FellmError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "page-cache" | "page_cache" => Ok(Self::PageCache),
            "mmap" | "mmap-copy" | "mmap_copy" => Ok(Self::Mmap),
            "buffered" | "pread" => Ok(Self::Buffered),
            "direct" | "o_direct" | "o-direct" => Ok(Self::Direct),
            "io-uring" | "io_uring" => Ok(Self::IoUring),
            "gds" => Ok(Self::Gds),
            other => Err(FellmError::other(format!(
                "unknown storage provider '{other}' (expected auto|page-cache|mmap-copy|buffered|direct|io-uring|gds)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageProviderKind {
    PageCache,
    Mmap,
    Buffered,
    Direct,
    IoUring,
    Gds,
}

impl StorageProviderKind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PageCache => "page-cache",
            Self::Mmap => "mmap-copy",
            Self::Buffered => "buffered-pread",
            Self::Direct => "direct-io",
            Self::IoUring => "io-uring-direct",
            Self::Gds => "gds",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StorageWorkload {
    /// Expected uses before an object leaves the active schedule window.
    pub reuse_count: u32,
    /// Fraction of usable host memory already committed after reserves.
    pub host_pressure: f64,
    /// True only when every physical object satisfies provider alignment constraints.
    pub direct_io_aligned: bool,
    /// Direct-to-device is useful only for a CUDA consumer with stable staging addresses.
    pub device_consumer: bool,
}

/// Select a storage provider from measured capabilities and workload behavior.
/// Forced choices never silently fall back; automatic selection always returns a correct provider.
pub fn select_storage_provider(
    requested: StorageProviderRequest,
    capabilities: TransferCapabilities,
    workload: StorageWorkload,
) -> Result<StorageProviderKind> {
    if requested != StorageProviderRequest::Auto {
        let selected = match requested {
            StorageProviderRequest::Auto => unreachable!(),
            StorageProviderRequest::PageCache if capabilities.async_file => {
                StorageProviderKind::PageCache
            }
            StorageProviderRequest::Mmap if capabilities.mmap => StorageProviderKind::Mmap,
            StorageProviderRequest::Buffered if capabilities.async_file => {
                StorageProviderKind::Buffered
            }
            StorageProviderRequest::Direct
                if capabilities.direct_io && workload.direct_io_aligned =>
            {
                StorageProviderKind::Direct
            }
            StorageProviderRequest::IoUring
                if capabilities.io_uring && workload.direct_io_aligned =>
            {
                StorageProviderKind::IoUring
            }
            StorageProviderRequest::Gds
                if capabilities.gds && workload.device_consumer && workload.direct_io_aligned =>
            {
                StorageProviderKind::Gds
            }
            forced => {
                return Err(FellmError::other(format!(
                    "forced storage provider {forced:?} is unavailable or incompatible with the model layout"
                )));
            }
        };
        return Ok(selected);
    }

    if workload.device_consumer
        && workload.direct_io_aligned
        && capabilities.gds
        && workload.reuse_count <= 1
    {
        return Ok(StorageProviderKind::Gds);
    }
    if workload.direct_io_aligned && workload.host_pressure >= 0.80 {
        if capabilities.io_uring {
            return Ok(StorageProviderKind::IoUring);
        }
        if capabilities.direct_io {
            return Ok(StorageProviderKind::Direct);
        }
    }
    if capabilities.async_file && workload.reuse_count > 1 && workload.host_pressure < 0.70 {
        return Ok(StorageProviderKind::PageCache);
    }
    Ok(StorageProviderKind::Buffered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> TransferCapabilities {
        TransferCapabilities {
            mmap: true,
            async_file: true,
            direct_io: true,
            io_uring: true,
            gds: true,
        }
    }

    #[test]
    fn auto_uses_page_cache_for_reuse_and_direct_under_pressure() {
        assert_eq!(
            select_storage_provider(
                StorageProviderRequest::Auto,
                all(),
                StorageWorkload {
                    reuse_count: 3,
                    host_pressure: 0.2,
                    direct_io_aligned: true,
                    device_consumer: true,
                }
            )
            .unwrap(),
            StorageProviderKind::PageCache
        );
        assert_eq!(
            select_storage_provider(
                StorageProviderRequest::Auto,
                all(),
                StorageWorkload {
                    reuse_count: 1,
                    host_pressure: 0.9,
                    direct_io_aligned: true,
                    device_consumer: true,
                }
            )
            .unwrap(),
            StorageProviderKind::Gds
        );
    }

    #[test]
    fn forced_direct_rejects_unaligned_gguf_objects() {
        assert!(
            select_storage_provider(
                StorageProviderRequest::Direct,
                all(),
                StorageWorkload {
                    reuse_count: 1,
                    host_pressure: 1.0,
                    direct_io_aligned: false,
                    device_consumer: true,
                }
            )
            .is_err()
        );
    }
}
