//! Runtime CPU topology discovery for cache-aware kernels.
//!
//! Zero-config: everything is measured at process start. No env vars.

use std::sync::OnceLock;

/// Detected CPU hardware characteristics used to size attention tiles.
#[derive(Debug, Clone, Copy)]
pub struct CpuHardwareProfile {
    /// L2 cache size per core (bytes).
    pub l2_bytes_per_core: usize,
    /// Shared L3 cache size (bytes).
    pub l3_bytes: usize,
    /// Physical cores (no hyperthreads).
    pub physical_cores: usize,
    /// Logical threads (incl. HT).
    pub logical_threads: usize,
    /// AVX-512F available.
    pub has_avx512: bool,
    /// AVX2 available.
    pub has_avx2: bool,
    /// ARM NEON available.
    pub has_neon: bool,
    /// Intel AMX tile extensions (detected; unused by attention v1).
    pub has_amx: bool,
    /// Default KV-sequence tile size (tokens) for a typical `head_dim=128`.
    pub kv_tile: usize,
    /// Preferred f32 SIMD lane count.
    pub simd_f32_lanes: u32,
}

impl CpuHardwareProfile {
    /// Detect once and cache.
    #[must_use]
    pub fn get() -> &'static Self {
        static PROFILE: OnceLock<CpuHardwareProfile> = OnceLock::new();
        PROFILE.get_or_init(Self::detect)
    }

    /// Fresh detection (also used by tests).
    #[must_use]
    pub fn detect() -> Self {
        let logical = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            detect_x86(logical)
        }

        #[cfg(target_arch = "aarch64")]
        {
            return detect_aarch64(logical);
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            fallback_profile(logical)
        }
    }

    /// KV tile for a concrete `head_dim`, capped by sequence length.
    ///
    /// Working set target: ≤ 75% of L2 for one head's Q + K/V tile + softmax scratch.
    #[must_use]
    pub fn kv_tile_for(&self, head_dim: usize, seq: usize) -> usize {
        let budget = (self.l2_bytes_per_core.saturating_mul(3) / 4).max(16 * 1024);
        let hd = head_dim.max(1);
        // bytes ≈ 4 * (hd + 2*T*hd + 3*T) = 4*((2T+1)*hd + 3T)
        let mut best = 16usize;
        let mut t = 16usize;
        while t <= 512 {
            let bytes = 4 * ((2 * t + 1) * hd + 3 * t);
            if bytes <= budget {
                best = t;
                t *= 2;
            } else {
                break;
            }
        }
        best.min(seq.max(1)).max(1)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn fallback_profile(logical: usize) -> CpuHardwareProfile {
    let physical = logical.max(1);
    let l2 = 1024 * 1024;
    let mut p = CpuHardwareProfile {
        l2_bytes_per_core: l2,
        l3_bytes: 8 * 1024 * 1024,
        physical_cores: physical,
        logical_threads: logical,
        has_avx512: false,
        has_avx2: false,
        has_neon: false,
        has_amx: false,
        kv_tile: 64,
        simd_f32_lanes: 8,
    };
    p.kv_tile = p.kv_tile_for(128, usize::MAX);
    p
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_x86(logical: usize) -> CpuHardwareProfile {
    use raw_cpuid::{CacheType, CpuId};

    let cpuid = CpuId::new();
    let mut l2 = 0usize;
    let mut l3 = 0usize;

    if let Some(iter) = cpuid.get_cache_parameters() {
        for cache in iter {
            let typ = cache.cache_type();
            if matches!(
                typ,
                CacheType::Null | CacheType::Instruction | CacheType::Reserved
            ) {
                continue;
            }
            let size = cache.associativity()
                * cache.physical_line_partitions()
                * cache.coherency_line_size()
                * cache.sets();
            match cache.level() {
                2 => l2 = l2.max(size),
                3 => l3 = l3.max(size),
                _ => {}
            }
        }
    }

    // AMD extended leaf fallback for L2 size (KiB).
    if l2 == 0
        && let Some(l2l3) = cpuid.get_l2_l3_cache_and_tlb_info()
    {
        l2 = (l2l3.l2cache_size() as usize) * 1024;
        // L3 is often reported in 512KB units on AMD; treat as best-effort.
        let l3_raw = l2l3.l3cache_size() as usize;
        if l3_raw > 0 {
            l3 = l3.max(l3_raw * 512 * 1024);
        }
    }

    if l2 == 0 {
        l2 = 1024 * 1024;
    }
    if l3 == 0 {
        l3 = 8 * 1024 * 1024;
    }

    let feat = cpuid.get_feature_info();
    let has_avx2 = cpuid
        .get_extended_feature_info()
        .map(|e| e.has_avx2())
        .unwrap_or(false);
    let has_avx512 = cpuid
        .get_extended_feature_info()
        .map(|e| e.has_avx512f())
        .unwrap_or(false);
    let has_amx = cpuid
        .get_extended_feature_info()
        .map(|e| e.has_amx_bf16() || e.has_amx_tile() || e.has_amx_int8())
        .unwrap_or(false);

    let ht = feat.as_ref().map(|f| f.has_htt()).unwrap_or(false);
    let has_avx = feat.as_ref().map(|f| f.has_avx()).unwrap_or(false);

    let smt_per_core = cpuid
        .get_extended_topology_info_v2()
        .or_else(|| cpuid.get_extended_topology_info())
        .and_then(|mut iter| {
            iter.find(|l| l.level_type() != raw_cpuid::TopologyType::Invalid)
                .map(|l| l.processors().max(1) as usize)
        });
    let physical = match smt_per_core {
        Some(per_core) => (logical / per_core).max(1),
        None => {
            let per_core = if ht && logical > 1 { 2 } else { 1 };
            (logical / per_core).max(1)
        }
    };

    let simd_f32_lanes = if has_avx512 {
        16
    } else if has_avx2 || has_avx {
        8
    } else {
        4
    };

    let mut p = CpuHardwareProfile {
        l2_bytes_per_core: l2,
        l3_bytes: l3,
        physical_cores: physical,
        logical_threads: logical.max(1),
        has_avx512,
        has_avx2,
        has_neon: false,
        has_amx,
        kv_tile: 64,
        simd_f32_lanes,
    };
    p.kv_tile = p.kv_tile_for(128, usize::MAX);
    tracing::info!(
        l2_kib = l2 / 1024,
        l3_kib = l3 / 1024,
        physical_cores = physical,
        logical_threads = logical,
        has_avx512,
        has_avx2,
        has_amx,
        kv_tile = p.kv_tile,
        "CPU hardware profile"
    );
    p
}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64(logical: usize) -> CpuHardwareProfile {
    let mut l2 = 1024 * 1024usize;
    let mut l3 = 8 * 1024 * 1024usize;

    // Best-effort Linux sysfs probe.
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Some(b) = parse_sysfs_size(&s) {
                l2 = b;
            }
        }
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            if let Some(b) = parse_sysfs_size(&s) {
                l3 = b;
            }
        }
    }

    let physical = logical.max(1);
    let mut p = CpuHardwareProfile {
        l2_bytes_per_core: l2,
        l3_bytes: l3,
        physical_cores: physical,
        logical_threads: logical.max(1),
        has_avx512: false,
        has_avx2: false,
        has_neon: true,
        has_amx: false,
        kv_tile: 64,
        simd_f32_lanes: 4,
    };
    p.kv_tile = p.kv_tile_for(128, usize::MAX);
    tracing::info!(
        l2_kib = l2 / 1024,
        l3_kib = l3 / 1024,
        physical_cores = physical,
        kv_tile = p.kv_tile,
        "CPU hardware profile (aarch64)"
    );
    p
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn parse_sysfs_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(n) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
        return n.parse::<usize>().ok().map(|x| x * 1024);
    }
    if let Some(n) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        return n.parse::<usize>().ok().map(|x| x * 1024 * 1024);
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_sane_values() {
        let p = CpuHardwareProfile::detect();
        assert!(p.l2_bytes_per_core >= 64 * 1024);
        assert!(p.physical_cores >= 1);
        assert!(p.logical_threads >= p.physical_cores);
        assert!(p.kv_tile >= 1);
    }

    #[test]
    fn tile_shrinks_for_large_head_dim() {
        let p = CpuHardwareProfile {
            l2_bytes_per_core: 256 * 1024,
            l3_bytes: 8 * 1024 * 1024,
            physical_cores: 4,
            logical_threads: 8,
            has_avx512: false,
            has_avx2: true,
            has_neon: false,
            has_amx: false,
            kv_tile: 64,
            simd_f32_lanes: 8,
        };
        let t64 = p.kv_tile_for(64, 4096);
        let t256 = p.kv_tile_for(256, 4096);
        assert!(t256 <= t64);
        assert_eq!(p.kv_tile_for(128, 7), 7);
    }
}
