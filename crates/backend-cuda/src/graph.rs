//! Bucketed CUDA graph capture for decode steps.
//!
//! `CudaGraph` is not Sync in cudarc; FeLLM's inference worker is single-threaded,
//! so we mark [`GraphCache`] Send+Sync under that ownership contract.

#[cfg(feature = "cuda")]
use fellm_core::error::FellmError;
use fellm_core::error::Result;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use crate::device::CudaDeviceState;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaGraph, sys};

/// Past-length bucket for a captured graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphBucket {
    /// Inclusive lower bound on `past_len`.
    pub min_past: u32,
    /// Exclusive upper bound on `past_len`.
    pub max_past: u32,
}

impl GraphBucket {
    /// Standard power-of-two buckets up to `max_ctx`.
    #[must_use]
    pub fn buckets_for_ctx(max_ctx: u32) -> Vec<Self> {
        let mut out = Vec::new();
        let mut lo = 0u32;
        let mut span = 128u32;
        while lo < max_ctx {
            let hi = (lo + span).min(max_ctx + 1);
            out.push(Self {
                min_past: lo,
                max_past: hi,
            });
            lo = hi;
            span = span.saturating_mul(2).min(2048);
        }
        out
    }

    /// Whether `past_len` falls in this bucket.
    #[must_use]
    pub fn contains(self, past_len: u32) -> bool {
        past_len >= self.min_past && past_len < self.max_past
    }
}

/// Cache of instantiated CUDA graphs keyed by past-length bucket.
pub struct GraphCache {
    #[cfg(feature = "cuda")]
    graphs: HashMap<GraphBucket, CudaGraph>,
}

// SAFETY: FeLLM runs CUDA graph capture/launch only on the dedicated inference
// thread (see fellm-server worker). Concurrent access is forbidden by design.
unsafe impl Send for GraphCache {}
unsafe impl Sync for GraphCache {}

impl GraphCache {
    /// Empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            graphs: HashMap::new(),
        }
    }

    /// Look up a graph for `past_len`.
    #[cfg(feature = "cuda")]
    pub fn get(&self, past_len: u32) -> Option<&CudaGraph> {
        self.graphs
            .iter()
            .find(|(b, _)| b.contains(past_len))
            .map(|(_, g)| g)
    }

    /// True if a graph covering `past_len` is cached.
    #[must_use]
    pub fn has(&self, past_len: u32) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.get(past_len).is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = past_len;
            false
        }
    }

    /// Capture `body` on the device stream into `bucket`.
    ///
    /// `body` must only enqueue GPU work (no host sync).
    #[cfg(feature = "cuda")]
    pub fn capture<F>(
        &mut self,
        device: &CudaDeviceState,
        bucket: GraphBucket,
        body: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let stream = device.stream();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| FellmError::other(format!("begin_capture: {e}")))?;
        let body_result = body();
        let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream
            .end_capture(flags)
            .map_err(|e| FellmError::other(format!("end_capture: {e}")))?;
        body_result?;
        let Some(graph) = graph else {
            return Err(FellmError::other("end_capture returned no graph"));
        };
        let _ = graph.upload();
        self.graphs.insert(bucket, graph);
        Ok(())
    }

    /// Launch a previously captured graph for `past_len`.
    ///
    /// Returns `true` if a graph was found and launched.
    pub fn launch(&self, past_len: u32) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            let Some(graph) = self.get(past_len) else {
                return Ok(false);
            };
            graph
                .launch()
                .map_err(|e| FellmError::other(format!("graph.launch: {e}")))?;
            Ok(true)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = past_len;
            Ok(false)
        }
    }

    /// Number of captured graphs.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.graphs.len()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_ctx() {
        let buckets = GraphBucket::buckets_for_ctx(1000);
        assert!(!buckets.is_empty());
        assert!(buckets[0].contains(0));
        assert!(buckets.iter().any(|b| b.contains(999)));
    }
}
