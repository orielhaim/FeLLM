//! CUDA Graph capture and replay on the device plugin's compute stream.

use fellm_core::error::{FellmError, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, result, sys};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn synchronize_external_stream(
    device: &crate::CudaDeviceState,
    stream: u64,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        device
            .context()
            .bind_to_thread()
            .map_err(|error| FellmError::other(format!("bind CUDA stream context: {error}")))?;
        let stream = stream as usize as sys::CUstream;
        // SAFETY: the active plugin exports this live stream from this context.
        unsafe { result::stream::synchronize(stream) }
            .map_err(|error| FellmError::other(format!("synchronize CUDA plugin stream: {error}")))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, stream);
        Err(FellmError::other("cuda feature disabled"))
    }
}

/// An in-progress stream capture.
///
/// Kernel launches issued through the backend between construction and
/// [`Self::finish`] become one executable graph. Dropping an unfinished
/// capture closes and destroys it so the plugin stream is never left in
/// capture mode after an error.
pub struct CudaGraphCapture {
    #[cfg(feature = "cuda")]
    context: Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    stream: sys::CUstream,
    #[cfg(feature = "cuda")]
    active: bool,
}

impl CudaGraphCapture {
    pub(crate) fn begin(device: &crate::CudaDeviceState, stream: u64) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let stream = stream as usize as sys::CUstream;
            device
                .context()
                .bind_to_thread()
                .map_err(|error| FellmError::other(format!("bind CUDA graph context: {error}")))?;
            // SAFETY: the plugin exported this live stream from the same CUDA context.
            unsafe {
                result::stream::begin_capture(
                    stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            }
            .map_err(|error| FellmError::other(format!("begin CUDA graph capture: {error}")))?;
            Ok(Self {
                context: Arc::clone(device.context()),
                stream,
                active: true,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, stream);
            Err(FellmError::other("cuda feature disabled"))
        }
    }

    /// Finish capture, instantiate it, and upload the executable graph.
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    pub fn finish(mut self) -> Result<CudaGraphExec> {
        #[cfg(feature = "cuda")]
        {
            // SAFETY: this object exclusively owns the active capture lifecycle.
            let graph = unsafe { result::stream::end_capture(self.stream) }
                .map_err(|error| FellmError::other(format!("end CUDA graph capture: {error}")))?;
            self.active = false;
            let mut executable = std::ptr::null_mut();
            // SAFETY: graph is live and executable points to writable output storage.
            let instantiate =
                unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, graph, 0) };
            if instantiate != sys::CUresult::CUDA_SUCCESS {
                // SAFETY: graph was returned by the successful capture above.
                let _ = unsafe { result::graph::destroy(graph) };
                return Err(FellmError::other(format!(
                    "instantiate CUDA graph: {instantiate:?}"
                )));
            }
            // SAFETY: both handles are live and the stream belongs to this context.
            if let Err(error) = unsafe { result::graph::upload(executable, self.stream) } {
                let _ = unsafe { result::graph::exec_destroy(executable) };
                let _ = unsafe { result::graph::destroy(graph) };
                return Err(FellmError::other(format!("upload CUDA graph: {error}")));
            }
            Ok(CudaGraphExec {
                context: Arc::clone(&self.context),
                stream: self.stream,
                graph,
                executable,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(FellmError::other("cuda feature disabled"))
        }
    }
}

impl Drop for CudaGraphCapture {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.active {
            // SAFETY: best-effort cleanup of the capture owned by this object.
            if let Ok(graph) = unsafe { result::stream::end_capture(self.stream) } {
                let _ = unsafe { result::graph::destroy(graph) };
            }
        }
    }
}

/// Instantiated CUDA graph with stable topology and kernel arguments.
pub struct CudaGraphExec {
    #[cfg(feature = "cuda")]
    context: Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    stream: sys::CUstream,
    #[cfg(feature = "cuda")]
    graph: sys::CUgraph,
    #[cfg(feature = "cuda")]
    executable: sys::CUgraphExec,
}

impl CudaGraphExec {
    /// Enqueue one replay on the plugin compute stream.
    pub fn launch(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.context
                .bind_to_thread()
                .map_err(|error| FellmError::other(format!("bind CUDA graph context: {error}")))?;
            // SAFETY: both handles remain owned by this object.
            unsafe { result::graph::launch(self.executable, self.stream) }
                .map_err(|error| FellmError::other(format!("launch CUDA graph: {error}")))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(FellmError::other("cuda feature disabled"))
        }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let _ = self.context.bind_to_thread();
            // SAFETY: these handles are uniquely owned and destroyed once here.
            let _ = unsafe { result::graph::exec_destroy(self.executable) };
            let _ = unsafe { result::graph::destroy(self.graph) };
        }
    }
}
