//! Compile-time CUDA fold + runtime backend selection with CPU fallback.

use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::traits::Backend;
use std::fmt;

/// Which backend the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Prefer CUDA when compiled in and healthy; otherwise CPU.
    #[default]
    Auto,
    /// Force CPU.
    Cpu,
    /// Force CUDA (fails if not compiled / unhealthy, unless fallback allowed).
    Cuda,
}

impl BackendPreference {
    /// Parse `auto` / `cpu` / `cuda` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(FellmError::other(format!(
                "unknown backend preference '{other}' (expected auto|cpu|cuda)"
            ))),
        }
    }
}

impl fmt::Display for BackendPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        })
    }
}

/// Runtime backend selection policy.
#[derive(Debug, Clone, Copy)]
pub struct BackendSelect {
    /// Requested backend.
    pub preference: BackendPreference,
    /// If CUDA init fails (or CUDA not compiled), fall back to CPU.
    /// When `false`, selection errors instead.
    pub allow_cpu_fallback: bool,
}

impl Default for BackendSelect {
    fn default() -> Self {
        Self::from_env()
    }
}

impl BackendSelect {
    /// Defaults: `auto`, CPU fallback on. Overridden by env:
    /// - `FELLM_BACKEND` = `auto` | `cpu` | `cuda`
    /// - `FELLM_CPU_FALLBACK` = `0`/`false`/`off` to disable fallback
    #[must_use]
    pub fn from_env() -> Self {
        let preference = std::env::var("FELLM_BACKEND")
            .ok()
            .and_then(|s| BackendPreference::parse(&s).ok())
            .unwrap_or_default();
        let allow_cpu_fallback = match std::env::var("FELLM_CPU_FALLBACK") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "0" | "false" | "off" | "no")
            }
            Err(_) => true,
        };
        Self {
            preference,
            allow_cpu_fallback,
        }
    }

    /// Explicit policy.
    #[must_use]
    pub fn new(preference: BackendPreference, allow_cpu_fallback: bool) -> Self {
        Self {
            preference,
            allow_cpu_fallback,
        }
    }

    /// True if this build included the `backend-cuda` Cargo feature.
    #[must_use]
    pub fn cuda_compiled() -> bool {
        cfg!(feature = "backend-cuda")
    }

    /// Resolve and construct a backend.
    pub fn resolve(self) -> Result<Box<dyn Backend>> {
        match self.preference {
            BackendPreference::Cpu => Ok(Box::new(backend_cpu::CpuBackend::new())),
            BackendPreference::Cuda => self.resolve_cuda_required(),
            BackendPreference::Auto => self.resolve_auto(),
        }
    }

    fn resolve_auto(self) -> Result<Box<dyn Backend>> {
        if !Self::cuda_compiled() {
            tracing::info!("backend=cpu (CUDA not compiled into this binary)");
            return Ok(Box::new(backend_cpu::CpuBackend::new()));
        }
        match try_cuda() {
            Ok(b) => {
                tracing::info!(backend = b.id(), "selected CUDA backend");
                Ok(b)
            }
            Err(e) => {
                if self.allow_cpu_fallback {
                    tracing::warn!(
                        error = %e,
                        "CUDA unavailable; falling back to CPU (set FELLM_BACKEND=cuda and FELLM_CPU_FALLBACK=0 to error instead)"
                    );
                    Ok(Box::new(backend_cpu::CpuBackend::new()))
                } else {
                    Err(FellmError::other(format!(
                        "CUDA required (auto) but init failed and CPU fallback disabled: {e}"
                    )))
                }
            }
        }
    }

    fn resolve_cuda_required(self) -> Result<Box<dyn Backend>> {
        if !Self::cuda_compiled() {
            if self.allow_cpu_fallback {
                tracing::warn!(
                    "FELLM_BACKEND=cuda but this binary was built without --features backend-cuda; using CPU"
                );
                return Ok(Box::new(backend_cpu::CpuBackend::new()));
            }
            return Err(FellmError::other(
                "FELLM_BACKEND=cuda but binary built without feature `backend-cuda` \
                 (rebuild with: cargo build -p fellm-cli --features fellm-runtime/backend-cuda)",
            ));
        }
        match try_cuda() {
            Ok(b) => {
                tracing::info!(backend = b.id(), "forced CUDA backend");
                Ok(b)
            }
            Err(e) => {
                if self.allow_cpu_fallback {
                    tracing::warn!(error = %e, "CUDA init failed; falling back to CPU");
                    Ok(Box::new(backend_cpu::CpuBackend::new()))
                } else {
                    Err(FellmError::other(format!(
                        "FELLM_BACKEND=cuda and CPU fallback disabled: {e}"
                    )))
                }
            }
        }
    }
}

/// Probe CUDA: construct backend and run a light health check.
#[cfg(feature = "backend-cuda")]
fn try_cuda() -> Result<Box<dyn Backend>> {
    let backend = backend_cuda::CudaBackend::new()?;
    // Health: device + stream handles must be non-zero on a real GPU context.
    // (cudarc Arc pointer cast is non-null after successful CudaContext::new.)
    if backend.device().device_handle() == 0 {
        return Err(FellmError::other("CUDA device handle is null"));
    }
    Ok(Box::new(backend))
}

#[cfg(not(feature = "backend-cuda"))]
fn try_cuda() -> Result<Box<dyn Backend>> {
    Err(FellmError::other(
        "CUDA support not compiled into this binary",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_preference() {
        assert_eq!(
            BackendPreference::parse("AUTO").unwrap(),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse("cpu").unwrap(),
            BackendPreference::Cpu
        );
        assert_eq!(
            BackendPreference::parse("cuda").unwrap(),
            BackendPreference::Cuda
        );
    }

    #[test]
    fn cpu_always_resolves() {
        let b = BackendSelect::new(BackendPreference::Cpu, false)
            .resolve()
            .unwrap();
        assert_eq!(b.id(), "cpu");
    }
}
