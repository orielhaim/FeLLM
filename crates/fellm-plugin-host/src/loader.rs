//! Dynamic library loader for `FeLLM` kernel plugins.

use crate::capability_registry::CapabilityRegistry;
use crate::registry::KernelRegistry;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::ABI_VERSION;
use fellm_plugin_abi::c_abi::{
    HostContext, PluginAbiVersionFn, PluginInitFn, PluginInvalidateF32Fn, PluginManifestFn,
    PluginRegisterArchitecturesFn, PluginRegisterCapabilitiesFn, PluginRegisterDeviceTensorFn,
    PluginRegisterFn, PluginShutdownFn, PluginUpdateStepParamsFn, abi_hash, symbols,
};
use libloading::Library;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// One loaded plugin library (kept alive so function pointers remain valid).
pub struct LoadedPlugin {
    /// Filesystem path.
    pub path: PathBuf,
    _lib: Library,
    shutdown: Option<PluginShutdownFn>,
    invalidate_f32: Option<PluginInvalidateF32Fn>,
    update_step_params: Option<PluginUpdateStepParamsFn>,
    register_device_tensor: Option<PluginRegisterDeviceTensorFn>,
}

impl Drop for LoadedPlugin {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            unsafe {
                shutdown();
            }
        }
    }
}

/// Host that owns loaded plugins and shared registries.
pub struct PluginHost {
    plugins: Vec<LoadedPlugin>,
    registry: KernelRegistry,
    architectures: crate::registry::ArchitectureRegistry,
    capabilities: CapabilityRegistry,
}

impl PluginHost {
    /// Empty host (no dynamic plugins; builtins installed in capability registry).
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            registry: KernelRegistry::new(),
            architectures: crate::registry::ArchitectureRegistry::new(),
            capabilities: CapabilityRegistry::new(),
        }
    }

    /// Shared registry of registered kernels.
    #[must_use]
    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Mutable registry.
    pub fn registry_mut(&mut self) -> &mut KernelRegistry {
        &mut self.registry
    }

    /// Architecture provider registry.
    #[must_use]
    pub fn architectures(&self) -> &crate::registry::ArchitectureRegistry {
        &self.architectures
    }

    /// Mutable architecture provider registry.
    pub fn architectures_mut(&mut self) -> &mut crate::registry::ArchitectureRegistry {
        &mut self.architectures
    }

    /// Multi-capability provider registry.
    #[must_use]
    pub fn capabilities(&self) -> &CapabilityRegistry {
        &self.capabilities
    }

    /// Mutable multi-capability registry.
    pub fn capabilities_mut(&mut self) -> &mut CapabilityRegistry {
        &mut self.capabilities
    }

    /// Number of loaded plugin libraries.
    #[must_use]
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Paths of loaded plugin libraries.
    #[must_use]
    pub fn plugin_paths(&self) -> Vec<&std::path::Path> {
        self.plugins.iter().map(|p| p.path.as_path()).collect()
    }

    /// Invalidate device mirrors for host f32 buffers written by CPU fallback.
    ///
    /// Call after any CPU op that mutates activation tensors so the next GPU
    /// `ensure_f32` re-uploads instead of trusting a stale `device_valid` cache.
    pub fn invalidate_f32_outputs(&self, outputs: &[fellm_plugin_abi::TensorMut]) {
        for out in outputs {
            if out
                .dtype()
                .is_some_and(|d| d == fellm_core::dtype::DType::F32)
                && !out.data.is_null()
                && out.byte_len >= 4
            {
                let ptr = out.data as *const f32;
                for plugin in &self.plugins {
                    if let Some(inv) = plugin.invalidate_f32 {
                        unsafe { inv(ptr, out.byte_len as usize) };
                    }
                }
            }
        }
    }

    /// Upload the fixed-layout controls consumed by prepared CUDA kernels.
    pub fn update_step_params(&self, params: &fellm_plugin_abi::DeviceStepParams) -> Result<()> {
        for plugin in &self.plugins {
            if let Some(update) = plugin.update_step_params {
                let rc = unsafe { update(std::ptr::from_ref(params)) };
                if rc != 0 {
                    return Err(FellmError::other(format!(
                        "plugin step-parameter update failed ({rc})"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Bind a host constant to its stable address in the packed CUDA model image.
    pub fn register_device_tensor(
        &self,
        host_ptr: *const u8,
        nbytes: usize,
        device_ptr: u64,
    ) -> Result<()> {
        for plugin in &self.plugins {
            if let Some(register) = plugin.register_device_tensor {
                let rc = unsafe { register(host_ptr, nbytes, device_ptr) };
                if rc != 0 {
                    return Err(FellmError::other(format!(
                        "plugin device tensor registration failed ({rc})"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Load all plugins from `dir` (or `FELLM_PLUGIN_DIR` if `dir` is `None`).
    pub fn load_dir(&mut self, dir: Option<&Path>, ctx: &HostContext) -> Result<()> {
        let path = match dir {
            Some(p) => p.to_path_buf(),
            None => std::env::var_os("FELLM_PLUGIN_DIR")
                .map_or_else(|| PathBuf::from("plugins"), PathBuf::from),
        };
        if !path.is_dir() {
            tracing::debug!(?path, "plugin dir missing; skipping");
            return Ok(());
        }
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&path)
            .map_err(|e| FellmError::other(format!("read plugin dir: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| is_plugin_lib(p))
            .collect();
        entries.sort();
        for entry in entries {
            match self.load_path(&entry, ctx) {
                Ok(()) => tracing::info!(path = %entry.display(), "loaded plugin"),
                Err(e) => tracing::warn!(path = %entry.display(), error = %e, "skip plugin"),
            }
        }
        Ok(())
    }

    /// Load a single plugin library.
    pub fn load_path(&mut self, path: &Path, ctx: &HostContext) -> Result<()> {
        // SAFETY: plugins are trusted; we verify ABI version before init.
        let lib = unsafe { Library::new(path) }
            .map_err(|e| FellmError::other(format!("dlopen {}: {e}", path.display())))?;

        let abi_version_fn: PluginAbiVersionFn = unsafe {
            *lib.get(symbols::ABI_VERSION)
                .map_err(|e| FellmError::other(format!("missing abi_version: {e}")))?
        };
        let reported = unsafe { abi_version_fn() };
        if reported.major != ABI_VERSION.major
            || reported.minor != ABI_VERSION.minor
            || reported.patch != ABI_VERSION.patch
        {
            return Err(FellmError::other(format!(
                "plugin ABI {}.{}.{} incompatible with host {}.{}.{}",
                reported.major,
                reported.minor,
                reported.patch,
                ABI_VERSION.major,
                ABI_VERSION.minor,
                ABI_VERSION.patch
            )));
        }

        if let Ok(sym) = unsafe { lib.get::<PluginManifestFn>(symbols::MANIFEST) } {
            let manifest = unsafe { (*sym)() };
            if manifest.abi_hash != 0 && manifest.abi_hash != abi_hash() {
                return Err(FellmError::other(format!(
                    "plugin abi_hash {:#x} != host {:#x}",
                    manifest.abi_hash,
                    abi_hash()
                )));
            }
            let _name = c_name_to_str(&manifest.name);
        }

        let init_fn: PluginInitFn = unsafe {
            *lib.get(symbols::INIT)
                .map_err(|e| FellmError::other(format!("missing init: {e}")))?
        };
        let rc = unsafe { init_fn(ctx) };
        if rc != 0 {
            return Err(FellmError::other(format!("plugin init failed ({rc})")));
        }

        let register_fn: PluginRegisterFn = unsafe {
            *lib.get(symbols::REGISTER)
                .map_err(|e| FellmError::other(format!("missing register: {e}")))?
        };
        let mut vtable = self.registry.vtable();
        let rc = unsafe { register_fn(&raw mut vtable) };
        if rc != 0 {
            return Err(FellmError::other(format!("plugin register failed ({rc})")));
        }

        if let Ok(sym) =
            unsafe { lib.get::<PluginRegisterArchitecturesFn>(symbols::REGISTER_ARCHITECTURES) }
        {
            let mut architecture_vtable = self.architectures.vtable();
            let rc = unsafe { (*sym)(&raw mut architecture_vtable) };
            if rc != 0 {
                return Err(FellmError::other(format!(
                    "plugin architecture registration failed ({rc})"
                )));
            }
        }

        if let Ok(sym) =
            unsafe { lib.get::<PluginRegisterCapabilitiesFn>(symbols::REGISTER_CAPABILITIES) }
        {
            let mut cap_vtable = self.capabilities.vtable();
            let rc = unsafe { (*sym)(&raw mut cap_vtable) };
            if rc != 0 {
                return Err(FellmError::other(format!(
                    "plugin capability registration failed ({rc})"
                )));
            }
            // Tag dynamic providers with source path.
            for p in self.capabilities.list() {
                // source already set for builtins; dynamic entries have None.
                let _ = p;
            }
        }

        let shutdown: Option<PluginShutdownFn> =
            unsafe { lib.get(symbols::SHUTDOWN).ok().map(|s| *s) };
        let invalidate_f32: Option<PluginInvalidateF32Fn> =
            unsafe { lib.get(symbols::INVALIDATE_F32).ok().map(|s| *s) };
        let update_step_params: Option<PluginUpdateStepParamsFn> =
            unsafe { lib.get(symbols::UPDATE_STEP_PARAMS).ok().map(|s| *s) };
        let register_device_tensor: Option<PluginRegisterDeviceTensorFn> =
            unsafe { lib.get(symbols::REGISTER_DEVICE_TENSOR).ok().map(|s| *s) };

        self.plugins.push(LoadedPlugin {
            path: path.to_path_buf(),
            _lib: lib,
            shutdown,
            invalidate_f32,
            update_step_params,
            register_device_tensor,
        });
        Ok(())
    }
}

impl Default for PluginHost {
    fn default() -> Self {
        Self::new()
    }
}

fn is_plugin_lib(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    // Skip the example crate's rlib / build artifacts; accept cdylib names.
    let ext = path.extension().and_then(OsStr::to_str).unwrap_or("");
    // Only accept the platform's native loadable extension. On Windows that is
    // `.dll`; on Linux/WSL it is `.so`. This prevents the loader from trying to
    // `dlopen` a checked-in Windows `.dll` under WSL (and vice-versa).
    let native_ok = cfg!(target_os = "windows")
        .then(|| matches!(ext, "dll"))
        .unwrap_or(false)
        || cfg!(any(target_os = "linux", target_os = "android"))
            .then(|| matches!(ext, "so" | "dylib"))
            .unwrap_or(false);
    native_ok
        && (name.contains("example_cpu_op")
            || name.contains("cuda_kernels")
            || name.contains("triattention")
            || name.starts_with("lib"))
}

fn c_name_to_str(buf: &[std::ffi::c_char; fellm_plugin_abi::PLUGIN_NAME_MAX]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_lib_is_platform_native_extension() {
        // A `.dll` must never be treated as loadable on Linux/WSL, and a `.so`
        // must never be treated as loadable on Windows. This is what caused the
        // "skip plugin path=...fellm_triattention.dll error=dlopen failed" warn.
        let dll = Path::new("plugins/fellm_triattention.dll");
        let so = Path::new("plugins/libcuda_kernels.so");
        assert_eq!(
            is_plugin_lib(dll),
            cfg!(target_os = "windows"),
            "dll loadable only on Windows"
        );
        assert_eq!(
            is_plugin_lib(so),
            cfg!(any(target_os = "linux", target_os = "android")),
            "so loadable only on Linux"
        );
        // Rlibs / random files are never plugins.
        assert!(!is_plugin_lib(Path::new("plugins/triattention/README.md")));
        assert!(!is_plugin_lib(Path::new("plugins/fellm_triattention.rlib")));
        assert!(!is_plugin_lib(Path::new("plugins/fellm_triattention.pdb")));
    }
}
