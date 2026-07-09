//! Dynamic library loader for FeLLM kernel plugins.

use crate::registry::KernelRegistry;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::ABI_VERSION;
use fellm_plugin_abi::c_abi::{
    HostContext, PluginAbiVersionFn, PluginInitFn, PluginManifestFn, PluginRegisterFn,
    PluginShutdownFn, abi_hash, symbols,
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

/// Host that owns loaded plugins and a shared [`KernelRegistry`].
pub struct PluginHost {
    plugins: Vec<LoadedPlugin>,
    registry: KernelRegistry,
}

impl PluginHost {
    /// Empty host (no plugins).
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            registry: KernelRegistry::new(),
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

    /// Number of loaded plugin libraries.
    #[must_use]
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Load all plugins from `dir` (or `FELLM_PLUGIN_DIR` if `dir` is `None`).
    pub fn load_dir(&mut self, dir: Option<&Path>, ctx: &HostContext) -> Result<()> {
        let path = match dir {
            Some(p) => p.to_path_buf(),
            None => std::env::var_os("FELLM_PLUGIN_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("plugins")),
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
        if reported.major != ABI_VERSION.major || reported.minor != ABI_VERSION.minor {
            return Err(FellmError::other(format!(
                "plugin ABI {}.{} incompatible with host {}.{}",
                reported.major, reported.minor, ABI_VERSION.major, ABI_VERSION.minor
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
        let rc = unsafe { register_fn(&mut vtable) };
        if rc != 0 {
            return Err(FellmError::other(format!("plugin register failed ({rc})")));
        }

        let shutdown: Option<PluginShutdownFn> =
            unsafe { lib.get(symbols::SHUTDOWN).ok().map(|s| *s) };

        self.plugins.push(LoadedPlugin {
            path: path.to_path_buf(),
            _lib: lib,
            shutdown,
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
    matches!(ext, "so" | "dll" | "dylib")
        && (name.contains("example_cpu_op")
            || name.contains("cuda_kernels")
            || name.starts_with("lib")
            || name.ends_with(".dll"))
}

fn c_name_to_str(buf: &[std::ffi::c_char; fellm_plugin_abi::PLUGIN_NAME_MAX]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
