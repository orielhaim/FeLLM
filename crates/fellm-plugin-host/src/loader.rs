//! Dynamic library discovery and activation for FeLLM plugins.

use crate::capability_registry::CapabilityRegistry;
use crate::manifest::{PluginComponentKind, PluginManifest, parse_manifest};
use crate::registry::KernelRegistry;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::ABI_VERSION;
use fellm_plugin_abi::c_abi::{
    HostContext, PluginAbiVersionFn, PluginDeviceStreamFn, PluginInitFn, PluginInvalidateF32Fn,
    PluginManifestJsonFn, PluginPrefetchWeightsFn, PluginRegisterArchitecturesFn,
    PluginRegisterCapabilitiesFn, PluginRegisterDeviceTensorFn, PluginRegisterKernelsFn,
    PluginSetWeightCacheBudgetFn, PluginShutdownFn, PluginUpdateStepParamsFn,
    PluginWeightCacheMetrics, PluginWeightCacheMetricsFn, symbols,
};
use libloading::Library;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;

/// A plugin found during discovery, before it has been initialized.
#[derive(Debug, Clone)]
pub struct DiscoveredPlugin {
    /// Filesystem path of the dynamic library.
    pub path: PathBuf,
    /// Parsed manifest embedded in the library.
    pub manifest: PluginManifest,
}

/// Catalog of plugins discovered from a directory.
#[derive(Debug, Clone, Default)]
pub struct PluginCatalog {
    plugins: Vec<DiscoveredPlugin>,
}

impl PluginCatalog {
    /// Discover all native dynamic libraries in `dir` without initializing
    /// or registering any plugin.
    pub fn discover_dir(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            return Ok(Self::default());
        }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| FellmError::other(format!("read plugin dir: {e}")))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| is_dynamic_library(path))
            .collect();
        paths.sort();

        let mut plugins = Vec::with_capacity(paths.len());
        let mut ids = HashSet::new();
        for path in paths {
            match Self::discover_path(&path) {
                Ok(plugin) => {
                    if !ids.insert(plugin.manifest.id.clone()) {
                        return Err(FellmError::other(format!(
                            "duplicate plugin id {}",
                            plugin.manifest.id
                        )));
                    }
                    plugins.push(plugin);
                }
                Err(error) => {
                    tracing::warn!(path = %path.display(), %error, "skip non-FeLLM dynamic library");
                }
            }
        }
        Ok(Self { plugins })
    }

    /// Discover one native dynamic library without initializing it.
    pub fn discover_path(path: &Path) -> Result<DiscoveredPlugin> {
        if !is_dynamic_library(path) {
            return Err(FellmError::other(format!(
                "not a native dynamic library: {}",
                path.display()
            )));
        }
        let lib = unsafe { Library::new(path) }
            .map_err(|e| FellmError::other(format!("dlopen {}: {e}", path.display())))?;
        verify_abi(&lib, path)?;
        let manifest = read_manifest(&lib, path)?;
        Ok(DiscoveredPlugin {
            path: path.to_path_buf(),
            manifest,
        })
    }

    /// Discovered plugins in deterministic path order.
    #[must_use]
    pub fn plugins(&self) -> &[DiscoveredPlugin] {
        &self.plugins
    }
}

/// One activated plugin library. The library remains alive while all function
/// pointers registered in host registries may be called.
pub struct LoadedPlugin {
    /// Filesystem path.
    pub path: PathBuf,
    /// Declarative metadata used during activation.
    pub manifest: PluginManifest,
    _lib: Library,
    shutdown: PluginShutdownFn,
    invalidate_f32: Option<PluginInvalidateF32Fn>,
    update_step_params: Option<PluginUpdateStepParamsFn>,
    register_device_tensor: Option<PluginRegisterDeviceTensorFn>,
    set_weight_cache_budget: Option<PluginSetWeightCacheBudgetFn>,
    prefetch_weights: Option<PluginPrefetchWeightsFn>,
    weight_cache_metrics: Option<PluginWeightCacheMetricsFn>,
    device_stream: Option<PluginDeviceStreamFn>,
}

impl Drop for LoadedPlugin {
    fn drop(&mut self) {
        unsafe { (self.shutdown)() };
    }
}

/// Host that owns activated plugins and shared registries.
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
    pub fn plugin_paths(&self) -> Vec<&Path> {
        self.plugins
            .iter()
            .map(|plugin| plugin.path.as_path())
            .collect()
    }

    /// Invalidate device mirrors for host f32 buffers written by CPU fallback.
    pub fn invalidate_f32_outputs(&self, outputs: &[fellm_plugin_abi::TensorMut]) {
        for out in outputs {
            if out
                .dtype()
                .is_some_and(|dtype| dtype == fellm_core::dtype::DType::F32)
                && !out.data.is_null()
                && out.byte_len >= 4
            {
                let ptr = out.data as *const f32;
                for plugin in &self.plugins {
                    if let Some(invalidate) = plugin.invalidate_f32 {
                        unsafe { invalidate(ptr, out.byte_len as usize) };
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

    /// Publish a host view's resident device replica to device plugins.
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

    /// Configure the bounded device working set used by streaming weight providers.
    pub fn set_weight_cache_budget(&self, bytes: u64, buffer_count: u32) -> Result<()> {
        for plugin in &self.plugins {
            if let Some(set_budget) = plugin.set_weight_cache_budget {
                let rc = unsafe { set_budget(bytes, buffer_count) };
                if rc != 0 {
                    return Err(FellmError::other(format!(
                        "plugin weight-cache budget failed ({rc})"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Enqueue an architecture-neutral future execution group's immutable weights.
    pub fn prefetch_weight_group(
        &self,
        group_id: u64,
        weights: &[fellm_plugin_abi::TensorRef],
    ) -> Result<()> {
        if weights.is_empty() {
            return Ok(());
        }
        for plugin in &self.plugins {
            if let Some(prefetch) = plugin.prefetch_weights {
                let rc = unsafe { prefetch(group_id, weights.as_ptr(), weights.len()) };
                if rc != 0 {
                    return Err(FellmError::other(format!(
                        "plugin weight prefetch failed ({rc})"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Aggregate current weight-provider telemetry across loaded device plugins.
    #[must_use]
    pub fn weight_cache_metrics(&self) -> PluginWeightCacheMetrics {
        let mut aggregate = PluginWeightCacheMetrics::default();
        for plugin in &self.plugins {
            let Some(read) = plugin.weight_cache_metrics else {
                continue;
            };
            let mut snapshot = PluginWeightCacheMetrics::default();
            if unsafe { read(std::ptr::from_mut(&mut snapshot)) } == 0 {
                aggregate.resident_bytes = aggregate
                    .resident_bytes
                    .saturating_add(snapshot.resident_bytes);
                aggregate.h2d_bytes = aggregate.h2d_bytes.saturating_add(snapshot.h2d_bytes);
                aggregate.prefetch_hits = aggregate
                    .prefetch_hits
                    .saturating_add(snapshot.prefetch_hits);
                aggregate.prefetch_misses = aggregate
                    .prefetch_misses
                    .saturating_add(snapshot.prefetch_misses);
                aggregate.evictions = aggregate.evictions.saturating_add(snapshot.evictions);
            }
        }
        aggregate
    }

    /// Capture-capable stream exported by the active device plugin.
    #[must_use]
    pub fn device_stream(&self) -> Option<fellm_plugin_abi::StreamHandle> {
        self.plugins
            .iter()
            .find_map(|plugin| plugin.device_stream.map(|get| unsafe { get() }))
            .filter(|stream| *stream != 0)
    }

    /// Discover and activate all plugins in `dir` (or `FELLM_PLUGIN_DIR` if
    /// `dir` is `None`). Discovery completes before the first plugin is
    /// initialized.
    pub fn load_dir(&mut self, dir: Option<&Path>, ctx: &HostContext) -> Result<()> {
        let path = match dir {
            Some(path) => path.to_path_buf(),
            None => std::env::var_os("FELLM_PLUGIN_DIR")
                .map_or_else(|| PathBuf::from("plugins"), PathBuf::from),
        };
        let catalog = PluginCatalog::discover_dir(&path)?;
        for plugin in catalog.plugins() {
            match self.activate_path(plugin, ctx) {
                Ok(()) => tracing::info!(
                    path = %plugin.path.display(),
                    id = %plugin.manifest.id,
                    "activated plugin"
                ),
                Err(error) => tracing::warn!(
                    path = %plugin.path.display(),
                    id = %plugin.manifest.id,
                    %error,
                    "skip plugin activation"
                ),
            }
        }
        Ok(())
    }

    /// Discover and activate one plugin library.
    pub fn load_path(&mut self, path: &Path, ctx: &HostContext) -> Result<()> {
        let discovered = PluginCatalog::discover_path(path)?;
        self.activate_path(&discovered, ctx)
    }

    fn activate_path(&mut self, discovered: &DiscoveredPlugin, ctx: &HostContext) -> Result<()> {
        if self
            .plugins
            .iter()
            .any(|plugin| plugin.manifest.id == discovered.manifest.id)
        {
            return Err(FellmError::other(format!(
                "plugin id {} is already activated",
                discovered.manifest.id
            )));
        }

        let lib = unsafe { Library::new(&discovered.path) }
            .map_err(|e| FellmError::other(format!("dlopen {}: {e}", discovered.path.display())))?;
        verify_abi(&lib, &discovered.path)?;
        let manifest = read_manifest(&lib, &discovered.path)?;
        if manifest.id != discovered.manifest.id {
            return Err(FellmError::other(format!(
                "plugin manifest changed during activation: {} -> {}",
                discovered.manifest.id, manifest.id
            )));
        }

        let init: PluginInitFn = unsafe {
            *lib.get(symbols::INIT)
                .map_err(|e| FellmError::other(format!("missing _fellm_plugin_init: {e}")))?
        };
        let shutdown: PluginShutdownFn = unsafe {
            *lib.get(symbols::SHUTDOWN)
                .map_err(|e| FellmError::other(format!("missing _fellm_plugin_shutdown: {e}")))?
        };
        let registrations = resolve_registrations(&lib, &manifest)?;
        let invalidate_f32 = unsafe { lib.get(symbols::INVALIDATE_F32).ok().map(|s| *s) };
        let update_step_params = unsafe { lib.get(symbols::UPDATE_STEP_PARAMS).ok().map(|s| *s) };
        let register_device_tensor =
            unsafe { lib.get(symbols::REGISTER_DEVICE_TENSOR).ok().map(|s| *s) };
        let set_weight_cache_budget =
            unsafe { lib.get(symbols::SET_WEIGHT_CACHE_BUDGET).ok().map(|s| *s) };
        let prefetch_weights = unsafe { lib.get(symbols::PREFETCH_WEIGHTS).ok().map(|s| *s) };
        let weight_cache_metrics =
            unsafe { lib.get(symbols::WEIGHT_CACHE_METRICS).ok().map(|s| *s) };
        let device_stream = unsafe { lib.get(symbols::DEVICE_STREAM).ok().map(|s| *s) };

        let result = (|| {
            let rc = unsafe { init(ctx) };
            if rc != 0 {
                return Err(FellmError::other(format!("plugin init failed ({rc})")));
            }
            register_components(self, &registrations)
        })();
        if let Err(error) = result {
            unsafe { shutdown() };
            return Err(error);
        }

        self.plugins.push(LoadedPlugin {
            path: discovered.path.clone(),
            manifest,
            _lib: lib,
            shutdown,
            invalidate_f32,
            update_step_params,
            register_device_tensor,
            set_weight_cache_budget,
            prefetch_weights,
            weight_cache_metrics,
            device_stream,
        });
        Ok(())
    }
}

impl Default for PluginHost {
    fn default() -> Self {
        Self::new()
    }
}

struct Registrations {
    kernels: Option<PluginRegisterKernelsFn>,
    architectures: Option<PluginRegisterArchitecturesFn>,
    capabilities: Option<PluginRegisterCapabilitiesFn>,
}

fn register_components(host: &mut PluginHost, registrations: &Registrations) -> Result<()> {
    if let Some(register) = registrations.kernels {
        let mut vtable = host.registry.vtable();
        let rc = unsafe { register(&raw mut vtable) };
        if rc != 0 {
            return Err(FellmError::other(format!(
                "plugin kernel registration failed ({rc})"
            )));
        }
    }
    if let Some(register) = registrations.architectures {
        let mut vtable = host.architectures.vtable();
        let rc = unsafe { register(&raw mut vtable) };
        if rc != 0 {
            return Err(FellmError::other(format!(
                "plugin architecture registration failed ({rc})"
            )));
        }
    }
    if let Some(register) = registrations.capabilities {
        let mut vtable = host.capabilities.vtable();
        let rc = unsafe { register(&raw mut vtable) };
        if rc != 0 {
            return Err(FellmError::other(format!(
                "plugin capability registration failed ({rc})"
            )));
        }
    }
    Ok(())
}

fn resolve_registrations(lib: &Library, manifest: &PluginManifest) -> Result<Registrations> {
    let mut kernels = false;
    let mut architectures = false;
    let mut capabilities = false;
    for component in &manifest.provides {
        let (required, symbol) = match component.kind() {
            PluginComponentKind::Backend | PluginComponentKind::Unknown(_) => continue,
            PluginComponentKind::Kernels => {
                kernels = true;
                (
                    component.entrypoint.as_deref(),
                    "_fellm_plugin_register_kernels",
                )
            }
            PluginComponentKind::Architecture => {
                architectures = true;
                (
                    component.entrypoint.as_deref(),
                    "_fellm_plugin_register_architectures",
                )
            }
            PluginComponentKind::Capability => {
                capabilities = true;
                (
                    component.entrypoint.as_deref(),
                    "_fellm_plugin_register_capabilities",
                )
            }
        };
        if required.is_some_and(|entrypoint| !entrypoint_matches(entrypoint, symbol)) {
            return Err(FellmError::other(format!(
                "plugin {} declares unsupported entrypoint for {} (expected {symbol})",
                manifest.id, component.component_type
            )));
        }
    }

    let kernels = if kernels {
        Some(unsafe {
            *lib.get(symbols::REGISTER_KERNELS).map_err(|e| {
                FellmError::other(format!(
                    "manifest declares kernels but symbol is missing: {e}"
                ))
            })?
        })
    } else {
        None
    };
    let architectures = if architectures {
        Some(unsafe {
            *lib.get(symbols::REGISTER_ARCHITECTURES).map_err(|e| {
                FellmError::other(format!(
                    "manifest declares architecture but symbol is missing: {e}"
                ))
            })?
        })
    } else {
        None
    };
    let capabilities = if capabilities {
        Some(unsafe {
            *lib.get(symbols::REGISTER_CAPABILITIES).map_err(|e| {
                FellmError::other(format!(
                    "manifest declares capability but symbol is missing: {e}"
                ))
            })?
        })
    } else {
        None
    };
    Ok(Registrations {
        kernels,
        architectures,
        capabilities,
    })
}

fn entrypoint_matches(declared: &str, symbol: &str) -> bool {
    declared == symbol
        || symbol
            .strip_prefix("_fellm_plugin_")
            .is_some_and(|expected| declared == expected)
}

fn verify_abi(lib: &Library, path: &Path) -> Result<()> {
    let abi_version: PluginAbiVersionFn = unsafe {
        *lib.get(symbols::ABI_VERSION).map_err(|e| {
            FellmError::other(format!(
                "missing _fellm_plugin_abi_version in {}: {e}",
                path.display()
            ))
        })?
    };
    let reported = unsafe { abi_version() };
    if reported != ABI_VERSION {
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
    Ok(())
}

fn read_manifest(lib: &Library, path: &Path) -> Result<PluginManifest> {
    let manifest_json: PluginManifestJsonFn = unsafe {
        *lib.get(symbols::MANIFEST_JSON).map_err(|e| {
            FellmError::other(format!(
                "missing _fellm_plugin_manifest_json in {}: {e}",
                path.display()
            ))
        })?
    };
    let raw = unsafe { manifest_json() };
    if raw.ptr.is_null() || raw.len == 0 || raw.len > MAX_MANIFEST_BYTES {
        return Err(FellmError::other(format!(
            "invalid embedded manifest buffer in {}",
            path.display()
        )));
    }
    let bytes = unsafe { std::slice::from_raw_parts(raw.ptr, raw.len) };
    parse_manifest(bytes)
}

fn is_dynamic_library(path: &Path) -> bool {
    let extension = path.extension().and_then(|extension| extension.to_str());
    match () {
        _ if cfg!(target_os = "windows") => extension == Some("dll"),
        _ if cfg!(target_os = "macos") => extension == Some("dylib"),
        _ if cfg!(unix) => extension == Some("so"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_library_filter_uses_only_native_extension() {
        assert_eq!(
            is_dynamic_library(Path::new("plugins/plugin.dll")),
            cfg!(target_os = "windows")
        );
        assert_eq!(
            is_dynamic_library(Path::new("plugins/plugin.dylib")),
            cfg!(target_os = "macos")
        );
        assert_eq!(
            is_dynamic_library(Path::new("plugins/plugin.so")),
            cfg!(unix)
        );
        assert!(!is_dynamic_library(Path::new("plugins/plugin.rlib")));
        assert!(!is_dynamic_library(Path::new("plugins/plugin.dll.bak")));
    }

    #[test]
    fn entrypoint_accepts_manifest_convention_or_exported_symbol() {
        assert!(entrypoint_matches(
            "register_architectures",
            "_fellm_plugin_register_architectures"
        ));
        assert!(entrypoint_matches(
            "_fellm_plugin_register_kernels",
            "_fellm_plugin_register_kernels"
        ));
        assert!(!entrypoint_matches(
            "register_backend",
            "_fellm_plugin_register_kernels"
        ));
    }
}
