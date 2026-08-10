//! Dynamic loader and kernel/architecture registries for `FeLLM` plugins.

#![deny(missing_docs)]

mod builtin_attention;
mod capability_registry;
mod loader;
mod manifest;
mod registry;

pub use builtin_attention::{CudaAttentionProvider, HostTiledAttentionProvider};
pub use capability_registry::{CapabilityRegistry, RegisteredProvider};
pub use loader::{DiscoveredPlugin, LoadedPlugin, PluginCatalog, PluginHost};
pub use manifest::{
    MANIFEST_SCHEMA, PluginComponent, PluginComponentKind, PluginManifest, PluginRequirement,
    parse_manifest,
};
pub use registry::{
    ArchitectureRegistry, KernelKey, KernelRegistry, RegisteredArchitecture, RegisteredKernel,
};
