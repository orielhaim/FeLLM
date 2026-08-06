//! Dynamic loader and kernel/architecture registries for `FeLLM` plugins.

#![deny(missing_docs)]

mod loader;
mod registry;

pub use loader::{LoadedPlugin, PluginHost};
pub use registry::{
    ArchitectureRegistry, KernelKey, KernelRegistry, RegisteredArchitecture, RegisteredKernel,
};
