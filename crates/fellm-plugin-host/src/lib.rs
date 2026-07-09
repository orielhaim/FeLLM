//! Dynamic loader and kernel registry for FeLLM operator plugins.

#![deny(missing_docs)]

mod loader;
mod registry;

pub use loader::{LoadedPlugin, PluginHost};
pub use registry::{KernelKey, KernelRegistry, RegisteredKernel};
