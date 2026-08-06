//! In-process kernel registry filled by dynamic plugins.

use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::c_abi::{
    ArchitecturePluginRegistration, ArchitectureRegistryVtable, KernelRegistryVtable,
    PLUGIN_MAX_INPUT_DTYPES, PluginLaunchFn, PluginOpRegistration,
};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::{ArchitectureProvider, StreamHandle, TensorMut, TensorRef};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;

/// Key for a registered plugin kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    /// Op kind.
    pub op: OpKind,
    /// Input dtypes (ggml codes truncated to registration length).
    pub input_dtypes: Vec<u32>,
    /// Output dtype ggml code.
    pub output_dtype: u32,
}

/// One resolved plugin launch entry.
#[derive(Clone, Copy)]
pub struct RegisteredKernel {
    /// C launch function.
    pub launch: PluginLaunchFn,
}

/// One dynamically registered architecture provider.
pub struct RegisteredArchitecture {
    /// Stable architecture id.
    pub architecture_id: String,
    /// Exact current C ABI callback set.
    pub registration: ArchitecturePluginRegistration,
}

/// Host-side architecture provider registry, separate from kernels.
#[derive(Default)]
pub struct ArchitectureRegistry {
    entries: HashMap<String, RegisteredArchitecture>,
    providers: HashMap<String, Arc<dyn ArchitectureProvider>>,
}

impl ArchitectureRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of registered providers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len() + self.providers.len()
    }

    /// Register a statically linked provider.
    pub fn register_provider(&mut self, provider: Arc<dyn ArchitectureProvider>) -> Result<()> {
        let id = provider.architecture_id().trim();
        if id.is_empty() {
            return Err(FellmError::other("architecture provider id is empty"));
        }
        if self.entries.contains_key(id) || self.providers.contains_key(id) {
            return Err(FellmError::other(format!(
                "duplicate architecture provider {id}"
            )));
        }
        self.providers.insert(id.to_owned(), provider);
        Ok(())
    }

    /// Register an exact C ABI provider record; incomplete providers fail.
    pub fn register(&mut self, registration: &ArchitecturePluginRegistration) -> c_int {
        let id = c_name_to_string(&registration.architecture_id);
        if id.is_empty() {
            return -1;
        }
        if registration.probe.is_none()
            || registration.compile.is_none()
            || registration.create_generation_driver.is_none()
        {
            return -2;
        }
        if self.entries.contains_key(&id) || self.providers.contains_key(&id) {
            return -3;
        }
        self.entries.insert(
            id.clone(),
            RegisteredArchitecture {
                architecture_id: id,
                registration: *registration,
            },
        );
        0
    }

    /// Find a statically linked provider.
    #[must_use]
    pub fn provider(&self, id: &str) -> Option<Arc<dyn ArchitectureProvider>> {
        self.providers.get(id).cloned()
    }

    /// Find a dynamic provider record.
    #[must_use]
    pub fn dynamic(&self, id: &str) -> Option<&RegisteredArchitecture> {
        self.entries.get(id)
    }

    /// Build the registration vtable.
    #[must_use]
    pub fn vtable(&mut self) -> ArchitectureRegistryVtable {
        ArchitectureRegistryVtable {
            registry: std::ptr::from_mut::<ArchitectureRegistry>(self).cast::<c_void>(),
            register_architecture: architecture_register,
        }
    }
}

/// Host-side registry of plugin kernels.
#[derive(Default)]
pub struct KernelRegistry {
    entries: HashMap<KernelKey, RegisteredKernel>,
    /// Sequential handles for [`fellm_plugin_abi::KernelHandle`] encoding.
    next_handle: u64,
    /// handle → key for launch dispatch.
    by_handle: HashMap<u64, KernelKey>,
}

impl KernelRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of registered ops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Register from a C ABI record. Returns `0` on success.
    pub fn register(&mut self, reg: &PluginOpRegistration) -> c_int {
        let Some(op) = OpKind::from_u32(reg.op_kind) else {
            return -1;
        };
        let Some(launch) = reg.launch else {
            return -2;
        };
        let n = reg.n_input_dtypes as usize;
        if n > PLUGIN_MAX_INPUT_DTYPES {
            return -3;
        }
        let key = KernelKey {
            op,
            input_dtypes: reg.input_dtypes[..n].to_vec(),
            output_dtype: reg.output_dtype,
        };
        let handle = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1).max(1);
        self.by_handle.insert(handle, key.clone());
        self.entries.insert(key, RegisteredKernel { launch });
        0
    }

    /// Look up a kernel by op + dtypes.
    #[must_use]
    pub fn lookup(
        &self,
        op: OpKind,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<(u64, RegisteredKernel)> {
        let key = KernelKey {
            op,
            input_dtypes: input_dtypes.iter().map(|d| *d as u32).collect(),
            output_dtype: output_dtype as u32,
        };
        let kern = *self.entries.get(&key)?;
        let handle = self
            .by_handle
            .iter()
            .find(|(_, k)| *k == &key)
            .map(|(h, _)| *h)?;
        Some((handle, kern))
    }

    /// Launch by previously assigned handle.
    pub fn launch(
        &self,
        handle: u64,
        attrs: &OpAttrs,
        inputs: &[TensorRef],
        outputs: &mut [TensorMut],
        stream: StreamHandle,
    ) -> Result<()> {
        let key = self
            .by_handle
            .get(&handle)
            .ok_or_else(|| FellmError::other(format!("unknown plugin kernel handle {handle}")))?;
        let kern = self
            .entries
            .get(key)
            .ok_or_else(|| FellmError::other("plugin kernel missing"))?;
        let rc = unsafe {
            (kern.launch)(
                std::ptr::from_ref::<OpAttrs>(attrs),
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                stream,
            )
        };
        if rc != 0 {
            return Err(FellmError::other(format!(
                "plugin kernel launch failed (code {rc})"
            )));
        }
        Ok(())
    }

    /// Build a C vtable that writes into `self`.
    #[must_use]
    pub fn vtable(&mut self) -> KernelRegistryVtable {
        KernelRegistryVtable {
            registry: std::ptr::from_mut::<KernelRegistry>(self).cast::<c_void>(),
            register_op: registry_register_op,
        }
    }
}

unsafe extern "C" fn registry_register_op(
    registry: *mut c_void,
    reg: *const PluginOpRegistration,
) -> c_int {
    if registry.is_null() || reg.is_null() {
        return -100;
    }
    // SAFETY: host passes a valid KernelRegistry pointer for the duration of register.
    let reg_ref = unsafe { &*reg };
    let host = unsafe { &mut *registry.cast::<KernelRegistry>() };
    host.register(reg_ref)
}

unsafe extern "C" fn architecture_register(
    registry: *mut c_void,
    registration: *const ArchitecturePluginRegistration,
) -> c_int {
    if registry.is_null() || registration.is_null() {
        return -100;
    }
    // SAFETY: the host owns both pointers for the duration of registration.
    let host = unsafe { &mut *registry.cast::<ArchitectureRegistry>() };
    let record = unsafe { &*registration };
    host.register(record)
}

fn c_name_to_string(
    buf: &[std::ffi::c_char; fellm_plugin_abi::c_abi::ARCHITECTURE_ID_MAX],
) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fellm_plugin_abi::c_abi::ARCHITECTURE_ID_MAX;

    unsafe extern "C" fn probe(_source: *const c_void, _config: *mut c_void) -> c_int {
        0
    }
    unsafe extern "C" fn compile(
        _source: *const c_void,
        _config: *const c_void,
        _backend: *const c_void,
        _program: *mut c_void,
    ) -> c_int {
        0
    }
    unsafe extern "C" fn driver(
        _program: *const c_void,
        _request: *const c_void,
        _driver: *mut c_void,
    ) -> c_int {
        0
    }

    fn registration(id: &str) -> ArchitecturePluginRegistration {
        let mut architecture_id = [0i8; ARCHITECTURE_ID_MAX];
        for (dst, byte) in architecture_id.iter_mut().zip(id.bytes()) {
            *dst = byte as i8;
        }
        ArchitecturePluginRegistration {
            architecture_id,
            probe: Some(probe),
            compile: Some(compile),
            create_generation_driver: Some(driver),
        }
    }

    #[test]
    fn architecture_registration_requires_complete_current_contract() {
        let mut registry = ArchitectureRegistry::new();
        let mut incomplete = registration("test-arch");
        incomplete.compile = None;
        assert_eq!(registry.register(&incomplete), -2);
        assert_eq!(registry.register(&registration("test-arch")), 0);
        assert_eq!(registry.register(&registration("test-arch")), -3);
        assert!(registry.dynamic("test-arch").is_some());
    }
}
