//! In-process kernel registry filled by dynamic plugins.

use fellm_core::dtype::DType;
use fellm_core::error::{FellmError, Result};
use fellm_plugin_abi::c_abi::{
    KernelRegistryVtable, PLUGIN_MAX_INPUT_DTYPES, PluginLaunchFn, PluginOpRegistration,
};
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::{StreamHandle, TensorMut, TensorRef};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_int;

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
                attrs as *const OpAttrs,
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
            registry: self as *mut KernelRegistry as *mut c_void,
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
    let host = unsafe { &mut *(registry as *mut KernelRegistry) };
    host.register(reg_ref)
}
