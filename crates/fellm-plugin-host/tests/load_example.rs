//! Integration: build + load `example_cpu_op` and launch identity Cast.

use fellm_core::dtype::DType;
use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::{TensorMut, TensorRef};
use fellm_plugin_host::PluginHost;
use std::path::PathBuf;

fn plugin_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates
    p.pop(); // workspace root
    let target = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| p.join("target").to_string_lossy().into_owned());
    let mut lib = PathBuf::from(target).join("debug");
    #[cfg(windows)]
    lib.push("example_cpu_op.dll");
    #[cfg(target_os = "linux")]
    lib.push("libexample_cpu_op.so");
    #[cfg(target_os = "macos")]
    lib.push("libexample_cpu_op.dylib");
    lib
}

#[test]
fn load_and_launch_identity_cast() {
    let path = plugin_path();
    if !path.exists() {
        eprintln!("skip: plugin not built at {}", path.display());
        return;
    }
    let mut host = PluginHost::new();
    let ctx = HostContext::new(0, 0, std::ptr::null_mut(), "cpu");
    host.load_path(&path, &ctx).expect("load plugin");
    assert!(!host.registry().is_empty());

    let (handle, _) = host
        .registry()
        .lookup(OpKind::Cast, &[DType::F32], DType::F32)
        .expect("cast registered");

    let src = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut dst = vec![0.0f32; 4];
    let inputs = [unsafe {
        TensorRef::from_raw(
            DType::F32,
            &[4],
            &[1],
            src.as_ptr() as *const u8,
            src.len() * 4,
        )
    }];
    let mut outputs = [unsafe {
        TensorMut::from_raw(
            DType::F32,
            &[4],
            &[1],
            dst.as_mut_ptr() as *mut u8,
            dst.len() * 4,
        )
    }];
    let attrs = OpAttrs::default();
    host.registry()
        .launch(handle, &attrs, &inputs, &mut outputs, 0)
        .expect("launch");
    assert_eq!(dst, src);
}
