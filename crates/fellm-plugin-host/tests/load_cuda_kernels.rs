//! Load `cuda_kernels` if present; must register zero ops until oxide lands.

use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_host::PluginHost;
use std::path::PathBuf;

fn candidate_paths() -> Vec<PathBuf> {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    vec![
        root.join("plugins/dist/libcuda_kernels.so"),
        root.join("plugins/cuda_kernels/target/release/libcuda_kernels.so"),
        root.join("plugins/cuda_kernels/target/debug/libcuda_kernels.so"),
    ]
}

#[test]
fn load_cuda_kernels_registers_zero_ops() {
    let path = candidate_paths().into_iter().find(|p| p.exists());
    let Some(path) = path else {
        eprintln!("skip: cuda_kernels not built");
        return;
    };
    let mut host = PluginHost::new();
    let ctx = HostContext::new(0, 0, std::ptr::null_mut(), "cuda");
    host.load_path(&path, &ctx).expect("load cuda_kernels");
    assert_eq!(
        host.registry().len(),
        0,
        "cuda_kernels must not register ops until oxide kernels are ready"
    );
}
