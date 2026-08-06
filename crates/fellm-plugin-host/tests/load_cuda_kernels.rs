//! Load `cuda_kernels` if present; expects oxide ops including Add/Embedding.

use fellm_core::dtype::DType;
use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_abi::op::OpKind;
use fellm_plugin_host::PluginHost;
use std::path::PathBuf;

fn candidate_paths() -> Vec<PathBuf> {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    if cfg!(windows) {
        vec![
            root.join("plugins/dist/cuda_kernels.dll"),
            root.join("plugins/cuda_kernels/target/release/cuda_kernels.dll"),
            root.join("plugins/cuda_kernels/target/debug/cuda_kernels.dll"),
        ]
    } else {
        vec![
            root.join("plugins/dist/libcuda_kernels.so"),
            root.join("plugins/cuda_kernels/target/release/libcuda_kernels.so"),
            root.join("plugins/cuda_kernels/target/debug/libcuda_kernels.so"),
        ]
    }
}

#[test]
fn load_cuda_kernels_registers_b1_ops() {
    let path = candidate_paths().into_iter().find(|p| p.exists());
    let Some(path) = path else {
        eprintln!("skip: cuda_kernels not built");
        return;
    };
    let mut host = PluginHost::new();
    let ctx = HostContext::new(0, 0, std::ptr::null_mut(), "cuda");
    host.load_path(&path, &ctx).expect("load cuda_kernels");
    let n = host.registry().len();
    // RmsNorm, SiluGate, Rope, MatMul(Q4K), Attention, KvWrite, Add, Embedding×2.
    assert!(
        n >= 8,
        "expected ≥8 oxide ops, got {n} from {}",
        path.display()
    );
    assert!(
        host.registry()
            .lookup(OpKind::RmsNorm, &[DType::F32, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::SiluGate, &[DType::F32, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::Rope, &[DType::F32, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::MatMul, &[DType::Q4K, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(
                OpKind::Attention,
                &[DType::F32, DType::F32, DType::F32],
                DType::F32
            )
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::KvWrite, &[DType::F32, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::Add, &[DType::F32, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::Embedding, &[DType::F32, DType::U32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::Embedding, &[DType::Q4K, DType::U32], DType::F32)
            .is_some()
    );
}
