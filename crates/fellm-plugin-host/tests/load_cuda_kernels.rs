//! Load `cuda_kernels` if present; expects oxide ops including Add/Embedding.

use fellm_core::dtype::DType;
use fellm_plugin_abi::c_abi::HostContext;
use fellm_plugin_abi::op::OpKind;
#[cfg(not(windows))]
use fellm_plugin_abi::{OpAttrs, TensorMut, TensorRef};
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
    assert!(
        host.registry()
            .lookup(OpKind::MatMul, &[DType::Q8_0, DType::F32], DType::F32)
            .is_some()
    );
    assert!(
        host.registry()
            .lookup(OpKind::Embedding, &[DType::Q8_0, DType::U32], DType::F32)
            .is_some()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn q8_0_batched_matmul_matches_reference() {
    // Numerical tests explicitly request host-visible outputs; production strict
    // execution keeps intermediate tensors resident.
    unsafe { std::env::set_var("FELLM_CUDA_SYNC_OUTPUTS", "1") };
    let Some(path) = candidate_paths().into_iter().find(|p| p.exists()) else {
        return;
    };
    let mut host = PluginHost::new();
    host.load_path(&path, &HostContext::new(0, 0, std::ptr::null_mut(), "cuda"))
        .expect("load cuda kernels");
    let (handle, _) = host
        .registry()
        .lookup(OpKind::MatMul, &[DType::Q8_0, DType::F32], DType::F32)
        .expect("Q8_0 matmul registration");

    // One 32-element Q8_0 row: scale=1, values 1..=32. Two activation rows.
    let mut weights = vec![0u8; 34];
    weights[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    for (index, value) in weights[2..].iter_mut().enumerate() {
        *value = (index + 1) as u8;
    }
    let x = [vec![1.0f32; 32], vec![2.0f32; 32]].concat();
    let mut output = vec![0.0f32; 2];
    let weight_dims = [1u64, 32];
    let weight_strides = [32u64, 1];
    let x_dims = [2u64, 32];
    let x_strides = [32u64, 1];
    let out_dims = [2u64, 1];
    let out_strides = [1u64, 1];
    let inputs = unsafe {
        [
            TensorRef::from_raw(
                DType::Q8_0,
                &weight_dims,
                &weight_strides,
                weights.as_ptr(),
                weights.len(),
            ),
            TensorRef::from_raw(
                DType::F32,
                &x_dims,
                &x_strides,
                x.as_ptr().cast(),
                x.len() * 4,
            ),
        ]
    };
    let mut outputs = [unsafe {
        TensorMut::from_raw(
            DType::F32,
            &out_dims,
            &out_strides,
            output.as_mut_ptr().cast(),
            output.len() * 4,
        )
    }];
    host.registry()
        .launch(handle, &OpAttrs::default(), &inputs, &mut outputs, 0)
        .expect("Q8_0 CUDA launch");
    assert!((output[0] - 528.0).abs() < 1e-3, "{}", output[0]);
    assert!((output[1] - 1056.0).abs() < 1e-3, "{}", output[1]);
}

#[cfg(target_os = "linux")]
#[test]
fn q6k_batched_matmul_matches_reference() {
    unsafe { std::env::set_var("FELLM_CUDA_SYNC_OUTPUTS", "1") };
    let Some(path) = candidate_paths().into_iter().find(|p| p.exists()) else {
        return;
    };
    let mut host = PluginHost::new();
    host.load_path(&path, &HostContext::new(0, 0, std::ptr::null_mut(), "cuda"))
        .expect("load cuda kernels");
    let (handle, _) = host
        .registry()
        .lookup(OpKind::MatMul, &[DType::Q6K, DType::F32], DType::F32)
        .expect("Q6_K matmul registration");

    // One Q6_K block. Low/high quant bits are zero, hence every quantized
    // value is -32. All sixteen signed group scales and the f16 block scale
    // are one, so dot(ones) is exactly 256 * -32.
    let mut weights = vec![0u8; 210];
    weights[192..208].fill(1);
    weights[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
    let x = [vec![1.0f32; 256], vec![2.0f32; 256]].concat();
    let mut output = vec![0.0f32; 2];
    let weight_dims = [1u64, 256];
    let weight_strides = [256u64, 1];
    let x_dims = [2u64, 256];
    let x_strides = [256u64, 1];
    let out_dims = [2u64, 1];
    let out_strides = [1u64, 1];
    let inputs = unsafe {
        [
            TensorRef::from_raw(
                DType::Q6K,
                &weight_dims,
                &weight_strides,
                weights.as_ptr(),
                weights.len(),
            ),
            TensorRef::from_raw(
                DType::F32,
                &x_dims,
                &x_strides,
                x.as_ptr().cast(),
                x.len() * 4,
            ),
        ]
    };
    let mut outputs = [unsafe {
        TensorMut::from_raw(
            DType::F32,
            &out_dims,
            &out_strides,
            output.as_mut_ptr().cast(),
            output.len() * 4,
        )
    }];
    host.registry()
        .launch(handle, &OpAttrs::default(), &inputs, &mut outputs, 0)
        .expect("Q6_K CUDA launch");
    assert!((output[0] + 8192.0).abs() < 1e-3, "{}", output[0]);
    assert!((output[1] + 16384.0).abs() < 1e-3, "{}", output[1]);
}

#[cfg(target_os = "linux")]
#[test]
fn q4k_batched_matmul_matches_reference() {
    unsafe { std::env::set_var("FELLM_CUDA_SYNC_OUTPUTS", "1") };
    let Some(path) = candidate_paths().into_iter().find(|p| p.exists()) else {
        return;
    };
    let mut host = PluginHost::new();
    host.load_path(&path, &HostContext::new(0, 0, std::ptr::null_mut(), "cuda"))
        .expect("load cuda kernels");
    let (handle, _) = host
        .registry()
        .lookup(OpKind::MatMul, &[DType::Q4K, DType::F32], DType::F32)
        .expect("Q4_K matmul registration");

    // d=1, dmin=0, all eight packed scales=1, all minima=0, and every
    // four-bit quant=1. Each 256-element row dotted with ones is 256.
    let mut weights = vec![0u8; 144];
    weights[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    weights[4..8].fill(1);
    weights[12..16].fill(1);
    weights[16..].fill(0x11);
    let x = [vec![1.0f32; 256], vec![2.0f32; 256]].concat();
    let mut output = vec![0.0f32; 2];
    let weight_dims = [1u64, 256];
    let weight_strides = [256u64, 1];
    let x_dims = [2u64, 256];
    let x_strides = [256u64, 1];
    let out_dims = [2u64, 1];
    let out_strides = [1u64, 1];
    let inputs = unsafe {
        [
            TensorRef::from_raw(
                DType::Q4K,
                &weight_dims,
                &weight_strides,
                weights.as_ptr(),
                weights.len(),
            ),
            TensorRef::from_raw(
                DType::F32,
                &x_dims,
                &x_strides,
                x.as_ptr().cast(),
                x.len() * 4,
            ),
        ]
    };
    let mut outputs = [unsafe {
        TensorMut::from_raw(
            DType::F32,
            &out_dims,
            &out_strides,
            output.as_mut_ptr().cast(),
            output.len() * 4,
        )
    }];
    host.registry()
        .launch(handle, &OpAttrs::default(), &inputs, &mut outputs, 0)
        .expect("Q4_K CUDA launch");
    assert!((output[0] - 256.0).abs() < 1e-3, "{}", output[0]);
    assert!((output[1] - 512.0).abs() < 1e-3, "{}", output[1]);
}
