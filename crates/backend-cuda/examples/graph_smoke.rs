//! Live CUDA Graph capture/replay smoke for the plugin stream.

use backend_cuda::CudaBackend;
use fellm_core::dtype::DType;
use fellm_plugin_abi::op::{OpAttrs, OpKind};
use fellm_plugin_abi::traits::Backend;
use fellm_plugin_abi::{DeviceStepParams, TensorMut, TensorRef};

fn main() -> fellm_core::error::Result<()> {
    let backend = CudaBackend::new()?;
    let descriptor = backend
        .resolve_kernel(OpKind::Add, &[DType::F32, DType::F32], DType::F32)
        .ok_or_else(|| fellm_core::error::FellmError::other("CUDA Add kernel is unavailable"))?;
    let lhs = [1.0f32, 2.0, 3.0, 4.0];
    let rhs = [10.0f32, 20.0, 30.0, 40.0];
    let mut output = [0.0f32; 4];
    let output_ptr = output.as_mut_ptr();
    let dims = [4u64];
    let strides = [4u64];
    // SAFETY: all arrays outlive every launch and describe contiguous f32 storage.
    let inputs = unsafe {
        [
            TensorRef::from_raw(
                DType::F32,
                &dims,
                &strides,
                lhs.as_ptr().cast(),
                size_of_val(&lhs),
            ),
            TensorRef::from_raw(
                DType::F32,
                &dims,
                &strides,
                rhs.as_ptr().cast(),
                size_of_val(&rhs),
            ),
        ]
    };
    let output_view = || unsafe {
        TensorMut::from_raw(
            DType::F32,
            &dims,
            &strides,
            output_ptr.cast(),
            size_of_val(&output),
        )
    };
    let output_ref = || unsafe {
        TensorRef::from_raw(
            DType::F32,
            &dims,
            &strides,
            output_ptr.cast_const().cast(),
            size_of_val(&output),
        )
    };

    // Warmup resolves and allocates plugin-side device buffers before capture.
    backend.launch(
        descriptor.handle,
        &OpAttrs::default(),
        &inputs,
        &mut [output_view()],
        0,
    )?;
    backend.synchronize()?;

    let capture = backend.begin_graph_capture()?;
    backend.launch(
        descriptor.handle,
        &OpAttrs::default(),
        &inputs,
        &mut [output_view()],
        0,
    )?;
    let graph = capture.finish()?;
    graph.launch()?;
    backend.synchronize()?;
    backend.materialize(output_ref(), output_view())?;
    if output != [11.0, 22.0, 33.0, 44.0] {
        return Err(fellm_core::error::FellmError::other(format!(
            "CUDA Graph replay mismatch: {output:?}"
        )));
    }

    let rope = backend
        .resolve_kernel(OpKind::Rope, &[DType::F32, DType::F32], DType::F32)
        .ok_or_else(|| fellm_core::error::FellmError::other("CUDA RoPE kernel is unavailable"))?;
    let rope_x = [1.0f32, 0.0, 1.0, 0.0];
    let inv_freqs = [1.0f32, 0.1];
    let mut rope_output = [0.0f32; 4];
    let rope_output_ptr = rope_output.as_mut_ptr();
    let inv_dims = [2u64];
    let rope_inputs = unsafe {
        [
            TensorRef::from_raw(
                DType::F32,
                &dims,
                &strides,
                rope_x.as_ptr().cast(),
                size_of_val(&rope_x),
            ),
            TensorRef::from_raw(
                DType::F32,
                &inv_dims,
                &strides,
                inv_freqs.as_ptr().cast(),
                size_of_val(&inv_freqs),
            ),
        ]
    };
    let rope_output_view = || unsafe {
        TensorMut::from_raw(
            DType::F32,
            &dims,
            &strides,
            rope_output_ptr.cast(),
            size_of_val(&rope_output),
        )
    };
    let rope_output_ref = || unsafe {
        TensorRef::from_raw(
            DType::F32,
            &dims,
            &strides,
            rope_output_ptr.cast_const().cast(),
            size_of_val(&rope_output),
        )
    };
    let rope_attrs = OpAttrs {
        n_heads: 1,
        head_dim: 4,
        rope_dim: 4,
        position: 999,
        ..OpAttrs::default()
    };
    backend.update_step_params(&DeviceStepParams {
        position: 1,
        ..DeviceStepParams::default()
    })?;
    backend.launch(
        rope.handle,
        &rope_attrs,
        &rope_inputs,
        &mut [rope_output_view()],
        0,
    )?;
    backend.synchronize()?;

    let rope_capture = backend.begin_graph_capture()?;
    backend.launch(
        rope.handle,
        &rope_attrs,
        &rope_inputs,
        &mut [rope_output_view()],
        0,
    )?;
    let rope_graph = rope_capture.finish()?;
    backend.update_step_params(&DeviceStepParams {
        position: 2,
        ..DeviceStepParams::default()
    })?;
    rope_graph.launch()?;
    backend.synchronize()?;
    backend.materialize(rope_output_ref(), rope_output_view())?;
    let expected = [2.0f32.cos(), 2.0f32.sin(), 0.2f32.cos(), 0.2f32.sin()];
    if rope_output
        .iter()
        .zip(expected)
        .any(|(actual, expected)| (actual - expected).abs() > 1e-5)
    {
        return Err(fellm_core::error::FellmError::other(format!(
            "controlled CUDA Graph replay mismatch: {rope_output:?}"
        )));
    }
    println!(
        "CUDA Graph capture/replay passed; device-controlled RoPE position=2: {rope_output:?}"
    );
    Ok(())
}
