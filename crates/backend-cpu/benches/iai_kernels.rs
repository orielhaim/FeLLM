//! Deterministic instruction-count benchmarks for the CPU kernels.
//!
//! Run on Linux with Valgrind/Callgrind:
//! `cargo bench -p backend-cpu --bench iai_kernels`

use backend_cpu::kernels::matmul;
use fellm_core::dtype::DType;
use iai_callgrind::library_benchmark;

const ROWS: usize = 32;
const OUT: usize = 256;
const IN: usize = 256;

#[library_benchmark]
fn f32_canvas_gemm() -> f32 {
    let weights = vec![0.001f32; OUT * IN];
    let input = vec![0.002f32; ROWS * IN];
    let mut output = vec![0.0f32; ROWS * OUT];
    matmul::matmul_f32_batch(&weights, &input, &mut output, ROWS, OUT, IN).unwrap();
    std::hint::black_box(output[0])
}

#[library_benchmark]
fn q4k_canvas_gemm() -> f32 {
    let weights = vec![0u8; DType::Q4K.byte_size(OUT * IN)];
    let input = vec![0.002f32; ROWS * IN];
    let mut output = vec![0.0f32; ROWS * OUT];
    matmul::matmul_quant_batch(&weights, DType::Q4K, &input, &mut output, ROWS, OUT, IN).unwrap();
    std::hint::black_box(output[0])
}

iai_callgrind::library_benchmark_group!(
    name = cpu_kernel_group;
    benchmarks = f32_canvas_gemm, q4k_canvas_gemm
);

iai_callgrind::main!(library_benchmark_groups = cpu_kernel_group);
