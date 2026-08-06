//! Microbenchmarks for the CPU canvas kernels.
//!
//! Run with:
//! `cargo bench -p backend-cpu --bench cpu_kernels`

use backend_cpu::kernels::matmul;
use divan::black_box;
use fellm_core::dtype::DType;

fn main() {
    divan::main();
}

#[divan::bench]
fn f32_canvas_gemm(bencher: divan::Bencher) {
    const ROWS: usize = 256;
    const OUT: usize = 1024;
    const IN: usize = 1024;
    let weights = vec![0.001f32; OUT * IN];
    let input = vec![0.002f32; ROWS * IN];
    let mut output = vec![0.0f32; ROWS * OUT];
    bencher.bench_local(|| {
        matmul::matmul_f32_batch(
            black_box(&weights),
            black_box(&input),
            black_box(&mut output),
            ROWS,
            OUT,
            IN,
        )
        .unwrap();
    });
}

#[divan::bench]
fn q4k_canvas_gemm(bencher: divan::Bencher) {
    const ROWS: usize = 256;
    const OUT: usize = 1024;
    const IN: usize = 1024;
    let weights = vec![0u8; DType::Q4K.byte_size(OUT * IN)];
    let input = vec![0.002f32; ROWS * IN];
    let mut output = vec![0.0f32; ROWS * OUT];
    bencher.bench_local(|| {
        matmul::matmul_quant_batch(
            black_box(&weights),
            DType::Q4K,
            black_box(&input),
            black_box(&mut output),
            ROWS,
            OUT,
            IN,
        )
        .unwrap();
    });
}
