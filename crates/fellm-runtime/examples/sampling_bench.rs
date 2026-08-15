use fellm_runtime::block_diffusion::{BlockDiffusionConfig, EntropyBoundSampler, SamplerStep};
use fellm_runtime::sampling::{SamplingOptions, SamplingWorkspace, sample, sample_with_workspace};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct CountingAllocator;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: the system allocator receives the caller-provided layout unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr/layout came from the matching System allocation above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    const VOCAB: usize = 128_256;
    const ITERATIONS: u64 = 1_000;
    let original: Vec<f32> = (0..VOCAB)
        .map(|index| (index % 997) as f32 / 997.0)
        .collect();
    let recent = [1, 7, 42, 99];

    let baseline = measure(ITERATIONS, || {
        std::hint::black_box(sample(&original, 0.8, 80, 0.95, 17, 1.05, &recent));
    });

    let mut workspace = SamplingWorkspace::default();
    let optimized = measure(ITERATIONS, || {
        std::hint::black_box(sample_with_workspace(
            &original,
            SamplingOptions {
                temperature: 0.8,
                top_k: 80,
                top_p: 0.95,
                min_p: 0.0,
                seed: 17,
                repetition_penalty: 1.05,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logit_bias: &[],
                grammar: None,
                recent_tokens: &recent,
            },
            &mut workspace,
        ));
    });

    println!("mode,allocations_per_token,bytes_per_token,nanoseconds_per_token");
    println!(
        "baseline,{:.3},{:.1},{:.1}",
        baseline.0 as f64 / ITERATIONS as f64,
        baseline.1 as f64 / ITERATIONS as f64,
        baseline.2 as f64 / ITERATIONS as f64
    );
    println!(
        "workspace,{:.3},{:.1},{:.1}",
        optimized.0 as f64 / ITERATIONS as f64,
        optimized.1 as f64 / ITERATIONS as f64,
        optimized.2 as f64 / ITERATIONS as f64
    );

    let config = BlockDiffusionConfig {
        canvas_length: 32,
        ..BlockDiffusionConfig::default()
    };
    let mut diffusion = EntropyBoundSampler::new(config, 17);
    let canvas = diffusion.initialize_canvas(4096).expect("valid canvas");
    let logits: Vec<f32> = (0..32 * 4096)
        .map(|index| (index % 991) as f32 / 991.0)
        .collect();
    let mut output = SamplerStep::default();
    let diffusion_result = measure(ITERATIONS, || {
        diffusion
            .step_into(&canvas, &logits, 4096, 0, &mut output)
            .expect("diffusion sample");
    });
    println!("mode,allocations_per_step,bytes_per_step,nanoseconds_per_step");
    println!(
        "diffusion_workspace,{:.3},{:.1},{:.1}",
        diffusion_result.0 as f64 / ITERATIONS as f64,
        diffusion_result.1 as f64 / ITERATIONS as f64,
        diffusion_result.2 as f64 / ITERATIONS as f64
    );
}

fn measure(mut iterations: u64, mut operation: impl FnMut()) -> (u64, u64, u128) {
    operation();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    let started = Instant::now();
    while iterations > 0 {
        operation();
        iterations -= 1;
    }
    let elapsed = started.elapsed().as_nanos();
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        elapsed,
    )
}
