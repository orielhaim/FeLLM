# Runtime architecture validation notes

Measured on 2026-08-10. Release commands used a separate Windows target directory; CUDA builds and runs were performed only in WSL2.

## Sampling microbenchmark

`cargo run -p fellm-runtime --release --example sampling_bench`

| Path | Allocations | Bytes | Latency |
| --- | ---: | ---: | ---: |
| allocating baseline, per token | 3.000 | 1360.0 | 49.34 us |
| reusable workspace, per token | 0.000 | 0.0 | 48.46 us |
| diffusion workspace, per 32x4096 step | 0.000 | 0.0 | 476.77 us |

The workspace path removes the full-logit mutation/copy and all steady-state sampler allocations. The focused benchmark improved by 1.8%; one short end-to-end CPU CLI run is not enough to claim a statistically stable model-throughput change.

## Runtime smoke evidence

- Attention model, CPU and strict CUDA: two requests completed in 7 scheduler iterations; the largest plan contained 2 requests / 16 tokens and used chunked prefill with `n_batch=32`, `n_ubatch=8`.
- LFM2.5 hybrid, CPU and strict CUDA: two requests completed in 14 iterations and shared physical batches of 2 rows. ShortConv state is request-owned and repacked/scattered at each batch boundary.
- Recurrent prompt rows are intentionally one token per sequence because each ShortConv token consumes the preceding recurrent state. Multiple recurrent sequences still execute together; attention-only prompts use multi-token physical chunks.
- Strict CUDA CLI model load and 4-token generation completed with the oxide plugin and no CPU fallback. CPU CLI generation also completed.

## Validation

- `cargo test --workspace --no-fail-fast`: passed.
- WSL stable CUDA host build: passed.
- WSL pinned-nightly cuda-oxide plugin build: passed.
- Strict-CUDA attention and LFM2.5 two-request batch smokes: passed.
- Live server cancellation was exercised by dropping an SSE client; the request was removed and `fellm_requests_cancelled_total` increased. The server was not used for model-generation validation.
