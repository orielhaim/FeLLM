# FeLLM

[![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)](https://github.com/orielhaim/TunTun)
[![Discord](https://img.shields.io/badge/discord-join%20server-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/y5bNc3MYKz)

**A Rust-native inference engine built for what comes after the Transformer monoculture.**

FeLLM is an LLM inference engine and serving runtime written entirely in Rust. It exists because the model landscape has fractured: attention, SSMs, mixture-of-experts, block-diffusion, hybrid stacks - and existing tools force you to choose between edge performance and datacenter scale. FeLLM refuses the tradeoff.

## What it does

FeLLM runs large language models. It loads them, schedules inference, serves completions over an OpenAI-compatible API, and streams tokens back. That's it - no training, no fine-tuning, no model zoo. Just inference, done well.

The engine is designed around two promises that usually don't coexist:

**Run anywhere.** A quantized model on a Raspberry Pi, a MacBook, a Ryzen laptop with no discrete GPU. FeLLM's CPU path is not an afterthought - it ships hand-tuned SIMD kernels, first-class GGUF quantization support, and compiles to a single static binary with no Python runtime, no heavyweight dependencies, and a cold start measured in milliseconds. If llama.cpp runs it, FeLLM should too.

**Scale to anything.** Continuous batching, paged KV caching, prefix caching, chunked prefill, speculative decoding, multi-token prediction, prefill/decode disaggregation - the full arsenal of serving optimizations that tools like vLLM pioneered, implemented natively in Rust. On a single H100 or a multi-node cluster, FeLLM is built to saturate the hardware.

## How it gets there

**Backend folding.** You don't compile one bloated binary that tries to talk to every GPU vendor at runtime. You compile `fellm-cpu` or `fellm-cuda`, and you get a single, tight binary where the compiler has inlined and optimized everything for exactly that target. Nothing you didn't ask for is in the binary.

**Plugin system.** New model architectures - Mamba-3, block-diffusion models, hybrid attention-SSM stacks, whatever comes next month - ship as plugins. Drop a shared library into `plugins/` and the engine picks it up. No forking, no months-long integration projects, no waiting for upstream. The plugin runs its kernels on the same GPU context as the core with zero-copy buffer sharing, so the abstraction doesn't cost you performance where it matters.

**Flexible core.** The execution model isn't hardcoded to "predict one token, append, repeat." It's a general-purpose generator abstraction that handles autoregressive decoding, block-diffusion, multi-token prediction, speculative decoding, and reasoning-budget loops through one uniform interface. The scheduler, the memory allocator, and the graph executor don't care *how* a model produces tokens - they just care that it does.

## Current status

FeLLM is in active early development. Two backends are available today:

- **CPU** - with SIMD acceleration (AVX2, AVX-512, NEON) and full GGUF quantization support.
- **CUDA** - targeting NVIDIA GPUs from Ampere onward. (linux only currently)

*Metal, ROCm, and Vulkan backends are planned.*

### Supported architectures

- **dense attention** - the standard Transformer architecture.
- **MoE** - mixture-of-experts.
- **Diffusion** - block-diffusion models. (plugin)

### KV Fabric

Unified KV Fabric that virtualizes and manages KV state across memory tiers, with sharing, paging, residency-aware scheduling, and efficient reuse. It is designed to benefit both high-concurrency servers and individual users running long-context workloads on limited hardware.

## Building

```bash
# CPU-only edge build
cargo build --release

# CUDA build
cargo build --release --features backend-cuda
```

## License

[Apache-2.0](LICENSE)
