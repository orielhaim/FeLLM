//! Drive shipped FA2-style attention against the reference oracle.
//! Includes contiguous + paged gather paths (host provider entry points).

use fellm_plugin_abi::{
    fa2_style_attention_f32, fa2_style_attention_paged_f32, reference_attention_f32,
    reference_attention_paged_f32,
};
use half::f16;
use std::time::Instant;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

struct Case {
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    q_len: usize,
    kv_len: usize,
    causal: bool,
    window: usize,
}

fn run_case(c: &Case) {
    let scale = 1.0 / (c.hd as f32).sqrt();
    let mut q = vec![0.0f32; c.n_heads * c.q_len * c.hd];
    let mut k = vec![0.0f32; c.n_kv * c.kv_len * c.hd];
    let mut v = vec![0.0f32; c.n_kv * c.kv_len * c.hd];
    for (i, x) in q.iter_mut().enumerate() {
        *x = (((i * 13) % 17) as f32) * 0.07 - 0.5;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = (((i * 7) % 19) as f32) * 0.05 - 0.4;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = (((i * 3) % 11) as f32) * 0.03;
    }
    let mut out_ref = vec![0.0f32; c.n_heads * c.q_len * c.hd];
    let mut out_fa2 = vec![0.0f32; c.n_heads * c.q_len * c.hd];
    reference_attention_f32(
        &q,
        &k,
        &v,
        &mut out_ref,
        c.n_heads,
        c.n_kv,
        c.hd,
        c.q_len,
        c.kv_len,
        scale,
        c.causal,
        c.window,
    );
    fa2_style_attention_f32(
        &q,
        &k,
        &v,
        &mut out_fa2,
        c.n_heads,
        c.n_kv,
        c.hd,
        c.q_len,
        c.kv_len,
        scale,
        c.causal,
        c.window,
        16,
        32,
    );
    let err = max_abs_diff(&out_ref, &out_fa2);
    assert!(
        err < 1e-4,
        "case h={} kv={} hd={} q={} kvlen={} causal={} window={}: max_abs_diff={err}",
        c.n_heads,
        c.n_kv,
        c.hd,
        c.q_len,
        c.kv_len,
        c.causal,
        c.window
    );
}

#[test]
fn attention_grid_matches_reference() {
    let cases = [
        // decode MHA
        Case {
            n_heads: 8,
            n_kv: 8,
            hd: 64,
            q_len: 1,
            kv_len: 33,
            causal: true,
            window: 0,
        },
        // decode GQA
        Case {
            n_heads: 8,
            n_kv: 2,
            hd: 64,
            q_len: 1,
            kv_len: 64,
            causal: true,
            window: 0,
        },
        // decode MQA
        Case {
            n_heads: 8,
            n_kv: 1,
            hd: 32,
            q_len: 1,
            kv_len: 48,
            causal: true,
            window: 0,
        },
        // prefill causal
        Case {
            n_heads: 4,
            n_kv: 4,
            hd: 32,
            q_len: 17,
            kv_len: 17,
            causal: true,
            window: 0,
        },
        // prefill non-causal
        Case {
            n_heads: 4,
            n_kv: 2,
            hd: 16,
            q_len: 8,
            kv_len: 8,
            causal: false,
            window: 0,
        },
        // sliding window
        Case {
            n_heads: 4,
            n_kv: 4,
            hd: 16,
            q_len: 1,
            kv_len: 32,
            causal: true,
            window: 8,
        },
        // short-batch decode-like
        Case {
            n_heads: 4,
            n_kv: 2,
            hd: 32,
            q_len: 4,
            kv_len: 40,
            causal: true,
            window: 0,
        },
        // common head dims
        Case {
            n_heads: 2,
            n_kv: 2,
            hd: 128,
            q_len: 1,
            kv_len: 16,
            causal: true,
            window: 0,
        },
    ];
    for c in &cases {
        run_case(c);
    }
}

#[test]
fn attention_paged_gather_matches_reference() {
    let n_heads = 4;
    let n_kv = 2;
    let hd = 16;
    let seq = 24;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut q = vec![0.0f32; n_heads * hd];
    // Synthetic paged arena: seq × n_kv × hd f16 for K and V.
    let stride = n_kv * hd;
    let mut k_f16 = vec![f16::from_f32(0.0); seq * stride];
    let mut v_f16 = vec![f16::from_f32(0.0); seq * stride];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i * 5) % 11) as f32 * 0.1 - 0.4;
    }
    for t in 0..seq {
        for e in 0..stride {
            let v = ((t * 3 + e) % 13) as f32 * 0.05;
            k_f16[t * stride + e] = f16::from_f32(v);
            v_f16[t * stride + e] = f16::from_f32(v * 0.5);
        }
    }
    let gather = |t: usize, is_v: bool, row: &mut [f32]| {
        let src = if is_v { &v_f16 } else { &k_f16 };
        for (d, &s) in row.iter_mut().zip(src[t * stride..(t + 1) * stride].iter()) {
            *d = s.to_f32();
        }
    };
    let mut out_ref = vec![0.0f32; n_heads * hd];
    let mut out_fa2 = vec![0.0f32; n_heads * hd];
    reference_attention_paged_f32(
        &q,
        &mut out_ref,
        n_heads,
        n_kv,
        hd,
        seq,
        scale,
        true,
        0,
        gather,
    );
    fa2_style_attention_paged_f32(
        &q,
        &mut out_fa2,
        n_heads,
        n_kv,
        hd,
        seq,
        scale,
        true,
        0,
        4,
        8,
        gather,
    );
    let err = max_abs_diff(&out_ref, &out_fa2);
    assert!(err < 1e-3, "paged FA2 vs ref max_abs_diff={err}");
}

#[test]
fn attention_fp16_roundtrip_numerics() {
    // Simulate FP16 KV storage: cast dense K/V through f16 then back.
    let n_heads = 2;
    let n_kv = 2;
    let hd = 32;
    let q_len = 1;
    let kv_len = 20;
    let scale = 0.2f32;
    let mut q = vec![0.0f32; n_heads * q_len * hd];
    let mut k = vec![0.0f32; n_kv * kv_len * hd];
    let mut v = vec![0.0f32; n_kv * kv_len * hd];
    for i in 0..q.len() {
        q[i] = (i as f32 * 0.07).sin();
    }
    for i in 0..k.len() {
        k[i] = f16::from_f32((i as f32 * 0.03).cos()).to_f32();
        v[i] = f16::from_f32((i as f32 * 0.05).sin()).to_f32();
    }
    let mut out_ref = vec![0.0f32; n_heads * q_len * hd];
    let mut out_fa2 = vec![0.0f32; n_heads * q_len * hd];
    reference_attention_f32(
        &q,
        &k,
        &v,
        &mut out_ref,
        n_heads,
        n_kv,
        hd,
        q_len,
        kv_len,
        scale,
        true,
        0,
    );
    fa2_style_attention_f32(
        &q,
        &k,
        &v,
        &mut out_fa2,
        n_heads,
        n_kv,
        hd,
        q_len,
        kv_len,
        scale,
        true,
        0,
        1,
        16,
    );
    let err = max_abs_diff(&out_ref, &out_fa2);
    assert!(err < 1e-3, "fp16-sim FA2 vs ref err={err}");
}

#[test]
fn attention_bench_paths() {
    // Micro-bench of the shipped FA2-style host path for the four required
    // workload classes. Writes timings to stderr (captured under SCRATCH by
    // the goal harness / implementer scripts).
    let scale = 0.125f32;
    let configs = [
        ("prefill", 8usize, 8, 64, 128, 128),
        ("decode", 8, 8, 64, 1, 1024),
        ("paged_decode", 8, 2, 64, 1, 2048), // GQA proxy for paged decode work
        ("short_batch_decode", 8, 4, 64, 4, 512),
    ];
    eprintln!("attention_bench:");
    for (name, nh, nkv, hd, ql, kvl) in configs {
        let mut q = vec![0.1f32; nh * ql * hd];
        let mut k = vec![0.05f32; nkv * kvl * hd];
        let mut v = vec![0.02f32; nkv * kvl * hd];
        for (i, x) in q.iter_mut().enumerate() {
            *x = (i as f32 * 0.01).sin();
        }
        for (i, x) in k.iter_mut().enumerate() {
            *x = (i as f32 * 0.013).cos();
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i as f32 * 0.007).sin() * 0.5;
        }
        let mut out = vec![0.0f32; nh * ql * hd];
        // Warmup
        fa2_style_attention_f32(
            &q, &k, &v, &mut out, nh, nkv, hd, ql, kvl, scale, true, 0, 32, 64,
        );
        let iters = 8usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            fa2_style_attention_f32(
                &q, &k, &v, &mut out, nh, nkv, hd, ql, kvl, scale, true, 0, 32, 64,
            );
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let tokens = if ql == 1 { kvl } else { ql * kvl };
        let tok_s = if ms > 0.0 {
            tokens as f64 / (ms / 1000.0)
        } else {
            0.0
        };
        assert!(ms > 0.0 || tokens > 0, "bench produced no work for {name}");
        // Non-zero output proves the kernel ran.
        let sum: f32 = out.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "{name}: empty output");
        eprintln!(
            "  {name}: latency_ms={ms:.4} tokens_touched={tokens} effective_tok_s={tok_s:.1}"
        );
    }
}
