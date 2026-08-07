//! Oxide `#[cuda_module]` kernels for FeLLM B1 ops.

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

/// Q4_K super-block size (GGUF / ggml).
pub const Q4K_BLOCK_BYTES: u32 = 144;
/// Elements per Q4_K super-block.
pub const Q4K_BLOCK_ELEMS: u32 = 256;
/// Q6_K super-block size (GGUF / ggml).
pub const Q6K_BLOCK_BYTES: u32 = 210;
/// Elements per Q6_K super-block.
pub const Q6K_BLOCK_ELEMS: u32 = 256;
/// Q8_0 block size (scale f16 + 32 signed bytes).
pub const Q8_0_BLOCK_BYTES: u32 = 34;
/// Elements per Q8_0 block.
pub const Q8_0_BLOCK_ELEMS: u32 = 32;
pub const Q5_0_BLOCK_BYTES: u32 = 22;
pub const Q5_0_BLOCK_ELEMS: u32 = 32;

#[cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
    use cuda_device::dotprod::dp4a_s32;

    #[inline]
    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7f_ffff;
        if exp == 255 {
            return (sign | 0x7c00 | (if frac != 0 { 0x200 } else { 0 })) as u16;
        }
        let exp16 = exp - 127 + 15;
        if exp16 >= 31 {
            return (sign | 0x7c00) as u16;
        }
        if exp16 <= 0 {
            if exp16 < -10 {
                return sign as u16;
            }
            let frac32 = (frac | 0x800000) >> (1 - exp16);
            let half = (frac32 >> 13) + if (frac32 & 0x1000) != 0 { 1 } else { 0 };
            return (sign | half) as u16;
        }
        let half = ((exp16 as u32) << 10) | (frac >> 13);
        let round = if (frac & 0x1000) != 0 { 1u32 } else { 0 };
        (sign | (half + round)) as u16
    }

    /// `out[i] = silu(gate[i]) * up[i]`.
    #[kernel]
    pub fn silu_gate(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let g = gate[i];
            let u = up[i];
            // silu(x) = x / (1 + exp(-x))
            let s = g / (1.0 + (-g).exp());
            *o = s * u;
        }
    }

    /// One RMSNorm group: `x`/`w` length = `n`, write scaled result to `out`.
    ///
    /// Launch with `block_dim = (256,1,1)`, `grid = (1,1,1)` per group (or
    /// grid.x = number of groups with strided `x`/`out` via `group_stride`).
    #[kernel]
    pub fn rmsnorm_group(
        x: &[f32],
        w: &[f32],
        eps: f32,
        n: u32,
        group_stride: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let gid = thread::blockIdx_x();
        let base = (gid * group_stride) as usize;
        let n_usz = n as usize;

        // Partial sum of squares.
        let mut local = 0.0f32;
        let mut i = tid as usize;
        while i < n_usz {
            let v = x[base + i];
            local += v * v;
            i += 256;
        }
        unsafe {
            PARTIAL[tid as usize] = local;
        }
        thread::sync_threads();

        // Block reduce in shared memory (256 → 1).
        let mut stride = 128u32;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid as usize] += PARTIAL[(tid + stride) as usize];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let sumsq = unsafe { PARTIAL[0] };
        let scale = 1.0 / ((sumsq / n as f32) + eps).sqrt();

        let mut i = tid as usize;
        while i < n_usz {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale * w[i];
            }
            i += 256;
        }
    }

    /// RoPE: rotate pairs in the first `rope_dim` of each head.
    ///
    /// `x` / `out` layout `[n_heads * head_dim]`. `inv_freqs` length `rope_dim/2`.
    #[kernel]
    pub fn rope(
        x: &[f32],
        inv_freqs: &[f32],
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        position: f32,
        total: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= total as usize {
            return;
        }
        let d = i % head_dim as usize;
        let base = i - d;
        let row = i / (n_heads as usize * head_dim as usize);

        // Copy non-rotated dims; rotate even/odd pairs for d < rope_dim.
        if d >= rope_dim as usize {
            unsafe {
                *out.get_unchecked_mut(i) = x[i];
            }
            return;
        }
        // Only even indices perform the pair write (to avoid races).
        if d % 2 == 1 {
            return;
        }
        let pair = d / 2;
        let theta = (position + row as f32) * inv_freqs[pair];
        let (s, c) = (theta.sin(), theta.cos());
        let a = x[base + d];
        let b = x[base + d + 1];
        unsafe {
            *out.get_unchecked_mut(base + d) = a * c - b * s;
            *out.get_unchecked_mut(base + d + 1) = a * s + b * c;
        }
    }

    #[inline]
    fn dot_q4k(w: &[u8], row_off: usize, x: &[f32], x_base: usize, n_blocks: u32) -> f32 {
        let mut acc = 0.0f32;
        let mut b = 0u32;
        while b < n_blocks {
            let blk = row_off + b as usize * Q4K_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([w[blk], w[blk + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([w[blk + 2], w[blk + 3]]));
            let (scales, mins) = decode_scales_mins(
                w[blk + 4],
                w[blk + 5],
                w[blk + 6],
                w[blk + 7],
                w[blk + 8],
                w[blk + 9],
                w[blk + 10],
                w[blk + 11],
                w[blk + 12],
                w[blk + 13],
                w[blk + 14],
                w[blk + 15],
            );
            let xb = x_base + b as usize * Q4K_BLOCK_ELEMS as usize;
            let mut group = 0usize;
            while group < 8 {
                let mut sum = 0.0f32;
                let mut lane = 0usize;
                while lane < 32 {
                    sum += x[xb + group * 32 + lane];
                    lane += 1;
                }
                acc -= dmin * mins[group] as f32 * sum;
                group += 1;
            }
            let qs = blk + 16;
            let mut chunk = 0usize;
            while chunk < 4 {
                let qbase = qs + chunk * 32;
                let mut lo = 0.0f32;
                let mut hi = 0.0f32;
                let mut lane = 0usize;
                while lane < 32 {
                    let q = w[qbase + lane];
                    lo += (q & 15) as f32 * x[xb + chunk * 64 + lane];
                    hi += (q >> 4) as f32 * x[xb + chunk * 64 + 32 + lane];
                    lane += 1;
                }
                acc += d * scales[chunk * 2] as f32 * lo;
                acc += d * scales[chunk * 2 + 1] as f32 * hi;
                chunk += 1;
            }
            b += 1;
        }
        acc
    }

    #[inline]
    fn dot_q6k(w: &[u8], row_off: usize, x: &[f32], x_base: usize, n_blocks: u32) -> f32 {
        let mut acc = 0.0f32;
        let mut b = 0u32;
        while b < n_blocks {
            let blk = row_off + b as usize * Q6K_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([w[blk + 208], w[blk + 209]]));
            let xb = x_base + b as usize * Q6K_BLOCK_ELEMS as usize;
            let mut half = 0usize;
            while half < 2 {
                let ql = blk + half * 64;
                let qh = blk + 128 + half * 32;
                let sc = blk + 192 + half * 8;
                let y0 = xb + half * 128;
                let mut lane = 0usize;
                while lane < 32 {
                    let si = lane / 16;
                    let q1 = ((w[ql + lane] & 15) as i32 | (((w[qh + lane]) & 3) as i32) << 4) - 32;
                    let q2 = ((w[ql + lane + 32] & 15) as i32
                        | (((w[qh + lane] >> 2) & 3) as i32) << 4)
                        - 32;
                    let q3 =
                        ((w[ql + lane] >> 4) as i32 | (((w[qh + lane] >> 4) & 3) as i32) << 4) - 32;
                    let q4 = ((w[ql + lane + 32] >> 4) as i32
                        | (((w[qh + lane] >> 6) & 3) as i32) << 4)
                        - 32;
                    acc += d * w[sc + si] as i8 as f32 * q1 as f32 * x[y0 + lane];
                    acc += d * w[sc + si + 2] as i8 as f32 * q2 as f32 * x[y0 + lane + 32];
                    acc += d * w[sc + si + 4] as i8 as f32 * q3 as f32 * x[y0 + lane + 64];
                    acc += d * w[sc + si + 6] as i8 as f32 * q4 as f32 * x[y0 + lane + 96];
                    lane += 1;
                }
                half += 1;
            }
            b += 1;
        }
        acc
    }

    #[inline]
    fn dot_q6k_lane(
        w: &[u8],
        row_off: usize,
        x: &[f32],
        x_base: usize,
        n_blocks: u32,
        lane: usize,
    ) -> f32 {
        let mut acc = 0.0f32;
        let mut block = 0usize;
        while block < n_blocks as usize {
            let blk = row_off + block * Q6K_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([w[blk + 208], w[blk + 209]]));
            let xb = x_base + block * Q6K_BLOCK_ELEMS as usize;
            let mut half = 0usize;
            while half < 2 {
                let ql = blk + half * 64;
                let qh = blk + 128 + half * 32;
                let sc = blk + 192 + half * 8;
                let y0 = xb + half * 128;
                let is = lane / 16;
                let q1 = ((w[ql + lane] & 0xF) as i32
                    | (((w[qh + lane] >> 0) & 3) as i32) << 4)
                    - 32;
                let q2 = ((w[ql + lane + 32] & 0xF) as i32
                    | (((w[qh + lane] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((w[ql + lane] >> 4) as i32
                    | (((w[qh + lane] >> 4) & 3) as i32) << 4)
                    - 32;
                let q4 = ((w[ql + lane + 32] >> 4) as i32
                    | (((w[qh + lane] >> 6) & 3) as i32) << 4)
                    - 32;
                acc += d * w[sc + is] as i8 as f32 * q1 as f32 * x[y0 + lane];
                acc += d * w[sc + is + 2] as i8 as f32 * q2 as f32 * x[y0 + lane + 32];
                acc += d * w[sc + is + 4] as i8 as f32 * q3 as f32 * x[y0 + lane + 64];
                acc += d * w[sc + is + 6] as i8 as f32 * q4 as f32 * x[y0 + lane + 96];
                half += 1;
            }
            block += 1;
        }
        acc
    }

    /// One output row of Q4_K × f32 GEMV.
    ///
    /// `w` is the full weight matrix bytes (`out_dim` rows × `n_blocks` × 144).
    /// `x` is the activation vector (`n_blocks * 256` f32).
    /// Launch: one thread per output row (`grid` covering `out_dim`).
    #[kernel]
    pub fn q4k_gemv_row(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= out_dim as usize * batch_rows as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = (n_blocks * Q4K_BLOCK_BYTES) as usize;
        let row_off = row * row_bytes;
        let mut acc = 0.0f32;

        let mut b = 0u32;
        while b < n_blocks {
            let blk = row_off + (b as usize) * Q4K_BLOCK_BYTES as usize;
            // d, dmin as f16 at bytes 0..4
            let d_bits = u16::from_le_bytes([w[blk], w[blk + 1]]);
            let dm_bits = u16::from_le_bytes([w[blk + 2], w[blk + 3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dm_bits);

            // Decode scales/mins from 12 bytes at blk+4 (ggml utmp).
            let (scales, mins) = decode_scales_mins(
                w[blk + 4],
                w[blk + 5],
                w[blk + 6],
                w[blk + 7],
                w[blk + 8],
                w[blk + 9],
                w[blk + 10],
                w[blk + 11],
                w[blk + 12],
                w[blk + 13],
                w[blk + 14],
                w[blk + 15],
            );

            let qs = blk + 16;
            let x_off = batch * n_blocks as usize * Q4K_BLOCK_ELEMS as usize
                + (b as usize) * Q4K_BLOCK_ELEMS as usize;

            // Min term: each min covers 32 weights (2×16).
            let mut sum_min = 0.0f32;
            let mut j = 0usize;
            while j < 8 {
                let mut s = 0.0f32;
                let mut t = 0usize;
                while t < 32 {
                    s += x[x_off + j * 32 + t];
                    t += 1;
                }
                sum_min += mins[j] as f32 * s;
                j += 1;
            }
            acc -= dmin * sum_min;

            // Weight × activation with per-32-group scales.
            // qs layout: 4 chunks of 32 bytes; each byte → lo nibble then (after 32) hi.
            let mut is = 0usize;
            let mut off = 0usize;
            let mut chunk = 0usize;
            while chunk < 4 {
                let qbase = qs + chunk * 32;
                // low nibbles → 32 weights
                let scale_lo = scales[is] as f32;
                is += 1;
                let mut dot = 0.0f32;
                let mut l = 0usize;
                while l < 32 {
                    let q = (w[qbase + l] & 0x0F) as f32;
                    dot += q * x[x_off + off + l];
                    l += 1;
                }
                acc += d * scale_lo * dot;
                off += 32;

                // high nibbles → next 32 weights
                let scale_hi = scales[is] as f32;
                is += 1;
                let mut dot = 0.0f32;
                let mut l = 0usize;
                while l < 32 {
                    let q = (w[qbase + l] >> 4) as f32;
                    dot += q * x[x_off + off + l];
                    l += 1;
                }
                acc += d * scale_hi * dot;
                off += 32;
                chunk += 1;
            }
            b += 1;
        }

        if let Some(o) = out.get_mut(idx) {
            *o = acc;
        }
    }

    /// Q6_K × f32 GEMV: one thread per output row.
    ///
    /// Weight layout matches ggml `block_q6_K` (210 bytes / 256 elems).
    #[kernel]
    pub fn q6k_gemv_row(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= out_dim as usize * batch_rows as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = (n_blocks * Q6K_BLOCK_BYTES) as usize;
        let row_off = row * row_bytes;
        let mut acc = 0.0f32;

        let mut b = 0u32;
        while b < n_blocks {
            let blk = row_off + (b as usize) * Q6K_BLOCK_BYTES as usize;
            let d_bits = u16::from_le_bytes([w[blk + 208], w[blk + 209]]);
            let d = f16_to_f32(d_bits);
            let x_off = batch * n_blocks as usize * Q6K_BLOCK_ELEMS as usize
                + (b as usize) * Q6K_BLOCK_ELEMS as usize;

            let mut half = 0usize;
            while half < 2 {
                let ql = blk + half * 64;
                let qh = blk + 128 + half * 32;
                let sc = blk + 192 + half * 8;
                let y0 = x_off + half * 128;

                let mut l = 0usize;
                while l < 32 {
                    let is = l / 16;
                    let q1 = ((w[ql + l] & 0xF) as i32 | (((w[qh + l] >> 0) & 3) as i32) << 4) - 32;
                    let q2 =
                        ((w[ql + l + 32] & 0xF) as i32 | (((w[qh + l] >> 2) & 3) as i32) << 4) - 32;
                    let q3 = ((w[ql + l] >> 4) as i32 | (((w[qh + l] >> 4) & 3) as i32) << 4) - 32;
                    let q4 =
                        ((w[ql + l + 32] >> 4) as i32 | (((w[qh + l] >> 6) & 3) as i32) << 4) - 32;
                    let s0 = w[sc + is] as i8 as f32;
                    let s1 = w[sc + is + 2] as i8 as f32;
                    let s2 = w[sc + is + 4] as i8 as f32;
                    let s3 = w[sc + is + 6] as i8 as f32;
                    acc += d * s0 * (q1 as f32) * x[y0 + l];
                    acc += d * s1 * (q2 as f32) * x[y0 + l + 32];
                    acc += d * s2 * (q3 as f32) * x[y0 + l + 64];
                    acc += d * s3 * (q4 as f32) * x[y0 + l + 96];
                    l += 1;
                }
                half += 1;
            }
            b += 1;
        }

        if let Some(o) = out.get_mut(idx) {
            *o = acc;
        }
    }

    /// Q6_K x f32 GEMM/GEMV with one cooperative warp per output row.
    #[kernel]
    pub fn q6k_gemm_warp(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= out_dim as usize * batch_rows as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = n_blocks as usize * Q6K_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = 0usize;
        while block < n_blocks as usize {
            let blk = row * row_bytes + block * Q6K_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([w[blk + 208], w[blk + 209]]));
            let xb = (batch * n_blocks as usize + block) * Q6K_BLOCK_ELEMS as usize;
            let mut half = 0usize;
            while half < 2 {
                let ql = blk + half * 64;
                let qh = blk + 128 + half * 32;
                let sc = blk + 192 + half * 8;
                let y0 = xb + half * 128;
                let is = tid / 16;
                let q1 =
                    ((w[ql + tid] & 0xF) as i32 | (((w[qh + tid] >> 0) & 3) as i32) << 4)
                        - 32;
                let q2 = ((w[ql + tid + 32] & 0xF) as i32
                    | (((w[qh + tid] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((w[ql + tid] >> 4) as i32
                    | (((w[qh + tid] >> 4) & 3) as i32) << 4)
                    - 32;
                let q4 = ((w[ql + tid + 32] >> 4) as i32
                    | (((w[qh + tid] >> 6) & 3) as i32) << 4)
                    - 32;
                let s0 = w[sc + is] as i8 as f32;
                let s1 = w[sc + is + 2] as i8 as f32;
                let s2 = w[sc + is + 4] as i8 as f32;
                let s3 = w[sc + is + 6] as i8 as f32;
                acc += d * s0 * q1 as f32 * x[y0 + tid];
                acc += d * s1 * q2 as f32 * x[y0 + tid + 32];
                acc += d * s2 * q3 as f32 * x[y0 + tid + 64];
                acc += d * s3 * q4 as f32 * x[y0 + tid + 96];
                half += 1;
            }
            block += 1;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(linear) = PARTIAL[0];
            }
        }
    }

    /// Q6_K decode kernel computing four adjacent output rows per warp. This
    /// amortizes scheduling and reuses the activation vector across four rows.
    #[kernel]
    pub fn q6k_gemv_warp4(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let groups = (out_dim as usize + 3) / 4;
        let linear_group = thread::blockIdx_x() as usize;
        if linear_group >= groups * batch_rows as usize {
            return;
        }
        let batch = linear_group / groups;
        let row0 = (linear_group % groups) * 4;
        let row_bytes = n_blocks as usize * Q6K_BLOCK_BYTES as usize;
        let x_base = batch * n_blocks as usize * Q6K_BLOCK_ELEMS as usize;
        let a0 = if row0 < out_dim as usize {
            dot_q6k_lane(w, row0 * row_bytes, x, x_base, n_blocks, lane)
        } else { 0.0 };
        let a1 = if row0 + 1 < out_dim as usize {
            dot_q6k_lane(w, (row0 + 1) * row_bytes, x, x_base, n_blocks, lane)
        } else { 0.0 };
        let a2 = if row0 + 2 < out_dim as usize {
            dot_q6k_lane(w, (row0 + 2) * row_bytes, x, x_base, n_blocks, lane)
        } else { 0.0 };
        let a3 = if row0 + 3 < out_dim as usize {
            dot_q6k_lane(w, (row0 + 3) * row_bytes, x, x_base, n_blocks, lane)
        } else { 0.0 };
        unsafe {
            PARTIAL[lane] = a0;
            PARTIAL[32 + lane] = a1;
            PARTIAL[64 + lane] = a2;
            PARTIAL[96 + lane] = a3;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL[lane] += PARTIAL[lane + stride];
                    PARTIAL[32 + lane] += PARTIAL[32 + lane + stride];
                    PARTIAL[64 + lane] += PARTIAL[64 + lane + stride];
                    PARTIAL[96 + lane] += PARTIAL[96 + lane + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            let base = batch * out_dim as usize + row0;
            unsafe {
                if row0 < out_dim as usize { *out.get_unchecked_mut(base) = PARTIAL[0]; }
                if row0 + 1 < out_dim as usize { *out.get_unchecked_mut(base + 1) = PARTIAL[32]; }
                if row0 + 2 < out_dim as usize { *out.get_unchecked_mut(base + 2) = PARTIAL[64]; }
                if row0 + 3 < out_dim as usize { *out.get_unchecked_mut(base + 3) = PARTIAL[96]; }
            }
        }
    }

    /// Quantize f32 activation chunks to symmetric Q8 plus one scale per 32
    /// values, matching the integer-dot decode path used by optimized K-quants.
    #[kernel]
    pub fn quantize_q8_32(
        input: &[f32],
        mut quants: DisjointSlice<i8>,
        mut scales: DisjointSlice<f32>,
    ) {
        static mut MAXABS: SharedArray<f32, 32> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let chunk = thread::blockIdx_x() as usize;
        let index = chunk * 32 + lane;
        if index >= input.len() {
            return;
        }
        unsafe { MAXABS[lane] = input[index].abs(); }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if lane < stride {
                unsafe { MAXABS[lane] = MAXABS[lane].max(MAXABS[lane + stride]); }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let scale = unsafe { MAXABS[0] } / 127.0;
        let quant = if scale > 0.0 {
            (input[index] / scale).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
        unsafe { *quants.get_unchecked_mut(index) = quant; }
        if lane == 0 {
            unsafe { *scales.get_unchecked_mut(chunk) = scale; }
        }
    }

    #[inline]
    fn q6k_quant(w: &[u8], blk: usize, j: usize) -> i8 {
        let half = j / 128;
        let local = j % 128;
        let lane = local % 32;
        let quad = local / 32;
        let ql = blk + half * 64;
        let qh = blk + 128 + half * 32;
        let shift = (quad * 2) as u32;
        let ql_idx = lane + if quad == 1 || quad == 3 { 32 } else { 0 };
        let packed = unsafe { *w.get_unchecked(ql + ql_idx) };
        let low = if quad < 2 { packed & 15 } else { packed >> 4 };
        (low as i32
            | ((((unsafe { *w.get_unchecked(qh + lane) } >> shift) & 3) as i32) << 4)
            - 32) as i8
    }

    #[inline]
    fn q6k_scale(w: &[u8], blk: usize, j: usize) -> i8 {
        let half = j / 128;
        let local = j % 128;
        let lane = local % 32;
        let quad = local / 32;
        unsafe { *w.get_unchecked(blk + 192 + half * 8 + lane / 16 + quad * 2) as i8 }
    }

    #[inline]
    fn q6k_q8_chunk(w: &[u8], blk: usize, j: usize, xp: u32, xs: f32) -> f32 {
        let qp = u32::from_le_bytes([
            q6k_quant(w, blk, j) as u8,
            q6k_quant(w, blk, j + 1) as u8,
            q6k_quant(w, blk, j + 2) as u8,
            q6k_quant(w, blk, j + 3) as u8,
        ]);
        let d = f16_to_f32(u16::from_le_bytes(unsafe {
            [*w.get_unchecked(blk + 208), *w.get_unchecked(blk + 209)]
        }));
        d * q6k_scale(w, blk, j) as f32 * xs * dp4a_s32(qp, xp, 0) as f32
    }

    /// Q6_K matvec using Q8 activation blocks and Ada's packed signed-byte
    /// dot product. One warp computes four adjacent output rows.
    #[kernel]
    pub fn q6k_q8_gemv_warp4(
        w: &[u8],
        qx: &[i8],
        x_scales: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let groups = (out_dim as usize + 3) / 4;
        let group = thread::blockIdx_x() as usize;
        if group >= groups * batch_rows as usize { return; }
        let batch = group / groups;
        let row0 = (group % groups) * 4;
        let row_bytes = n_blocks as usize * Q6K_BLOCK_BYTES as usize;
        let x_base = batch * n_blocks as usize * Q6K_BLOCK_ELEMS as usize;
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let mut block = 0usize;
        while block < n_blocks as usize {
            let j0 = lane * 8;
            let mut pair = 0usize;
            while pair < 2 {
                let j = j0 + pair * 4;
                let xi = x_base + block * 256 + j;
                let xp = u32::from_le_bytes(unsafe {
                    [*qx.get_unchecked(xi) as u8, *qx.get_unchecked(xi + 1) as u8,
                     *qx.get_unchecked(xi + 2) as u8, *qx.get_unchecked(xi + 3) as u8]
                });
                let xs = unsafe { *x_scales.get_unchecked(xi / 32) };
                if row0 < out_dim as usize {
                    sum0 += q6k_q8_chunk(w, row0 * row_bytes + block * Q6K_BLOCK_BYTES as usize, j, xp, xs);
                }
                if row0 + 1 < out_dim as usize {
                    sum1 += q6k_q8_chunk(w, (row0 + 1) * row_bytes + block * Q6K_BLOCK_BYTES as usize, j, xp, xs);
                }
                if row0 + 2 < out_dim as usize {
                    sum2 += q6k_q8_chunk(w, (row0 + 2) * row_bytes + block * Q6K_BLOCK_BYTES as usize, j, xp, xs);
                }
                if row0 + 3 < out_dim as usize {
                    sum3 += q6k_q8_chunk(w, (row0 + 3) * row_bytes + block * Q6K_BLOCK_BYTES as usize, j, xp, xs);
                }
                pair += 1;
            }
            block += 1;
        }
        unsafe {
            PARTIAL[lane] = sum0; PARTIAL[32 + lane] = sum1;
            PARTIAL[64 + lane] = sum2; PARTIAL[96 + lane] = sum3;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL[lane] += PARTIAL[lane + stride];
                    PARTIAL[32 + lane] += PARTIAL[32 + lane + stride];
                    PARTIAL[64 + lane] += PARTIAL[64 + lane + stride];
                    PARTIAL[96 + lane] += PARTIAL[96 + lane + stride];
                }
            }
            thread::sync_threads(); stride /= 2;
        }
        if lane == 0 {
            let base = batch * out_dim as usize + row0;
            unsafe {
                if row0 < out_dim as usize { *out.get_unchecked_mut(base) = PARTIAL[0]; }
                if row0 + 1 < out_dim as usize { *out.get_unchecked_mut(base + 1) = PARTIAL[32]; }
                if row0 + 2 < out_dim as usize { *out.get_unchecked_mut(base + 2) = PARTIAL[64]; }
                if row0 + 3 < out_dim as usize { *out.get_unchecked_mut(base + 3) = PARTIAL[96]; }
            }
        }
    }

    #[inline]
    fn q4k_quant(w: &[u8], blk: usize, j: usize) -> i8 {
        let lane = j % 32;
        let chunk = j / 64;
        let packed = unsafe { *w.get_unchecked(blk + 16 + chunk * 32 + lane) };
        let q = if j % 64 < 32 { packed & 15 } else { packed >> 4 };
        q as i8 - 8
    }

    #[inline]
    fn q4k_scale(w: &[u8], blk: usize, group: usize) -> u8 {
        if group < 4 {
            unsafe { *w.get_unchecked(blk + 4 + group) & 63 }
        } else {
            let i = group - 4;
            unsafe {
                (*w.get_unchecked(blk + 12 + i) & 15)
                    | ((*w.get_unchecked(blk + 4 + i) >> 6) << 4)
            }
        }
    }

    #[inline]
    fn q4k_min(w: &[u8], blk: usize, group: usize) -> u8 {
        if group < 4 {
            unsafe { *w.get_unchecked(blk + 8 + group) & 63 }
        } else {
            let i = group - 4;
            unsafe {
                (*w.get_unchecked(blk + 12 + i) >> 4)
                    | ((*w.get_unchecked(blk + 8 + i) >> 6) << 4)
            }
        }
    }

    #[inline]
    fn q4k_q8_chunk(w: &[u8], blk: usize, j: usize, xp: u32, sx: i32, xs: f32) -> f32 {
        let qp = u32::from_le_bytes([
            q4k_quant(w, blk, j) as u8,
            q4k_quant(w, blk, j + 1) as u8,
            q4k_quant(w, blk, j + 2) as u8,
            q4k_quant(w, blk, j + 3) as u8,
        ]);
        let dot = dp4a_s32(qp, xp, 0) + 8 * sx;
        let d = f16_to_f32(u16::from_le_bytes(unsafe {
            [*w.get_unchecked(blk), *w.get_unchecked(blk + 1)]
        }));
        let dm = f16_to_f32(u16::from_le_bytes(unsafe {
            [*w.get_unchecked(blk + 2), *w.get_unchecked(blk + 3)]
        }));
        let group = j / 32;
        xs * (d * q4k_scale(w, blk, group) as f32 * dot as f32
            - dm * q4k_min(w, blk, group) as f32 * sx as f32)
    }

    /// Q4_K matvec using Q8 activation chunks and packed signed-byte dots.
    /// The affine K-quant minimum term is retained exactly in the Q8 domain.
    #[kernel]
    pub fn q4k_q8_gemv_warp4(
        w: &[u8],
        qx: &[i8],
        x_scales: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let groups = (out_dim as usize + 3) / 4;
        let group = thread::blockIdx_x() as usize;
        if group >= groups * batch_rows as usize { return; }
        let batch = group / groups;
        let row0 = (group % groups) * 4;
        let row_bytes = n_blocks as usize * Q4K_BLOCK_BYTES as usize;
        let x_base = batch * n_blocks as usize * Q4K_BLOCK_ELEMS as usize;
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let mut block = 0usize;
        while block < n_blocks as usize {
            let mut half = 0usize;
            while half < 2 {
                let j = half * 128 + lane * 4;
                let xi = x_base + block * 256 + j;
                let x0 = unsafe { *qx.get_unchecked(xi) };
                let x1 = unsafe { *qx.get_unchecked(xi + 1) };
                let x2 = unsafe { *qx.get_unchecked(xi + 2) };
                let x3 = unsafe { *qx.get_unchecked(xi + 3) };
                let xp = u32::from_le_bytes([x0 as u8, x1 as u8, x2 as u8, x3 as u8]);
                let sx = x0 as i32 + x1 as i32 + x2 as i32 + x3 as i32;
                let xs = unsafe { *x_scales.get_unchecked(xi / 32) };
                if row0 < out_dim as usize {
                    sum0 += q4k_q8_chunk(w, row0 * row_bytes + block * Q4K_BLOCK_BYTES as usize, j, xp, sx, xs);
                }
                if row0 + 1 < out_dim as usize {
                    sum1 += q4k_q8_chunk(w, (row0 + 1) * row_bytes + block * Q4K_BLOCK_BYTES as usize, j, xp, sx, xs);
                }
                if row0 + 2 < out_dim as usize {
                    sum2 += q4k_q8_chunk(w, (row0 + 2) * row_bytes + block * Q4K_BLOCK_BYTES as usize, j, xp, sx, xs);
                }
                if row0 + 3 < out_dim as usize {
                    sum3 += q4k_q8_chunk(w, (row0 + 3) * row_bytes + block * Q4K_BLOCK_BYTES as usize, j, xp, sx, xs);
                }
                half += 1;
            }
            block += 1;
        }
        unsafe {
            PARTIAL[lane] = sum0;
            PARTIAL[32 + lane] = sum1;
            PARTIAL[64 + lane] = sum2;
            PARTIAL[96 + lane] = sum3;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL[lane] += PARTIAL[lane + stride];
                    PARTIAL[32 + lane] += PARTIAL[32 + lane + stride];
                    PARTIAL[64 + lane] += PARTIAL[64 + lane + stride];
                    PARTIAL[96 + lane] += PARTIAL[96 + lane + stride];
                }
            }
            thread::sync_threads(); stride /= 2;
        }
        if lane == 0 {
            let base = batch * out_dim as usize + row0;
            unsafe {
                if row0 < out_dim as usize { *out.get_unchecked_mut(base) = PARTIAL[0]; }
                if row0 + 1 < out_dim as usize { *out.get_unchecked_mut(base + 1) = PARTIAL[32]; }
                if row0 + 2 < out_dim as usize { *out.get_unchecked_mut(base + 2) = PARTIAL[64]; }
                if row0 + 3 < out_dim as usize { *out.get_unchecked_mut(base + 3) = PARTIAL[96]; }
            }
        }
    }

    /// Direct Q8_0 x f32 GEMM; one thread per `(batch,row)` output.
    #[kernel]
    pub fn q8_0_gemm_element(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= out_dim as usize * batch_rows as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = n_blocks as usize * Q8_0_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = 0usize;
        while block < n_blocks as usize {
            let woff = row * row_bytes + block * Q8_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([w[woff], w[woff + 1]]));
            let xoff = (batch * n_blocks as usize + block) * Q8_0_BLOCK_ELEMS as usize;
            let mut lane = 0usize;
            while lane < Q8_0_BLOCK_ELEMS as usize {
                acc += d * (w[woff + 2 + lane] as i8 as f32) * x[xoff + lane];
                lane += 1;
            }
            block += 1;
        }
        if let Some(value) = out.get_mut(idx) {
            *value = acc;
        }
    }

    #[kernel]
    pub fn q5_0_gemm_element(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= out_dim as usize * batch_rows as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = n_blocks as usize * Q5_0_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = 0usize;
        while block < n_blocks as usize {
            let wb = row * row_bytes + block * Q5_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([w[wb], w[wb + 1]]));
            let qh = u32::from_le_bytes([w[wb + 2], w[wb + 3], w[wb + 4], w[wb + 5]]);
            let xb = (batch * n_blocks as usize + block) * Q5_0_BLOCK_ELEMS as usize;
            let mut lane = 0usize;
            while lane < 16 {
                let packed = w[wb + 6 + lane];
                let lo = ((((qh >> lane) & 1) as i32) << 4 | (packed & 15) as i32) - 16;
                let hi = ((((qh >> (lane + 16)) & 1) as i32) << 4 | (packed >> 4) as i32) - 16;
                acc += d * lo as f32 * x[xb + lane];
                acc += d * hi as f32 * x[xb + lane + 16];
                lane += 1;
            }
            block += 1;
        }
        if let Some(dst) = out.get_mut(idx) {
            *dst = acc;
        }
    }

    #[kernel]
    pub fn q5_0_gemm_warp(
        weights: &[u8],
        input: &[f32],
        out_dim: u32,
        n_blocks: u32,
        rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= rows as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = n_blocks as usize * Q5_0_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = tid;
        while block < n_blocks as usize {
            let wb = row * row_bytes + block * Q5_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([weights[wb], weights[wb + 1]]));
            let qh = u32::from_le_bytes([
                weights[wb + 2],
                weights[wb + 3],
                weights[wb + 4],
                weights[wb + 5],
            ]);
            let xb = (batch * n_blocks as usize + block) * 32;
            let mut lane = 0usize;
            while lane < 16 {
                let p = weights[wb + 6 + lane];
                let lo = ((((qh >> lane) & 1) as i32) << 4 | (p & 15) as i32) - 16;
                let hi = ((((qh >> (lane + 16)) & 1) as i32) << 4 | (p >> 4) as i32) - 16;
                acc += d * lo as f32 * input[xb + lane] + d * hi as f32 * input[xb + lane + 16];
                lane += 1;
            }
            block += 32;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(linear) = PARTIAL[0];
            }
        }
    }

    #[kernel]
    pub fn q8_0_gemm_warp(
        weights: &[u8],
        input: &[f32],
        out_dim: u32,
        n_blocks: u32,
        rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= rows as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = n_blocks as usize * Q8_0_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = tid;
        while block < n_blocks as usize {
            let wb = row * row_bytes + block * Q8_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([weights[wb], weights[wb + 1]]));
            let xb = (batch * n_blocks as usize + block) * 32;
            let mut lane = 0usize;
            while lane < 32 {
                acc += d * weights[wb + 2 + lane] as i8 as f32 * input[xb + lane];
                lane += 1;
            }
            block += 32;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(linear) = PARTIAL[0];
            }
        }
    }

    #[kernel]
    pub fn add_in_place_f32(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(dst) = output.get_mut(idx) {
            *dst += input[i];
        }
    }

    /// Router projection, activation, and top-k selection. One device thread
    /// owns a token so expert IDs never cross the host boundary.
    #[kernel]
    pub fn moe_route_topk(
        x: &[f32],
        router: &[f32],
        bias: &[f32],
        tokens: u32,
        n_embd: u32,
        n_experts: u32,
        top_k: u32,
        gating_func: u32,
        normalize: u32,
        routed_scale: f32,
        mut ids: DisjointSlice<u32>,
        mut scores: DisjointSlice<f32>,
    ) {
        let token = thread::index_1d().get();
        if token >= tokens as usize {
            return;
        }
        let xb = token * n_embd as usize;
        let mut denominator = 0.0f32;
        let mut max_logit = -3.402823466e38f32;
        let mut expert = 0usize;
        while expert < n_experts as usize {
            let mut logit = if bias.len() >= n_experts as usize {
                bias[expert]
            } else {
                0.0
            };
            let mut d = 0usize;
            while d < n_embd as usize {
                logit += router[expert * n_embd as usize + d] * x[xb + d];
                d += 1;
            }
            if logit > max_logit {
                max_logit = logit;
            }
            expert += 1;
        }
        if gating_func != 2 {
            expert = 0;
            while expert < n_experts as usize {
                let mut logit = if bias.len() >= n_experts as usize {
                    bias[expert]
                } else {
                    0.0
                };
                let mut d = 0usize;
                while d < n_embd as usize {
                    logit += router[expert * n_embd as usize + d] * x[xb + d];
                    d += 1;
                }
                denominator += (logit - max_logit).exp();
                expert += 1;
            }
        }
        let mut selected_sum = 0.0f32;
        let mut slot = 0usize;
        while slot < top_k as usize {
            let mut best_id = 0usize;
            let mut best = -3.402823466e38f32;
            expert = 0;
            while expert < n_experts as usize {
                let mut used = false;
                let mut prior = 0usize;
                while prior < slot {
                    if unsafe { *ids.get_unchecked_mut(token * top_k as usize + prior) } as usize
                        == expert
                    {
                        used = true;
                    }
                    prior += 1;
                }
                if !used {
                    let mut logit = if bias.len() >= n_experts as usize {
                        bias[expert]
                    } else {
                        0.0
                    };
                    let mut d = 0usize;
                    while d < n_embd as usize {
                        logit += router[expert * n_embd as usize + d] * x[xb + d];
                        d += 1;
                    }
                    let score = if gating_func == 2 {
                        1.0 / (1.0 + (-logit).exp())
                    } else {
                        (logit - max_logit).exp() / denominator
                    };
                    if score > best {
                        best = score;
                        best_id = expert;
                    }
                }
                expert += 1;
            }
            unsafe {
                *ids.get_unchecked_mut(token * top_k as usize + slot) = best_id as u32;
                *scores.get_unchecked_mut(token * top_k as usize + slot) = best;
            }
            selected_sum += best;
            slot += 1;
        }
        slot = 0;
        while slot < top_k as usize {
            let at = token * top_k as usize + slot;
            let value = unsafe { *scores.get_unchecked_mut(at) };
            let norm = if normalize != 0 && selected_sum > 0.0 {
                value / selected_sum
            } else {
                value
            };
            unsafe {
                *scores.get_unchecked_mut(at) = norm * routed_scale;
            }
            slot += 1;
        }
    }

    #[kernel]
    pub fn fill_u32(value: u32, mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    #[kernel]
    pub fn moe_count_assignments(ids: &[u32], counts: &[DeviceAtomicU32]) {
        let i = thread::index_1d().get();
        if i < ids.len() {
            counts[ids[i] as usize].fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub fn moe_prefix_offsets(
        counts: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut cursors: DisjointSlice<u32>,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut sum = 0u32;
        let mut expert = 0usize;
        while expert < counts.len() {
            unsafe {
                *offsets.get_unchecked_mut(expert) = sum;
                *cursors.get_unchecked_mut(expert) = 0;
            }
            sum += counts[expert];
            expert += 1;
        }
    }

    #[kernel]
    pub fn moe_scatter_assignments(
        ids: &[u32],
        offsets: &[u32],
        cursors: &[DeviceAtomicU32],
        mut order: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= ids.len() {
            return;
        }
        let expert = ids[i] as usize;
        let local = cursors[expert].fetch_add(1, AtomicOrdering::Relaxed) as usize;
        unsafe {
            *order.get_unchecked_mut(offsets[expert] as usize + local) = i as u32;
        }
    }

    /// Q4_K expert projection over device-resident `(token,slot)` assignments.
    #[kernel]
    pub fn moe_q4k_project(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        expert_rows: u32,
        row_offset: u32,
        input_is_grouped: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let linear = thread::index_1d().get();
        let total = tokens as usize * top_k as usize * out_dim as usize;
        if linear >= total {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = linear / out_dim as usize;
        let token = assignment / top_k as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim / Q4K_BLOCK_ELEMS;
        let row_bytes = blocks as usize * Q4K_BLOCK_BYTES as usize;
        let expert_bytes = expert_rows as usize * row_bytes;
        let xb = if input_is_grouped != 0 {
            assignment * in_dim as usize
        } else {
            token * in_dim as usize
        };
        let value = dot_q4k(
            weights,
            expert * expert_bytes + (row_offset as usize + row) * row_bytes,
            input,
            xb,
            blocks,
        );
        if let Some(dst) = out.get_mut(thread::index_1d()) {
            *dst = value;
        }
    }

    #[kernel]
    pub fn moe_q6k_project(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let linear = thread::index_1d().get();
        let total = tokens as usize * top_k as usize * out_dim as usize;
        if linear >= total {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = linear / out_dim as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim / Q6K_BLOCK_ELEMS;
        let row_bytes = blocks as usize * Q6K_BLOCK_BYTES as usize;
        let expert_bytes = out_dim as usize * row_bytes;
        let value = dot_q6k(
            weights,
            expert * expert_bytes + row * row_bytes,
            input,
            assignment * in_dim as usize,
            blocks,
        );
        if let Some(dst) = out.get_mut(thread::index_1d()) {
            *dst = value;
        }
    }

    #[kernel]
    pub fn moe_q5_0_project(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= tokens as usize * top_k as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = linear / out_dim as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim as usize / Q5_0_BLOCK_ELEMS as usize;
        let row_bytes = blocks * Q5_0_BLOCK_BYTES as usize;
        let wb0 = (expert * out_dim as usize + row) * row_bytes;
        let xb0 = assignment * in_dim as usize;
        let mut acc = 0.0f32;
        let mut block = 0usize;
        while block < blocks {
            let wb = wb0 + block * Q5_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([weights[wb], weights[wb + 1]]));
            let qh = u32::from_le_bytes([
                weights[wb + 2],
                weights[wb + 3],
                weights[wb + 4],
                weights[wb + 5],
            ]);
            let xb = xb0 + block * Q5_0_BLOCK_ELEMS as usize;
            let mut lane = 0usize;
            while lane < 16 {
                let p = weights[wb + 6 + lane];
                let lo = ((((qh >> lane) & 1) as i32) << 4 | (p & 15) as i32) - 16;
                let hi = ((((qh >> (lane + 16)) & 1) as i32) << 4 | (p >> 4) as i32) - 16;
                acc += d * lo as f32 * input[xb + lane] + d * hi as f32 * input[xb + lane + 16];
                lane += 1;
            }
            block += 1;
        }
        if let Some(dst) = out.get_mut(idx) {
            *dst = acc;
        }
    }

    #[kernel]
    pub fn moe_q8_0_project(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= tokens as usize * top_k as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = linear / out_dim as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim as usize / Q8_0_BLOCK_ELEMS as usize;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES as usize;
        let wb0 = (expert * out_dim as usize + row) * row_bytes;
        let xb0 = assignment * in_dim as usize;
        let mut acc = 0.0f32;
        let mut block = 0usize;
        while block < blocks {
            let wb = wb0 + block * Q8_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([weights[wb], weights[wb + 1]]));
            let xb = xb0 + block * Q8_0_BLOCK_ELEMS as usize;
            let mut lane = 0usize;
            while lane < 32 {
                acc += d * weights[wb + 2 + lane] as i8 as f32 * input[xb + lane];
                lane += 1;
            }
            block += 1;
        }
        if let Some(dst) = out.get_mut(idx) {
            *dst = acc;
        }
    }

    #[kernel]
    pub fn moe_weighted_reduce(
        expert_out: &[f32],
        scores: &[f32],
        tokens: u32,
        top_k: u32,
        n_embd: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let linear = thread::index_1d().get();
        if linear >= tokens as usize * n_embd as usize {
            return;
        }
        let token = linear / n_embd as usize;
        let dim = linear % n_embd as usize;
        let mut value = 0.0f32;
        let mut slot = 0usize;
        while slot < top_k as usize {
            let assignment = token * top_k as usize + slot;
            value += scores[assignment] * expert_out[assignment * n_embd as usize + dim];
            slot += 1;
        }
        if let Some(dst) = out.get_mut(thread::index_1d()) {
            *dst = value;
        }
    }

    /// Dequantize one Q8_0 embedding row directly on the device.
    #[kernel]
    pub fn embedding_q8_0_row(
        table: &[u8],
        token_id: u32,
        dim: u32,
        n_blocks: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let col = idx.get();
        if col >= dim as usize {
            return;
        }
        let row_bytes = n_blocks as usize * Q8_0_BLOCK_BYTES as usize;
        let block = col / Q8_0_BLOCK_ELEMS as usize;
        let lane = col % Q8_0_BLOCK_ELEMS as usize;
        let off = token_id as usize * row_bytes + block * Q8_0_BLOCK_BYTES as usize;
        let d = f16_to_f32(u16::from_le_bytes([table[off], table[off + 1]]));
        if let Some(value) = out.get_mut(idx) {
            *value = d * table[off + 2 + lane] as i8 as f32;
        }
    }

    /// Tiled Q4_K × f32 GEMV: 32 threads per output row, shared-memory reduce.
    ///
    /// Launch: `grid = (out_dim, 1, 1)`, `block = (32, 1, 1)`.
    #[kernel]
    pub fn q4k_gemv_row_tiled(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let row = thread::blockIdx_x();
        if row >= out_dim {
            return;
        }
        let row_usz = row as usize;
        let row_bytes = (n_blocks * Q4K_BLOCK_BYTES) as usize;
        let row_off = row_usz * row_bytes;
        let mut acc = 0.0f32;

        let mut b = tid;
        while b < n_blocks {
            let blk = row_off + (b as usize) * Q4K_BLOCK_BYTES as usize;
            let d_bits = u16::from_le_bytes([w[blk], w[blk + 1]]);
            let dm_bits = u16::from_le_bytes([w[blk + 2], w[blk + 3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dm_bits);

            let (scales, mins) = decode_scales_mins(
                w[blk + 4],
                w[blk + 5],
                w[blk + 6],
                w[blk + 7],
                w[blk + 8],
                w[blk + 9],
                w[blk + 10],
                w[blk + 11],
                w[blk + 12],
                w[blk + 13],
                w[blk + 14],
                w[blk + 15],
            );

            let qs = blk + 16;
            let x_off = (b as usize) * Q4K_BLOCK_ELEMS as usize;

            let mut sum_min = 0.0f32;
            let mut j = 0usize;
            while j < 8 {
                let mut s = 0.0f32;
                let mut t = 0usize;
                while t < 32 {
                    s += x[x_off + j * 32 + t];
                    t += 1;
                }
                sum_min += mins[j] as f32 * s;
                j += 1;
            }
            acc -= dmin * sum_min;

            let mut is = 0usize;
            let mut off = 0usize;
            let mut chunk = 0usize;
            while chunk < 4 {
                let qbase = qs + chunk * 32;
                let scale_lo = scales[is] as f32;
                is += 1;
                let mut dot = 0.0f32;
                let mut l = 0usize;
                while l < 32 {
                    let q = (w[qbase + l] & 0x0F) as f32;
                    dot += q * x[x_off + off + l];
                    l += 1;
                }
                acc += d * scale_lo * dot;
                off += 32;

                let scale_hi = scales[is] as f32;
                is += 1;
                let mut dot = 0.0f32;
                let mut l = 0usize;
                while l < 32 {
                    let q = (w[qbase + l] >> 4) as f32;
                    dot += q * x[x_off + off + l];
                    l += 1;
                }
                acc += d * scale_hi * dot;
                off += 32;
                chunk += 1;
            }
            b += 32;
        }

        unsafe {
            PARTIAL[tid as usize] = acc;
        }
        thread::sync_threads();

        let mut stride = 16u32;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid as usize] += PARTIAL[(tid + stride) as usize];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(row_usz) = PARTIAL[0];
            }
        }
    }

    /// Warp-parallel batched Q4_K projection. One block owns one output
    /// element and splits compact blocks across its 32 lanes.
    #[kernel]
    pub fn q4k_gemm_warp(
        w: &[u8],
        x: &[f32],
        out_dim: u32,
        n_blocks: u32,
        batch_rows: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= out_dim as usize * batch_rows as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let batch = linear / out_dim as usize;
        let row_bytes = n_blocks as usize * Q4K_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = tid;
        while block < n_blocks as usize {
            acc += dot_q4k(
                w,
                row * row_bytes + block * Q4K_BLOCK_BYTES as usize,
                x,
                (batch * n_blocks as usize + block) * Q4K_BLOCK_ELEMS as usize,
                1,
            );
            block += 32;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(linear) = PARTIAL[0];
            }
        }
    }

    #[kernel]
    pub fn moe_q4k_project_warp(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        order: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        expert_rows: u32,
        row_offset: u32,
        input_is_grouped: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= tokens as usize * top_k as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = order[linear / out_dim as usize] as usize;
        let token = assignment / top_k as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim as usize / Q4K_BLOCK_ELEMS as usize;
        let row_bytes = blocks * Q4K_BLOCK_BYTES as usize;
        let expert_bytes = expert_rows as usize * row_bytes;
        let input_row = if input_is_grouped != 0 {
            assignment
        } else {
            token
        };
        let mut acc = 0.0f32;
        let mut block = tid;
        while block < blocks {
            acc += dot_q4k(
                weights,
                expert * expert_bytes
                    + (row_offset as usize + row) * row_bytes
                    + block * Q4K_BLOCK_BYTES as usize,
                input,
                (input_row * blocks + block) * Q4K_BLOCK_ELEMS as usize,
                1,
            );
            block += 32;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(assignment * out_dim as usize + row) = PARTIAL[0];
            }
        }
    }

    #[kernel]
    pub fn moe_q5_0_project_warp(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        order: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= tokens as usize * top_k as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = order[linear / out_dim as usize] as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim as usize / Q5_0_BLOCK_ELEMS as usize;
        let row_bytes = blocks * Q5_0_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = tid;
        while block < blocks {
            let wb =
                (expert * out_dim as usize + row) * row_bytes + block * Q5_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([weights[wb], weights[wb + 1]]));
            let qh = u32::from_le_bytes([
                weights[wb + 2],
                weights[wb + 3],
                weights[wb + 4],
                weights[wb + 5],
            ]);
            let xb = (assignment * blocks + block) * 32;
            let mut lane = 0usize;
            while lane < 16 {
                let p = weights[wb + 6 + lane];
                let lo = ((((qh >> lane) & 1) as i32) << 4 | (p & 15) as i32) - 16;
                let hi = ((((qh >> (lane + 16)) & 1) as i32) << 4 | (p >> 4) as i32) - 16;
                acc += d * lo as f32 * input[xb + lane] + d * hi as f32 * input[xb + lane + 16];
                lane += 1;
            }
            block += 32;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(assignment * out_dim as usize + row) = PARTIAL[0];
            }
        }
    }

    #[kernel]
    pub fn moe_q8_0_project_warp(
        weights: &[u8],
        input: &[f32],
        ids: &[u32],
        order: &[u32],
        tokens: u32,
        top_k: u32,
        out_dim: u32,
        in_dim: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let linear = thread::blockIdx_x() as usize;
        if linear >= tokens as usize * top_k as usize * out_dim as usize {
            return;
        }
        let row = linear % out_dim as usize;
        let assignment = order[linear / out_dim as usize] as usize;
        let expert = ids[assignment] as usize;
        let blocks = in_dim as usize / Q8_0_BLOCK_ELEMS as usize;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut block = tid;
        while block < blocks {
            let wb =
                (expert * out_dim as usize + row) * row_bytes + block * Q8_0_BLOCK_BYTES as usize;
            let d = f16_to_f32(u16::from_le_bytes([weights[wb], weights[wb + 1]]));
            let xb = (assignment * blocks + block) * 32;
            let mut lane = 0usize;
            while lane < 32 {
                acc += d * weights[wb + 2 + lane] as i8 as f32 * input[xb + lane];
                lane += 1;
            }
            block += 32;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(assignment * out_dim as usize + row) = PARTIAL[0];
            }
        }
    }

    /// Elementwise add: `out[i] = a[i] + b[i]`.
    #[kernel]
    pub fn add_f32(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i] + b[i];
        }
    }

    #[kernel]
    pub fn mul_f32(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(dst) = out.get_mut(idx) {
            *dst = a[i] * b[i];
        }
    }

    /// Sparse self-conditioning: softmax packed `(token,logit)` candidates
    /// and accumulate Q6_K embedding rows without materializing a dense vocab.
    #[kernel]
    pub fn weighted_embedding_q6k_topk(
        weights: &[u8],
        packed: &[f32],
        rows: u32,
        top_k: u32,
        dim: u32,
        vocab: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= rows as usize * dim as usize {
            return;
        }
        let row = linear / dim as usize;
        let col = linear % dim as usize;
        let base = row * top_k as usize * 2;
        let mut max = -3.402823466e38f32;
        let mut slot = 0usize;
        while slot < top_k as usize {
            let logit = packed[base + slot * 2 + 1];
            if logit > max {
                max = logit;
            }
            slot += 1;
        }
        let mut denom = 0.0f32;
        slot = 0;
        while slot < top_k as usize {
            denom += (packed[base + slot * 2 + 1] - max).exp();
            slot += 1;
        }
        let blocks = dim as usize / Q6K_BLOCK_ELEMS as usize;
        let row_bytes = blocks * Q6K_BLOCK_BYTES as usize;
        let block = col / Q6K_BLOCK_ELEMS as usize;
        let j = col % Q6K_BLOCK_ELEMS as usize;
        let mut value = 0.0f32;
        slot = 0;
        while slot < top_k as usize {
            let token = packed[base + slot * 2] as u32;
            if token < vocab && denom > 0.0 {
                let blk = token as usize * row_bytes + block * Q6K_BLOCK_BYTES as usize;
                let d = f16_to_f32(u16::from_le_bytes([weights[blk + 208], weights[blk + 209]]));
                let half = j / 128;
                let local = j % 128;
                let lane = local % 32;
                let quad = local / 32;
                let ql = blk + half * 64;
                let qh = blk + 128 + half * 32;
                let sc = blk + 192 + half * 8;
                let shift = quad as u32 * 2;
                let qi = if quad == 0 || quad == 2 {
                    lane
                } else {
                    lane + 32
                };
                let nibble = if quad < 2 {
                    (weights[ql + qi] & 15) as i32
                } else {
                    (weights[ql + qi] >> 4) as i32
                };
                let q = (nibble | ((((weights[qh + lane] >> shift) & 3) as i32) << 4)) - 32;
                let decoded = d * weights[sc + lane / 16 + quad * 2] as i8 as f32 * q as f32;
                value += ((packed[base + slot * 2 + 1] - max).exp() / denom) * decoded;
            }
            slot += 1;
        }
        if let Some(dst) = out.get_mut(idx) {
            *dst = value;
        }
    }

    /// Fuse the recurrent part of LFM ShortConv and update its device state.
    #[kernel]
    pub fn shortconv_mix(
        bcx: &[f32],
        conv: &[f32],
        mut state: DisjointSlice<f32>,
        n_embd: u32,
        l_cache: u32,
        mut y_pre: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let c = idx.get();
        if c >= n_embd as usize || l_cache == 0 {
            return;
        }
        let n = n_embd as usize;
        let history = l_cache as usize - 1;
        let b = bcx[c];
        let gate = bcx[n + c];
        let projected = bcx[2 * n + c];
        let bx = b * projected;
        let mut mixed = bx * conv[c * l_cache as usize + history];
        let mut t = 0usize;
        while t < history {
            mixed +=
                unsafe { *state.get_unchecked_mut(t * n + c) } * conv[c * l_cache as usize + t];
            t += 1;
        }
        if let Some(value) = y_pre.get_mut(idx) {
            *value = gate * mixed;
        }
        t = 0;
        while t < history {
            let next = if t + 1 < history {
                unsafe { *state.get_unchecked_mut((t + 1) * n + c) }
            } else {
                bx
            };
            unsafe {
                *state.get_unchecked_mut(t * n + c) = next;
            }
            t += 1;
        }
    }

    /// Copy embedding row `token_id` from an f32 `[vocab, dim]` table.
    #[kernel]
    pub fn embedding_f32(table: &[f32], token_id: u32, dim: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dim as usize {
            return;
        }
        let src = (token_id as usize) * (dim as usize) + i;
        if let Some(o) = out.get_mut(idx) {
            *o = table[src];
        }
    }

    /// Dequantize one Q4_K embedding row into f32.
    ///
    /// `w` is the full `[vocab, dim]` Q4_K matrix. Launch with `for_num_elems(dim)`.
    #[kernel]
    pub fn embedding_q4k_row(
        w: &[u8],
        token_id: u32,
        dim: u32,
        n_blocks: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dim as usize {
            return;
        }
        let row_bytes = (n_blocks * Q4K_BLOCK_BYTES) as usize;
        let row_off = (token_id as usize) * row_bytes;
        let b = i / Q4K_BLOCK_ELEMS as usize;
        let j = i % Q4K_BLOCK_ELEMS as usize;
        let blk = row_off + b * Q4K_BLOCK_BYTES as usize;

        let d_bits = u16::from_le_bytes([w[blk], w[blk + 1]]);
        let dm_bits = u16::from_le_bytes([w[blk + 2], w[blk + 3]]);
        let d = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dm_bits);

        let (scales, mins) = decode_scales_mins(
            w[blk + 4],
            w[blk + 5],
            w[blk + 6],
            w[blk + 7],
            w[blk + 8],
            w[blk + 9],
            w[blk + 10],
            w[blk + 11],
            w[blk + 12],
            w[blk + 13],
            w[blk + 14],
            w[blk + 15],
        );

        // j in 0..256: four chunks of 64 (lo32 + hi32), scale/min index = j/32.
        let group = j / 32;
        let lane = j % 32;
        let scale = scales[group] as f32;
        let min_v = mins[group] as f32;
        let qs = blk + 16;
        // chunk 0: j 0..63, chunk 1: 64..127, ...
        let chunk = j / 64;
        let within = j % 64;
        let qbase = qs + chunk * 32;
        let q = if within < 32 {
            (w[qbase + lane] & 0x0F) as f32
        } else {
            (w[qbase + lane] >> 4) as f32
        };
        if let Some(o) = out.get_mut(idx) {
            *o = d * scale * q - dmin * min_v;
        }
    }

    /// Dequantize one Q6_K embedding row into f32.
    #[kernel]
    pub fn embedding_q6k_row(
        w: &[u8],
        token_id: u32,
        dim: u32,
        n_blocks: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dim as usize {
            return;
        }
        let row_bytes = (n_blocks * Q6K_BLOCK_BYTES) as usize;
        let row_off = (token_id as usize) * row_bytes;
        let b = i / Q6K_BLOCK_ELEMS as usize;
        let j = i % Q6K_BLOCK_ELEMS as usize;
        let blk = row_off + b * Q6K_BLOCK_BYTES as usize;

        let d_bits = u16::from_le_bytes([w[blk + 208], w[blk + 209]]);
        let d = f16_to_f32(d_bits);

        let half = j / 128;
        let local = j % 128;
        let ql = blk + half * 64;
        let qh = blk + 128 + half * 32;
        let sc = blk + 192 + half * 8;

        // Inverse of the 4-way pack in dequantize_q6_k_block.
        let lane = local % 32;
        let quad = local / 32; // 0..3 → q1..q4
        let is = lane / 16;
        let qh_shift = match quad {
            0 => 0u32,
            1 => 2,
            2 => 4,
            _ => 6,
        };
        let ql_idx = if quad == 0 || quad == 2 {
            lane
        } else {
            lane + 32
        };
        let nibble = if quad == 0 || quad == 1 {
            (w[ql + ql_idx] & 0xF) as i32
        } else {
            (w[ql + ql_idx] >> 4) as i32
        };
        let q = (nibble | ((((w[qh + lane] >> qh_shift) & 3) as i32) << 4)) - 32;
        let scale = w[sc + is + quad * 2] as i8 as f32;
        // sc indexing: q1→sc[is], q2→sc[is+2], q3→sc[is+4], q4→sc[is+6]
        // quad*2 gives 0,2,4,6 ✓

        if let Some(o) = out.get_mut(idx) {
            *o = d * scale * (q as f32);
        }
    }

    #[kernel]
    pub fn embedding_q6k_rows(
        w: &[u8],
        ids: &[u32],
        rows: u32,
        dim: u32,
        n_blocks: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= rows as usize * dim as usize {
            return;
        }
        let row = linear / dim as usize;
        let i = linear % dim as usize;
        let row_bytes = n_blocks as usize * Q6K_BLOCK_BYTES as usize;
        let b = i / Q6K_BLOCK_ELEMS as usize;
        let j = i % Q6K_BLOCK_ELEMS as usize;
        let blk = ids[row] as usize * row_bytes + b * Q6K_BLOCK_BYTES as usize;
        let d = f16_to_f32(u16::from_le_bytes([w[blk + 208], w[blk + 209]]));
        let half = j / 128;
        let local = j % 128;
        let ql = blk + half * 64;
        let qh = blk + 128 + half * 32;
        let sc = blk + 192 + half * 8;
        let lane = local % 32;
        let quad = local / 32;
        let shift = quad as u32 * 2;
        let qi = if quad == 0 || quad == 2 {
            lane
        } else {
            lane + 32
        };
        let nibble = if quad < 2 {
            (w[ql + qi] & 15) as i32
        } else {
            (w[ql + qi] >> 4) as i32
        };
        let q = (nibble | ((((w[qh + lane] >> shift) & 3) as i32) << 4)) - 32;
        if let Some(dst) = out.get_mut(idx) {
            *dst = d * w[sc + lane / 16 + quad * 2] as i8 as f32 * q as f32;
        }
    }

    /// Bidirectional canvas attention over device-resident canvas K/V. Prefix
    /// K/V is supplied separately when available; the normal CUDA lowering
    /// uses the paged variant for the persistent prompt prefix.
    #[kernel]
    pub fn attention_canvas_heads(
        q: &[f32],
        prefix_k: &[f32],
        prefix_v: &[f32],
        canvas_k: &[f32],
        canvas_v: &[f32],
        rows: u32,
        prefix_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let linear = thread::index_1d().get();
        if linear >= rows as usize * n_heads as usize {
            return;
        }
        let row = linear / n_heads as usize;
        let head = linear % n_heads as usize;
        let hd = head_dim as usize;
        let nkh = n_kv_heads.max(1) as usize;
        let kvh = head / (n_heads as usize / nkh).max(1);
        let qb = row * n_heads as usize * hd + head * hd;
        let ob = qb;
        let mut d = 0usize;
        while d < hd {
            unsafe {
                *out.get_unchecked_mut(ob + d) = 0.0;
            }
            d += 1;
        }
        let total = prefix_len as usize + rows as usize;
        let mut m = 0.0f32;
        let mut sum = 0.0f32;
        let mut started = false;
        let mut t = 0usize;
        while t < total {
            let (k, v, kb) = if t < prefix_len as usize {
                (prefix_k, prefix_v, t * nkh * hd + kvh * hd)
            } else {
                let ct = t - prefix_len as usize;
                (canvas_k, canvas_v, ct * nkh * hd + kvh * hd)
            };
            let mut score = 0.0f32;
            d = 0;
            while d < hd {
                score += q[qb + d] * k[kb + d];
                d += 1;
            }
            score *= scale;
            let (nm, alpha) = if !started {
                (score, 0.0)
            } else if score > m {
                (score, (m - score).exp())
            } else {
                (m, 1.0)
            };
            let p = (score - nm).exp();
            sum = sum * alpha + p;
            d = 0;
            while d < hd {
                unsafe {
                    let dst = out.get_unchecked_mut(ob + d);
                    *dst = *dst * alpha + p * v[kb + d];
                }
                d += 1;
            }
            m = nm;
            started = true;
            t += 1;
        }
        if sum > 0.0 {
            d = 0;
            while d < hd {
                unsafe {
                    *out.get_unchecked_mut(ob + d) /= sum;
                }
                d += 1;
            }
        }
    }

    #[kernel]
    pub fn attention_canvas_paged_heads(
        q: &[f32],
        arena: &[u8],
        table: &[u32],
        canvas_k: &[f32],
        canvas_v: &[f32],
        rows: u32,
        prefix_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        layer: u32,
        n_logical: u32,
        block_size: u32,
        block_bytes: u32,
        tokens_stride: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let linear = thread::index_1d().get();
        if linear >= rows as usize * n_heads as usize {
            return;
        }
        let row = linear / n_heads as usize;
        let head = linear % n_heads as usize;
        let hd = head_dim as usize;
        let nkh = n_kv_heads.max(1) as usize;
        let kvh = head / (n_heads as usize / nkh).max(1);
        let qb = row * n_heads as usize * hd + head * hd;
        let ob = qb;
        let mut d = 0usize;
        while d < hd {
            unsafe {
                *out.get_unchecked_mut(ob + d) = 0.0;
            }
            d += 1;
        }
        let mut m = 0.0f32;
        let mut sum = 0.0f32;
        let mut started = false;
        let mut t = 0usize;
        while t < prefix_len as usize + rows as usize {
            let mut score = 0.0f32;
            if t < prefix_len as usize {
                let logical = t / block_size as usize;
                let slot = t % block_size as usize;
                let phys = table[layer as usize * n_logical as usize + logical] as usize;
                let row_bytes = tokens_stride as usize * 2;
                let kb = phys * block_bytes as usize + slot * row_bytes + kvh * hd * 2;
                d = 0;
                while d < hd {
                    let bits = arena[kb + d * 2] as u16 | ((arena[kb + d * 2 + 1] as u16) << 8);
                    score += q[qb + d] * f16_to_f32(bits);
                    d += 1;
                }
                score *= scale;
                let (nm, alpha) = if !started {
                    (score, 0.0)
                } else if score > m {
                    (score, (m - score).exp())
                } else {
                    (m, 1.0)
                };
                let p = (score - nm).exp();
                sum = sum * alpha + p;
                let vb = phys * block_bytes as usize
                    + block_size as usize * row_bytes
                    + slot * row_bytes
                    + kvh * hd * 2;
                d = 0;
                while d < hd {
                    let bits = arena[vb + d * 2] as u16 | ((arena[vb + d * 2 + 1] as u16) << 8);
                    unsafe {
                        let o = out.get_unchecked_mut(ob + d);
                        *o = *o * alpha + p * f16_to_f32(bits);
                    }
                    d += 1;
                }
                m = nm;
                started = true;
            } else {
                let ct = t - prefix_len as usize;
                let kb = ct * nkh * hd + kvh * hd;
                d = 0;
                while d < hd {
                    score += q[qb + d] * canvas_k[kb + d];
                    d += 1;
                }
                score *= scale;
                let (nm, alpha) = if !started {
                    (score, 0.0)
                } else if score > m {
                    (score, (m - score).exp())
                } else {
                    (m, 1.0)
                };
                let p = (score - nm).exp();
                sum = sum * alpha + p;
                d = 0;
                while d < hd {
                    unsafe {
                        let o = out.get_unchecked_mut(ob + d);
                        *o = *o * alpha + p * canvas_v[kb + d];
                    }
                    d += 1;
                }
                m = nm;
                started = true;
            }
            t += 1;
        }
        if sum > 0.0 {
            d = 0;
            while d < hd {
                unsafe {
                    *out.get_unchecked_mut(ob + d) /= sum;
                }
                d += 1;
            }
        }
    }

    /// Contiguous multi-head attention (online softmax).
    ///
    /// Launch with `for_num_elems(n_heads)` — one thread per head.
    /// `q`/`out`: `[n_heads * head_dim]`, `k`/`v`: `[seq * n_kv_heads * head_dim]`.
    #[kernel]
    pub fn attention_heads(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let head = thread::index_1d().get() as u32;
        if head >= n_heads {
            return;
        }
        let hd = head_dim as usize;
        let seq_u = seq as usize;
        let n_kv = n_kv_heads.max(1);
        let kv_group = (n_heads / n_kv).max(1);
        let kv_h = (head / kv_group) as usize;

        let q_base = (head as usize) * hd;
        let out_base = q_base;
        let mut d = 0usize;
        while d < hd {
            unsafe {
                *out.get_unchecked_mut(out_base + d) = 0.0;
            }
            d += 1;
        }

        // Avoid −∞ / is_finite on device: track first score explicitly.
        let mut m = 0.0f32;
        let mut l = 0.0f32;
        let mut started = false;

        let mut t = 0usize;
        while t < seq_u {
            let k_base = t * (n_kv as usize) * hd + kv_h * hd;
            let mut score = 0.0f32;
            let mut d = 0usize;
            while d < hd {
                score += q[q_base + d] * k[k_base + d];
                d += 1;
            }
            score *= scale;

            let (m_new, alpha) = if !started {
                (score, 0.0f32)
            } else if score > m {
                (score, (m - score).exp())
            } else {
                (m, 1.0f32)
            };
            let p = (score - m_new).exp();
            l = l * alpha + p;

            let v_base = t * (n_kv as usize) * hd + kv_h * hd;
            let mut d = 0usize;
            while d < hd {
                unsafe {
                    let o = out.get_unchecked_mut(out_base + d);
                    *o = *o * alpha + p * v[v_base + d];
                }
                d += 1;
            }
            m = m_new;
            started = true;
            t += 1;
        }
        if l > 0.0 {
            let inv_l = 1.0 / l;
            let mut d = 0usize;
            while d < hd {
                unsafe {
                    *out.get_unchecked_mut(out_base + d) *= inv_l;
                }
                d += 1;
            }
        }
    }

    /// Write one f32 KV row into the paged f16 arena (device).
    ///
    /// Launch with `for_num_elems(tokens_stride)`.
    #[kernel]
    pub fn kv_write_row(
        row: &[f32],
        mut arena: DisjointSlice<u8>,
        block_table: &[u32],
        layer: u32,
        position: u32,
        is_v: u32,
        n_logical: u32,
        tokens_stride: u32,
        row_len: u32,
        block_size: u32,
        block_bytes: u32,
    ) {
        let i = thread::index_1d().get() as u32;
        if i >= row_len {
            return;
        }
        let logical = position / block_size;
        let slot = position % block_size;
        let table_idx = (layer * n_logical + logical) as usize;
        let phys = block_table[table_idx] as usize;
        let row_bytes = (tokens_stride as usize) * 2;
        let v_base = if is_v != 0 {
            (block_size as usize) * row_bytes
        } else {
            0
        };
        let base = phys * (block_bytes as usize) + v_base + (slot as usize) * row_bytes;
        let bits = f32_to_f16_bits(row[i as usize]);
        let lo = (bits & 0xff) as u8;
        let hi = (bits >> 8) as u8;
        let off = base + (i as usize) * 2;
        unsafe {
            *arena.get_unchecked_mut(off) = lo;
            *arena.get_unchecked_mut(off + 1) = hi;
        }
    }

    /// Paged multi-head attention reading f16 K/V from the device arena.
    #[kernel]
    pub fn attention_paged_heads(
        q: &[f32],
        arena: &[u8],
        block_table: &[u32],
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq: u32,
        scale: f32,
        layer: u32,
        n_logical: u32,
        block_size: u32,
        block_bytes: u32,
        tokens_stride: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let head = thread::index_1d().get() as u32;
        if head >= n_heads {
            return;
        }
        let hd = head_dim as usize;
        let seq_u = seq as usize;
        let n_kv = n_kv_heads.max(1);
        let kv_group = (n_heads / n_kv).max(1);
        let kv_h = (head / kv_group) as usize;
        let stride = tokens_stride as usize;
        let bs = block_size as usize;
        let bb = block_bytes as usize;
        let row_bytes = stride * 2;
        let v_off0 = bs * row_bytes;

        let q_base = (head as usize) * hd;
        let out_base = q_base;
        let mut d = 0usize;
        while d < hd {
            unsafe {
                *out.get_unchecked_mut(out_base + d) = 0.0;
            }
            d += 1;
        }

        let mut m = 0.0f32;
        let mut l = 0.0f32;
        let mut started = false;

        let mut t = 0usize;
        while t < seq_u {
            let logical = t / bs;
            let slot = t % bs;
            let table_idx = (layer as usize) * (n_logical as usize) + logical;
            let phys = block_table[table_idx] as usize;
            let k_base = phys * bb + slot * row_bytes + kv_h * hd * 2;

            let mut score = 0.0f32;
            let mut d = 0usize;
            while d < hd {
                let b0 = arena[k_base + d * 2] as u16;
                let b1 = arena[k_base + d * 2 + 1] as u16;
                let kv = f16_to_f32(b0 | (b1 << 8));
                score += q[q_base + d] * kv;
                d += 1;
            }
            score *= scale;

            let (m_new, alpha) = if !started {
                (score, 0.0f32)
            } else if score > m {
                (score, (m - score).exp())
            } else {
                (m, 1.0f32)
            };
            let p = (score - m_new).exp();
            l = l * alpha + p;

            let v_base = phys * bb + v_off0 + slot * row_bytes + kv_h * hd * 2;
            let mut d = 0usize;
            while d < hd {
                let b0 = arena[v_base + d * 2] as u16;
                let b1 = arena[v_base + d * 2 + 1] as u16;
                let vv = f16_to_f32(b0 | (b1 << 8));
                unsafe {
                    let o = out.get_unchecked_mut(out_base + d);
                    *o = *o * alpha + p * vv;
                }
                d += 1;
            }
            m = m_new;
            started = true;
            t += 1;
        }
        if l > 0.0 {
            let inv_l = 1.0 / l;
            let mut d = 0usize;
            while d < hd {
                unsafe {
                    *out.get_unchecked_mut(out_base + d) *= inv_l;
                }
                d += 1;
            }
        }
    }

    /// Smoke kernel.
    #[kernel]
    pub fn scale_f32(factor: f32, input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[i] * factor;
        }
    }
}

/// IEEE754 half → f32 (device-safe, no `half` crate on device).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut f = frac;
            let mut e = 127 - 15 + 1;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            (sign << 31) | ((e as u32) << 23) | (f << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

#[inline]
fn decode_scales_mins(
    b0: u8,
    b1: u8,
    b2: u8,
    b3: u8,
    b4: u8,
    b5: u8,
    b6: u8,
    b7: u8,
    b8: u8,
    b9: u8,
    b10: u8,
    b11: u8,
) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes([b0, b1, b2, b3]);
    utmp[1] = u32::from_le_bytes([b4, b5, b6, b7]);
    utmp[2] = u32::from_le_bytes([b8, b9, b10, b11]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    let s0 = utmp[0].to_le_bytes();
    let s1 = utmp[1].to_le_bytes();
    let m0 = utmp[2].to_le_bytes();
    let m1 = utmp[3].to_le_bytes();
    (
        [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]],
        [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]],
    )
}
