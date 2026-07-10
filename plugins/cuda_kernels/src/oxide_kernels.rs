//! Oxide `#[cuda_module]` kernels for FeLLM B1 ops.

use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

/// Q4_K super-block size (GGUF / ggml).
pub const Q4K_BLOCK_BYTES: u32 = 144;
/// Elements per Q4_K super-block.
pub const Q4K_BLOCK_ELEMS: u32 = 256;

#[cuda_module]
pub mod kernels {
    use super::*;

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
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let total = (n_heads * head_dim) as usize;
        if i >= total {
            return;
        }
        let head = i / head_dim as usize;
        let d = i % head_dim as usize;
        let base = head * head_dim as usize;

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
        let theta = position * inv_freqs[pair];
        let (s, c) = (theta.sin(), theta.cos());
        let a = x[base + d];
        let b = x[base + d + 1];
        unsafe {
            *out.get_unchecked_mut(base + d) = a * c - b * s;
            *out.get_unchecked_mut(base + d + 1) = a * s + b * c;
        }
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
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        if row >= out_dim as usize {
            return;
        }
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
            let x_off = (b as usize) * Q4K_BLOCK_ELEMS as usize;

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
        block_size: u32,
        block_bytes: u32,
    ) {
        let i = thread::index_1d().get() as u32;
        if i >= tokens_stride {
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
