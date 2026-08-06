//! Generic floating-point primitives dispatched once through `pulp`.
//!
//! Quantized Q4_K/Q6_K kernels intentionally live outside this module: their
//! nibble unpacking and integer dot products are format-specific.  Everything
//! here is ordinary f32/f16 elementwise or reduction work and benefits from
//! one runtime-selected `pulp::Arch`.

use half::f16;
use pulp::{Arch, Simd, WithSimd};

/// Runtime-selected CPU SIMD implementation.
///
/// `Arch::new()` is called by `CpuBackend::new()` exactly once. The value is
/// `Copy`, so passing it through hot kernels does not repeat CPUID detection.
#[derive(Clone, Copy, Debug)]
pub struct PulpDispatch {
    arch: Arch,
}

impl PulpDispatch {
    #[must_use]
    pub fn new() -> Self {
        Self { arch: Arch::new() }
    }

    #[inline(always)]
    fn dispatch<Op: WithSimd>(&self, op: Op) -> Op::Output {
        self.arch.dispatch(op)
    }
}

impl Default for PulpDispatch {
    fn default() -> Self {
        Self::new()
    }
}

struct DotF32<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl WithSimd for DotF32<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);
        let mut acc = simd.splat_f32s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            // Keep the old non-fused accumulation order for transformer
            // logits. FMA is faster, but changing the rounding here can move
            // a greedy token across a tie; pulp still supplies the runtime
            // SIMD implementation and vector loads/reductions.
            acc = simd.add_f32s(simd.mul_f32s(a, b), acc);
        }
        simd.reduce_sum_f32s(acc) + a_tail.iter().zip(b_tail).map(|(&a, &b)| a * b).sum::<f32>()
    }
}

struct DotF32F16<'a> {
    a: &'a [f32],
    b: &'a [f16],
}

impl WithSimd for DotF32F16<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let lanes = S::F32_LANES;
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let mut acc = simd.splat_f32s(0.0);
        let mut b_offset = 0;
        for &a in a_head {
            let b_chunk = &self.b[b_offset..b_offset + lanes];
            // pulp's largest f32 register is 512-bit (16 lanes); keeping this
            // conversion scratch at the actual upper bound avoids clearing a
            // larger temporary for every f16 attention dot.
            let mut converted = [0.0f32; 16];
            for (dst, src) in converted[..lanes].iter_mut().zip(b_chunk) {
                *dst = src.to_f32();
            }
            let b = simd.partial_load_f32s(&converted[..lanes]);
            acc = simd.add_f32s(simd.mul_f32s(a, b), acc);
            b_offset += lanes;
        }
        simd.reduce_sum_f32s(acc)
            + a_tail
                .iter()
                .zip(&self.b[b_offset..])
                .map(|(&a, b)| a * b.to_f32())
                .sum::<f32>()
    }
}

struct AxpyF32<'a> {
    out: &'a mut [f32],
    v: &'a [f32],
    scale: f32,
}

impl WithSimd for AxpyF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let n = self.out.len().min(self.v.len());
        let (out_head, out_tail) = S::as_mut_simd_f32s(&mut self.out[..n]);
        let (v_head, v_tail) = S::as_simd_f32s(&self.v[..n]);
        let scale = simd.splat_f32s(self.scale);
        for (out, &v) in out_head.iter_mut().zip(v_head) {
            *out = simd.add_f32s(simd.mul_f32s(v, scale), *out);
        }
        for (out, &v) in out_tail.iter_mut().zip(v_tail) {
            *out += self.scale * v;
        }
    }
}

struct AxpyF32F16<'a> {
    out: &'a mut [f32],
    v: &'a [f16],
    scale: f32,
}

impl WithSimd for AxpyF32F16<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let n = self.out.len().min(self.v.len());
        let lanes = S::F32_LANES;
        let (out_head, out_tail) = S::as_mut_simd_f32s(&mut self.out[..n]);
        let mut offset = 0;
        let scale = simd.splat_f32s(self.scale);
        for out in out_head {
            let chunk = &self.v[offset..offset + lanes];
            let mut converted = [0.0f32; 16];
            for (dst, src) in converted[..lanes].iter_mut().zip(chunk) {
                *dst = src.to_f32();
            }
            let v = simd.partial_load_f32s(&converted[..lanes]);
            *out = simd.add_f32s(simd.mul_f32s(v, scale), *out);
            offset += lanes;
        }
        for (out, v) in out_tail.iter_mut().zip(&self.v[offset..n]) {
            *out += self.scale * v.to_f32();
        }
    }
}

struct ScaleF32<'a> {
    out: &'a mut [f32],
    scale: f32,
}

impl WithSimd for ScaleF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head, tail) = S::as_mut_simd_f32s(self.out);
        let scale = simd.splat_f32s(self.scale);
        for value in head {
            *value = simd.mul_f32s(*value, scale);
        }
        for value in tail {
            *value *= self.scale;
        }
    }
}

struct AddF32<'a> {
    a: &'a [f32],
    b: &'a [f32],
    y: &'a mut [f32],
}

impl WithSimd for AddF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let n = self.y.len().min(self.a.len()).min(self.b.len());
        let (a, a_tail) = S::as_simd_f32s(&self.a[..n]);
        let (b, b_tail) = S::as_simd_f32s(&self.b[..n]);
        let (y, y_tail) = S::as_mut_simd_f32s(&mut self.y[..n]);
        for ((y, &a), &b) in y.iter_mut().zip(a).zip(b) {
            *y = simd.add_f32s(a, b);
        }
        for ((y, &a), &b) in y_tail.iter_mut().zip(a_tail).zip(b_tail) {
            *y = a + b;
        }
    }
}

struct MulF32<'a> {
    a: &'a [f32],
    b: &'a [f32],
    y: &'a mut [f32],
}

impl WithSimd for MulF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let n = self.y.len().min(self.a.len()).min(self.b.len());
        let (a, a_tail) = S::as_simd_f32s(&self.a[..n]);
        let (b, b_tail) = S::as_simd_f32s(&self.b[..n]);
        let (y, y_tail) = S::as_mut_simd_f32s(&mut self.y[..n]);
        for ((y, &a), &b) in y.iter_mut().zip(a).zip(b) {
            *y = simd.mul_f32s(a, b);
        }
        for ((y, &a), &b) in y_tail.iter_mut().zip(a_tail).zip(b_tail) {
            *y = a * b;
        }
    }
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], dispatch: PulpDispatch) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    dispatch.dispatch(DotF32 { a, b })
}

#[inline]
pub fn dot_f32_f16(a: &[f32], b: &[f16], dispatch: PulpDispatch) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    dispatch.dispatch(DotF32F16 { a, b })
}

#[inline]
pub fn axpy_f32(out: &mut [f32], v: &[f32], scale: f32, dispatch: PulpDispatch) {
    dispatch.dispatch(AxpyF32 { out, v, scale });
}

#[inline]
pub fn axpy_f32_f16(out: &mut [f32], v: &[f16], scale: f32, dispatch: PulpDispatch) {
    dispatch.dispatch(AxpyF32F16 { out, v, scale });
}

#[inline]
pub fn scale_f32(out: &mut [f32], scale: f32, dispatch: PulpDispatch) {
    dispatch.dispatch(ScaleF32 { out, scale });
}

#[inline]
pub fn add_f32(a: &[f32], b: &[f32], y: &mut [f32], dispatch: PulpDispatch) {
    dispatch.dispatch(AddF32 { a, b, y });
}

#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], y: &mut [f32], dispatch: PulpDispatch) {
    dispatch.dispatch(MulF32 { a, b, y });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| 1.0 - i as f32 * 0.03).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = dot_f32(&a, &b, PulpDispatch::new());
        assert!((got - expected).abs() < 1e-4, "{got} vs {expected}");
    }

    #[test]
    fn add_mul_match_scalar() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let dispatch = PulpDispatch::new();
        let mut y_add = vec![0.0; a.len()];
        let mut y_mul = vec![0.0; a.len()];
        add_f32(&a, &b, &mut y_add, dispatch);
        mul_f32(&a, &b, &mut y_mul, dispatch);
        assert_eq!(y_add, vec![6.0, 6.0, 6.0, 6.0, 6.0]);
        assert_eq!(y_mul, vec![5.0, 8.0, 9.0, 8.0, 5.0]);
    }
}
