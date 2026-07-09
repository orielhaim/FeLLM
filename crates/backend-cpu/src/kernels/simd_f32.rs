#![allow(unsafe_op_in_unsafe_fn)]

use crate::cpu_profile::CpuHardwareProfile;
use half::f16;
use wide::f32x8;

/// Prefer 16-wide AVX-512 loops when the runtime profile says so.
#[inline]
pub fn use_avx512(profile: &CpuHardwareProfile) -> bool {
    profile.has_avx512 && profile.simd_f32_lanes >= 16
}

/// Contiguous f32 · f32 dot product.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], profile: &CpuHardwareProfile) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        return unsafe { dot_f32_avx512(a, b) };
    }
    let _ = profile;
    dot_f32_x8(a, b)
}

/// Contiguous f32 · f16 dot (dequant in-register).
#[inline]
pub fn dot_f32_f16(a: &[f32], b: &[f16], profile: &CpuHardwareProfile) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        return unsafe { dot_f32_f16_avx512(a, b) };
    }
    let _ = profile;
    dot_f32_f16_x8(a, b)
}

/// `out[i] += scale * v[i]` with SIMD FMA-style accumulation.
#[inline]
pub fn axpy_f32(out: &mut [f32], v: &[f32], scale: f32, profile: &CpuHardwareProfile) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        unsafe {
            axpy_f32_avx512(out, v, scale);
        }
        return;
    }
    let _ = profile;
    axpy_f32_x8(out, v, scale);
}

/// `out[i] += scale * v[i].to_f32()` for f16 V rows.
#[inline]
pub fn axpy_f32_f16(out: &mut [f32], v: &[f16], scale: f32, profile: &CpuHardwareProfile) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        unsafe {
            axpy_f32_f16_avx512(out, v, scale);
        }
        return;
    }
    let _ = profile;
    axpy_f32_f16_x8(out, v, scale);
}

/// `out[i] *= scale` (online-softmax rescale).
#[inline]
pub fn scale_f32(out: &mut [f32], scale: f32, profile: &CpuHardwareProfile) {
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        unsafe {
            scale_f32_avx512(out, scale);
        }
        return;
    }
    let _ = profile;
    scale_f32_x8(out, scale);
}

/// Elementwise `y[i] = a[i] + b[i]`.
#[inline]
pub fn add_f32(a: &[f32], b: &[f32], y: &mut [f32], profile: &CpuHardwareProfile) {
    let n = y.len().min(a.len()).min(b.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        unsafe {
            add_f32_avx512(&a[..n], &b[..n], &mut y[..n]);
        }
        return;
    }
    let _ = profile;
    add_f32_x8(&a[..n], &b[..n], &mut y[..n]);
}

/// Elementwise `y[i] = a[i] * b[i]`.
#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], y: &mut [f32], profile: &CpuHardwareProfile) {
    let n = y.len().min(a.len()).min(b.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx512(profile) {
        // SAFETY: gated on runtime CPUID via profile.
        unsafe {
            mul_f32_avx512(&a[..n], &b[..n], &mut y[..n]);
        }
        return;
    }
    let _ = profile;
    mul_f32_x8(&a[..n], &b[..n], &mut y[..n]);
}

#[inline]
fn dot_f32_x8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + 8 <= n {
        let av = f32x8::from(*<&[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let bv = f32x8::from(*<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        acc += av * bv;
        i += 8;
    }
    let mut s = acc.reduce_add();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
fn dot_f32_f16_x8(a: &[f32], b: &[f16]) -> f32 {
    let n = a.len();
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + 8 <= n {
        let av = f32x8::from(*<&[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let bv = f32x8::new([
            b[i].to_f32(),
            b[i + 1].to_f32(),
            b[i + 2].to_f32(),
            b[i + 3].to_f32(),
            b[i + 4].to_f32(),
            b[i + 5].to_f32(),
            b[i + 6].to_f32(),
            b[i + 7].to_f32(),
        ]);
        acc += av * bv;
        i += 8;
    }
    let mut s = acc.reduce_add();
    while i < n {
        s += a[i] * b[i].to_f32();
        i += 1;
    }
    s
}

#[inline]
fn axpy_f32_x8(out: &mut [f32], v: &[f32], scale: f32) {
    let n = out.len();
    let sv = f32x8::splat(scale);
    let mut i = 0;
    while i + 8 <= n {
        let ov = f32x8::from(*<&[f32; 8]>::try_from(&out[i..i + 8]).unwrap());
        let vv = f32x8::from(*<&[f32; 8]>::try_from(&v[i..i + 8]).unwrap());
        let r = ov + vv * sv;
        let arr: [f32; 8] = r.into();
        out[i..i + 8].copy_from_slice(&arr);
        i += 8;
    }
    while i < n {
        out[i] += scale * v[i];
        i += 1;
    }
}

#[inline]
fn axpy_f32_f16_x8(out: &mut [f32], v: &[f16], scale: f32) {
    let n = out.len();
    let sv = f32x8::splat(scale);
    let mut i = 0;
    while i + 8 <= n {
        let ov = f32x8::from(*<&[f32; 8]>::try_from(&out[i..i + 8]).unwrap());
        let vv = f32x8::new([
            v[i].to_f32(),
            v[i + 1].to_f32(),
            v[i + 2].to_f32(),
            v[i + 3].to_f32(),
            v[i + 4].to_f32(),
            v[i + 5].to_f32(),
            v[i + 6].to_f32(),
            v[i + 7].to_f32(),
        ]);
        let r = ov + vv * sv;
        let arr: [f32; 8] = r.into();
        out[i..i + 8].copy_from_slice(&arr);
        i += 8;
    }
    while i < n {
        out[i] += scale * v[i].to_f32();
        i += 1;
    }
}

#[inline]
fn scale_f32_x8(out: &mut [f32], scale: f32) {
    let n = out.len();
    let sv = f32x8::splat(scale);
    let mut i = 0;
    while i + 8 <= n {
        let ov = f32x8::from(*<&[f32; 8]>::try_from(&out[i..i + 8]).unwrap());
        let r = ov * sv;
        let arr: [f32; 8] = r.into();
        out[i..i + 8].copy_from_slice(&arr);
        i += 8;
    }
    while i < n {
        out[i] *= scale;
        i += 1;
    }
}

#[inline]
fn add_f32_x8(a: &[f32], b: &[f32], y: &mut [f32]) {
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        let av = f32x8::from(*<&[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let bv = f32x8::from(*<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        let r = av + bv;
        let arr: [f32; 8] = r.into();
        y[i..i + 8].copy_from_slice(&arr);
        i += 8;
    }
    while i < n {
        y[i] = a[i] + b[i];
        i += 1;
    }
}

#[inline]
fn mul_f32_x8(a: &[f32], b: &[f32], y: &mut [f32]) {
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        let av = f32x8::from(*<&[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let bv = f32x8::from(*<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        let r = av * bv;
        let arr: [f32; 8] = r.into();
        y[i..i + 8].copy_from_slice(&arr);
        i += 8;
    }
    while i < n {
        y[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(av, bv, acc);
        i += 16;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_f16_avx512(a: &[f32], b: &[f16]) -> f32 {
    // Convert f16 → f32 scalar then load; avoids requiring AVX-512-FP16.
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    let mut tmp = [0.0f32; 16];
    while i + 16 <= n {
        for lane in 0..16 {
            tmp[lane] = b[i + lane].to_f32();
        }
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(tmp.as_ptr());
        acc = _mm512_fmadd_ps(av, bv, acc);
        i += 16;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    while i < n {
        s += a[i] * b[i].to_f32();
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn axpy_f32_avx512(out: &mut [f32], v: &[f32], scale: f32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let sv = _mm512_set1_ps(scale);
    let mut i = 0;
    while i + 16 <= n {
        let ov = _mm512_loadu_ps(out.as_ptr().add(i));
        let vv = _mm512_loadu_ps(v.as_ptr().add(i));
        let r = _mm512_fmadd_ps(vv, sv, ov);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < n {
        out[i] += scale * v[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn axpy_f32_f16_avx512(out: &mut [f32], v: &[f16], scale: f32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let sv = _mm512_set1_ps(scale);
    let mut i = 0;
    let mut tmp = [0.0f32; 16];
    while i + 16 <= n {
        for lane in 0..16 {
            tmp[lane] = v[i + lane].to_f32();
        }
        let ov = _mm512_loadu_ps(out.as_ptr().add(i));
        let vv = _mm512_loadu_ps(tmp.as_ptr());
        let r = _mm512_fmadd_ps(vv, sv, ov);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < n {
        out[i] += scale * v[i].to_f32();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scale_f32_avx512(out: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let sv = _mm512_set1_ps(scale);
    let mut i = 0;
    while i + 16 <= n {
        let ov = _mm512_loadu_ps(out.as_ptr().add(i));
        let r = _mm512_mul_ps(ov, sv);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < n {
        out[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_f32_avx512(a: &[f32], b: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = y.len();
    let mut i = 0;
    while i + 16 <= n {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        let r = _mm512_add_ps(av, bv);
        _mm512_storeu_ps(y.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < n {
        y[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn mul_f32_avx512(a: &[f32], b: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = y.len();
    let mut i = 0;
    while i + 16 <= n {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        let r = _mm512_mul_ps(av, bv);
        _mm512_storeu_ps(y.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < n {
        y[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let mut expected = 0.0f32;
        for i in 0..a.len() {
            expected += a[i] * b[i];
        }
        let profile = CpuHardwareProfile::detect();
        let got = dot_f32(&a, &b, &profile);
        assert!(
            (got - expected).abs() < 1e-4,
            "got {got} expected {expected}"
        );
    }

    #[test]
    fn add_mul_match_scalar() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..19).map(|i| (i as f32) * 0.5).collect();
        let mut y_add = vec![0.0f32; 19];
        let mut y_mul = vec![0.0f32; 19];
        let profile = CpuHardwareProfile::detect();
        add_f32(&a, &b, &mut y_add, &profile);
        mul_f32(&a, &b, &mut y_mul, &profile);
        for i in 0..19 {
            assert!((y_add[i] - (a[i] + b[i])).abs() < 1e-6);
            assert!((y_mul[i] - (a[i] * b[i])).abs() < 1e-6);
        }
    }
}
