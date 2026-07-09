//! RMSNorm.

use wide::f32x8;

/// `y[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)` per row.
pub fn rmsnorm_row(x: &[f32], weight: &[f32], eps: f32, y: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut sumsq = 0.0f32;
    let mut i = 0;
    let mut acc = f32x8::ZERO;
    while i + 8 <= n {
        let arr: [f32; 8] = x[i..i + 8].try_into().unwrap();
        let v = f32x8::from(arr);
        acc = acc + v * v;
        i += 8;
    }
    sumsq += acc.reduce_add();
    while i < n {
        sumsq += x[i] * x[i];
        i += 1;
    }
    let scale = 1.0 / (sumsq / n as f32 + eps).sqrt();
    let mut i = 0;
    let scale_v = f32x8::splat(scale);
    while i + 8 <= n {
        let xa: [f32; 8] = x[i..i + 8].try_into().unwrap();
        let wa: [f32; 8] = weight[i..i + 8].try_into().unwrap();
        let out = f32x8::from(xa) * scale_v * f32x8::from(wa);
        let arr: [f32; 8] = out.into();
        y[i..i + 8].copy_from_slice(&arr);
        i += 8;
    }
    while i < n {
        y[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_identity_scale() {
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let w = vec![1.0f32; 8];
        let mut y = vec![0.0f32; 8];
        rmsnorm_row(&x, &w, 1e-6, &mut y);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 8.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms = {rms}");
    }
}
