//! Activation and logit sanity checks used while locating silent-zero bugs.

use fellm_core::error::{FellmError, Result};
use std::cell::Cell;

thread_local! {
    static PROBE_STEP: Cell<bool> = const { Cell::new(false) };
}

/// Enable per-op TRACE stats for the current forward step.
pub fn set_probe_step(enabled: bool) {
    PROBE_STEP.with(|flag| flag.set(enabled));
}

#[must_use]
pub fn probe_step_enabled() -> bool {
    PROBE_STEP.with(Cell::get)
}

#[derive(Debug, Clone, Copy)]
pub struct ActivationStats {
    pub len: usize,
    pub finite: usize,
    pub zeros: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub l2: f32,
}

impl ActivationStats {
    #[must_use]
    pub fn collect(values: &[f32]) -> Self {
        let mut finite = 0usize;
        let mut zeros = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &value in values {
            if !value.is_finite() {
                continue;
            }
            finite += 1;
            if value == 0.0 {
                zeros += 1;
            }
            min = min.min(value);
            max = max.max(value);
            sum += f64::from(value);
            sum_sq += f64::from(value) * f64::from(value);
        }
        if finite == 0 {
            min = f32::NAN;
            max = f32::NAN;
        }
        Self {
            len: values.len(),
            finite,
            zeros,
            min,
            max,
            mean: if finite == 0 {
                f32::NAN
            } else {
                (sum / finite as f64) as f32
            },
            l2: sum_sq.sqrt() as f32,
        }
    }

    #[must_use]
    pub fn zero_pct(&self) -> f32 {
        if self.len == 0 {
            0.0
        } else {
            (self.zeros as f32) * 100.0 / self.len as f32
        }
    }

    #[must_use]
    pub fn all_zero(&self) -> bool {
        self.len > 0 && self.finite == self.len && self.zeros == self.len
    }
}

pub fn log_activation(name: &str, values: &[f32]) -> ActivationStats {
    let stats = ActivationStats::collect(values);
    tracing::debug!(
        name,
        len = stats.len,
        finite = stats.finite,
        min = stats.min,
        max = stats.max,
        mean = stats.mean,
        zero_pct = stats.zero_pct(),
        l2 = stats.l2,
        "activation stats"
    );
    stats
}

pub fn require_nonzero(name: &str, values: &[f32]) -> Result<ActivationStats> {
    let stats = log_activation(name, values);
    if stats.len == 0 {
        return Err(FellmError::other(format!("{name} is empty")));
    }
    if stats.finite == 0 {
        return Err(FellmError::other(format!(
            "{name} has no finite values; inference result is invalid"
        )));
    }
    if stats.all_zero() {
        if name == "logits" {
            tracing::error!("all output logits are zero; inference result is invalid");
            return Err(FellmError::other(
                "all output logits are zero; inference result is invalid",
            ));
        }
        tracing::error!("{name} is all zeros; inference result is invalid");
        return Err(FellmError::other(format!(
            "{name} is all zeros; inference result is invalid"
        )));
    }
    Ok(stats)
}

pub fn log_f32_ref(name: &str, data: *const u8, byte_len: u64) {
    if data.is_null() || byte_len < 4 {
        return;
    }
    let len = (byte_len as usize) / 4;
    // SAFETY: caller guarantees a live contiguous f32 buffer.
    let values = unsafe { std::slice::from_raw_parts(data.cast::<f32>(), len) };
    let stats = ActivationStats::collect(values);
    tracing::trace!(
        name,
        len = stats.len,
        finite = stats.finite,
        min = stats.min,
        max = stats.max,
        mean = stats.mean,
        zero_pct = stats.zero_pct(),
        l2 = stats.l2,
        "op activation"
    );
}
