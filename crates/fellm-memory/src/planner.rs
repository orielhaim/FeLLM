use crate::{
    ConsumerBudget, FabricPlan, HardwareProfile, ModelProfile, ResidencyClass, WeightPlacement,
};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningError {
    NoDeviceWorkingSet,
    NoHostStaging,
}

/// Joint planner for weights, KV, activations, and bounded streaming buffers.
pub struct FabricPlanner;

impl FabricPlanner {
    pub fn plan(
        hardware: &HardwareProfile,
        model: &ModelProfile,
    ) -> Result<FabricPlan, PlanningError> {
        let device_reserve = (hardware.device_total / 20).max(256 << 20);
        let host_reserve = (hardware.host_total / 10).max(512 << 20);
        let device_usable = hardware.device_available.saturating_sub(device_reserve);
        let host_usable = hardware.host_available.saturating_sub(host_reserve);
        let activation = model.activation_bytes.min(device_usable);
        let after_activation = device_usable.saturating_sub(activation);

        // Preserve a useful KV budget before weights consume the device. It may be smaller than
        // the requested context, which higher-level admission control handles explicitly.
        let kv_device = model.kv_bytes.min(after_activation / 3);
        let weight_device_capacity = after_activation.saturating_sub(kv_device);
        let largest_group = model.groups.iter().map(|g| g.byte_len).max().unwrap_or(0);
        if largest_group > 0
            && hardware.device_total > 0
            && weight_device_capacity < largest_group.saturating_mul(2)
        {
            return Err(PlanningError::NoDeviceWorkingSet);
        }

        let device_buffer_count = if hardware.device_total == 0 {
            0
        } else if weight_device_capacity >= largest_group.saturating_mul(3) {
            3
        } else if weight_device_capacity >= largest_group.saturating_mul(2) {
            2
        } else {
            0
        };
        let device_buffer_bytes = if device_buffer_count == 0 {
            0
        } else {
            largest_group
        };
        let permanent_capacity = weight_device_capacity
            .saturating_sub(device_buffer_bytes.saturating_mul(u64::from(device_buffer_count)));

        let host_staging_each = largest_group.max(4 << 20).min(256 << 20);
        let host_buffer_count = 2u8;
        let host_staging = host_staging_each.saturating_mul(u64::from(host_buffer_count));
        if largest_group > 0 && host_usable < host_staging_each {
            return Err(PlanningError::NoHostStaging);
        }
        let kv_host = model
            .kv_bytes
            .saturating_sub(kv_device)
            .min(host_usable / 4);
        let host_weight_capacity = host_usable
            .saturating_sub(host_staging)
            .saturating_sub(kv_host);

        let mut candidates = model.groups.iter().collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            let av = f64::from(a.reuse_count.max(1)) / a.byte_len.max(1) as f64;
            let bv = f64::from(b.reuse_count.max(1)) / b.byte_len.max(1) as f64;
            bv.total_cmp(&av).then_with(|| a.id.cmp(&b.id))
        });
        let mut device_used = 0u64;
        let mut host_used = 0u64;
        let mut placements = Vec::with_capacity(candidates.len());
        for group in candidates {
            let h2d = transfer_time(
                group.byte_len,
                hardware.h2d_bytes_per_second,
                Duration::ZERO,
            );
            let storage = transfer_time(
                group.byte_len,
                hardware.storage_bytes_per_second,
                hardware.storage_latency,
            );
            let host_fits = host_used.saturating_add(group.byte_len) <= host_weight_capacity;
            let (class, transfer) =
                if device_used.saturating_add(group.byte_len) <= permanent_capacity {
                    device_used += group.byte_len;
                    (ResidencyClass::PermanentDevice, Duration::ZERO)
                } else if host_fits && group.cpu_compute_time.is_some_and(|cpu| cpu < h2d) {
                    (ResidencyClass::CpuCompute, Duration::ZERO)
                } else if host_fits {
                    host_used += group.byte_len;
                    (ResidencyClass::HostResident, h2d)
                } else if group
                    .cpu_compute_time
                    .is_some_and(|cpu| cpu < h2d.saturating_add(storage))
                {
                    (ResidencyClass::CpuCompute, Duration::ZERO)
                } else if hardware.transfers.gds {
                    (ResidencyClass::StorageStream, storage)
                } else {
                    (ResidencyClass::StorageStream, storage.saturating_add(h2d))
                };
            placements.push(WeightPlacement {
                group: group.id,
                class,
                bytes: group.byte_len,
                estimated_transfer: transfer,
            });
        }
        placements.sort_by_key(|p| p.group);
        Ok(FabricPlan {
            budget: ConsumerBudget {
                weights_device: device_used,
                weights_host: host_used,
                kv_device,
                kv_host,
                activation_device: activation,
                device_staging: device_buffer_bytes.saturating_mul(u64::from(device_buffer_count)),
                host_staging,
                device_reserve,
                host_reserve,
            },
            placements,
            device_buffer_count,
            device_buffer_bytes,
            host_buffer_count,
            host_buffer_bytes: host_staging_each,
        })
    }
}

fn transfer_time(bytes: u64, bandwidth: u64, latency: Duration) -> Duration {
    if bandwidth == 0 {
        return Duration::MAX;
    }
    latency.saturating_add(Duration::from_secs_f64(bytes as f64 / bandwidth as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionGroup, TransferCapabilities};
    fn group(id: u32, bytes: u64, reuse: u32) -> ExecutionGroup {
        ExecutionGroup {
            id,
            weights: vec![],
            byte_len: bytes,
            first_op: id,
            last_op: id,
            reuse_count: reuse,
            cpu_compute_time: None,
        }
    }
    #[test]
    fn independently_budgets_device_and_host_working_sets() {
        let hw = HardwareProfile {
            device_total: 8 << 30,
            device_available: 7 << 30,
            host_total: 32 << 30,
            host_available: 24 << 30,
            h2d_bytes_per_second: 24 << 30,
            storage_bytes_per_second: 3 << 30,
            storage_latency: Duration::from_micros(100),
            cpu_score: 1.0,
            transfers: TransferCapabilities {
                mmap: true,
                async_file: true,
                ..Default::default()
            },
        };
        let model = ModelProfile {
            weight_bytes: 20 << 30,
            kv_bytes: 4 << 30,
            activation_bytes: 1 << 30,
            groups: vec![
                group(0, 1 << 30, 100),
                group(1, 1 << 30, 1),
                group(2, 1 << 30, 1),
            ],
        };
        let plan = FabricPlanner::plan(&hw, &model).unwrap();
        assert!(plan.budget.kv_device > 0);
        assert!(plan.budget.device_staging > 0);
        assert!(
            plan.placements
                .iter()
                .any(|p| p.class == ResidencyClass::HostResident)
        );
    }

    #[test]
    fn rejects_streaming_slots_smaller_than_an_execution_group() {
        let hw = HardwareProfile {
            device_total: 2 << 30,
            device_available: 1500 << 20,
            host_total: 16 << 30,
            host_available: 8 << 30,
            h2d_bytes_per_second: 24 << 30,
            storage_bytes_per_second: 3 << 30,
            storage_latency: Duration::from_micros(100),
            cpu_score: 1.0,
            transfers: TransferCapabilities::default(),
        };
        let model = ModelProfile {
            weight_bytes: 1 << 30,
            kv_bytes: 256 << 20,
            activation_bytes: 700 << 20,
            groups: vec![group(0, 256 << 20, 1)],
        };
        assert!(matches!(
            FabricPlanner::plan(&hw, &model),
            Err(PlanningError::NoDeviceWorkingSet)
        ));
    }
}
