//! Capability registry: multi-provider coexistence, prepare-time handles.

use fellm_core::dtype::DType;
use fellm_plugin_abi::attention_provider::{AttentionWorkload, DeviceCapabilityView};
use fellm_plugin_abi::capability::{
    CapabilityKind, FeatureId, FeatureSet, PreparedProviderId, ProviderSelection,
};
use fellm_plugin_host::{CapabilityRegistry, HostTiledAttentionProvider};
use std::sync::Arc;

#[test]
fn two_attention_providers_coexist_and_resolve_by_name() {
    let reg = CapabilityRegistry::new();
    // Builtins already install host + cuda attention + kv.full
    let list = reg.list_capability(CapabilityKind::Attention);
    assert!(list.len() >= 2, "expected host + cuda attention providers");
    let names: Vec<_> = list.iter().map(|p| p.descriptor.name.as_str()).collect();
    assert!(names.contains(&"attention.host_tiled"));
    assert!(names.contains(&"attention.cuda_tiled"));

    let id_host = reg.id_of("attention.host_tiled").unwrap();
    let id_cuda = reg.id_of("attention.cuda_tiled").unwrap();
    assert_ne!(id_host, id_cuda);
    assert_ne!(id_host, PreparedProviderId::NONE);
}

#[test]
fn prepare_uses_prepared_ids_not_names_on_hot_path() {
    let mut reg = CapabilityRegistry::new();
    let mut device = DeviceCapabilityView::default();
    // CPU-like: no hopper → host preferred via applicability
    let features = FeatureSet::from_ids([
        FeatureId::ATTN_PAGED_KV,
        FeatureId::ATTN_DECODE,
        FeatureId::ATTN_PREFILL,
        FeatureId::ATTN_INDIRECT_POSITIONS,
        FeatureId::ATTN_PER_HEAD_KV_VIEWS,
        FeatureId::ATTN_GQA,
        FeatureId::ATTN_FP16,
    ]);
    device.features = features.clone();
    let workloads = [AttentionWorkload {
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 64,
        query_len: 1,
        kv_len: 512,
        dtype: DType::F16,
        causal: true,
        window: 0,
        paged: true,
        indirect_positions: true,
    }];
    let sel = ProviderSelection::new();
    let report = reg
        .prepare(&sel, &features, &device, &workloads, 16)
        .expect("prepare");
    let attn_name = report.selected.get(&CapabilityKind::Attention).unwrap();
    let attn_id = reg.selected(CapabilityKind::Attention).unwrap();
    assert_eq!(reg.name_of(attn_id).unwrap(), attn_name.as_str());
    // Prepared plan keyed by id + path, not by string.
    let plan = reg.prepared_attention(attn_id, 2); // Decode
    assert!(plan.is_some(), "decode plan must be prepared");
    assert_eq!(plan.unwrap().provider, attn_id);
}

#[test]
fn explicit_provider_not_found_errors() {
    let mut reg = CapabilityRegistry::new();
    let mut sel = ProviderSelection::new();
    sel.attention = Some("attention.nope".into());
    let features = FeatureSet::new();
    let device = DeviceCapabilityView::default();
    let err = reg.prepare(&sel, &features, &device, &[], 1).unwrap_err();
    assert!(err.to_string().contains("nope"));
}

#[test]
fn paged_flash_decode_not_a_macro_kind_name() {
    // Structural: open MacroOpKind must expose semantic attention names.
    use fellm_plugin_abi::MacroOpKind;
    assert_eq!(
        MacroOpKind::PagedAttentionDecode.name(),
        "paged_attention_decode"
    );
    // The old algorithm-branded name must not exist as a constant.
    // (compile-time: PagedFlashDecode removed; runtime string check)
    assert!(!MacroOpKind::PagedAttentionDecode.name().contains("flash"));
}

#[test]
fn host_provider_register_extra_instance() {
    // Multiple implementations of same capability can be registered under
    // different names.
    let mut reg = CapabilityRegistry::new();
    // Host is already registered; re-registering same name fails.
    let again = Arc::new(HostTiledAttentionProvider::new());
    let err = reg.register_attention_provider(again);
    assert!(err.is_err());
}
