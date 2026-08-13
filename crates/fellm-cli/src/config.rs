use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FellmConfig {
    pub log: Option<String>,
    #[serde(default)]
    pub run: RunConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub plugins: PluginsConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    pub device_memory_limit: Option<u64>,
    pub host_memory_limit: Option<u64>,
    pub h2d_bytes_per_second: Option<u64>,
    pub storage_bytes_per_second: Option<u64>,
    pub storage_latency_micros: Option<u64>,
    pub storage_provider: Option<String>,
    pub host_weight_cache: Option<u64>,
    pub storage_overlap: Option<bool>,
    pub router_trace_capacity: Option<usize>,
    pub disable_cpu_partitions: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub model: Option<PathBuf>,
    pub prompt: Option<String>,
    pub system: Option<String>,
    pub completion: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub repetition_penalty: Option<f32>,
    pub ctx_size: Option<usize>,
    pub batch_size: Option<usize>,
    pub ubatch_size: Option<usize>,
    pub backend: Option<String>,
    pub cpu_fallback: Option<bool>,
    pub attention: Option<String>,
    pub kv_policy: Option<String>,
    pub plugin_config: Option<Vec<String>>,
    pub plugin_dir: Option<PathBuf>,
    pub kv_device_budget: Option<u64>,
    pub kv_host_budget: Option<u64>,
    pub kv_memory_fraction: Option<f64>,
    pub kv_safety_reserve_bytes: Option<u64>,
    pub kv_mode: Option<String>,
    pub kv_addressing: Option<String>,
    pub kv_prefix_sharing: Option<bool>,
    pub kv_prefetch: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginsConfig {
    pub plugin_dir: Option<PathBuf>,
}

impl FellmConfig {
    pub fn load(path: &Path, explicitly_selected: bool) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let mut config: Self = toml::from_str(&text)
                    .map_err(|error| format!("invalid config {}: {error}", path.display()))?;
                let base = path.parent().unwrap_or_else(|| Path::new("."));
                resolve_relative(&mut config.run.model, base);
                resolve_relative(&mut config.run.plugin_dir, base);
                resolve_relative(&mut config.plugins.plugin_dir, base);
                Ok(config)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !explicitly_selected => {
                Ok(Self::default())
            }
            Err(error) => Err(format!("cannot read config {}: {error}", path.display())),
        }
    }
}

fn resolve_relative(path: &mut Option<PathBuf>, base: &Path) {
    if let Some(value) = path {
        if value.is_relative() {
            *value = base.join(&*value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_configuration() {
        assert!(toml::from_str::<FellmConfig>("[run]\nunknown = true").is_err());
    }

    #[test]
    fn deserializes_all_run_sections() {
        let config: FellmConfig = toml::from_str(
            r#"
                log = "debug"
                [run]
                model = "model.gguf"
                prompt = "hello"
                backend = "cpu"
                kv_prefix_sharing = false
                plugin_config = ["provider.key=value"]
                [plugins]
                plugin_dir = "plugins"
            "#,
        )
        .unwrap();
        assert_eq!(config.run.backend.as_deref(), Some("cpu"));
        assert_eq!(config.run.kv_prefix_sharing, Some(false));
        assert_eq!(config.run.plugin_config.unwrap().len(), 1);
    }
}
