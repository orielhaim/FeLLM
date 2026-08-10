//! Declarative metadata embedded in a FeLLM plugin library.

use fellm_core::error::{FellmError, Result};
use serde::Deserialize;

/// The manifest schema understood by this host.
pub const MANIFEST_SCHEMA: u32 = 1;

/// A component kind declared by a plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginComponentKind {
    /// A backend implementation or backend metadata provider.
    Backend,
    /// A model architecture provider.
    Architecture,
    /// A kernel provider registered through the kernel vtable.
    Kernels,
    /// A capability provider registered through the capability vtable.
    Capability,
    /// A future component kind not understood by this host yet.
    Unknown(String),
}

impl PluginComponentKind {
    fn parse(value: &str) -> Self {
        match value {
            "backend" => Self::Backend,
            "architecture" => Self::Architecture,
            "kernels" | "kernel-provider" => Self::Kernels,
            "capability" | "capabilities" => Self::Capability,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

/// One item in a plugin's `provides` list.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginComponent {
    /// Extensible component type string from the manifest.
    #[serde(rename = "type")]
    pub component_type: String,
    /// Stable component id, when the component has one.
    #[serde(default)]
    pub id: Option<String>,
    /// Backend id for kernel-provider declarations.
    #[serde(default)]
    pub backend: Option<String>,
    /// Optional aliases for an architecture or backend id.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Optional explicit ABI entrypoint. V1 normally uses the convention for
    /// the component kind.
    #[serde(default)]
    pub entrypoint: Option<String>,
}

impl PluginComponent {
    /// Resolve the extensible JSON type to a host-known kind.
    #[must_use]
    pub fn kind(&self) -> PluginComponentKind {
        PluginComponentKind::parse(&self.component_type)
    }

    /// The identifier used for diagnostics and catalog queries.
    #[must_use]
    pub fn identifier(&self) -> Option<&str> {
        self.id.as_deref().or(self.backend.as_deref())
    }
}

/// A generic dependency declaration reserved for manifest evolution.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginRequirement {
    /// Extensible requirement type.
    #[serde(rename = "type")]
    pub requirement_type: String,
    /// Required stable id.
    pub id: String,
}

/// Parsed declarative metadata for one discovered plugin.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    /// V1 schema field used by the original proposal.
    #[serde(default)]
    pub schema: Option<u32>,
    /// V1 alias accepted for forward-compatible manifests.
    #[serde(default)]
    pub schema_version: Option<u32>,
    /// Stable plugin id.
    pub id: String,
    /// Human-readable plugin name.
    #[serde(default)]
    pub name: Option<String>,
    /// Plugin release version.
    pub version: String,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// Components explicitly provided by this plugin.
    #[serde(default)]
    pub provides: Vec<PluginComponent>,
    /// Reserved dependency declarations. Resolution is not activated in V1.
    #[serde(default)]
    pub requires: Vec<PluginRequirement>,
    /// Optional platform declarations reserved for catalog filtering.
    #[serde(default)]
    pub platforms: Vec<String>,
}

impl PluginManifest {
    /// Validate the fields required by the V1 loader.
    pub fn validate(&self) -> Result<()> {
        let schema = self
            .schema
            .or(self.schema_version)
            .ok_or_else(|| FellmError::other("plugin manifest is missing schema/schema_version"))?;
        if self.schema.is_some()
            && self.schema_version.is_some()
            && self.schema != self.schema_version
        {
            return Err(FellmError::other("plugin manifest schema fields disagree"));
        }
        if schema != MANIFEST_SCHEMA {
            return Err(FellmError::other(format!(
                "unsupported plugin manifest schema {schema}"
            )));
        }
        if self.id.trim().is_empty() {
            return Err(FellmError::other("plugin manifest id is empty"));
        }
        if self.version.trim().is_empty() {
            return Err(FellmError::other(format!(
                "plugin {} has an empty version",
                self.id
            )));
        }
        if self.provides.is_empty() {
            return Err(FellmError::other(format!(
                "plugin {} declares no provided components",
                self.id
            )));
        }
        for component in &self.provides {
            match component.kind() {
                PluginComponentKind::Backend => {
                    if component.id.as_deref().is_none_or(str::is_empty) {
                        return Err(FellmError::other(format!(
                            "plugin {} backend component is missing id",
                            self.id
                        )));
                    }
                }
                PluginComponentKind::Unknown(_) => {}
                PluginComponentKind::Architecture | PluginComponentKind::Capability => {
                    if component.id.as_deref().is_none_or(str::is_empty) {
                        return Err(FellmError::other(format!(
                            "plugin {} component {} is missing id",
                            self.id, component.component_type
                        )));
                    }
                }
                PluginComponentKind::Kernels => {
                    if component.identifier().is_none_or(str::is_empty) {
                        return Err(FellmError::other(format!(
                            "plugin {} kernels component needs id or backend",
                            self.id
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Display name used in logs.
    #[must_use]
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }
}

/// Parse and validate UTF-8 JSON manifest bytes.
pub fn parse_manifest(bytes: &[u8]) -> Result<PluginManifest> {
    let manifest: PluginManifest = serde_json::from_slice(bytes)
        .map_err(|e| FellmError::other(format!("invalid plugin manifest JSON: {e}")))?;
    manifest.validate()?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_composite_manifest_without_single_plugin_type() {
        let manifest = parse_manifest(
            br#"{
                "schema": 1,
                "id": "fellm.cuda",
                "name": "FeLLM CUDA Backend",
                "version": "0.3.0",
                "provides": [
                    {"type": "backend", "id": "cuda"},
                    {"type": "kernels", "backend": "cuda"},
                    {"type": "capability", "id": "attention.flash"}
                ]
            }"#,
        )
        .expect("manifest should parse");

        assert_eq!(manifest.id, "fellm.cuda");
        assert_eq!(manifest.provides.len(), 3);
        assert_eq!(manifest.provides[1].kind(), PluginComponentKind::Kernels);
    }

    #[test]
    fn unknown_component_types_are_preserved_for_future_schema_versions() {
        let manifest = parse_manifest(
            br#"{
                "schema_version": 1,
                "id": "fellm.future",
                "version": "1.0.0",
                "provides": [{"type": "tokenizer", "id": "sentencepiece"}]
            }"#,
        )
        .expect("future component should be accepted");

        assert_eq!(
            manifest.provides[0].kind(),
            PluginComponentKind::Unknown("tokenizer".into())
        );
    }

    #[test]
    fn missing_declared_kernel_id_is_rejected() {
        let error = parse_manifest(
            br#"{
                "schema": 1,
                "id": "fellm.invalid",
                "version": "1.0.0",
                "provides": [{"type": "kernels"}]
            }"#,
        )
        .expect_err("kernel id should be required");
        assert!(error.to_string().contains("needs id or backend"));
    }
}
