use serde::{Deserialize, Serialize};

use crate::budget::ModelProfile;

/// How to specify the model: by preset name or custom parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelProfileConfig {
    /// A preset name like "local_small", "local_large", "cloud_strong".
    Preset(String),
    /// Custom model parameters.
    Custom {
        context_window: u32,
        max_output_tokens: u32,
        effective_fraction: f32,
    },
}

/// Session management configuration, loadable from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Model profile name or custom parameters.
    pub model: ModelProfileConfig,

    /// Override the profile's effective_fraction if set.
    #[serde(default)]
    pub effective_fraction: Option<f32>,

    /// Sliding window trigger as fraction of usable budget (default 0.50).
    #[serde(default = "default_sliding")]
    pub sliding_window_fraction: f32,

    /// Hard reset trigger as fraction of usable budget (default 0.60).
    #[serde(default = "default_reset")]
    pub hard_reset_fraction: f32,

    /// Minimum tokens to keep in the tail after trim (0 = use profile default).
    #[serde(default)]
    pub tail_tokens: u32,

    /// Minimum turns to keep in the tail after trim (default 2).
    #[serde(default = "default_min_tail_turns")]
    pub min_tail_turns: usize,
}

fn default_sliding() -> f32 {
    0.50
}

fn default_reset() -> f32 {
    0.60
}

fn default_min_tail_turns() -> usize {
    2
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model: ModelProfileConfig::Preset("local_large".into()),
            effective_fraction: None,
            sliding_window_fraction: 0.50,
            hard_reset_fraction: 0.60,
            tail_tokens: 0,
            min_tail_turns: 2,
        }
    }
}

impl SessionConfig {
    /// Resolve the model profile from the config.
    pub fn resolve_profile(&self) -> ModelProfile {
        match &self.model {
            ModelProfileConfig::Preset(name) => {
                ModelProfile::by_name(name).unwrap_or_else(|| {
                    tracing::warn!("unknown model profile '{name}', falling back to local_large");
                    ModelProfile::local_large()
                })
            }
            ModelProfileConfig::Custom {
                context_window,
                max_output_tokens,
                effective_fraction,
            } => ModelProfile {
                name: "custom".into(),
                context_window: *context_window,
                max_output_tokens: *max_output_tokens,
                effective_fraction: *effective_fraction,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.sliding_window_fraction, 0.50);
        assert_eq!(config.hard_reset_fraction, 0.60);
        assert_eq!(config.min_tail_turns, 2);
        assert_eq!(config.tail_tokens, 0);
        assert!(config.effective_fraction.is_none());
    }

    #[test]
    fn test_config_deserialize_preset() {
        let toml_str = r#"
model = "cloud_strong"
sliding_window_fraction = 0.55
hard_reset_fraction = 0.65
tail_tokens = 5000
min_tail_turns = 3
"#;
        let config: SessionConfig = toml::from_str(toml_str).expect("parse TOML");
        match &config.model {
            ModelProfileConfig::Preset(name) => assert_eq!(name, "cloud_strong"),
            _ => panic!("expected Preset"),
        }
        assert_eq!(config.sliding_window_fraction, 0.55);
        assert_eq!(config.hard_reset_fraction, 0.65);
        assert_eq!(config.tail_tokens, 5000);
        assert_eq!(config.min_tail_turns, 3);
    }

    #[test]
    fn test_config_deserialize_custom_model() {
        let toml_str = r#"
[model]
context_window = 65536
max_output_tokens = 8192
effective_fraction = 0.65
"#;
        let config: SessionConfig = toml::from_str(toml_str).expect("parse TOML");
        match &config.model {
            ModelProfileConfig::Custom {
                context_window,
                max_output_tokens,
                effective_fraction,
            } => {
                assert_eq!(*context_window, 65536);
                assert_eq!(*max_output_tokens, 8192);
                assert!((effective_fraction - 0.65).abs() < f32::EPSILON);
            }
            _ => panic!("expected Custom"),
        }
        // Defaults should apply
        assert_eq!(config.sliding_window_fraction, 0.50);
        assert_eq!(config.hard_reset_fraction, 0.60);
        assert_eq!(config.min_tail_turns, 2);
    }

    #[test]
    fn test_config_resolve_profile_preset() {
        let config = SessionConfig {
            model: ModelProfileConfig::Preset("local_small".into()),
            ..SessionConfig::default()
        };
        let profile = config.resolve_profile();
        assert_eq!(profile.name, "local_small");
        assert_eq!(profile.context_window, 8192);
    }

    #[test]
    fn test_config_resolve_profile_custom() {
        let config = SessionConfig {
            model: ModelProfileConfig::Custom {
                context_window: 65536,
                max_output_tokens: 8192,
                effective_fraction: 0.65,
            },
            ..SessionConfig::default()
        };
        let profile = config.resolve_profile();
        assert_eq!(profile.name, "custom");
        assert_eq!(profile.context_window, 65536);
    }

    #[test]
    fn test_config_resolve_profile_unknown_preset() {
        let config = SessionConfig {
            model: ModelProfileConfig::Preset("nonexistent".into()),
            ..SessionConfig::default()
        };
        let profile = config.resolve_profile();
        // Falls back to local_large
        assert_eq!(profile.name, "local_large");
    }

    #[test]
    fn test_config_deserialize_with_effective_fraction_override() {
        let toml_str = r#"
model = "local_large"
effective_fraction = 0.50
"#;
        let config: SessionConfig = toml::from_str(toml_str).expect("parse TOML");
        assert_eq!(config.effective_fraction, Some(0.50));
    }

    #[test]
    fn test_config_serialize_roundtrip() {
        let config = SessionConfig {
            model: ModelProfileConfig::Preset("cloud_weak".into()),
            effective_fraction: Some(0.75),
            sliding_window_fraction: 0.55,
            hard_reset_fraction: 0.65,
            tail_tokens: 4000,
            min_tail_turns: 3,
        };
        let toml_str = toml::to_string(&config).expect("serialize");
        let back: SessionConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back.effective_fraction, Some(0.75));
        assert_eq!(back.sliding_window_fraction, 0.55);
        assert_eq!(back.tail_tokens, 4000);
    }
}
