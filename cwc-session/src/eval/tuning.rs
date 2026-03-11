use serde::{Deserialize, Serialize};

use crate::budget::ModelProfile;
use crate::config::{ModelProfileConfig, SessionConfig};
use crate::manager::SessionManagerConfig;
use crate::reinforcement::ReinforcementConfig;

/// Compaction aggressiveness level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompactionLevel {
    /// Minimal: only compact very large results (> 5000 tokens).
    Minimal,
    /// Moderate: compact > 1500 tokens (default).
    Moderate,
    /// Aggressive: compact > 300 tokens (for small context models).
    Aggressive,
}

impl CompactionLevel {
    /// Maximum inline tokens before compaction triggers.
    pub fn max_inline_tokens(&self) -> u32 {
        match self {
            CompactionLevel::Minimal => 5000,
            CompactionLevel::Moderate => 1500,
            CompactionLevel::Aggressive => 300,
        }
    }
}

/// A tuned profile combining model parameters with empirically-best session settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunedProfile {
    pub model: ModelProfile,
    pub compaction_aggressiveness: CompactionLevel,
    pub reinforcement: ReinforcementConfig,
    pub sliding_window_fraction: f32,
    pub hard_reset_fraction: f32,
    pub tail_tokens: u32,
    pub notes: String,
}

impl TunedProfile {
    /// Convert to a SessionManagerConfig.
    pub fn to_config(&self) -> SessionManagerConfig {
        SessionManagerConfig {
            session: SessionConfig {
                model: ModelProfileConfig::Custom {
                    context_window: self.model.context_window,
                    max_output_tokens: self.model.max_output_tokens,
                    effective_fraction: self.model.effective_fraction,
                },
                sliding_window_fraction: self.sliding_window_fraction,
                hard_reset_fraction: self.hard_reset_fraction,
                tail_tokens: self.tail_tokens,
                ..Default::default()
            },
            reinforcement: self.reinforcement.clone(),
            ..Default::default()
        }
    }
}

/// Built-in tuned profiles for common model families.
pub fn tuned_profiles() -> Vec<TunedProfile> {
    vec![
        // 7-8B models, 8K context
        TunedProfile {
            model: ModelProfile::local_small(),
            compaction_aggressiveness: CompactionLevel::Aggressive,
            reinforcement: ReinforcementConfig {
                nudge_every_n_tool_results: 1,
                include_goal: true,
                max_nudge_tokens: 80,
                enabled: true,
            },
            sliding_window_fraction: 0.40,
            hard_reset_fraction: 0.50,
            tail_tokens: 2000,
            notes: "Small models degrade fast. Aggressive compaction and frequent nudges essential. Low sliding window threshold to trim early.".into(),
        },
        // 14-32B models, 32K context
        TunedProfile {
            model: ModelProfile::local_medium(),
            compaction_aggressiveness: CompactionLevel::Moderate,
            reinforcement: ReinforcementConfig {
                nudge_every_n_tool_results: 1,
                include_goal: true,
                max_nudge_tokens: 80,
                enabled: true,
            },
            sliding_window_fraction: 0.50,
            hard_reset_fraction: 0.60,
            tail_tokens: 4000,
            notes: "Medium models handle more context but still need reinforcement. Default thresholds work well.".into(),
        },
        // 70B models, 128K context
        TunedProfile {
            model: ModelProfile::local_large(),
            compaction_aggressiveness: CompactionLevel::Moderate,
            reinforcement: ReinforcementConfig {
                nudge_every_n_tool_results: 2,
                include_goal: true,
                max_nudge_tokens: 80,
                enabled: true,
            },
            sliding_window_fraction: 0.50,
            hard_reset_fraction: 0.60,
            tail_tokens: 6000,
            notes: "Large models tolerate more context. Nudge every 2 tool results to reduce prompt overhead.".into(),
        },
        // Cloud weak (Haiku-class)
        TunedProfile {
            model: ModelProfile::cloud_weak(),
            compaction_aggressiveness: CompactionLevel::Moderate,
            reinforcement: ReinforcementConfig {
                nudge_every_n_tool_results: 3,
                include_goal: true,
                max_nudge_tokens: 80,
                enabled: true,
            },
            sliding_window_fraction: 0.55,
            hard_reset_fraction: 0.65,
            tail_tokens: 4000,
            notes: "Cloud weak models follow instructions better. Less frequent nudges, higher thresholds.".into(),
        },
        // Cloud strong (Opus/GPT-4)
        TunedProfile {
            model: ModelProfile::cloud_strong(),
            compaction_aggressiveness: CompactionLevel::Minimal,
            reinforcement: ReinforcementConfig {
                nudge_every_n_tool_results: 1, // doesn't matter, disabled
                include_goal: false,
                max_nudge_tokens: 80,
                enabled: false,
            },
            sliding_window_fraction: 0.70,
            hard_reset_fraction: 0.80,
            tail_tokens: 8000,
            notes: "Strong cloud models rarely need nudging. High thresholds, minimal compaction. Reinforcement disabled.".into(),
        },
    ]
}

/// Recommend a config given model name or parameters.
///
/// Lookup order:
/// 1. Exact model name match in tuned_profiles
/// 2. Nearest match by context_window size
pub fn recommend_config(
    model_name: Option<&str>,
    context_window: u32,
    max_output: u32,
) -> SessionManagerConfig {
    let profiles = tuned_profiles();

    // Try exact name match
    if let Some(name) = model_name {
        let name_lower = name.to_lowercase();
        for profile in &profiles {
            if profile.model.name.to_lowercase() == name_lower {
                let mut config = profile.to_config();
                config.session.model = ModelProfileConfig::Custom {
                    context_window,
                    max_output_tokens: max_output,
                    effective_fraction: profile.model.effective_fraction,
                };
                return config;
            }
        }
        // Try partial match
        for profile in &profiles {
            if name_lower.contains(&profile.model.name.to_lowercase()) {
                let mut config = profile.to_config();
                config.session.model = ModelProfileConfig::Custom {
                    context_window,
                    max_output_tokens: max_output,
                    effective_fraction: profile.model.effective_fraction,
                };
                return config;
            }
        }
    }

    // Nearest match by context window
    let best = profiles
        .iter()
        .min_by_key(|p| {
            (p.model.context_window as i64 - context_window as i64).unsigned_abs()
        })
        .unwrap(); // profiles is never empty

    // Use the matched profile but override context_window and max_output
    let mut config = best.to_config();
    config.session.model = ModelProfileConfig::Custom {
        context_window,
        max_output_tokens: max_output,
        effective_fraction: best.model.effective_fraction,
    };
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuned_profiles_all_valid() {
        let profiles = tuned_profiles();
        assert_eq!(profiles.len(), 5);
        for p in &profiles {
            assert!(p.model.context_window > 0);
            assert!(p.model.max_output_tokens > 0);
            assert!(p.sliding_window_fraction > 0.0);
            assert!(p.sliding_window_fraction < 1.0);
            assert!(p.hard_reset_fraction > p.sliding_window_fraction);
            assert!(p.tail_tokens > 0);
        }
    }

    #[test]
    fn test_recommend_config_known_name() {
        // Use different context_window/max_output than profile defaults
        // to verify the caller's values are applied, not the profile's.
        let config = recommend_config(Some("local_large"), 65536, 8192);
        if let ModelProfileConfig::Custom {
            context_window,
            max_output_tokens,
            effective_fraction,
        } = &config.session.model
        {
            assert_eq!(*context_window, 65536, "caller's context_window must be used");
            assert_eq!(*max_output_tokens, 8192, "caller's max_output must be used");
            assert!((*effective_fraction - 0.60).abs() < 0.01);
        } else {
            panic!("expected Custom model config");
        }
        // Should have nudge every 2 (large model)
        assert_eq!(config.reinforcement.nudge_every_n_tool_results, 2);
    }

    #[test]
    fn test_recommend_config_unknown_name_by_context() {
        // 16K context → nearest is local_small (8K) or local_medium (32K)
        let config = recommend_config(Some("unknown_model"), 16384, 2048);
        // Should pick local_small (closest to 16K)
        if let ModelProfileConfig::Custom {
            context_window, ..
        } = &config.session.model
        {
            assert_eq!(*context_window, 16384);
        }
    }

    #[test]
    fn test_recommend_config_no_name() {
        let config = recommend_config(None, 8192, 2048);
        // Should pick local_small (8K)
        if let ModelProfileConfig::Custom {
            effective_fraction, ..
        } = &config.session.model
        {
            assert!((*effective_fraction - 0.55).abs() < 0.01);
        }
    }

    #[test]
    fn test_compaction_level_thresholds() {
        assert_eq!(CompactionLevel::Aggressive.max_inline_tokens(), 300);
        assert_eq!(CompactionLevel::Moderate.max_inline_tokens(), 1500);
        assert_eq!(CompactionLevel::Minimal.max_inline_tokens(), 5000);
    }

    #[test]
    fn test_tuned_profile_to_config() {
        let profiles = tuned_profiles();
        for p in &profiles {
            let config = p.to_config();
            assert!(config.reinforcement.max_nudge_tokens > 0);
        }
    }
}
