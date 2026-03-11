use serde::{Deserialize, Serialize};

use crate::config::SessionConfig;

/// Model-specific context parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub context_window: u32,
    pub max_output_tokens: u32,
    /// Fraction of context window that's actually usable before degradation.
    /// 0.60 for most local models, 0.85 for strong cloud models.
    pub effective_fraction: f32,
}

impl ModelProfile {
    /// 7-8B models, 8K context, 2K output, 0.55 effective.
    pub fn local_small() -> Self {
        Self {
            name: "local_small".into(),
            context_window: 8192,
            max_output_tokens: 2048,
            effective_fraction: 0.55,
        }
    }

    /// 14-32B models, 32K context, 4K output, 0.60 effective.
    pub fn local_medium() -> Self {
        Self {
            name: "local_medium".into(),
            context_window: 32768,
            max_output_tokens: 4096,
            effective_fraction: 0.60,
        }
    }

    /// 70B models, 128K context, 16K output, 0.60 effective.
    pub fn local_large() -> Self {
        Self {
            name: "local_large".into(),
            context_window: 131072,
            max_output_tokens: 16384,
            effective_fraction: 0.60,
        }
    }

    /// Strong cloud models (Claude Opus, GPT-4), 200K context, 16K output, 0.85 effective.
    pub fn cloud_strong() -> Self {
        Self {
            name: "cloud_strong".into(),
            context_window: 200000,
            max_output_tokens: 16384,
            effective_fraction: 0.85,
        }
    }

    /// Weaker cloud models (Haiku-class), 32K context, 4K output, 0.70 effective.
    pub fn cloud_weak() -> Self {
        Self {
            name: "cloud_weak".into(),
            context_window: 32768,
            max_output_tokens: 4096,
            effective_fraction: 0.70,
        }
    }

    /// Look up a profile by name.
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "local_small" => Some(Self::local_small()),
            "local_medium" => Some(Self::local_medium()),
            "local_large" => Some(Self::local_large()),
            "cloud_strong" => Some(Self::cloud_strong()),
            "cloud_weak" => Some(Self::cloud_weak()),
            _ => None,
        }
    }
}

/// Budget thresholds computed from a model profile.
#[derive(Debug, Clone)]
pub struct SessionBudget {
    pub context_window: u32,
    pub max_output: u32,
    /// context_window * effective_fraction - max_output
    pub usable_budget: u32,
    /// 50% of usable (default)
    pub sliding_window_trigger: u32,
    /// 60% of usable (default)
    pub hard_reset_trigger: u32,
    /// 85% of usable (last resort)
    pub compaction_trigger: u32,
    /// Minimum tokens to preserve in tail (default 6000)
    pub tail_tokens: u32,
    /// Minimum turns to preserve in tail (default 2)
    pub min_tail_turns: usize,
}

impl SessionBudget {
    /// Compute budget from a model profile with default trigger fractions.
    pub fn from_profile(profile: &ModelProfile) -> Self {
        Self::compute(
            profile.context_window,
            profile.max_output_tokens,
            profile.effective_fraction,
            0.50,
            0.60,
            0.85,
            default_tail_tokens(profile),
            2,
        )
    }

    /// Compute budget from a full session config.
    pub fn from_config(config: &SessionConfig) -> Self {
        let profile = config.resolve_profile();
        let effective = config.effective_fraction.unwrap_or(profile.effective_fraction);
        let tail = if config.tail_tokens > 0 {
            config.tail_tokens
        } else {
            default_tail_tokens(&profile)
        };
        Self::compute(
            profile.context_window,
            profile.max_output_tokens,
            effective,
            config.sliding_window_fraction,
            config.hard_reset_fraction,
            0.85,
            tail,
            config.min_tail_turns,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compute(
        context_window: u32,
        max_output: u32,
        effective_fraction: f32,
        sliding_frac: f32,
        reset_frac: f32,
        compaction_frac: f32,
        tail_tokens: u32,
        min_tail_turns: usize,
    ) -> Self {
        let effective_ctx = (context_window as f64 * effective_fraction as f64) as u32;
        let usable = effective_ctx.saturating_sub(max_output);
        Self {
            context_window,
            max_output,
            usable_budget: usable,
            sliding_window_trigger: (usable as f64 * sliding_frac as f64) as u32,
            hard_reset_trigger: (usable as f64 * reset_frac as f64) as u32,
            compaction_trigger: (usable as f64 * compaction_frac as f64) as u32,
            tail_tokens,
            min_tail_turns,
        }
    }

    /// Which action should be taken given current token usage?
    pub fn action_needed(&self, current_tokens: u32) -> BudgetAction {
        if current_tokens >= self.compaction_trigger {
            BudgetAction::Compaction
        } else if current_tokens >= self.hard_reset_trigger {
            BudgetAction::HardReset
        } else if current_tokens >= self.sliding_window_trigger {
            BudgetAction::SlidingWindow
        } else {
            BudgetAction::None
        }
    }
}

/// Default tail_tokens based on model size.
fn default_tail_tokens(profile: &ModelProfile) -> u32 {
    if profile.context_window <= 8192 {
        2000
    } else if profile.context_window <= 32768 {
        4000
    } else {
        6000
    }
}

/// Action to take based on token budget thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetAction {
    /// Under threshold — no action needed.
    None,
    /// Sliding window: trim oldest turns.
    SlidingWindow,
    /// Hard reset: rebuild from scratch.
    HardReset,
    /// LLM compaction: last resort.
    Compaction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_small_budget_math() {
        let profile = ModelProfile::local_small();
        let budget = SessionBudget::from_profile(&profile);
        // 8192 * 0.55 = 4505.6 → 4505, minus 2048 = 2457
        assert_eq!(budget.usable_budget, 2457);
        assert_eq!(budget.tail_tokens, 2000);
    }

    #[test]
    fn test_local_large_budget_math() {
        let profile = ModelProfile::local_large();
        let budget = SessionBudget::from_profile(&profile);
        // 131072 * 0.60 = 78643.2 → 78643, minus 16384 = 62259
        assert_eq!(budget.usable_budget, 62259);
        assert_eq!(budget.tail_tokens, 6000);
    }

    #[test]
    fn test_cloud_strong_budget_math() {
        let profile = ModelProfile::cloud_strong();
        let budget = SessionBudget::from_profile(&profile);
        // 200000 * 0.85 = 170000, minus 16384 = 153616
        assert_eq!(budget.usable_budget, 153616);
        assert_eq!(budget.tail_tokens, 6000);
    }

    #[test]
    fn test_action_needed_none() {
        let budget = SessionBudget::from_profile(&ModelProfile::local_large());
        assert_eq!(budget.action_needed(0), BudgetAction::None);
        assert_eq!(budget.action_needed(1000), BudgetAction::None);
        // Just below 50% trigger
        assert_eq!(
            budget.action_needed(budget.sliding_window_trigger - 1),
            BudgetAction::None
        );
    }

    #[test]
    fn test_action_needed_sliding_window() {
        let budget = SessionBudget::from_profile(&ModelProfile::local_large());
        assert_eq!(
            budget.action_needed(budget.sliding_window_trigger),
            BudgetAction::SlidingWindow
        );
        assert_eq!(
            budget.action_needed(budget.sliding_window_trigger + 100),
            BudgetAction::SlidingWindow
        );
        // Just below hard reset
        assert_eq!(
            budget.action_needed(budget.hard_reset_trigger - 1),
            BudgetAction::SlidingWindow
        );
    }

    #[test]
    fn test_action_needed_hard_reset() {
        let budget = SessionBudget::from_profile(&ModelProfile::local_large());
        assert_eq!(
            budget.action_needed(budget.hard_reset_trigger),
            BudgetAction::HardReset
        );
        assert_eq!(
            budget.action_needed(budget.compaction_trigger - 1),
            BudgetAction::HardReset
        );
    }

    #[test]
    fn test_action_needed_compaction() {
        let budget = SessionBudget::from_profile(&ModelProfile::local_large());
        assert_eq!(
            budget.action_needed(budget.compaction_trigger),
            BudgetAction::Compaction
        );
        assert_eq!(
            budget.action_needed(budget.usable_budget),
            BudgetAction::Compaction
        );
    }

    #[test]
    fn test_from_config_custom_profile() {
        use crate::config::{ModelProfileConfig, SessionConfig};
        let config = SessionConfig {
            model: ModelProfileConfig::Custom {
                context_window: 65536,
                max_output_tokens: 8192,
                effective_fraction: 0.65,
            },
            effective_fraction: None,
            sliding_window_fraction: 0.50,
            hard_reset_fraction: 0.60,
            tail_tokens: 5000,
            min_tail_turns: 3,
        };
        let budget = SessionBudget::from_config(&config);
        // 65536 * 0.65 = 42598.4 → 42598, minus 8192 = 34406
        assert_eq!(budget.usable_budget, 34406);
        assert_eq!(budget.tail_tokens, 5000);
        assert_eq!(budget.min_tail_turns, 3);
    }

    #[test]
    fn test_from_config_preset() {
        use crate::config::{ModelProfileConfig, SessionConfig};
        let config = SessionConfig {
            model: ModelProfileConfig::Preset("cloud_weak".into()),
            effective_fraction: None,
            sliding_window_fraction: 0.55,
            hard_reset_fraction: 0.65,
            tail_tokens: 0, // use default
            min_tail_turns: 2,
        };
        let budget = SessionBudget::from_config(&config);
        // cloud_weak: 32768 * 0.70 = 22937.6 → 22937, minus 4096 = 18841
        assert_eq!(budget.usable_budget, 18841);
        // Trigger fractions: 55% and 65% of usable
        assert_eq!(budget.sliding_window_trigger, (18841.0 * 0.55) as u32);
        assert_eq!(budget.hard_reset_trigger, (18841.0 * 0.65) as u32);
        assert_eq!(budget.tail_tokens, 4000); // default for 32K model
    }

    #[test]
    fn test_profile_by_name() {
        assert!(ModelProfile::by_name("local_small").is_some());
        assert!(ModelProfile::by_name("local_medium").is_some());
        assert!(ModelProfile::by_name("local_large").is_some());
        assert!(ModelProfile::by_name("cloud_strong").is_some());
        assert!(ModelProfile::by_name("cloud_weak").is_some());
        assert!(ModelProfile::by_name("unknown_model").is_none());
    }

    #[test]
    fn test_local_medium_budget() {
        let profile = ModelProfile::local_medium();
        let budget = SessionBudget::from_profile(&profile);
        // 32768 * 0.60 = 19660.8 → 19660, minus 4096 = 15564
        assert_eq!(budget.usable_budget, 15564);
        assert_eq!(budget.tail_tokens, 4000);
    }

    #[test]
    fn test_cloud_weak_budget() {
        let profile = ModelProfile::cloud_weak();
        let budget = SessionBudget::from_profile(&profile);
        // 32768 * 0.70 = 22937.6 → 22937, minus 4096 = 18841
        assert_eq!(budget.usable_budget, 18841);
    }

    #[test]
    fn test_budget_zero_usable_degenerate() {
        // max_output >= effective context → usable = 0 via saturating_sub
        let profile = ModelProfile {
            name: "degenerate".into(),
            context_window: 4096,
            max_output_tokens: 4096,
            effective_fraction: 0.50,
        };
        let budget = SessionBudget::from_profile(&profile);
        // 4096 * 0.50 = 2048, minus 4096 → saturates to 0
        assert_eq!(budget.usable_budget, 0);
        assert_eq!(budget.sliding_window_trigger, 0);
        assert_eq!(budget.hard_reset_trigger, 0);
        assert_eq!(budget.compaction_trigger, 0);
        // Any tokens at all → compaction (all triggers are 0)
        assert_eq!(budget.action_needed(0), BudgetAction::Compaction);
        assert_eq!(budget.action_needed(1), BudgetAction::Compaction);
    }

    #[test]
    fn test_budget_tail_tokens_exceeds_usable() {
        // tail_tokens > usable_budget — degenerate config, should not panic
        use crate::config::{ModelProfileConfig, SessionConfig};
        let config = SessionConfig {
            model: ModelProfileConfig::Preset("local_small".into()),
            effective_fraction: None,
            sliding_window_fraction: 0.50,
            hard_reset_fraction: 0.60,
            tail_tokens: 10000, // local_small usable is only ~2457
            min_tail_turns: 2,
        };
        let budget = SessionBudget::from_config(&config);
        assert_eq!(budget.tail_tokens, 10000);
        assert!(budget.tail_tokens > budget.usable_budget);
        // Budget still computes, no panic
    }

    #[test]
    fn test_budget_triggers_ordered() {
        // sliding < hard_reset < compaction for all standard profiles
        for name in &["local_small", "local_medium", "local_large", "cloud_strong", "cloud_weak"] {
            let profile = ModelProfile::by_name(name).unwrap();
            let budget = SessionBudget::from_profile(&profile);
            assert!(
                budget.sliding_window_trigger <= budget.hard_reset_trigger,
                "{name}: sliding {} > hard_reset {}",
                budget.sliding_window_trigger,
                budget.hard_reset_trigger
            );
            assert!(
                budget.hard_reset_trigger <= budget.compaction_trigger,
                "{name}: hard_reset {} > compaction {}",
                budget.hard_reset_trigger,
                budget.compaction_trigger
            );
            assert!(
                budget.compaction_trigger <= budget.usable_budget,
                "{name}: compaction {} > usable {}",
                budget.compaction_trigger,
                budget.usable_budget
            );
        }
    }

    #[test]
    fn test_effective_fraction_override() {
        use crate::config::{ModelProfileConfig, SessionConfig};
        let config = SessionConfig {
            model: ModelProfileConfig::Preset("local_large".into()),
            effective_fraction: Some(0.50), // Override from 0.60
            sliding_window_fraction: 0.50,
            hard_reset_fraction: 0.60,
            tail_tokens: 6000,
            min_tail_turns: 2,
        };
        let budget = SessionBudget::from_config(&config);
        // 131072 * 0.50 = 65536, minus 16384 = 49152
        assert_eq!(budget.usable_budget, 49152);
    }
}
