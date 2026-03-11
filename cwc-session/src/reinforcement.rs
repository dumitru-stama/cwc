use serde::{Deserialize, Serialize};

use crate::memory::SessionMemoryStore;
use crate::message::{MessageFlags, SessionMessage, SessionRole};

/// Templates for post-tool-result nudges.
/// Rotated to prevent the model from learning to ignore them.
const NUDGE_TEMPLATES: &[&str] = &[
    "Continue with your analysis. Do not repeat previous findings.",
    "Proceed to your next action. Build on what you've already discovered.",
    "What is your next step? Do not re-read files you have already seen.",
    "Continue working toward the goal. Use your findings so far.",
];

/// Configuration for reinforcement behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforcementConfig {
    /// Inject a nudge after every N tool results (default 1 = every tool result).
    pub nudge_every_n_tool_results: usize,
    /// Include goal text in nudges (default true).
    pub include_goal: bool,
    /// Maximum nudge length in tokens (default 80).
    pub max_nudge_tokens: u32,
    /// Enable reinforcement (can be disabled for cloud models).
    pub enabled: bool,
}

impl Default for ReinforcementConfig {
    fn default() -> Self {
        Self {
            nudge_every_n_tool_results: 1,
            include_goal: true,
            max_nudge_tokens: 80,
            enabled: true,
        }
    }
}

/// Generate a reinforcement nudge message.
///
/// Selects from NUDGE_TEMPLATES using a rotating index based on tool_result_count.
/// Appends the goal text if available and configured.
pub fn create_nudge(
    goal: Option<&str>,
    tool_result_count: usize,
    config: &ReinforcementConfig,
) -> SessionMessage {
    let template_idx = tool_result_count % NUDGE_TEMPLATES.len();
    let mut text = NUDGE_TEMPLATES[template_idx].to_string();

    if config.include_goal {
        if let Some(goal_text) = goal {
            if !goal_text.is_empty() {
                text.push_str(&format!("\nYour goal: {goal_text}"));
            }
        }
    }

    let mut msg = SessionMessage::text(SessionRole::User, text);
    msg.flags.insert(MessageFlags::IS_NUDGE);
    msg
}

/// Generate a nudge from a memory store (convenience wrapper).
pub fn create_nudge_from_store(
    store: &SessionMemoryStore,
    tool_result_count: usize,
    config: &ReinforcementConfig,
) -> SessionMessage {
    let goal = store.get("goal").map(|f| f.value.as_str());
    create_nudge(goal, tool_result_count, config)
}

/// Count tool results since the last nudge in the message list.
pub fn tool_results_since_last_nudge(messages: &[SessionMessage]) -> usize {
    let mut count = 0;
    for msg in messages.iter().rev() {
        if msg.flags.contains(MessageFlags::IS_NUDGE) {
            break;
        }
        if msg.role == SessionRole::Tool {
            count += 1;
        }
    }
    count
}

/// Determine if a nudge should be injected right now.
pub fn should_nudge(messages: &[SessionMessage], config: &ReinforcementConfig) -> bool {
    if !config.enabled {
        return false;
    }
    if config.nudge_every_n_tool_results == 0 {
        return false;
    }
    // Don't nudge if the last message is already a nudge
    if let Some(last) = messages.last() {
        if last.flags.contains(MessageFlags::IS_NUDGE) {
            return false;
        }
    }
    let count = tool_results_since_last_nudge(messages);
    count >= config.nudge_every_n_tool_results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ToolCall, ToolResult};

    fn tool_call_msg() -> SessionMessage {
        SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "file.grep".into(),
                arguments: serde_json::json!({}),
            }],
        )
    }

    fn tool_result_msg() -> SessionMessage {
        SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "file.grep".into(),
            output: "results".into(),
            is_error: false,
        })
    }

    #[test]
    fn test_create_nudge_rotates_templates() {
        let config = ReinforcementConfig {
            include_goal: false,
            ..Default::default()
        };
        let n0 = create_nudge(None, 0, &config);
        let n1 = create_nudge(None, 1, &config);
        let n2 = create_nudge(None, 2, &config);
        let n3 = create_nudge(None, 3, &config);
        // 4 templates → index 4 wraps to 0
        let n4 = create_nudge(None, 4, &config);

        assert_ne!(n0.content, n1.content);
        assert_ne!(n1.content, n2.content);
        assert_ne!(n2.content, n3.content);
        assert_eq!(n0.content, n4.content); // Wraps around
    }

    #[test]
    fn test_create_nudge_includes_goal() {
        let config = ReinforcementConfig::default();
        let nudge = create_nudge(Some("find bugs in parser"), 0, &config);
        assert!(nudge.content.contains("find bugs in parser"));
        assert!(nudge.content.contains("Your goal:"));
    }

    #[test]
    fn test_create_nudge_omits_goal_when_disabled() {
        let config = ReinforcementConfig {
            include_goal: false,
            ..Default::default()
        };
        let nudge = create_nudge(Some("find bugs"), 0, &config);
        assert!(!nudge.content.contains("find bugs"));
        assert!(!nudge.content.contains("Your goal:"));
    }

    #[test]
    fn test_create_nudge_has_is_nudge_flag() {
        let config = ReinforcementConfig::default();
        let nudge = create_nudge(None, 0, &config);
        assert!(nudge.flags.contains(MessageFlags::IS_NUDGE));
        assert_eq!(nudge.role, SessionRole::User);
    }

    #[test]
    fn test_should_nudge_after_n_tool_results() {
        let config = ReinforcementConfig {
            nudge_every_n_tool_results: 2,
            ..Default::default()
        };
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "go"),
            tool_call_msg(),
            tool_result_msg(),
        ];
        // Only 1 tool result, need 2
        assert!(!should_nudge(&messages, &config));

        let messages2 = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "go"),
            tool_call_msg(),
            tool_result_msg(),
            tool_call_msg(),
            tool_result_msg(),
        ];
        // 2 tool results, need 2
        assert!(should_nudge(&messages2, &config));
    }

    #[test]
    fn test_should_nudge_false_if_last_is_nudge() {
        let config = ReinforcementConfig::default();
        let mut nudge = SessionMessage::text(SessionRole::User, "continue");
        nudge.flags.insert(MessageFlags::IS_NUDGE);
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "go"),
            tool_call_msg(),
            tool_result_msg(),
            nudge,
        ];
        assert!(!should_nudge(&messages, &config));
    }

    #[test]
    fn test_should_nudge_false_when_disabled() {
        let config = ReinforcementConfig {
            enabled: false,
            ..Default::default()
        };
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "go"),
            tool_call_msg(),
            tool_result_msg(),
        ];
        assert!(!should_nudge(&messages, &config));
    }

    #[test]
    fn test_tool_results_since_last_nudge_counts_correctly() {
        let mut nudge = SessionMessage::text(SessionRole::User, "continue");
        nudge.flags.insert(MessageFlags::IS_NUDGE);
        let messages = vec![
            SessionMessage::system("sys"),
            SessionMessage::text(SessionRole::User, "go"),
            tool_call_msg(),
            tool_result_msg(),
            nudge,
            tool_call_msg(),
            tool_result_msg(),
            tool_call_msg(),
            tool_result_msg(),
        ];
        // 2 tool results after the nudge
        assert_eq!(tool_results_since_last_nudge(&messages), 2);
    }
}
