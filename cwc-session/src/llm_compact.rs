use cwc_core::traits::{LlmClient, TokenCounter};

use crate::budget::SessionBudget;
use crate::error::{Result, SessionError};
use crate::memory::types::{now_millis, FactPriority, MemoryFact, MemorySource};
use crate::memory::SessionMemoryStore;
use crate::message::{MessageFlags, SessionMessage, SessionRole};
use crate::rebuild::rebuild_conversation;
use crate::reset::ResetResult;
use crate::turn::parse_turns;

/// Last-resort LLM-based compaction.
///
/// Asks the LLM to produce a structured summary of the conversation,
/// then replaces all messages with: system + memory (from summary) + last turn.
pub async fn llm_compaction(
    messages: &[SessionMessage],
    memory_store: &mut SessionMemoryStore,
    llm: &dyn LlmClient,
    budget: &SessionBudget,
    tokenizer: &dyn TokenCounter,
) -> Result<ResetResult> {
    let tokens_before: u32 = messages.iter().map(|m| m.token_count).sum();

    // Build prompt and call LLM
    let prompt = build_compaction_prompt(messages);
    let schema = compaction_schema();
    let grammar_str = serde_json::to_string(&schema)?;
    let response = llm
        .generate(&prompt, Some(&grammar_str), budget.max_output)
        .await
        .map_err(|e| SessionError::Llm(format!("{e}")))?;

    // Parse structured response
    let parsed: serde_json::Value = serde_json::from_str(&response)
        .map_err(|e| SessionError::Llm(format!("failed to parse LLM response: {e}")))?;

    // Extract facts from LLM summary
    let ts = now_millis();
    let source = MemorySource::AssistantConclusion { message_index: 0 };

    if let Some(goal) = parsed["goal"].as_str() {
        if !goal.is_empty() {
            memory_store.upsert(MemoryFact {
                key: "goal".into(),
                value: goal.to_string(),
                source: source.clone(),
                created_at: ts,
                priority: FactPriority::Goal,
            });
        }
    }

    if let Some(findings) = parsed["findings"].as_array() {
        for (i, finding) in findings.iter().enumerate() {
            if let Some(text) = finding.as_str() {
                if !text.is_empty() {
                    memory_store.upsert(MemoryFact {
                        key: format!("finding:compact_{i}"),
                        value: text.to_string(),
                        source: source.clone(),
                        created_at: ts,
                        priority: FactPriority::Finding,
                    });
                }
            }
        }
    }

    memory_store.enforce_limits();

    // Keep system prompt + last turn
    let system_prompt = &messages[0];
    let turns = parse_turns(messages);
    let tail_messages = if let Some(last_turn) = turns.last() {
        // Skip system prompt if it's the last turn
        if last_turn.start == 0 && turns.len() == 1 {
            Vec::new()
        } else {
            messages[last_turn.start..=last_turn.end].to_vec()
        }
    } else {
        Vec::new()
    };

    // Rebuild
    let rebuilt = rebuild_conversation(system_prompt, memory_store, tail_messages, tokenizer);
    let tokens_after: u32 = rebuilt.iter().map(|m| m.token_count).sum();

    let extracted_facts = memory_store.all().to_vec();

    Ok(ResetResult {
        messages: rebuilt,
        extracted_facts,
        tokens_before,
        tokens_after,
    })
}

/// Build the prompt for LLM compaction.
///
/// Includes the conversation content and instructions to produce structured JSON.
fn build_compaction_prompt(messages: &[SessionMessage]) -> String {
    let mut prompt = String::from(
        "Summarize the following conversation as JSON.\n\
         Output ONLY valid JSON with this schema:\n\
         {\"goal\": \"string\", \"findings\": [\"string\"], \"next_steps\": [\"string\"]}\n\n\
         Conversation:\n",
    );

    for msg in messages {
        // Skip system prompt and old memory messages
        if msg.role == SessionRole::System && msg.flags.contains(MessageFlags::PRESERVE) {
            continue;
        }
        if msg.flags.contains(MessageFlags::IS_MEMORY) {
            continue;
        }
        let role = msg.role.to_string();
        let content = if msg.content.len() > 500 {
            // Use char boundary to avoid panic on multi-byte UTF-8
            let end = msg.content.char_indices()
                .nth(500)
                .map(|(i, _)| i)
                .unwrap_or(msg.content.len());
            format!("{}...", &msg.content[..end])
        } else {
            msg.content.clone()
        };
        prompt.push_str(&format!("{role}: {content}\n"));
    }

    prompt
}

/// JSON schema for the compaction output.
pub fn compaction_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "goal": { "type": "string" },
            "findings": {
                "type": "array",
                "items": { "type": "string" }
            },
            "next_steps": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["goal", "findings", "next_steps"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cwc_core::types::ChatMessage;

    struct WordCounter;
    impl TokenCounter for WordCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            text.split_whitespace().count() as u32
        }
        fn truncate_to_tokens(&self, text: &str, max: u32) -> String {
            text.split_whitespace()
                .take(max as usize)
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
    fn tc() -> &'static dyn TokenCounter {
        &WordCounter
    }

    struct MockLlm {
        response: String,
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn generate(
            &self,
            _prompt: &str,
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> cwc_core::error::Result<String> {
            Ok(self.response.clone())
        }
        async fn generate_chat(
            &self,
            _messages: &[ChatMessage],
            _grammar: Option<&str>,
            _max_tokens: u32,
        ) -> cwc_core::error::Result<String> {
            Ok(self.response.clone())
        }
    }

    fn msg(role: SessionRole, content: &str, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::text(role, content);
        m.token_count = tokens;
        m
    }

    fn sys_msg(tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::system("system prompt");
        m.token_count = tokens;
        m
    }

    fn test_budget() -> SessionBudget {
        SessionBudget {
            context_window: 32768,
            max_output: 4096,
            usable_budget: 10000,
            sliding_window_trigger: 5000,
            hard_reset_trigger: 6000,
            compaction_trigger: 8500,
            tail_tokens: 2000,
            min_tail_turns: 2,
        }
    }

    #[tokio::test]
    async fn test_llm_compaction_produces_system_summary_last_turn() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "find bugs", 100),
            msg(SessionRole::Assistant, "searching", 100),
            msg(SessionRole::User, "what did you find", 100),
            msg(SessionRole::Assistant, "found 3 bugs", 100),
        ];
        let llm = MockLlm {
            response: serde_json::json!({
                "goal": "find bugs in the codebase",
                "findings": ["3 bugs found in parser.rs"],
                "next_steps": ["fix the bugs"]
            })
            .to_string(),
        };
        let mut store = SessionMemoryStore::default();

        let result = llm_compaction(&messages, &mut store, &llm, &test_budget(), tc())
            .await
            .unwrap();

        // Should have system + memory + last turn
        assert_eq!(result.messages[0].role, SessionRole::System);
        assert!(result.messages[0].flags.contains(MessageFlags::PRESERVE));
        // Memory message at position 1
        assert!(result.messages[1].flags.contains(MessageFlags::IS_MEMORY));
        // Last turn present
        let has_last_content = result
            .messages
            .iter()
            .any(|m| m.content.contains("found 3 bugs"));
        assert!(has_last_content, "should keep last turn");
    }

    #[test]
    fn test_compaction_schema_valid() {
        let schema = compaction_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["goal"].is_object());
        assert!(schema["properties"]["findings"].is_object());
        assert!(schema["properties"]["next_steps"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 3);
    }

    #[test]
    fn test_build_compaction_prompt_includes_conversation() {
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, "find malloc usage", 100),
            msg(SessionRole::Assistant, "found 5 calls to malloc", 100),
        ];
        let prompt = build_compaction_prompt(&messages);
        assert!(prompt.contains("Summarize"), "should have summary instruction");
        assert!(prompt.contains("find malloc usage"), "should include user content");
        assert!(
            prompt.contains("found 5 calls to malloc"),
            "should include assistant content"
        );
        // System prompt should be skipped
        assert!(!prompt.contains("system prompt"), "should skip system prompt");
    }

    #[test]
    fn test_build_compaction_prompt_truncates_long_messages() {
        let long_content = "x".repeat(1000);
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, &long_content, 500),
        ];
        let prompt = build_compaction_prompt(&messages);
        // Content should be truncated to ~500 chars + "..."
        assert!(prompt.contains("..."), "should truncate long content");
    }

    #[test]
    fn test_build_compaction_prompt_multibyte_utf8_no_panic() {
        // 250 2-byte chars = 500 bytes but only 250 chars; at byte 500 we'd be mid-char
        let long_content = "é".repeat(600); // 600 chars, 1200 bytes
        let messages = vec![
            sys_msg(50),
            msg(SessionRole::User, &long_content, 500),
        ];
        // Must not panic — previously used byte indexing
        let prompt = build_compaction_prompt(&messages);
        assert!(prompt.contains("..."), "should truncate multi-byte content safely");
    }

    #[test]
    fn test_build_compaction_prompt_skips_memory_messages() {
        let mut mem = SessionMessage::text(SessionRole::User, "[WORKING_MEMORY] old data");
        mem.flags.insert(MessageFlags::IS_MEMORY);
        mem.token_count = 50;
        let messages = vec![
            sys_msg(50),
            mem,
            msg(SessionRole::User, "real question", 100),
        ];
        let prompt = build_compaction_prompt(&messages);
        assert!(!prompt.contains("WORKING_MEMORY"), "should skip memory messages");
        assert!(prompt.contains("real question"), "should include real messages");
    }
}
