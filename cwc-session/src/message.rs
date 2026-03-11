use std::sync::Arc;

use cwc_core::traits::TokenCounter;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SessionError};

/// Roles in a session conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionRole {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for SessionRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// A tool call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// A tool result returned to the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub tool_name: String,
    pub output: String,
    pub is_error: bool,
}

/// Flags on a session message controlling trim/drop behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MessageFlags {
    bits: u8,
}

impl MessageFlags {
    pub const NONE: u8 = 0;
    pub const IS_NUDGE: u8 = 1 << 0;
    pub const IS_COMPACTED: u8 = 1 << 1;
    pub const IS_MEMORY: u8 = 1 << 2;
    pub const PRESERVE: u8 = 1 << 3;

    pub fn empty() -> Self {
        Self { bits: 0 }
    }

    pub fn from_bits(bits: u8) -> Self {
        Self { bits }
    }

    pub fn contains(self, flag: u8) -> bool {
        self.bits & flag != 0
    }

    pub fn insert(&mut self, flag: u8) {
        self.bits |= flag;
    }

    pub fn remove(&mut self, flag: u8) {
        self.bits &= !flag;
    }

    pub fn bits(self) -> u8 {
        self.bits
    }
}

/// A single message in a session conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: SessionRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolResult>,
    pub token_count: u32,
    #[serde(default)]
    pub flags: MessageFlags,
}

impl SessionMessage {
    /// Create a simple text message.
    pub fn text(role: SessionRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_result: None,
            token_count: 0,
            flags: MessageFlags::empty(),
        }
    }

    /// Create a system message with PRESERVE flag.
    pub fn system(content: impl Into<String>) -> Self {
        let mut msg = Self::text(SessionRole::System, content);
        msg.flags.insert(MessageFlags::PRESERVE);
        msg
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_tool_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            role: SessionRole::Assistant,
            content: content.into(),
            tool_calls: calls,
            tool_result: None,
            token_count: 0,
            flags: MessageFlags::empty(),
        }
    }

    /// Create a tool result message.
    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: SessionRole::Tool,
            content: result.output.clone(),
            tool_calls: Vec::new(),
            tool_result: Some(result),
            token_count: 0,
            flags: MessageFlags::empty(),
        }
    }

    /// Create a nudge message.
    ///
    /// Uses User role (matching `create_nudge()` in reinforcement) so that the
    /// LLM interprets it as a user-directed instruction to continue.
    pub fn nudge(content: impl Into<String>) -> Self {
        let mut msg = Self::text(SessionRole::User, content);
        msg.flags.insert(MessageFlags::IS_NUDGE);
        msg
    }

    /// Compute token count from content + tool call/result text.
    ///
    /// For tool result messages, `content` already holds the output, so we
    /// skip counting `tool_result.output` separately to avoid double-counting.
    fn compute_tokens(&self, tokenizer: &dyn TokenCounter) -> u32 {
        let mut total = tokenizer.count_tokens(&self.content);
        for tc in &self.tool_calls {
            total += tokenizer.count_tokens(&tc.tool_name);
            let args_str = tc.arguments.to_string();
            total += tokenizer.count_tokens(&args_str);
        }
        // Overhead for role/delimiters (~4 tokens)
        total + 4
    }
}

/// A mutable conversation that tracks token usage as messages are added.
pub struct Session {
    messages: Vec<SessionMessage>,
    total_tokens: u32,
    tokenizer: Arc<dyn TokenCounter>,
}

impl Session {
    pub fn new(tokenizer: Arc<dyn TokenCounter>) -> Self {
        Self {
            messages: Vec::new(),
            total_tokens: 0,
            tokenizer,
        }
    }

    /// Append a message. Token count is computed and stored on the message.
    pub fn push(&mut self, mut msg: SessionMessage) {
        msg.token_count = msg.compute_tokens(&*self.tokenizer);
        self.total_tokens += msg.token_count;
        self.messages.push(msg);
    }

    /// Replace all messages (used after trim/reset). Recalculates total.
    pub fn replace(&mut self, messages: Vec<SessionMessage>) {
        self.total_tokens = messages.iter().map(|m| m.token_count).sum();
        self.messages = messages;
    }

    /// Current message count.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the session has no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Current total token count.
    pub fn total_tokens(&self) -> u32 {
        self.total_tokens
    }

    /// Borrow all messages.
    pub fn messages(&self) -> &[SessionMessage] {
        &self.messages
    }

    /// Mutable access to messages.
    pub fn messages_mut(&mut self) -> &mut Vec<SessionMessage> {
        &mut self.messages
    }

    /// Recalculate total token count from all messages.
    ///
    /// Call after modifying message token counts via `messages_mut()`.
    pub fn recalculate_total(&mut self) {
        self.total_tokens = self.messages.iter().map(|m| m.token_count).sum();
    }

    /// Take ownership of messages.
    pub fn into_messages(self) -> Vec<SessionMessage> {
        self.messages
    }

    /// Load from OpenAI-style messages JSON.
    ///
    /// Expected format: `[{"role": "...", "content": "...", ...}]`
    pub fn from_openai_messages(
        messages: &[serde_json::Value],
        tokenizer: Arc<dyn TokenCounter>,
    ) -> Result<Self> {
        let mut session = Self::new(tokenizer);
        for (i, msg) in messages.iter().enumerate() {
            let role_str = msg["role"]
                .as_str()
                .ok_or_else(|| SessionError::MissingField(format!("messages[{i}].role")))?;
            let role = match role_str {
                "system" => SessionRole::System,
                "user" => SessionRole::User,
                "assistant" => SessionRole::Assistant,
                "tool" => SessionRole::Tool,
                other => {
                    return Err(SessionError::InvalidFormat(format!(
                        "unknown role: {other}"
                    )))
                }
            };
            let content = msg["content"].as_str().unwrap_or("").to_string();

            let mut sm = SessionMessage::text(role, &content);

            // Set PRESERVE for system messages
            if role == SessionRole::System {
                sm.flags.insert(MessageFlags::PRESERVE);
            }

            // Parse tool_calls if present
            if let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                for call in calls {
                    let call_id = call["id"].as_str().unwrap_or("").to_string();
                    let tool_name = call["function"]["name"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    let arguments = call["function"]["arguments"].clone();
                    // Arguments may be a JSON string that needs parsing
                    let arguments = if let Some(s) = arguments.as_str() {
                        serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
                    } else {
                        arguments
                    };
                    sm.tool_calls.push(ToolCall {
                        call_id,
                        tool_name,
                        arguments,
                    });
                }
            }

            // Parse tool result if this is a tool message
            if role == SessionRole::Tool {
                let call_id = msg["tool_call_id"].as_str().unwrap_or("").to_string();
                let tool_name = msg["name"].as_str().unwrap_or("").to_string();
                sm.tool_result = Some(ToolResult {
                    call_id,
                    tool_name,
                    output: content.clone(),
                    is_error: false,
                });
            }

            session.push(sm);
        }
        Ok(session)
    }

    /// Export to OpenAI-style messages.
    pub fn to_openai_messages(&self) -> Vec<serde_json::Value> {
        self.messages
            .iter()
            .map(|m| {
                let mut obj = serde_json::json!({
                    "role": m.role.to_string(),
                    "content": m.content,
                });
                if !m.tool_calls.is_empty() {
                    let calls: Vec<serde_json::Value> = m
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.tool_name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    obj["tool_calls"] = serde_json::Value::Array(calls);
                }
                if let Some(tr) = &m.tool_result {
                    obj["tool_call_id"] = serde_json::Value::String(tr.call_id.clone());
                    obj["name"] = serde_json::Value::String(tr.tool_name.clone());
                }
                obj
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeTokenCounter;
    impl TokenCounter for FakeTokenCounter {
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

    fn tc() -> Arc<dyn TokenCounter> {
        Arc::new(FakeTokenCounter)
    }

    #[test]
    fn test_session_message_construct_all_roles() {
        let sys = SessionMessage::system("You are helpful");
        assert_eq!(sys.role, SessionRole::System);
        assert!(sys.flags.contains(MessageFlags::PRESERVE));

        let user = SessionMessage::text(SessionRole::User, "Hello");
        assert_eq!(user.role, SessionRole::User);

        let asst = SessionMessage::text(SessionRole::Assistant, "Hi");
        assert_eq!(asst.role, SessionRole::Assistant);

        let tr = SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "grep".into(),
            output: "found 3 matches".into(),
            is_error: false,
        });
        assert_eq!(tr.role, SessionRole::Tool);
        assert!(tr.tool_result.is_some());
    }

    #[test]
    fn test_session_message_serialize_deserialize_roundtrip() {
        let msg = SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "file.read".into(),
                arguments: serde_json::json!({"path": "main.rs"}),
            }],
        );
        let json = serde_json::to_string(&msg).expect("serialize");
        let back: SessionMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.role, SessionRole::Assistant);
        assert_eq!(back.tool_calls.len(), 1);
        assert_eq!(back.tool_calls[0].call_id, "c1");
        assert_eq!(back.tool_calls[0].tool_name, "file.read");
    }

    #[test]
    fn test_session_push_updates_total_tokens() {
        let mut session = Session::new(tc());
        assert_eq!(session.total_tokens(), 0);

        session.push(SessionMessage::text(SessionRole::User, "hello world"));
        let first = session.total_tokens();
        assert!(first > 0);

        session.push(SessionMessage::text(
            SessionRole::Assistant,
            "hi there friend",
        ));
        assert!(session.total_tokens() > first);
        assert_eq!(session.len(), 2);
    }

    #[test]
    fn test_session_from_openai_messages() {
        let msgs = serde_json::json!([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Rust?"},
            {"role": "assistant", "content": "A systems language."}
        ]);
        let session =
            Session::from_openai_messages(msgs.as_array().expect("array"), tc()).expect("parse");
        assert_eq!(session.len(), 3);
        assert_eq!(session.messages()[0].role, SessionRole::System);
        assert!(session.messages()[0].flags.contains(MessageFlags::PRESERVE));
        assert_eq!(session.messages()[1].role, SessionRole::User);
        assert_eq!(session.messages()[2].role, SessionRole::Assistant);
        assert!(session.total_tokens() > 0);
    }

    #[test]
    fn test_session_to_openai_messages() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("Be helpful"));
        session.push(SessionMessage::text(SessionRole::User, "Hi"));

        let exported = session.to_openai_messages();
        assert_eq!(exported.len(), 2);
        assert_eq!(exported[0]["role"], "system");
        assert_eq!(exported[0]["content"], "Be helpful");
        assert_eq!(exported[1]["role"], "user");
    }

    #[test]
    fn test_session_tool_call_linkage() {
        let msgs = serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "find malloc"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "file.grep",
                        "arguments": "{\"pattern\": \"malloc\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "content": "src/main.rs:10: malloc(256)",
                "tool_call_id": "call_1",
                "name": "file.grep"
            }
        ]);
        let session =
            Session::from_openai_messages(msgs.as_array().expect("array"), tc()).expect("parse");
        assert_eq!(session.len(), 4);

        // Assistant has tool_calls
        let asst = &session.messages()[2];
        assert_eq!(asst.tool_calls.len(), 1);
        assert_eq!(asst.tool_calls[0].call_id, "call_1");
        assert_eq!(asst.tool_calls[0].tool_name, "file.grep");

        // Tool result has matching call_id
        let tool = &session.messages()[3];
        assert_eq!(tool.role, SessionRole::Tool);
        let tr = tool.tool_result.as_ref().expect("tool_result");
        assert_eq!(tr.call_id, "call_1");
        assert_eq!(tr.tool_name, "file.grep");
    }

    #[test]
    fn test_session_replace_recalculates_total() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::text(SessionRole::User, "a long message with many words in it"));
        let before = session.total_tokens();
        assert!(before > 0);

        // Replace with a single short message (with pre-set token count)
        let mut short = SessionMessage::text(SessionRole::User, "hi");
        short.token_count = 5;
        session.replace(vec![short]);
        assert_eq!(session.total_tokens(), 5);
        assert_eq!(session.len(), 1);
    }

    #[test]
    fn test_message_flags_set_and_check() {
        let mut flags = MessageFlags::empty();
        assert!(!flags.contains(MessageFlags::IS_NUDGE));
        assert!(!flags.contains(MessageFlags::PRESERVE));

        flags.insert(MessageFlags::IS_NUDGE);
        assert!(flags.contains(MessageFlags::IS_NUDGE));
        assert!(!flags.contains(MessageFlags::PRESERVE));

        flags.insert(MessageFlags::PRESERVE);
        assert!(flags.contains(MessageFlags::IS_NUDGE));
        assert!(flags.contains(MessageFlags::PRESERVE));

        flags.remove(MessageFlags::IS_NUDGE);
        assert!(!flags.contains(MessageFlags::IS_NUDGE));
        assert!(flags.contains(MessageFlags::PRESERVE));
    }

    #[test]
    fn test_message_flags_serialize_roundtrip() {
        let mut flags = MessageFlags::empty();
        flags.insert(MessageFlags::IS_NUDGE);
        flags.insert(MessageFlags::IS_MEMORY);
        let json = serde_json::to_string(&flags).expect("ser");
        let back: MessageFlags = serde_json::from_str(&json).expect("de");
        assert!(back.contains(MessageFlags::IS_NUDGE));
        assert!(back.contains(MessageFlags::IS_MEMORY));
        assert!(!back.contains(MessageFlags::PRESERVE));
    }

    #[test]
    fn test_session_is_empty() {
        let session = Session::new(tc());
        assert!(session.is_empty());
    }

    #[test]
    fn test_session_into_messages() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::text(SessionRole::User, "test"));
        let msgs = session.into_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, SessionRole::User);
    }

    #[test]
    fn test_session_nudge_message() {
        let nudge = SessionMessage::nudge("Continue working.");
        assert_eq!(nudge.role, SessionRole::User);
        assert!(nudge.flags.contains(MessageFlags::IS_NUDGE));
        assert!(!nudge.flags.contains(MessageFlags::PRESERVE));
    }

    #[test]
    fn test_tool_result_tokens_not_double_counted() {
        // Bug fix: tool_result() copies output into both content and tool_result.
        // compute_tokens must not count it twice.
        let mut session = Session::new(tc());
        let tr = SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "grep".into(),
            output: "line one\nline two\nline three".into(),
            is_error: false,
        });
        // Verify content == tool_result.output (the duplication)
        assert_eq!(tr.content, tr.tool_result.as_ref().unwrap().output);

        session.push(tr);
        let tokens = session.messages()[0].token_count;

        // FakeTokenCounter: word count + 4 overhead
        // "line one\nline two\nline three" = 6 words → 6 + 4 = 10
        assert_eq!(tokens, 10);
    }

    #[test]
    fn test_compute_tokens_tool_call_includes_args() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::assistant_tool_calls(
            "thinking",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "file_grep".into(),
                arguments: serde_json::json!({"pattern": "malloc", "path": "src"}),
            }],
        ));
        let tokens = session.messages()[0].token_count;
        // "thinking" = 1 word, "file_grep" = 1 word,
        // arguments.to_string() = {"path":"src","pattern":"malloc"} (JSON, varies)
        // All counted via word-split + 4 overhead
        assert!(tokens > 4); // at minimum content + overhead
    }

    #[test]
    fn test_from_openai_messages_invalid_role() {
        let msgs = serde_json::json!([
            {"role": "narrator", "content": "Once upon a time"}
        ]);
        let result = Session::from_openai_messages(msgs.as_array().unwrap(), tc());
        match result {
            Err(SessionError::InvalidFormat(msg)) => {
                assert!(msg.contains("narrator"), "expected 'narrator' in: {msg}");
            }
            _ => panic!("expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_from_openai_messages_missing_role() {
        let msgs = serde_json::json!([
            {"content": "no role here"}
        ]);
        let result = Session::from_openai_messages(msgs.as_array().unwrap(), tc());
        match result {
            Err(SessionError::MissingField(msg)) => {
                assert!(msg.contains("messages[0].role"), "expected field path in: {msg}");
            }
            _ => panic!("expected MissingField error"),
        }
    }

    #[test]
    fn test_from_openai_messages_null_content() {
        // OpenAI assistant tool-call messages often have content: null
        let msgs = serde_json::json!([
            {"role": "assistant", "content": null}
        ]);
        let session = Session::from_openai_messages(msgs.as_array().unwrap(), tc()).unwrap();
        assert_eq!(session.messages()[0].content, "");
    }

    #[test]
    fn test_from_openai_messages_empty_array() {
        let msgs: Vec<serde_json::Value> = vec![];
        let session = Session::from_openai_messages(&msgs, tc()).unwrap();
        assert!(session.is_empty());
        assert_eq!(session.total_tokens(), 0);
    }

    #[test]
    fn test_session_recalculate_total() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::text(SessionRole::User, "hello world"));
        let before = session.total_tokens();

        // Manually modify a message's token count via messages_mut
        session.messages_mut()[0].token_count = 100;
        // Total is stale until recalculated
        assert_eq!(session.total_tokens(), before);

        session.recalculate_total();
        assert_eq!(session.total_tokens(), 100);
    }

    #[test]
    fn test_session_messages_mut_updates_reflected() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::text(SessionRole::User, "hello"));
        session.messages_mut()[0].flags.insert(MessageFlags::PRESERVE);
        assert!(session.messages()[0].flags.contains(MessageFlags::PRESERVE));
    }

    #[test]
    fn test_openai_roundtrip_with_tool_calls() {
        let mut session = Session::new(tc());
        session.push(SessionMessage::system("sys"));
        session.push(SessionMessage::text(SessionRole::User, "find bugs"));
        session.push(SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "grep".into(),
                arguments: serde_json::json!({"q": "bug"}),
            }],
        ));
        session.push(SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "grep".into(),
            output: "found 2".into(),
            is_error: false,
        }));

        let exported = session.to_openai_messages();
        assert_eq!(exported.len(), 4);

        // Re-parse
        let session2 = Session::from_openai_messages(&exported, tc()).expect("parse");
        assert_eq!(session2.len(), 4);
        assert_eq!(session2.messages()[2].tool_calls.len(), 1);
        assert!(session2.messages()[3].tool_result.is_some());
    }
}
