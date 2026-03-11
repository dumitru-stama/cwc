use crate::message::{MessageFlags, SessionMessage, SessionRole};

/// An atomic turn — a group of messages that must not be split.
#[derive(Debug, Clone)]
pub struct Turn {
    /// Index of first message in this turn.
    pub start: usize,
    /// Index of last message (inclusive).
    pub end: usize,
    /// Total tokens across all messages in the turn.
    pub tokens: u32,
    /// Contains a real user message (not nudge/system).
    pub has_user_request: bool,
    /// Contains tool call + result pairs.
    pub has_tool_calls: bool,
    /// Number of messages in this turn.
    pub message_count: usize,
}

/// Identify nudge/reinforcement messages.
pub fn is_nudge(msg: &SessionMessage) -> bool {
    msg.flags.contains(MessageFlags::IS_NUDGE)
}

/// Parse a message list into atomic turns.
///
/// Grouping rules:
/// - A User message (not flagged IS_NUDGE) starts a new turn
/// - Assistant messages with tool_calls belong to the current turn
/// - Tool result messages belong to the current turn
/// - An Assistant text-only message closes the current turn
/// - System and nudge messages are standalone 1-message turns
/// - Multiple consecutive tool call/result pairs stay in the same turn
pub fn parse_turns(messages: &[SessionMessage]) -> Vec<Turn> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut turns = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_tokens: u32 = 0;
    let mut current_has_user = false;
    let mut current_has_tools = false;

    for (i, msg) in messages.iter().enumerate() {
        match msg.role {
            SessionRole::System => {
                // Flush any open turn
                if let Some(start) = current_start.take() {
                    turns.push(Turn {
                        start,
                        end: i - 1,
                        tokens: current_tokens,
                        has_user_request: current_has_user,
                        has_tool_calls: current_has_tools,
                        message_count: i - start,
                    });
                    current_tokens = 0;
                    current_has_user = false;
                    current_has_tools = false;
                }
                // System/nudge is always a standalone turn
                turns.push(Turn {
                    start: i,
                    end: i,
                    tokens: msg.token_count,
                    has_user_request: false,
                    has_tool_calls: false,
                    message_count: 1,
                });
            }
            SessionRole::User => {
                // Flush any open turn
                if let Some(start) = current_start.take() {
                    turns.push(Turn {
                        start,
                        end: i - 1,
                        tokens: current_tokens,
                        has_user_request: current_has_user,
                        has_tool_calls: current_has_tools,
                        message_count: i - start,
                    });
                }
                // Start a new turn
                current_start = Some(i);
                current_tokens = msg.token_count;
                current_has_user = !is_nudge(msg);
                current_has_tools = false;
            }
            SessionRole::Assistant => {
                if current_start.is_none() {
                    // Orphaned assistant message — start a new turn
                    current_start = Some(i);
                    current_tokens = 0;
                    current_has_user = false;
                    current_has_tools = false;
                }
                current_tokens += msg.token_count;
                if !msg.tool_calls.is_empty() {
                    current_has_tools = true;
                    // Turn stays open — tool results will follow
                } else {
                    // Text-only assistant response — close the turn.
                    // current_start is guaranteed Some here (checked at line 91).
                    let start = match current_start.take() {
                        Some(s) => s,
                        None => unreachable!("current_start was checked above"),
                    };
                    turns.push(Turn {
                        start,
                        end: i,
                        tokens: current_tokens,
                        has_user_request: current_has_user,
                        has_tool_calls: current_has_tools,
                        message_count: i - start + 1,
                    });
                    current_tokens = 0;
                    current_has_user = false;
                    current_has_tools = false;
                }
            }
            SessionRole::Tool => {
                if current_start.is_none() {
                    // Orphaned tool result — standalone turn
                    turns.push(Turn {
                        start: i,
                        end: i,
                        tokens: msg.token_count,
                        has_user_request: false,
                        has_tool_calls: true,
                        message_count: 1,
                    });
                } else {
                    current_tokens += msg.token_count;
                    current_has_tools = true;
                }
            }
        }
    }

    // Flush any remaining open turn
    if let Some(start) = current_start {
        let end = messages.len() - 1;
        turns.push(Turn {
            start,
            end,
            tokens: current_tokens,
            has_user_request: current_has_user,
            has_tool_calls: current_has_tools,
            message_count: end - start + 1,
        });
    }

    turns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{SessionMessage, ToolCall, ToolResult};

    fn msg(role: SessionRole, content: &str, tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::text(role, content);
        m.token_count = tokens;
        m
    }

    fn tool_call_msg(tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::assistant_tool_calls(
            "",
            vec![ToolCall {
                call_id: "c1".into(),
                tool_name: "grep".into(),
                arguments: serde_json::json!({}),
            }],
        );
        m.token_count = tokens;
        m
    }

    fn tool_result_msg(tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::tool_result(ToolResult {
            call_id: "c1".into(),
            tool_name: "grep".into(),
            output: "results".into(),
            is_error: false,
        });
        m.token_count = tokens;
        m
    }

    fn nudge_msg(tokens: u32) -> SessionMessage {
        let mut m = SessionMessage::nudge("Continue.");
        m.token_count = tokens;
        m
    }

    #[test]
    fn test_turn_parser_user_assistant_single_turn() {
        let messages = vec![
            msg(SessionRole::User, "Hello", 10),
            msg(SessionRole::Assistant, "Hi there", 15),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 1);
        assert_eq!(turns[0].tokens, 25);
        assert!(turns[0].has_user_request);
        assert!(!turns[0].has_tool_calls);
        assert_eq!(turns[0].message_count, 2);
    }

    #[test]
    fn test_turn_parser_user_tool_call_result_assistant() {
        let messages = vec![
            msg(SessionRole::User, "find malloc", 10),
            tool_call_msg(5),
            tool_result_msg(50),
            msg(SessionRole::Assistant, "Found it", 10),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 3);
        assert_eq!(turns[0].tokens, 75);
        assert!(turns[0].has_user_request);
        assert!(turns[0].has_tool_calls);
        assert_eq!(turns[0].message_count, 4);
    }

    #[test]
    fn test_turn_parser_multiple_consecutive_tool_calls() {
        let messages = vec![
            msg(SessionRole::User, "analyze", 10),
            tool_call_msg(5),
            tool_result_msg(50),
            tool_call_msg(5),
            tool_result_msg(40),
            msg(SessionRole::Assistant, "Done", 8),
        ];
        let turns = parse_turns(&messages);
        // All in one turn: user + 2x(tool_call + tool_result) + assistant
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].tokens, 118);
        assert!(turns[0].has_tool_calls);
        assert_eq!(turns[0].message_count, 6);
    }

    #[test]
    fn test_turn_parser_system_standalone() {
        let messages = vec![
            msg(SessionRole::System, "You are helpful", 20),
            msg(SessionRole::User, "Hello", 10),
            msg(SessionRole::Assistant, "Hi", 8),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 0);
        assert!(!turns[0].has_user_request);
        assert_eq!(turns[1].start, 1);
        assert_eq!(turns[1].end, 2);
        assert!(turns[1].has_user_request);
    }

    #[test]
    fn test_turn_parser_nudge_standalone() {
        let messages = vec![
            msg(SessionRole::User, "Hello", 10),
            msg(SessionRole::Assistant, "Hi", 8),
            nudge_msg(5),
            msg(SessionRole::User, "More", 10),
            msg(SessionRole::Assistant, "Ok", 6),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 3);
        // Turn 0: user + assistant
        assert!(turns[0].has_user_request);
        // Turn 1: nudge (User IS_NUDGE, not a real user request)
        assert!(!turns[1].has_user_request);
        assert_eq!(turns[1].tokens, 5);
        // Turn 2: user + assistant
        assert!(turns[2].has_user_request);
    }

    #[test]
    fn test_turn_parser_orphaned_tool_result() {
        let messages = vec![
            tool_result_msg(50),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 1);
        assert!(!turns[0].has_user_request);
        assert!(turns[0].has_tool_calls);
        assert_eq!(turns[0].tokens, 50);
    }

    #[test]
    fn test_turn_parser_empty() {
        let turns = parse_turns(&[]);
        assert!(turns.is_empty());
    }

    #[test]
    fn test_turn_parser_token_counts_accumulated() {
        let messages = vec![
            msg(SessionRole::System, "sys", 100),
            msg(SessionRole::User, "q1", 20),
            tool_call_msg(10),
            tool_result_msg(200),
            msg(SessionRole::Assistant, "done", 15),
            msg(SessionRole::User, "q2", 30),
            msg(SessionRole::Assistant, "ok", 12),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].tokens, 100); // system
        assert_eq!(turns[1].tokens, 245); // user + tool_call + tool_result + assistant
        assert_eq!(turns[2].tokens, 42);  // user + assistant
    }

    #[test]
    fn test_is_nudge() {
        let nudge = SessionMessage::nudge("test");
        assert!(is_nudge(&nudge));

        let normal = SessionMessage::text(SessionRole::System, "test");
        assert!(!is_nudge(&normal));
    }

    #[test]
    fn test_turn_parser_two_user_turns() {
        let messages = vec![
            msg(SessionRole::User, "first question", 10),
            msg(SessionRole::Assistant, "first answer", 15),
            msg(SessionRole::User, "second question", 12),
            msg(SessionRole::Assistant, "second answer", 18),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 1);
        assert_eq!(turns[1].start, 2);
        assert_eq!(turns[1].end, 3);
    }

    #[test]
    fn test_turn_parser_orphaned_assistant() {
        // Assistant message with no preceding user
        let messages = vec![
            msg(SessionRole::Assistant, "I can help", 10),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 0);
        assert!(!turns[0].has_user_request);
        assert!(!turns[0].has_tool_calls);
        assert_eq!(turns[0].tokens, 10);
    }

    #[test]
    fn test_turn_parser_consecutive_system_messages() {
        let messages = vec![
            msg(SessionRole::System, "system prompt", 50),
            msg(SessionRole::System, "more system", 30),
            msg(SessionRole::User, "hi", 5),
            msg(SessionRole::Assistant, "hello", 5),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 3);
        // Two standalone system turns
        assert_eq!(turns[0].tokens, 50);
        assert_eq!(turns[0].message_count, 1);
        assert_eq!(turns[1].tokens, 30);
        assert_eq!(turns[1].message_count, 1);
        // User+assistant turn
        assert_eq!(turns[2].tokens, 10);
        assert!(turns[2].has_user_request);
    }

    #[test]
    fn test_turn_parser_system_splits_tool_sequence() {
        // System message injected between tool_call and tool_result
        // This breaks the atomic turn — by design (system always standalone)
        let messages = vec![
            msg(SessionRole::User, "find bugs", 10),
            tool_call_msg(5),
            msg(SessionRole::System, "nudge: keep going", 8),
            tool_result_msg(50),
            msg(SessionRole::Assistant, "Done", 10),
        ];
        let turns = parse_turns(&messages);
        // Turn 0: user + tool_call (flushed when system appears)
        // Turn 1: standalone system
        // Turn 2: orphaned tool_result (standalone)
        // Turn 3: orphaned assistant (text-only, closes immediately)
        assert_eq!(turns.len(), 4);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 1);
        assert!(turns[0].has_user_request);
        assert!(turns[0].has_tool_calls);
        assert_eq!(turns[1].start, 2);
        assert_eq!(turns[1].end, 2);
        assert_eq!(turns[2].start, 3);
        assert_eq!(turns[2].end, 3);
        assert!(turns[2].has_tool_calls); // orphaned tool result
        assert_eq!(turns[3].start, 4);
        assert_eq!(turns[3].end, 4);
    }

    #[test]
    fn test_turn_parser_user_only_at_end() {
        // Conversation ends with just a user message (waiting for response)
        let messages = vec![
            msg(SessionRole::User, "question", 10),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 0);
        assert!(turns[0].has_user_request);
        assert_eq!(turns[0].tokens, 10);
        assert_eq!(turns[0].message_count, 1);
    }

    #[test]
    fn test_turn_parser_invariant_start_le_end() {
        // Verify the start <= end invariant holds for a complex sequence
        let messages = vec![
            msg(SessionRole::System, "sys", 10),
            msg(SessionRole::User, "q1", 5),
            tool_call_msg(3),
            tool_result_msg(20),
            tool_call_msg(3),
            tool_result_msg(15),
            msg(SessionRole::Assistant, "a1", 8),
            nudge_msg(2),
            msg(SessionRole::User, "q2", 5),
            msg(SessionRole::Assistant, "a2", 7),
        ];
        let turns = parse_turns(&messages);
        for (idx, turn) in turns.iter().enumerate() {
            assert!(
                turn.start <= turn.end,
                "turn {idx}: start {} > end {}",
                turn.start,
                turn.end
            );
            assert_eq!(
                turn.message_count,
                turn.end - turn.start + 1,
                "turn {idx}: message_count {} != end - start + 1 = {}",
                turn.message_count,
                turn.end - turn.start + 1
            );
        }
    }

    #[test]
    fn test_turn_parser_open_turn_at_end() {
        // Conversation ends mid-turn (no closing assistant message)
        let messages = vec![
            msg(SessionRole::User, "question", 10),
            tool_call_msg(5),
            tool_result_msg(50),
        ];
        let turns = parse_turns(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns[0].end, 2);
        assert_eq!(turns[0].tokens, 65);
        assert!(turns[0].has_tool_calls);
    }
}
