use std::sync::LazyLock;

use cwc_core::error::{CwcError, Result};
use regex::Regex;
use serde_json::Value;

static CITATION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[S(\d+)\]").expect("citation regex"));

/// A parsed LLM response.
#[derive(Debug, Clone)]
pub struct ParsedResponse {
    /// Parsed JSON value (if a schema was expected and JSON was found).
    pub json: Option<Value>,
    /// The raw text from the LLM.
    pub raw_text: String,
    /// Extracted source citations like [S1], [S2].
    pub citations: Vec<String>,
    /// Whether the response indicates insufficient evidence.
    pub is_abstention: bool,
}

/// Parse an LLM response, handling common failure modes.
///
/// If `schema` is `Some`, attempts to extract and parse a JSON object from the
/// response. Handles markdown code fences, leading/trailing text, and partial JSON.
pub fn parse_structured_response(
    raw: &str,
    schema: Option<&Value>,
) -> Result<ParsedResponse> {
    if raw.trim().is_empty() {
        return Err(CwcError::Llm("empty LLM response".into()));
    }

    let is_abstention = raw.contains("INSUFFICIENT_EVIDENCE");
    let citations = extract_citations(raw);

    let json = if schema.is_some() {
        extract_json(raw)?
    } else {
        None
    };

    Ok(ParsedResponse {
        json,
        raw_text: raw.to_string(),
        citations,
        is_abstention,
    })
}

/// Extract [S1], [S2], etc. citation references from text.
fn extract_citations(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut citations = Vec::new();
    for cap in CITATION_RE.captures_iter(text) {
        let full = cap[0].to_string();
        if seen.insert(full.clone()) {
            citations.push(full);
        }
    }
    citations
}

/// Try to extract a JSON object from LLM output.
///
/// Handles:
/// - Clean JSON
/// - JSON wrapped in markdown code fences
/// - Extra text before/after JSON
/// - Partial JSON with unclosed braces/brackets
fn extract_json(raw: &str) -> Result<Option<Value>> {
    let trimmed = raw.trim();

    // Try direct parse first
    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
        if val.is_object() {
            return Ok(Some(val));
        }
    }

    // Strip markdown code fences
    let stripped = strip_code_fences(trimmed);
    if let Ok(val) = serde_json::from_str::<Value>(&stripped) {
        if val.is_object() {
            return Ok(Some(val));
        }
    }

    // Extract first {...} block
    if let Some(json_str) = extract_first_json_object(&stripped) {
        if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
            return Ok(Some(val));
        }

        // Try repairing partial JSON
        let repaired = repair_partial_json(&json_str);
        if let Ok(val) = serde_json::from_str::<Value>(&repaired) {
            tracing::warn!("repaired partial JSON from LLM response");
            return Ok(Some(val));
        }
    }

    // Try repair on full stripped text
    if let Some(json_str) = extract_first_json_object(trimmed) {
        let repaired = repair_partial_json(&json_str);
        if let Ok(val) = serde_json::from_str::<Value>(&repaired) {
            tracing::warn!("repaired partial JSON from LLM response");
            return Ok(Some(val));
        }
    }

    Ok(None)
}

/// Strip markdown code fences like ```json ... ``` or ``` ... ```
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();

    // Match ```json\n...\n``` or ```\n...\n```
    if trimmed.starts_with("```") {
        let after_opening = if let Some(nl) = trimmed.find('\n') {
            &trimmed[nl + 1..]
        } else {
            return trimmed.to_string();
        };

        if let Some(end) = after_opening.rfind("```") {
            return after_opening[..end].trim().to_string();
        }
        return after_opening.trim().to_string();
    }

    trimmed.to_string()
}

/// Find the first `{...}` block with balanced braces.
fn extract_first_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for i in start..bytes.len() {
        let c = bytes[i];

        if escape_next {
            escape_next = false;
            continue;
        }

        if c == b'\\' && in_string {
            escape_next = true;
            continue;
        }

        if c == b'"' {
            in_string = !in_string;
            continue;
        }

        if in_string {
            continue;
        }

        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }

    // Return what we have even if unbalanced (repair_partial_json will fix it)
    if depth > 0 {
        Some(text[start..].to_string())
    } else {
        None
    }
}

/// Attempt to repair partial JSON by closing unclosed braces and brackets.
/// Also removes trailing commas before `}` or `]` (common LLM quirk).
fn repair_partial_json(text: &str) -> String {
    let mut result = text.to_string();
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for c in text.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => open_braces += 1,
            '}' => open_braces -= 1,
            '[' => open_brackets += 1,
            ']' => open_brackets -= 1,
            _ => {}
        }
    }

    // If we ended inside a string, close it
    if in_string {
        result.push('"');
    }

    // Close unclosed brackets then braces
    for _ in 0..open_brackets {
        result.push(']');
    }
    for _ in 0..open_braces {
        result.push('}');
    }

    // Remove trailing commas before } or ] (common LLM output quirk)
    strip_trailing_commas(&result)
}

/// Remove trailing commas before `}` or `]` in JSON text.
fn strip_trailing_commas(text: &str) -> String {
    static TRAILING_COMMA_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r",(\s*[}\]])").expect("trailing comma regex"));
    TRAILING_COMMA_RE.replace_all(text, "$1").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_valid_json() {
        let raw = r#"{"answer": "Rust is a systems language", "citations": ["[S1]"]}"#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();

        assert!(parsed.json.is_some());
        let j = parsed.json.unwrap();
        assert_eq!(j["answer"], "Rust is a systems language");
    }

    #[test]
    fn test_parse_json_with_markdown_fences() {
        let raw = "```json\n{\"answer\": \"hello\", \"citations\": []}\n```";
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();

        assert!(parsed.json.is_some());
        assert_eq!(parsed.json.unwrap()["answer"], "hello");
    }

    #[test]
    fn test_parse_partial_json_unclosed_brace() {
        let raw = r#"{"answer": "partial", "citations": ["[S1]"]"#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();

        assert!(parsed.json.is_some());
        let j = parsed.json.unwrap();
        assert_eq!(j["answer"], "partial");
    }

    #[test]
    fn test_parse_citations_extracted() {
        let raw = "The answer [S1] is based on [S3] and [S1] again.";
        let parsed = parse_structured_response(raw, None).unwrap();

        assert_eq!(parsed.citations, vec!["[S1]", "[S3]"]);
        assert!(parsed.json.is_none());
    }

    #[test]
    fn test_parse_abstention_detected() {
        let raw = "INSUFFICIENT_EVIDENCE\n- Need more data about X\n- Missing Y";
        let parsed = parse_structured_response(raw, None).unwrap();

        assert!(parsed.is_abstention);
    }

    #[test]
    fn test_parse_empty_response_error() {
        let result = parse_structured_response("", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_with_surrounding_text() {
        let raw = "Here is the result:\n{\"answer\": \"test\", \"citations\": []}\nDone!";
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();

        assert!(parsed.json.is_some());
        assert_eq!(parsed.json.unwrap()["answer"], "test");
    }

    #[test]
    fn test_parse_no_schema_returns_none_json() {
        let raw = "Just a plain text response with no JSON.";
        let parsed = parse_structured_response(raw, None).unwrap();

        assert!(parsed.json.is_none());
        assert_eq!(parsed.raw_text, raw);
    }

    #[test]
    fn test_parse_no_citations_empty_vec() {
        let raw = "No citations here.";
        let parsed = parse_structured_response(raw, None).unwrap();
        assert!(parsed.citations.is_empty());
    }

    #[test]
    fn test_parse_partial_json_unclosed_bracket_and_brace() {
        let raw = r#"{"items": ["a", "b""#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();

        assert!(parsed.json.is_some());
    }

    #[test]
    fn test_parse_not_abstention_when_missing() {
        let raw = "The answer is 42.";
        let parsed = parse_structured_response(raw, None).unwrap();
        assert!(!parsed.is_abstention);
    }

    #[test]
    fn test_strip_code_fences_plain() {
        assert_eq!(strip_code_fences("hello"), "hello");
    }

    #[test]
    fn test_strip_code_fences_json() {
        let input = "```json\n{\"a\": 1}\n```";
        assert_eq!(strip_code_fences(input), "{\"a\": 1}");
    }

    #[test]
    fn test_extract_first_json_object_nested() {
        let text = "before {\"a\": {\"b\": 1}} after";
        let result = extract_first_json_object(text).unwrap();
        assert_eq!(result, "{\"a\": {\"b\": 1}}");
    }

    #[test]
    fn test_extract_first_json_object_with_string_braces() {
        let text = r#"{"text": "a { b } c"}"#;
        let result = extract_first_json_object(text).unwrap();
        assert_eq!(result, text);
    }

    #[test]
    fn test_repair_partial_json_closes_braces() {
        let partial = r#"{"a": 1"#;
        let repaired = repair_partial_json(partial);
        assert!(serde_json::from_str::<Value>(&repaired).is_ok());
    }

    #[test]
    fn test_citations_deduped() {
        let citations = extract_citations("[S1] and [S1] and [S2]");
        assert_eq!(citations, vec!["[S1]", "[S2]"]);
    }

    #[test]
    fn test_parse_whitespace_only_error() {
        let result = parse_structured_response("   \n\t  ", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_trailing_comma() {
        let raw = r#"{"answer": "test", "citations": ["[S1]"],}"#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert!(parsed.json.is_some());
        assert_eq!(parsed.json.unwrap()["answer"], "test");
    }

    #[test]
    fn test_parse_json_array_not_object() {
        // JSON array when expecting object — should return None for json
        let raw = "[1, 2, 3]";
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert!(parsed.json.is_none());
    }

    #[test]
    fn test_parse_code_fences_no_closing() {
        let raw = "```json\n{\"answer\": \"works\"}";
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert!(parsed.json.is_some());
        assert_eq!(parsed.json.unwrap()["answer"], "works");
    }

    #[test]
    fn test_parse_multiple_json_objects_takes_first() {
        let raw = r#"{"a": 1} and also {"b": 2}"#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert!(parsed.json.is_some());
        let j = parsed.json.unwrap();
        assert_eq!(j["a"], 1);
        assert!(j.get("b").is_none());
    }

    #[test]
    fn test_parse_json_with_escaped_quotes() {
        let raw = r#"{"answer": "He said \"hello\" to her"}"#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert!(parsed.json.is_some());
    }

    #[test]
    fn test_parse_abstention_with_json() {
        let raw = r#"{"answer": "INSUFFICIENT_EVIDENCE", "citations": []}"#;
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert!(parsed.is_abstention);
        assert!(parsed.json.is_some());
    }

    #[test]
    fn test_parse_raw_text_always_preserved() {
        let raw = "```json\n{\"a\": 1}\n```";
        let schema = json!({"type": "object"});
        let parsed = parse_structured_response(raw, Some(&schema)).unwrap();
        assert_eq!(parsed.raw_text, raw);
        assert!(parsed.json.is_some());
    }

    #[test]
    fn test_parse_citations_high_numbers() {
        let raw = "See [S1], [S15], and [S100].";
        let parsed = parse_structured_response(raw, None).unwrap();
        assert_eq!(parsed.citations, vec!["[S1]", "[S15]", "[S100]"]);
    }

    #[test]
    fn test_strip_trailing_commas_nested() {
        let input = r#"{"a": [1, 2,], "b": {"c": 3,},}"#;
        let result = strip_trailing_commas(input);
        let parsed: serde_json::Result<Value> = serde_json::from_str(&result);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_repair_partial_json_truncated_string() {
        let partial = r#"{"answer": "hello wor"#;
        let repaired = repair_partial_json(partial);
        let parsed = serde_json::from_str::<Value>(&repaired);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_extract_first_json_no_brace() {
        assert!(extract_first_json_object("no json here").is_none());
    }
}
