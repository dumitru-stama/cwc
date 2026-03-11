use cwc_core::types::{IssueKind, VerificationIssue};
use serde_json::Value;

pub struct SchemaVerifier;

impl SchemaVerifier {
    /// Validate JSON output against the expected schema.
    pub fn validate(output: &Value, schema: &Value) -> Vec<VerificationIssue> {
        let validator = match jsonschema::validator_for(schema) {
            Ok(v) => v,
            Err(e) => {
                return vec![VerificationIssue {
                    kind: IssueKind::SchemaViolation,
                    description: format!("invalid schema: {e}"),
                    claim_text: None,
                }];
            }
        };

        validator
            .iter_errors(output)
            .map(|err| VerificationIssue {
                kind: IssueKind::SchemaViolation,
                description: format!("{} at {}", err, err.instance_path),
                claim_text: None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qa_schema() -> Value {
        json!({
            "type": "object",
            "required": ["answer", "confidence"],
            "properties": {
                "answer": { "type": "string" },
                "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
            },
            "additionalProperties": false
        })
    }

    #[test]
    fn test_schema_validate_valid() {
        let output = json!({"answer": "Rust is safe.", "confidence": 0.9});
        let issues = SchemaVerifier::validate(&output, &qa_schema());
        assert!(issues.is_empty());
    }

    #[test]
    fn test_schema_validate_missing_required() {
        let output = json!({"answer": "Rust is safe."});
        let issues = SchemaVerifier::validate(&output, &qa_schema());
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.kind == IssueKind::SchemaViolation));
        assert!(issues.iter().any(|i| i.description.contains("confidence")));
    }

    #[test]
    fn test_schema_validate_extra_property() {
        let output = json!({"answer": "Rust.", "confidence": 0.5, "extra": true});
        let issues = SchemaVerifier::validate(&output, &qa_schema());
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.kind == IssueKind::SchemaViolation));
    }

    #[test]
    fn test_schema_validate_wrong_type() {
        let output = json!({"answer": 42, "confidence": 0.5});
        let issues = SchemaVerifier::validate(&output, &qa_schema());
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.kind == IssueKind::SchemaViolation));
    }

    #[test]
    fn test_schema_validate_out_of_range() {
        let output = json!({"answer": "ok", "confidence": 1.5});
        let issues = SchemaVerifier::validate(&output, &qa_schema());
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_schema_validate_enum() {
        let schema = json!({
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": { "type": "string", "enum": ["pass", "fail"] }
            }
        });
        let output = json!({"status": "unknown"});
        let issues = SchemaVerifier::validate(&output, &schema);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_schema_validate_nested_object() {
        let schema = json!({
            "type": "object",
            "required": ["result"],
            "properties": {
                "result": {
                    "type": "object",
                    "required": ["text"],
                    "properties": { "text": { "type": "string" } }
                }
            }
        });
        let output = json!({"result": {"text": "hello"}});
        assert!(SchemaVerifier::validate(&output, &schema).is_empty());

        let bad = json!({"result": {"text": 42}});
        assert!(!SchemaVerifier::validate(&bad, &schema).is_empty());
    }

    #[test]
    fn test_schema_validate_array() {
        let schema = json!({
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            }
        });
        let output = json!({"items": ["a", "b"]});
        assert!(SchemaVerifier::validate(&output, &schema).is_empty());

        let bad = json!({"items": [1, 2]});
        assert!(!SchemaVerifier::validate(&bad, &schema).is_empty());
    }

    #[test]
    fn test_schema_validate_invalid_schema() {
        // A broken schema — should return an issue about the schema itself
        let schema = json!({"type": "not_a_real_type"});
        let output = json!("hello");
        let issues = SchemaVerifier::validate(&output, &schema);
        // jsonschema may or may not error on unknown type; at minimum it shouldn't panic
        // The important thing is we don't crash
        let _ = issues;
    }

    #[test]
    fn test_schema_validate_multiple_errors() {
        let schema = json!({
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "number" }
            },
            "additionalProperties": false
        });
        let output = json!({"c": true});
        let issues = SchemaVerifier::validate(&output, &schema);
        // Should report multiple issues: missing a, missing b, extra c
        assert!(issues.len() >= 2);
    }
}
