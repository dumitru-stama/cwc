use cwc_core::error::{CwcError, Result};
use serde_json::Value;

/// Convert a JSON Schema to a GBNF grammar string for llama.cpp constrained decoding.
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut rules = Vec::new();
    let root_rule = schema_to_rule(schema, "root", &mut rules)?;
    rules.insert(0, root_rule);

    // Add common primitives
    rules.push(r#"ws ::= [ \t\n]*"#.to_string());
    rules.push(r#"string ::= "\"" ([^"\\] | "\\" .)* "\""#.to_string());
    rules.push(r#"number ::= "-"? [0-9]+ ("." [0-9]+)?"#.to_string());
    rules.push(r#"integer ::= "-"? [0-9]+"#.to_string());
    rules.push(r#"boolean ::= ("true" | "false")"#.to_string());
    rules.push(r#"null ::= "null""#.to_string());

    Ok(rules.join("\n"))
}

fn schema_to_rule(schema: &Value, name: &str, rules: &mut Vec<String>) -> Result<String> {
    let typ = schema.get("type").and_then(|v| v.as_str());

    // Handle enum constraint
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        let alts: Vec<String> = enum_values
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| {
                let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
                format!("\"\\\"{}\\\"\"", escaped)
            })
            .collect();
        if alts.is_empty() {
            return Err(CwcError::Llm(
                "enum has no string values (numeric/boolean enums not supported)".into(),
            ));
        }
        return Ok(format!("{name} ::= ({})", alts.join(" | ")));
    }

    match typ {
        Some("object") => object_rule(schema, name, rules),
        Some("array") => array_rule(schema, name, rules),
        Some("string") => Ok(format!("{name} ::= string")),
        Some("number") => Ok(format!("{name} ::= number")),
        Some("integer") => Ok(format!("{name} ::= integer")),
        Some("boolean") => Ok(format!("{name} ::= boolean")),
        Some("null") => Ok(format!("{name} ::= null")),
        Some(other) => Err(CwcError::Llm(format!("unsupported JSON Schema type: {other}"))),
        None => {
            // No type specified — accept any JSON value
            Ok(format!("{name} ::= (string | number | boolean | null)"))
        }
    }
}

fn object_rule(schema: &Value, name: &str, rules: &mut Vec<String>) -> Result<String> {
    let properties = schema.get("properties").and_then(|v| v.as_object());

    let props = match properties {
        Some(p) => p,
        None => {
            // Object with no properties — just braces
            return Ok(format!("{name} ::= \"{{\" ws \"}}\""));
        }
    };

    let mut prop_names: Vec<&String> = props.keys().collect();
    prop_names.sort(); // deterministic output

    // Build property rules. Note: GBNF grammars emit all properties (required
    // and optional) since GBNF has no clean way to express truly optional fields
    // without combinatorial alternation. For constrained decoding this is fine —
    // the LLM will generate all fields, which is preferable to missing them.
    let mut prop_parts = Vec::new();

    for key in &prop_names {
        let prop_schema = &props[*key];
        let rule_name = format!("{name}-{}", sanitize_rule_name(key));
        let prop_rule = schema_to_rule(prop_schema, &rule_name, rules)?;
        rules.push(prop_rule);

        let escaped_key = key.replace('\\', "\\\\").replace('"', "\\\"");
        prop_parts.push(format!("\"\\\"{}\\\"\" ws \":\" ws {rule_name}", escaped_key));
    }

    // Join with comma separators inside braces
    let mut parts = Vec::new();
    parts.push("\"{\"".to_string());
    parts.push("ws".to_string());

    for (i, part) in prop_parts.iter().enumerate() {
        if i > 0 {
            parts.push("\",\"".to_string());
            parts.push("ws".to_string());
        }
        parts.push(part.clone());
    }

    parts.push("ws".to_string());
    parts.push("\"}\"".to_string());

    Ok(format!("{name} ::= {}", parts.join(" ")))
}

/// Sanitize a property name for use in a GBNF rule name.
/// Replaces non-alphanumeric chars with hyphens.
fn sanitize_rule_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect()
}

fn array_rule(schema: &Value, name: &str, rules: &mut Vec<String>) -> Result<String> {
    let items_schema = schema.get("items");

    let item_rule_name = format!("{name}-item");
    match items_schema {
        Some(item_schema) => {
            let item_rule = schema_to_rule(item_schema, &item_rule_name, rules)?;
            rules.push(item_rule);
        }
        None => {
            // Untyped array items — accept any value
            rules.push(format!(
                "{item_rule_name} ::= (string | number | boolean | null)"
            ));
        }
    }

    Ok(format!(
        "{name} ::= \"[\" ws ({item_rule_name} (\",\" ws {item_rule_name})*)? \"]\"",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_gbnf_simple_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root ::="));
        assert!(gbnf.contains("\\\"answer\\\""));
        assert!(gbnf.contains("string ::="));
        assert!(gbnf.contains("ws ::="));
    }

    #[test]
    fn test_gbnf_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "integer" }
                    },
                    "required": ["value"]
                }
            },
            "required": ["result"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root-result ::="));
        assert!(gbnf.contains("\\\"value\\\""));
        assert!(gbnf.contains("integer ::="));
    }

    #[test]
    fn test_gbnf_array_of_strings() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["items"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root-items ::= \"[\""));
        assert!(gbnf.contains("root-items-item ::= string"));
    }

    #[test]
    fn test_gbnf_enum_constraint() {
        let schema = json!({
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"]
                }
            },
            "required": ["level"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root-level ::="));
        assert!(gbnf.contains("\\\"high\\\""));
        assert!(gbnf.contains("\\\"medium\\\""));
        assert!(gbnf.contains("\\\"low\\\""));
    }

    #[test]
    fn test_gbnf_qa_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" },
                "citations": {
                    "type": "array",
                    "items": { "type": "string", "pattern": "^\\[S\\d+\\]$" }
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"]
                },
                "reasoning": { "type": "string" }
            },
            "required": ["answer", "citations"],
            "additionalProperties": false
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root ::="));
        assert!(gbnf.contains("\\\"answer\\\""));
        assert!(gbnf.contains("\\\"citations\\\""));
        // Should have array rule for citations
        assert!(gbnf.contains("root-citations ::= \"[\""));
    }

    #[test]
    fn test_gbnf_boolean_and_number() {
        let schema = json!({
            "type": "object",
            "properties": {
                "score": { "type": "number" },
                "valid": { "type": "boolean" }
            },
            "required": ["score", "valid"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root-score ::= number"));
        assert!(gbnf.contains("root-valid ::= boolean"));
        assert!(gbnf.contains("number ::="));
        assert!(gbnf.contains("boolean ::="));
    }

    #[test]
    fn test_gbnf_empty_object() {
        let schema = json!({ "type": "object" });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root ::= \"{\" ws \"}\""));
    }

    #[test]
    fn test_gbnf_unsupported_type_error() {
        let schema = json!({ "type": "foobar" });
        let result = json_schema_to_gbnf(&schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_gbnf_no_type_fallback() {
        // Schema with no "type" field should produce a generic value rule
        let schema = json!({});
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root ::="));
        assert!(gbnf.contains("string"));
        assert!(gbnf.contains("number"));
    }

    #[test]
    fn test_gbnf_array_no_items_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "data": { "type": "array" }
            },
            "required": ["data"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        // Untyped array items should accept any value
        assert!(gbnf.contains("root-data-item ::= (string | number | boolean | null)"));
    }

    #[test]
    fn test_gbnf_property_with_underscore() {
        let schema = json!({
            "type": "object",
            "properties": {
                "my_field": { "type": "string" }
            },
            "required": ["my_field"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        // Rule name should have hyphen, key should keep underscore
        assert!(gbnf.contains("root-my-field ::= string"));
        assert!(gbnf.contains("\\\"my_field\\\""));
    }

    #[test]
    fn test_gbnf_deeply_nested() {
        // Object containing array of objects
        let schema = json!({
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "score": { "type": "number" }
                        },
                        "required": ["name", "score"]
                    }
                }
            },
            "required": ["results"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root-results ::= \"[\""));
        assert!(gbnf.contains("root-results-item ::="));
        assert!(gbnf.contains("root-results-item-name ::= string"));
        assert!(gbnf.contains("root-results-item-score ::= number"));
    }

    #[test]
    fn test_gbnf_all_properties_emitted() {
        // Optional properties are still emitted (by design)
        let schema = json!({
            "type": "object",
            "properties": {
                "required_field": { "type": "string" },
                "optional_field": { "type": "integer" }
            },
            "required": ["required_field"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        // Both fields appear in the grammar
        assert!(gbnf.contains("\\\"required_field\\\""));
        assert!(gbnf.contains("\\\"optional_field\\\""));
    }

    #[test]
    fn test_gbnf_null_type() {
        let schema = json!({
            "type": "object",
            "properties": {
                "empty": { "type": "null" }
            },
            "required": ["empty"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root-empty ::= null"));
        assert!(gbnf.contains("null ::= \"null\""));
    }
}
