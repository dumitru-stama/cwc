use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    QuestionAnswer,
    Summary,
    Extraction,
    Classification,
    FreeForm,
}

/// Generate JSON schema for the given task type.
pub fn schema_for_task(task: TaskType) -> serde_json::Value {
    match task {
        TaskType::QuestionAnswer => json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" },
                "citations": {
                    "type": "array",
                    "items": { "type": "string", "pattern": "^\\[S\\d+\\]$" }
                },
                "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
                "reasoning": { "type": "string" }
            },
            "required": ["answer", "citations"],
            "additionalProperties": false
        }),
        TaskType::Summary => json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" },
                "key_points": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "citations": {
                    "type": "array",
                    "items": { "type": "string", "pattern": "^\\[S\\d+\\]$" }
                }
            },
            "required": ["summary", "citations"],
            "additionalProperties": false
        }),
        TaskType::Extraction => json!({
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "type": { "type": "string" },
                            "value": { "type": "string" }
                        },
                        "required": ["name", "type"]
                    }
                },
                "citations": {
                    "type": "array",
                    "items": { "type": "string", "pattern": "^\\[S\\d+\\]$" }
                }
            },
            "required": ["entities", "citations"],
            "additionalProperties": false
        }),
        TaskType::Classification => json!({
            "type": "object",
            "properties": {
                "label": { "type": "string" },
                "reasoning": { "type": "string" },
                "citations": {
                    "type": "array",
                    "items": { "type": "string", "pattern": "^\\[S\\d+\\]$" }
                }
            },
            "required": ["label", "citations"],
            "additionalProperties": false
        }),
        TaskType::FreeForm => json!({
            "type": "object",
            "properties": {
                "response": { "type": "string" },
                "citations": {
                    "type": "array",
                    "items": { "type": "string", "pattern": "^\\[S\\d+\\]$" }
                }
            },
            "required": ["response", "citations"],
            "additionalProperties": false
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_qa_has_required_fields() {
        let schema = schema_for_task(TaskType::QuestionAnswer);
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("answer")));
        assert!(required.contains(&json!("citations")));
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn test_schema_summary_has_required_fields() {
        let schema = schema_for_task(TaskType::Summary);
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("summary")));
        assert!(required.contains(&json!("citations")));
    }

    #[test]
    fn test_schema_extraction_has_entities() {
        let schema = schema_for_task(TaskType::Extraction);
        assert!(schema["properties"]["entities"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("entities")));
    }

    #[test]
    fn test_schema_classification_has_label() {
        let schema = schema_for_task(TaskType::Classification);
        assert!(schema["properties"]["label"].is_object());
    }

    #[test]
    fn test_schema_freeform_has_response() {
        let schema = schema_for_task(TaskType::FreeForm);
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("response")));
    }

    #[test]
    fn test_schema_all_valid_json() {
        for task in [
            TaskType::QuestionAnswer,
            TaskType::Summary,
            TaskType::Extraction,
            TaskType::Classification,
            TaskType::FreeForm,
        ] {
            let schema = schema_for_task(task);
            assert_eq!(schema["type"], json!("object"));
            // Should serialize cleanly
            let s = serde_json::to_string_pretty(&schema).unwrap();
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_schema_citations_pattern() {
        let schema = schema_for_task(TaskType::QuestionAnswer);
        let pattern = schema["properties"]["citations"]["items"]["pattern"]
            .as_str()
            .unwrap();
        assert_eq!(pattern, "^\\[S\\d+\\]$");
    }
}
