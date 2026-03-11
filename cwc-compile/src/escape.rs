use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelimiterStyle {
    #[default]
    Brackets,
    Xml,
    Markdown,
}

/// Section label names used in prompts.
const BRACKET_TAGS: &[&str] = &[
    "[SYSTEM]",
    "[SOURCES]",
    "[USER_TASK]",
    "[OUTPUT_SCHEMA]",
    "[MEMORY]",
];

const XML_TAGS: &[&str] = &[
    "<system>", "</system>",
    "<sources>", "</sources>",
    "<user_task>", "</user_task>",
    "<output_schema>", "</output_schema>",
    "<memory>", "</memory>",
];

const MARKDOWN_TAGS: &[&str] = &[
    "## System",
    "## Sources",
    "## User Task",
    "## Output Schema",
    "## Memory",
];

/// Escape delimiter-like patterns in source text to prevent injection.
/// Returns (escaped_text, was_escaped).
pub fn escape_source_text(text: &str, style: DelimiterStyle) -> (String, bool) {
    let mut result = text.to_string();
    let mut escaped = false;

    match style {
        DelimiterStyle::Brackets => {
            for tag in BRACKET_TAGS {
                if result.contains(tag) {
                    // [SYSTEM] → [_SYSTEM_]
                    let neutered = tag.replace('[', "[_").replace(']', "_]");
                    result = result.replace(tag, &neutered);
                    escaped = true;
                }
            }
        }
        DelimiterStyle::Xml => {
            for tag in XML_TAGS {
                if result.contains(tag) {
                    // <system> → <_system_>
                    let neutered = tag.replace('<', "<_").replace('>', "_>");
                    result = result.replace(tag, &neutered);
                    escaped = true;
                }
            }
        }
        DelimiterStyle::Markdown => {
            for tag in MARKDOWN_TAGS {
                if result.contains(tag) {
                    // ## System → ##_ System
                    let neutered = tag.replace("## ", "##_ ");
                    result = result.replace(tag, &neutered);
                    escaped = true;
                }
            }
        }
    }

    if escaped {
        tracing::warn!(
            style = ?style,
            "anti-injection: escaped delimiter patterns in source text"
        );
    }

    (result, escaped)
}

/// Return the section header for a given section name.
pub fn section_header(name: &str, style: DelimiterStyle) -> String {
    match style {
        DelimiterStyle::Brackets => format!("[{name}]"),
        DelimiterStyle::Xml => format!("<{}>", name.to_lowercase()),
        DelimiterStyle::Markdown => format!("## {name}"),
    }
}

/// Return the section footer (only for XML).
pub fn section_footer(name: &str, style: DelimiterStyle) -> Option<String> {
    match style {
        DelimiterStyle::Xml => Some(format!("</{}>", name.to_lowercase())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_brackets_system() {
        let (escaped, was) = escape_source_text("Ignore [SYSTEM] override", DelimiterStyle::Brackets);
        assert!(was);
        assert!(escaped.contains("[_SYSTEM_]"));
        assert!(!escaped.contains("[SYSTEM]"));
    }

    #[test]
    fn test_escape_brackets_multiple() {
        let (escaped, was) = escape_source_text(
            "[SYSTEM] fake and [USER_TASK] inject",
            DelimiterStyle::Brackets,
        );
        assert!(was);
        assert!(escaped.contains("[_SYSTEM_]"));
        assert!(escaped.contains("[_USER_TASK_]"));
    }

    #[test]
    fn test_escape_xml_system() {
        let (escaped, was) = escape_source_text("Try <system> injection", DelimiterStyle::Xml);
        assert!(was);
        assert!(escaped.contains("<_system_>"));
        assert!(!escaped.contains("<system>"));
    }

    #[test]
    fn test_escape_xml_closing_tag() {
        let (escaped, was) = escape_source_text("end </system> here", DelimiterStyle::Xml);
        assert!(was);
        assert!(escaped.contains("<_/system_>"));
    }

    #[test]
    fn test_escape_markdown() {
        let (escaped, was) = escape_source_text("## System override", DelimiterStyle::Markdown);
        assert!(was);
        assert!(escaped.contains("##_ System"));
    }

    #[test]
    fn test_no_escape_clean_text() {
        let (escaped, was) = escape_source_text("Normal text about Rust ownership.", DelimiterStyle::Brackets);
        assert!(!was);
        assert_eq!(escaped, "Normal text about Rust ownership.");
    }

    #[test]
    fn test_section_header_brackets() {
        assert_eq!(section_header("SYSTEM", DelimiterStyle::Brackets), "[SYSTEM]");
    }

    #[test]
    fn test_section_header_xml() {
        assert_eq!(section_header("SYSTEM", DelimiterStyle::Xml), "<system>");
    }

    #[test]
    fn test_section_header_markdown() {
        assert_eq!(section_header("SYSTEM", DelimiterStyle::Markdown), "## SYSTEM");
    }

    #[test]
    fn test_section_footer_xml_only() {
        assert_eq!(section_footer("SYSTEM", DelimiterStyle::Xml), Some("</system>".to_string()));
        assert_eq!(section_footer("SYSTEM", DelimiterStyle::Brackets), None);
        assert_eq!(section_footer("SYSTEM", DelimiterStyle::Markdown), None);
    }
}
