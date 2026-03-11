use crate::types::Section;

/// Parse markdown content into sections based on heading hierarchy.
///
/// Returns `(front_matter, sections)`. Front-matter is YAML between `---`
/// delimiters at the start of the file.
pub fn parse_markdown(content: &str) -> (Option<String>, Vec<Section>) {
    let (front_matter, body) = extract_front_matter(content);
    let body_offset = content.len() - body.len();
    let sections = parse_headings(body, body_offset);
    (front_matter, sections)
}

fn extract_front_matter(content: &str) -> (Option<String>, &str) {
    if !content.starts_with("---") {
        return (None, content);
    }
    // Find closing ---
    let after_first = &content[3..];
    if let Some(end) = after_first.find("\n---") {
        let fm = after_first[..end].trim().to_string();
        let rest_start = 3 + end + 4; // skip "---" + "\n---"
        let rest = if rest_start < content.len() {
            // Skip optional newline after closing ---
            let r = &content[rest_start..];
            r.strip_prefix('\n').unwrap_or(r)
        } else {
            ""
        };
        (Some(fm), rest)
    } else {
        (None, content)
    }
}

fn parse_headings(body: &str, base_offset: usize) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut heading_stack: Vec<(usize, String)> = Vec::new(); // (level, title)
    let mut current_text = String::new();
    let mut section_start = 0;
    let mut in_code_block = false;
    let mut first = true;

    for line in body.lines() {
        let line_with_nl = if first {
            first = false;
            line.to_string()
        } else {
            format!("\n{line}")
        };

        // Track fenced code blocks
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            current_text.push_str(&line_with_nl);
            continue;
        }

        if in_code_block {
            current_text.push_str(&line_with_nl);
            continue;
        }

        // Check for ATX heading
        if let Some((level, title)) = parse_atx_heading(line) {
            // Flush previous section (only if it has content)
            let text = current_text.trim_end().to_string();
            if !text.is_empty() {
                let path = heading_stack.iter().map(|(_, t)| t.clone()).collect();
                sections.push(Section {
                    path,
                    text,
                    char_offset: base_offset + section_start,
                    char_len: current_text.len(),
                });
            }

            // Update heading stack
            while heading_stack
                .last()
                .is_some_and(|(l, _)| *l >= level)
            {
                heading_stack.pop();
            }
            heading_stack.push((level, title));
            current_text = String::new();
            section_start = (line.as_ptr() as usize) - (body.as_ptr() as usize);
        } else {
            current_text.push_str(&line_with_nl);
        }
    }

    // Flush final section
    let text = current_text.trim_end().to_string();
    if !text.is_empty() {
        let path = heading_stack.iter().map(|(_, t)| t.clone()).collect();
        sections.push(Section {
            path,
            text,
            char_offset: base_offset + section_start,
            char_len: current_text.len(),
        });
    }

    // If no headings found, return entire body as single section
    if sections.is_empty() && !body.is_empty() {
        sections.push(Section {
            path: vec![],
            text: body.to_string(),
            char_offset: base_offset,
            char_len: body.len(),
        });
    }

    sections
}

fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    // Must have space after hashes (or be just hashes)
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim().to_string();
    Some((hashes, title))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_heading_hierarchy() {
        let md = "# H1\n\nSome text.\n\n## H2a\n\nUnder H2a.\n\n## H2b\n\nUnder H2b.\n\n### H3\n\nDeep.\n";
        let (fm, sections) = parse_markdown(md);
        assert!(fm.is_none());

        assert_eq!(sections.len(), 4);
        assert_eq!(sections[0].path, vec!["H1"]);
        assert!(sections[0].text.contains("Some text."));
        assert_eq!(sections[1].path, vec!["H1", "H2a"]);
        assert!(sections[1].text.contains("Under H2a."));
        assert_eq!(sections[2].path, vec!["H1", "H2b"]);
        assert!(sections[2].text.contains("Under H2b."));
        assert_eq!(sections[3].path, vec!["H1", "H2b", "H3"]);
        assert!(sections[3].text.contains("Deep."));
    }

    #[test]
    fn test_markdown_front_matter() {
        let md = "---\ntitle: Test\nauthor: Alice\n---\n\n# Heading\n\nContent.\n";
        let (fm, sections) = parse_markdown(md);
        assert_eq!(fm.unwrap(), "title: Test\nauthor: Alice");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].path, vec!["Heading"]);
    }

    #[test]
    fn test_markdown_code_block_preserved() {
        let md = "# Code\n\nBefore.\n\n```rust\nfn main() {\n    # not a heading\n}\n```\n\nAfter.\n";
        let (_, sections) = parse_markdown(md);
        assert_eq!(sections.len(), 1);
        // Code block content should be in the section text, not split
        assert!(sections[0].text.contains("fn main()"));
        assert!(sections[0].text.contains("# not a heading"));
    }

    #[test]
    fn test_markdown_no_headings() {
        let md = "Just some plain text\nwith no headings at all.\n";
        let (_, sections) = parse_markdown(md);
        assert_eq!(sections.len(), 1);
        assert!(sections[0].path.is_empty());
        assert!(sections[0].text.contains("Just some plain text"));
    }

    #[test]
    fn test_markdown_heading_level_pop() {
        // H1 → H3 → H2 should pop H3 from stack
        let md = "# A\n\nA text.\n\n### Deep\n\nDeep text.\n\n## Sibling\n\nSibling text.\n";
        let (_, sections) = parse_markdown(md);
        assert_eq!(sections[0].path, vec!["A"]);
        assert_eq!(sections[1].path, vec!["A", "Deep"]);
        assert_eq!(sections[2].path, vec!["A", "Sibling"]);
    }

    #[test]
    fn test_markdown_empty() {
        let (fm, sections) = parse_markdown("");
        assert!(fm.is_none());
        assert!(sections.is_empty());
    }

    #[test]
    fn test_markdown_consecutive_headings_no_empty_sections() {
        // H1 has content, H2 has none, H3 has content
        let md = "# A\n\nContent A.\n\n## B\n\n## C\n\nContent C.\n";
        let (_, sections) = parse_markdown(md);
        // B has no content — should not produce an empty section
        for s in &sections {
            assert!(!s.text.is_empty(), "got empty section with path {:?}", s.path);
        }
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].path, vec!["A"]);
        assert!(sections[0].text.contains("Content A."));
        assert_eq!(sections[1].path, vec!["A", "C"]);
        assert!(sections[1].text.contains("Content C."));
    }

    #[test]
    fn test_markdown_heading_trailing_hashes() {
        let md = "## Title ##\n\nBody.\n";
        let (_, sections) = parse_markdown(md);
        assert_eq!(sections[0].path, vec!["Title"]);
    }

    #[test]
    fn test_atx_heading_hash_only() {
        assert!(parse_atx_heading("#").is_some());
        assert_eq!(parse_atx_heading("#").unwrap(), (1, String::new()));
    }

    #[test]
    fn test_atx_heading_no_space_is_not_heading() {
        // "#tag" is not a heading
        assert!(parse_atx_heading("#tag").is_none());
    }
}
