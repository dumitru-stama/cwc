use std::io::{self, BufRead, Write};

use crate::compiler::{ContextWindowCompiler, CwcOutput};

/// Parsed REPL command.
#[derive(Debug, PartialEq)]
pub enum ReplCommand {
    Query(String),
    Clear,
    Memory,
    Debug,
    Budget,
    Sources,
    Quit,
    Help,
    Empty,
}

/// Parse a REPL input line into a command.
pub fn parse_repl_command(input: &str) -> ReplCommand {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return ReplCommand::Empty;
    }

    match trimmed {
        "/clear" => ReplCommand::Clear,
        "/memory" => ReplCommand::Memory,
        "/debug" => ReplCommand::Debug,
        "/budget" => ReplCommand::Budget,
        "/sources" => ReplCommand::Sources,
        "/quit" | "/exit" | "/q" => ReplCommand::Quit,
        "/help" | "/?" => ReplCommand::Help,
        s if s.starts_with('/') => {
            eprintln!("Unknown command: {s}. Type /help for available commands.");
            ReplCommand::Empty
        }
        _ => ReplCommand::Query(trimmed.to_string()),
    }
}

/// Run the interactive REPL loop.
pub async fn run_repl(compiler: &ContextWindowCompiler) -> cwc_core::error::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut debug_mode = false;
    let mut last_output: Option<CwcOutput> = None;

    println!("CWC Interactive REPL (type /help for commands, /quit to exit)");
    println!();

    loop {
        print!("cwc> ");
        stdout.flush()?;

        let mut line = String::new();
        let bytes = stdin.lock().read_line(&mut line)?;
        if bytes == 0 {
            // EOF
            break;
        }

        match parse_repl_command(&line) {
            ReplCommand::Empty => continue,
            ReplCommand::Quit => {
                println!("Goodbye.");
                break;
            }
            ReplCommand::Help => {
                println!("Commands:");
                println!("  /clear   — Reset conversation history");
                println!("  /memory  — Show current memories");
                println!("  /debug   — Toggle debug output");
                println!("  /budget  — Show last budget report");
                println!("  /sources — Show source chunks from last query");
                println!("  /quit    — Exit");
                println!("  <text>   — Run a query");
            }
            ReplCommand::Clear => {
                compiler.clear_conversation()?;
                last_output = None;
                println!("Conversation history cleared.");
            }
            ReplCommand::Memory => {
                if let Some(store) = &compiler.long_term {
                    match store.list_all() {
                        Ok(entries) => {
                            if entries.is_empty() {
                                println!("No memories stored.");
                            } else {
                                for entry in &entries {
                                    println!(
                                        "  [{}] {} = {}",
                                        entry.category.as_str(),
                                        entry.key,
                                        entry.value,
                                    );
                                }
                            }
                        }
                        Err(e) => eprintln!("Error reading memories: {e}"),
                    }
                } else {
                    println!("No memory store configured.");
                }
            }
            ReplCommand::Debug => {
                debug_mode = !debug_mode;
                println!("Debug mode: {}", if debug_mode { "ON" } else { "OFF" });
            }
            ReplCommand::Budget => {
                if let Some(output) = &last_output {
                    let r = &output.budget_report;
                    println!("Budget Report:");
                    println!("  Total budget:     {} tokens", r.context_window);
                    println!("  Sources budget:   {} tokens", r.sources_budget);
                    println!("  Sources used:     {} tokens", r.sources_used);
                    println!("  Instruction:      {} tokens", r.instruction_tokens);
                    println!("  Memory:           {} tokens", r.memory_tokens);
                    println!("  Chunks selected:  {}", r.chunks_selected);
                    println!("  Chunks dropped:   {}", r.chunks_dropped);
                    println!("  Utilization:      {:.1}%", r.utilization * 100.0);
                } else {
                    println!("No query has been run yet.");
                }
            }
            ReplCommand::Sources => {
                if let Some(output) = &last_output {
                    if output.retrieval_hits.is_empty() {
                        println!("No source chunks retrieved.");
                    } else {
                        for (i, hit) in output.retrieval_hits.iter().enumerate() {
                            let preview: String = hit.chunk.text.chars().take(80).collect();
                            println!(
                                "  [S{}] [{:.3}] {} — {}",
                                i + 1,
                                hit.score_fused,
                                hit.chunk.source_path,
                                preview,
                            );
                        }
                    }
                } else {
                    println!("No query has been run yet.");
                }
            }
            ReplCommand::Query(query) => {
                // Use streaming: print tokens as they arrive
                let on_token: cwc_core::traits::StreamCallback = Box::new(|chunk: &str| {
                    use std::io::Write;
                    print!("{chunk}");
                    let _ = std::io::stdout().flush();
                });
                match compiler.query_streaming(&query, None, on_token).await {
                    Ok(output) => {
                        // Newline after streamed output
                        println!();
                        println!();

                        if !output.response.citations.is_empty() {
                            println!(
                                "Citations: {}",
                                output.response.citations.join(", ")
                            );
                        }

                        if output.response.is_abstention {
                            println!("(Model abstained: INSUFFICIENT_EVIDENCE)");
                        }

                        if debug_mode {
                            let t = &output.timing;
                            println!(
                                "Timing: retrieval={}ms compile={}ms generate={}ms verify={}ms total={}ms",
                                t.retrieval_ms, t.compilation_ms, t.generation_ms,
                                t.verification_ms, t.total_ms,
                            );
                            println!(
                                "Verdict: {} | Attempts: {} | Chunks: {}",
                                if output.verdict.is_pass() {
                                    "PASS"
                                } else if output.verdict.is_abstain() {
                                    "ABSTAIN"
                                } else {
                                    "FAIL"
                                },
                                output.attempts,
                                output.retrieval_hits.len(),
                            );
                        }

                        last_output = Some(output);
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_repl_commands() {
        assert_eq!(parse_repl_command("/clear"), ReplCommand::Clear);
        assert_eq!(parse_repl_command("/memory"), ReplCommand::Memory);
        assert_eq!(parse_repl_command("/debug"), ReplCommand::Debug);
        assert_eq!(parse_repl_command("/budget"), ReplCommand::Budget);
        assert_eq!(parse_repl_command("/sources"), ReplCommand::Sources);
        assert_eq!(parse_repl_command("/quit"), ReplCommand::Quit);
        assert_eq!(parse_repl_command("/exit"), ReplCommand::Quit);
        assert_eq!(parse_repl_command("/q"), ReplCommand::Quit);
        assert_eq!(parse_repl_command("/help"), ReplCommand::Help);
        assert_eq!(parse_repl_command("/?"), ReplCommand::Help);
        assert_eq!(parse_repl_command(""), ReplCommand::Empty);
        assert_eq!(parse_repl_command("  "), ReplCommand::Empty);
    }

    #[test]
    fn test_parse_repl_query() {
        assert_eq!(
            parse_repl_command("What is Rust?"),
            ReplCommand::Query("What is Rust?".into()),
        );
    }

    #[test]
    fn test_parse_repl_query_with_whitespace() {
        assert_eq!(
            parse_repl_command("  hello world  "),
            ReplCommand::Query("hello world".into()),
        );
    }

    #[test]
    fn test_parse_repl_unknown_slash_command() {
        // Unknown /commands should return Empty (with a warning printed to stderr)
        assert_eq!(parse_repl_command("/typo"), ReplCommand::Empty);
        assert_eq!(parse_repl_command("/unknown"), ReplCommand::Empty);
        assert_eq!(parse_repl_command("/clearr"), ReplCommand::Empty);
    }

    #[test]
    fn test_parse_repl_non_slash_prefix() {
        // Text starting with non-slash should be a query
        assert_eq!(
            parse_repl_command("?query"),
            ReplCommand::Query("?query".into()),
        );
    }
}
