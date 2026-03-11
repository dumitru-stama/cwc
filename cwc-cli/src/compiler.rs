use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing::info;

use cwc_compile::reorder::ReorderStrategy;
use cwc_compile::scaffold::PromptScaffold;
use cwc_compile::CompileParams;
use cwc_core::cache::{CacheStats, RetrievalCache, PromptCache};
use cwc_core::config::CwcConfig;
use cwc_core::error::{CwcError, Result};
use cwc_core::traits::{Embedder, LlmClient, Reranker, Retriever, TokenCounter, Verifier};
use cwc_core::types::{ChatMessage, CompiledContext, RetrievalHit, TokenBudget, Verdict};
use cwc_llm::grammar::json_schema_to_gbnf;
use cwc_llm::parse::{parse_structured_response, ParsedResponse};
use cwc_memory::budget::compile_memory;
use cwc_memory::conversation::ConversationMemory;
use cwc_memory::extract::extract_memories;
use cwc_memory::store::MemoryStore;
use cwc_verify::revision::RevisionLoop;

/// Output from a full query pipeline run.
pub struct CwcOutput {
    pub response: ParsedResponse,
    pub verdict: Verdict,
    pub budget_report: cwc_compile::report::BudgetReport,
    pub retrieval_hits: Vec<RetrievalHit>,
    pub attempts: usize,
    pub timing: PipelineTiming,
}

/// Timing of each pipeline stage in milliseconds.
#[derive(Debug, Clone, Default)]
pub struct PipelineTiming {
    pub retrieval_ms: u64,
    pub compilation_ms: u64,
    pub generation_ms: u64,
    pub verification_ms: u64,
    pub total_ms: u64,
}

/// Report from ingesting documents.
pub struct IngestReport {
    pub files_processed: usize,
    pub chunks_created: usize,
    pub index_path: PathBuf,
    pub chunks_path: PathBuf,
    /// Whether this was an incremental ingest.
    pub incremental: bool,
    /// Number of files skipped (unchanged) in incremental mode.
    pub files_skipped: usize,
}

/// The main orchestrator that ties all pipeline stages together.
#[allow(dead_code)]
pub struct ContextWindowCompiler {
    pub(crate) retriever: Arc<dyn Retriever>,
    pub(crate) reranker: Arc<dyn Reranker>,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) llm: Arc<dyn LlmClient>,
    pub(crate) verifier: Arc<dyn Verifier>,
    pub(crate) tokenizer: Arc<dyn TokenCounter>,
    pub(crate) scaffold: PromptScaffold,
    pub(crate) conversation: Mutex<ConversationMemory>,
    pub(crate) long_term: Option<Arc<dyn MemoryStore>>,
    pub(crate) retrieval_cache: RetrievalCache,
    pub(crate) prompt_cache: PromptCache,
    pub(crate) config: CwcConfig,
}

impl ContextWindowCompiler {
    /// Run the full query pipeline: retrieve → compile → generate → verify.
    pub async fn query(
        &self,
        input: &str,
        schema: Option<&serde_json::Value>,
    ) -> Result<CwcOutput> {
        let total_start = Instant::now();

        // 1. Retrieval (with cache)
        let retrieval_start = Instant::now();
        let retrieval_decision = cwc_retrieve::classify_retrieval_need(
            input,
            &self.config.retrieval,
        );

        let hits = if retrieval_decision == cwc_retrieve::RetrievalDecision::Skip {
            info!("retrieval skipped (procedural query)");
            vec![]
        } else if let Some(cached) = self.retrieval_cache.get(input, &[]) {
            info!("retrieval cache hit");
            cached
        } else {
            let mut raw_hits = self.retriever.retrieve(
                input,
                self.config.retrieval.final_top_k,
            )?;
            self.reranker.rerank(
                input,
                &mut raw_hits,
                self.config.retrieval.rerank_top_k,
            )?;
            self.retrieval_cache.put(input, &[], raw_hits.clone());
            raw_hits
        };
        let retrieval_ms = retrieval_start.elapsed().as_millis() as u64;
        let retrieval_hits = hits.clone();

        // 2. Memory extraction + compilation
        let compilation_start = Instant::now();

        // Extract memories from this query
        let pending = extract_memories(input, "");
        if let Some(store) = &self.long_term {
            for mem in &pending {
                if mem.confidence >= 0.7 {
                    let _ = store.upsert(&mem.key, &mem.value, mem.category.clone());
                }
            }
        }

        // Build memory text
        let memory_text = {
            let conv = self.conversation.lock().map_err(|e| {
                CwcError::Config(format!("conversation lock poisoned: {e}"))
            })?;
            let ltm_entries = self
                .long_term
                .as_ref()
                .map(|s| s.retrieve(input, 10))
                .transpose()?
                .unwrap_or_default();

            let memory_budget = (self.config.model.context_window as f32
                * self.config.budget.memory_fraction) as u32;
            compile_memory(&conv, &ltm_entries, memory_budget, self.tokenizer.as_ref())
        };

        // 3. Compile sources
        let (selected, report, source_ids) = cwc_compile::compile_sources(CompileParams {
            hits,
            context_window: self.config.model.context_window,
            max_output_tokens: self.config.model.max_output_tokens,
            instruction_text: "",
            memory_text: &memory_text,
            tokenizer: Arc::clone(&self.tokenizer),
            config: &self.config.budget,
            strategy: ReorderStrategy::EdgePlacement,
        });

        // 4. Build prompt
        let ctx = CompiledContext {
            instruction_block: String::new(),
            sources: selected.iter().map(|h| h.chunk.clone()).collect(),
            output_schema: schema.cloned(),
            budget: TokenBudget {
                total: self.config.model.context_window,
                instruction: report.instruction_tokens,
                sources: report.sources_budget,
                memory: report.memory_tokens,
                output_reserved: self.config.model.max_output_tokens,
                remaining: self.config.model.context_window.saturating_sub(
                    report.instruction_tokens
                        + report.sources_budget
                        + report.memory_tokens
                        + self.config.model.max_output_tokens,
                ),
            },
        };

        let memory_opt = if memory_text.is_empty() {
            None
        } else {
            Some(memory_text.as_str())
        };
        let messages = self
            .scaffold
            .compose_chat(&ctx, input, memory_opt, &source_ids);

        // 5. Build grammar constraint
        let grammar_str = schema.and_then(|s| json_schema_to_gbnf(s).ok());

        let compilation_ms = compilation_start.elapsed().as_millis() as u64;

        // 6. Generate
        let generation_start = Instant::now();
        let raw_response = self
            .llm
            .generate_chat(
                &messages,
                grammar_str.as_deref(),
                self.config.model.max_output_tokens,
            )
            .await?;
        let generation_ms = generation_start.elapsed().as_millis() as u64;

        // 7. Parse
        let parsed = parse_structured_response(&raw_response, schema)?;

        // 8. Verify
        let verification_start = Instant::now();
        let verdict = self.verifier.verify(&parsed.raw_text, &ctx.sources)?;
        let mut attempts = 1;

        // 9. Revision if needed
        let (final_parsed, final_verdict) = if verdict.is_fail() {
            let revision = RevisionLoop::new(
                Arc::clone(&self.llm),
                Arc::clone(&self.verifier),
                2,
            );
            let result = revision
                .revise(
                    &messages,
                    &parsed.raw_text,
                    &verdict,
                    &ctx.sources,
                    grammar_str.as_deref(),
                    self.config.model.max_output_tokens,
                )
                .await?;
            attempts += result.attempts;
            let revised_parsed = parse_structured_response(&result.output, schema)?;
            (revised_parsed, result.verdict)
        } else {
            (parsed, verdict)
        };
        let verification_ms = verification_start.elapsed().as_millis() as u64;

        // Update conversation history
        {
            let mut conv = self.conversation.lock().map_err(|e| {
                CwcError::Config(format!("conversation lock poisoned: {e}"))
            })?;
            conv.push(cwc_core::types::Role::User, input);
            conv.push(
                cwc_core::types::Role::Assistant,
                &final_parsed.raw_text,
            );
        }

        let total_ms = total_start.elapsed().as_millis() as u64;

        Ok(CwcOutput {
            response: final_parsed,
            verdict: final_verdict,
            budget_report: report,
            retrieval_hits,
            attempts,
            timing: PipelineTiming {
                retrieval_ms,
                compilation_ms,
                generation_ms,
                verification_ms,
                total_ms,
            },
        })
    }

    /// Compile only (no LLM call). Returns the compiled context and messages.
    pub fn compile(
        &self,
        input: &str,
        schema: Option<&serde_json::Value>,
    ) -> Result<(
        CompiledContext,
        Vec<ChatMessage>,
        cwc_compile::report::BudgetReport,
    )> {
        // Note: compile() does NOT use the retrieval cache. The cache stores
        // post-rerank results from query(), and compile() skips reranking.
        // Sharing the cache would cause query() to get un-reranked results.
        let hits = self
            .retriever
            .retrieve(input, self.config.retrieval.final_top_k)?;

        let memory_text = {
            let conv = self.conversation.lock().map_err(|e| {
                CwcError::Config(format!("conversation lock poisoned: {e}"))
            })?;
            let ltm_entries = self
                .long_term
                .as_ref()
                .map(|s| s.retrieve(input, 10))
                .transpose()?
                .unwrap_or_default();

            let memory_budget = (self.config.model.context_window as f32
                * self.config.budget.memory_fraction) as u32;
            compile_memory(&conv, &ltm_entries, memory_budget, self.tokenizer.as_ref())
        };

        let (selected, report, source_ids) = cwc_compile::compile_sources(CompileParams {
            hits,
            context_window: self.config.model.context_window,
            max_output_tokens: self.config.model.max_output_tokens,
            instruction_text: "",
            memory_text: &memory_text,
            tokenizer: Arc::clone(&self.tokenizer),
            config: &self.config.budget,
            strategy: ReorderStrategy::EdgePlacement,
        });

        let ctx = CompiledContext {
            instruction_block: String::new(),
            sources: selected.iter().map(|h| h.chunk.clone()).collect(),
            output_schema: schema.cloned(),
            budget: TokenBudget {
                total: self.config.model.context_window,
                instruction: report.instruction_tokens,
                sources: report.sources_budget,
                memory: report.memory_tokens,
                output_reserved: self.config.model.max_output_tokens,
                remaining: self.config.model.context_window.saturating_sub(
                    report.instruction_tokens
                        + report.sources_budget
                        + report.memory_tokens
                        + self.config.model.max_output_tokens,
                ),
            },
        };

        let memory_opt = if memory_text.is_empty() {
            None
        } else {
            Some(memory_text.as_str())
        };
        let messages = self
            .scaffold
            .compose_chat(&ctx, input, memory_opt, &source_ids);

        Ok((ctx, messages, report))
    }

    /// Ingest documents: load files, chunk, save, and index.
    ///
    /// When `incremental` is true, only added/modified files are re-chunked and
    /// the index is updated incrementally. A manifest tracks file state across runs.
    pub fn ingest(&self, paths: &[PathBuf], incremental: bool) -> Result<IngestReport> {
        let data_dir = PathBuf::from(&self.config.paths.data_dir);
        let index_dir = PathBuf::from(&self.config.paths.index_dir);
        let chunks_path = data_dir.join("chunks.jsonl");
        let manifest_path = data_dir.join("manifest.json");

        std::fs::create_dir_all(&data_dir)?;
        std::fs::create_dir_all(&index_dir)?;

        // Expand all input paths to individual files (recursive for dirs)
        let mut all_file_paths = Vec::new();
        for path in paths {
            if path.is_dir() {
                let docs = cwc_ingest::load_directory(path, true, None)
                    .map_err(|e| CwcError::Config(format!("ingest error: {e}")))?;
                for doc in &docs {
                    let p = PathBuf::from(&doc.source_path);
                    if let Ok(c) = p.canonicalize() {
                        all_file_paths.push(c);
                    }
                }
            } else if path.is_file() {
                if let Ok(c) = path.canonicalize() {
                    all_file_paths.push(c);
                }
            }
        }

        // Incremental: detect changes, only process added+modified
        let mut files_skipped = 0usize;
        let paths_to_process = if incremental {
            let manifest = cwc_index::incremental::Manifest::load(&manifest_path)
                .unwrap_or_default();
            match cwc_index::incremental::detect_changes(&manifest, &all_file_paths) {
                Ok(changes) => {
                    files_skipped = changes.unchanged;
                    info!(
                        added = changes.added.len(),
                        modified = changes.modified.len(),
                        deleted = changes.deleted.len(),
                        unchanged = changes.unchanged,
                        "incremental change detection"
                    );
                    changes.added.iter()
                        .chain(changes.modified.iter())
                        .cloned()
                        .collect()
                }
                Err(e) => {
                    info!("incremental detection failed ({e}), doing full rebuild");
                    all_file_paths.clone()
                }
            }
        } else {
            all_file_paths.clone()
        };

        let tokenizer = cwc_core::Tokenizer::default_tokenizer()?;
        let chunker = cwc_ingest::RecursiveChunker::new(512, 50);
        let mut all_chunks = Vec::new();

        for path in &paths_to_process {
            let doc = cwc_ingest::load_file(path)
                .map_err(|e| CwcError::Config(format!("ingest error: {e}")))?;
            let chunks = cwc_ingest::Chunker::chunk(&chunker, &doc, &tokenizer)
                .map_err(|e| CwcError::Config(format!("chunk error: {e}")))?;
            all_chunks.extend(chunks);
        }
        let files_processed = paths_to_process.len();
        let chunks_created = all_chunks.len();

        // Save chunks
        cwc_ingest::save_chunks(&all_chunks, &chunks_path)
            .map_err(|e| CwcError::Config(format!("save error: {e}")))?;

        // Build or update BM25 index
        if incremental {
            cwc_index::update_index(&all_chunks, &index_dir)
                .map_err(|e| CwcError::Retrieval(format!("index error: {e}")))?;
        } else {
            cwc_index::build_index(&all_chunks, &index_dir)
                .map_err(|e| CwcError::Retrieval(format!("index error: {e}")))?;
        }

        // Update manifest
        let mut manifest = if incremental {
            cwc_index::incremental::Manifest::load(&manifest_path).unwrap_or_default()
        } else {
            cwc_index::incremental::Manifest::default()
        };
        // Build doc_id map from chunks
        let mut doc_id_by_path: std::collections::HashMap<PathBuf, uuid::Uuid> =
            std::collections::HashMap::new();
        for chunk in &all_chunks {
            let p = PathBuf::from(&chunk.source_path);
            doc_id_by_path.entry(p).or_insert(chunk.doc_id);
        }
        // Update manifest for all input files (not just processed ones)
        for file_path in &all_file_paths {
            let key = file_path.to_string_lossy().to_string();
            if let (Ok(hash), Ok(meta)) = (
                cwc_index::incremental::file_hash(file_path),
                std::fs::metadata(file_path),
            ) {
                let mtime = meta.modified()
                    .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64)
                    .unwrap_or(0);
                let doc_id = doc_id_by_path.get(file_path)
                    .copied()
                    .or_else(|| manifest.entries.get(&key).map(|e| e.doc_id))
                    .unwrap_or_else(uuid::Uuid::new_v4);
                manifest.entries.insert(key, cwc_index::incremental::ManifestEntry {
                    hash,
                    mtime,
                    doc_id,
                });
            }
        }
        if let Err(e) = manifest.save(&manifest_path) {
            tracing::warn!("Failed to save ingest manifest: {e}");
        }

        // Invalidate caches — index content has changed
        self.clear_caches();

        info!(
            files = files_processed,
            chunks = chunks_created,
            skipped = files_skipped,
            "ingestion complete"
        );

        Ok(IngestReport {
            files_processed,
            chunks_created,
            index_path: index_dir,
            chunks_path,
            incremental,
            files_skipped,
        })
    }

    /// Clear conversation history.
    pub fn clear_conversation(&self) -> Result<()> {
        let mut conv = self.conversation.lock().map_err(|e| {
            CwcError::Config(format!("conversation lock poisoned: {e}"))
        })?;
        conv.clear();
        Ok(())
    }

    /// Get current conversation length.
    pub fn conversation_len(&self) -> Result<usize> {
        let conv = self.conversation.lock().map_err(|e| {
            CwcError::Config(format!("conversation lock poisoned: {e}"))
        })?;
        Ok(conv.len())
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &CwcConfig {
        &self.config
    }

    /// Get a reference to the retriever.
    pub fn retriever(&self) -> &dyn Retriever {
        &*self.retriever
    }

    /// Get the LLM client.
    pub fn llm(&self) -> &Arc<dyn LlmClient> {
        &self.llm
    }

    /// Get retrieval cache statistics.
    pub fn retrieval_cache_stats(&self) -> CacheStats {
        self.retrieval_cache.stats()
    }

    /// Get prompt cache statistics.
    pub fn prompt_cache_stats(&self) -> CacheStats {
        self.prompt_cache.stats()
    }

    /// Clear all caches (retrieval + prompt).
    pub fn clear_caches(&self) {
        self.retrieval_cache.clear();
        self.prompt_cache.clear();
    }

    /// Run query pipeline with streaming generation output.
    /// Calls `on_token` for each streamed chunk during generation.
    /// Verification happens after the full response is received.
    pub async fn query_streaming(
        &self,
        input: &str,
        schema: Option<&serde_json::Value>,
        on_token: cwc_core::traits::StreamCallback,
    ) -> Result<CwcOutput> {
        let total_start = Instant::now();

        // 1. Retrieval (same as query())
        let retrieval_start = Instant::now();
        let retrieval_decision = cwc_retrieve::classify_retrieval_need(
            input,
            &self.config.retrieval,
        );
        let hits = if retrieval_decision == cwc_retrieve::RetrievalDecision::Skip {
            info!("retrieval skipped (procedural query)");
            vec![]
        } else if let Some(cached) = self.retrieval_cache.get(input, &[]) {
            info!("retrieval cache hit");
            cached
        } else {
            let mut raw_hits = self.retriever.retrieve(
                input,
                self.config.retrieval.final_top_k,
            )?;
            self.reranker.rerank(
                input,
                &mut raw_hits,
                self.config.retrieval.rerank_top_k,
            )?;
            self.retrieval_cache.put(input, &[], raw_hits.clone());
            raw_hits
        };
        let retrieval_ms = retrieval_start.elapsed().as_millis() as u64;
        let retrieval_hits = hits.clone();

        // 2. Memory + compilation (same as query())
        let compilation_start = Instant::now();
        let pending = extract_memories(input, "");
        if let Some(store) = &self.long_term {
            for mem in &pending {
                if mem.confidence >= 0.7 {
                    let _ = store.upsert(&mem.key, &mem.value, mem.category.clone());
                }
            }
        }
        let memory_text = {
            let conv = self.conversation.lock().map_err(|e| {
                CwcError::Config(format!("conversation lock poisoned: {e}"))
            })?;
            let ltm_entries = self
                .long_term
                .as_ref()
                .map(|s| s.retrieve(input, 10))
                .transpose()?
                .unwrap_or_default();
            let memory_budget = (self.config.model.context_window as f32
                * self.config.budget.memory_fraction) as u32;
            compile_memory(&conv, &ltm_entries, memory_budget, self.tokenizer.as_ref())
        };

        let (selected, report, source_ids) = cwc_compile::compile_sources(CompileParams {
            hits,
            context_window: self.config.model.context_window,
            max_output_tokens: self.config.model.max_output_tokens,
            instruction_text: "",
            memory_text: &memory_text,
            tokenizer: Arc::clone(&self.tokenizer),
            config: &self.config.budget,
            strategy: ReorderStrategy::EdgePlacement,
        });

        let ctx = CompiledContext {
            instruction_block: String::new(),
            sources: selected.iter().map(|h| h.chunk.clone()).collect(),
            output_schema: schema.cloned(),
            budget: TokenBudget {
                total: self.config.model.context_window,
                instruction: report.instruction_tokens,
                sources: report.sources_budget,
                memory: report.memory_tokens,
                output_reserved: self.config.model.max_output_tokens,
                remaining: self.config.model.context_window.saturating_sub(
                    report.instruction_tokens
                        + report.sources_budget
                        + report.memory_tokens
                        + self.config.model.max_output_tokens,
                ),
            },
        };

        let memory_opt = if memory_text.is_empty() {
            None
        } else {
            Some(memory_text.as_str())
        };
        let messages = self
            .scaffold
            .compose_chat(&ctx, input, memory_opt, &source_ids);

        let grammar_str = schema.and_then(|s| json_schema_to_gbnf(s).ok());
        let compilation_ms = compilation_start.elapsed().as_millis() as u64;

        // 3. Generate with streaming
        let generation_start = Instant::now();
        let raw_response = self
            .llm
            .generate_chat_stream(
                &messages,
                grammar_str.as_deref(),
                self.config.model.max_output_tokens,
                on_token,
            )
            .await?;
        let generation_ms = generation_start.elapsed().as_millis() as u64;

        // 4. Parse + verify (after streaming completes)
        let parsed = parse_structured_response(&raw_response, schema)?;
        let verification_start = Instant::now();
        let verdict = self.verifier.verify(&parsed.raw_text, &ctx.sources)?;
        let verification_ms = verification_start.elapsed().as_millis() as u64;

        // Update conversation
        {
            let mut conv = self.conversation.lock().map_err(|e| {
                CwcError::Config(format!("conversation lock poisoned: {e}"))
            })?;
            conv.push(cwc_core::types::Role::User, input);
            conv.push(cwc_core::types::Role::Assistant, &parsed.raw_text);
        }

        let total_ms = total_start.elapsed().as_millis() as u64;

        Ok(CwcOutput {
            response: parsed,
            verdict,
            budget_report: report,
            retrieval_hits,
            attempts: 1,
            timing: PipelineTiming {
                retrieval_ms,
                compilation_ms,
                generation_ms,
                verification_ms,
                total_ms,
            },
        })
    }

    /// Full pipeline with session management:
    /// 1. Optimize the conversation (compact, trim, validate)
    /// 2. Extract the latest query from the optimized conversation
    /// 3. Retrieve evidence for the query
    /// 4. Compile the prompt with evidence + session memory
    /// 5. Generate + verify
    pub async fn query_with_session(
        &self,
        session: &mut cwc_session::Session,
        session_manager: &mut cwc_session::SessionManager,
    ) -> Result<CwcOutput> {
        // 1. Optimize the session
        let _report = session_manager.optimize(session).map_err(|e| {
            CwcError::Config(format!("session optimization failed: {e}"))
        })?;

        // 2. Extract the latest user query from the session
        let query = session
            .messages()
            .iter()
            .rev()
            .find(|m| {
                m.role == cwc_session::SessionRole::User
                    && !m.flags.contains(cwc_session::MessageFlags::IS_NUDGE)
                    && !m.flags.contains(cwc_session::MessageFlags::IS_MEMORY)
            })
            .map(|m| m.content.clone())
            .unwrap_or_default();

        if query.is_empty() {
            return Err(CwcError::Config("no user query found in session".into()));
        }

        // 3. Build session memory text from the session manager's extracted facts
        let session_memory = session_manager
            .memory()
            .render(2048, self.tokenizer.as_ref());

        // 4. Run the standard query pipeline with session memory prepended
        let total_start = Instant::now();
        let retrieval_start = Instant::now();

        let hits = {
            let decision = cwc_retrieve::classify_retrieval_need(
                &query,
                &self.config.retrieval,
            );
            if decision == cwc_retrieve::RetrievalDecision::Skip {
                vec![]
            } else if let Some(cached) = self.retrieval_cache.get(&query, &[]) {
                cached
            } else {
                let mut raw_hits = self.retriever.retrieve(
                    &query,
                    self.config.retrieval.final_top_k,
                )?;
                self.reranker.rerank(
                    &query,
                    &mut raw_hits,
                    self.config.retrieval.rerank_top_k,
                )?;
                self.retrieval_cache.put(&query, &[], raw_hits.clone());
                raw_hits
            }
        };
        let retrieval_ms = retrieval_start.elapsed().as_millis() as u64;
        let retrieval_hits = hits.clone();

        // 5. Compile with session memory
        let compilation_start = Instant::now();
        let combined_memory = if session_memory.is_empty() {
            String::new()
        } else {
            format!("[SESSION_MEMORY]\n{session_memory}")
        };

        let (selected, report, source_ids) = cwc_compile::compile_sources(CompileParams {
            hits,
            context_window: self.config.model.context_window,
            max_output_tokens: self.config.model.max_output_tokens,
            instruction_text: "",
            memory_text: &combined_memory,
            tokenizer: Arc::clone(&self.tokenizer),
            config: &self.config.budget,
            strategy: ReorderStrategy::EdgePlacement,
        });

        let ctx = CompiledContext {
            instruction_block: String::new(),
            sources: selected.iter().map(|h| h.chunk.clone()).collect(),
            output_schema: None,
            budget: TokenBudget {
                total: self.config.model.context_window,
                instruction: report.instruction_tokens,
                sources: report.sources_budget,
                memory: report.memory_tokens,
                output_reserved: self.config.model.max_output_tokens,
                remaining: self.config.model.context_window.saturating_sub(
                    report.instruction_tokens
                        + report.sources_budget
                        + report.memory_tokens
                        + self.config.model.max_output_tokens,
                ),
            },
        };

        let memory_opt = if combined_memory.is_empty() {
            None
        } else {
            Some(combined_memory.as_str())
        };
        let messages = self
            .scaffold
            .compose_chat(&ctx, &query, memory_opt, &source_ids);
        let compilation_ms = compilation_start.elapsed().as_millis() as u64;

        // 6. Generate
        let generation_start = Instant::now();
        let raw_response = self
            .llm
            .generate_chat(
                &messages,
                None,
                self.config.model.max_output_tokens,
            )
            .await?;
        let generation_ms = generation_start.elapsed().as_millis() as u64;

        // 7. Parse + verify
        let parsed = parse_structured_response(&raw_response, None)?;
        let verification_start = Instant::now();
        let verdict = self.verifier.verify(&parsed.raw_text, &ctx.sources)?;
        let verification_ms = verification_start.elapsed().as_millis() as u64;

        // 8. Add assistant response to session
        session.push(cwc_session::SessionMessage::text(
            cwc_session::SessionRole::Assistant,
            &parsed.raw_text,
        ));

        let total_ms = total_start.elapsed().as_millis() as u64;

        Ok(CwcOutput {
            response: parsed,
            verdict,
            budget_report: report,
            retrieval_hits,
            attempts: 1,
            timing: PipelineTiming {
                retrieval_ms,
                compilation_ms,
                generation_ms,
                verification_ms,
                total_ms,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_timing_default() {
        let timing = PipelineTiming::default();
        assert_eq!(timing.retrieval_ms, 0);
        assert_eq!(timing.compilation_ms, 0);
        assert_eq!(timing.generation_ms, 0);
        assert_eq!(timing.verification_ms, 0);
        assert_eq!(timing.total_ms, 0);
    }
}
