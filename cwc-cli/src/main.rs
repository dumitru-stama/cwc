use std::path::{Path, PathBuf};
use std::sync::Arc;

use cwc_cli::init;

use clap::{Parser, Subcommand};

use cwc_core::traits::TokenCounter;

#[derive(Parser)]
#[command(name = "cwc", about = "Context Window Compiler")]
struct Cli {
    /// Path to configuration file
    #[arg(long, default_value = "cwc.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new CWC project
    Init {
        /// Project name
        #[arg(long)]
        name: Option<String>,
        /// LLM endpoint URL
        #[arg(long)]
        endpoint: Option<String>,
        /// Project directory (defaults to current directory)
        #[arg(long, default_value = ".")]
        dir: PathBuf,
    },
    /// Ingest and index documents
    Ingest {
        /// Paths to files or directories to ingest
        #[arg(long, num_args = 1..)]
        path: Vec<PathBuf>,
        /// Only process changed files (requires previous manifest)
        #[arg(long)]
        incremental: bool,
        /// Force full re-index, ignoring manifest
        #[arg(long)]
        full_rebuild: bool,
    },
    /// Run a full query pipeline (retrieve → compile → generate → verify)
    Query {
        /// The query text
        query: String,
        /// Skip verification
        #[arg(long)]
        no_verify: bool,
        /// Show raw LLM response without parsing
        #[arg(long)]
        raw: bool,
        /// Show verbose timing and debug info
        #[arg(long, short)]
        verbose: bool,
        /// Task type for output schema
        #[arg(long)]
        task_type: Option<String>,
        /// Session file for context management
        #[arg(long)]
        session: Option<PathBuf>,
    },
    /// Search the index directly
    Search {
        #[command(subcommand)]
        kind: SearchKind,
    },
    /// Compile a prompt without generating
    Compile {
        /// User query
        #[arg(long)]
        query: String,
        /// Path to chunks.jsonl
        #[arg(long)]
        chunks: PathBuf,
        /// Model context window size
        #[arg(long, default_value = "4096")]
        model_ctx: u32,
        /// Max output tokens
        #[arg(long, default_value = "1024")]
        max_output: u32,
        /// Output format: prompt, chat, debug
        #[arg(long, default_value = "prompt")]
        format: String,
        /// Task type for output schema
        #[arg(long)]
        task_type: Option<String>,
    },
    /// Generate a response using an LLM backend
    Generate {
        /// User query
        #[arg(long)]
        query: String,
        /// Path to chunks.jsonl
        #[arg(long)]
        chunks: PathBuf,
        /// LLM server endpoint
        #[arg(long, default_value = "http://localhost:8080")]
        endpoint: String,
        /// Backend: llamacpp or vllm
        #[arg(long, default_value = "llamacpp")]
        backend: String,
        /// Model name (required for vllm)
        #[arg(long, default_value = "default")]
        model: String,
        /// Model context window size
        #[arg(long, default_value = "4096")]
        model_ctx: u32,
        /// Max output tokens
        #[arg(long, default_value = "1024")]
        max_output: u32,
        /// Task type for output schema
        #[arg(long)]
        task_type: Option<String>,
        /// Temperature
        #[arg(long, default_value = "0.1")]
        temperature: f32,
    },
    /// Manage long-term memory
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },
    /// Show project status (index stats, config)
    Status,
    /// Interactive query loop
    Repl {
        /// Enable session management
        #[arg(long)]
        session: bool,
    },
    /// Manage caches
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    /// Run evaluation benchmarks
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },
    /// Session context window management
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// Embed chunks or a query
    Embed {
        /// Path to chunks.jsonl
        #[arg(long, group = "input")]
        chunks: Option<PathBuf>,
        /// Single query to embed
        #[arg(long, group = "input")]
        query: Option<String>,
        /// Path to ONNX model directory
        #[arg(long, default_value = "./models/bge-small-en-v1.5")]
        model: PathBuf,
        /// Output file for chunk embeddings
        #[arg(long, default_value = "./data/embeddings.bin")]
        output: PathBuf,
        /// Embedding cache directory
        #[arg(long, default_value = "./data/embed_cache")]
        cache_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum SearchKind {
    /// BM25 sparse search
    Sparse {
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "10")]
        top_k: usize,
        #[arg(long, default_value = "./data/sparse_index")]
        index_dir: PathBuf,
    },
    /// Dense vector search
    Dense {
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "10")]
        top_k: usize,
        #[arg(long, group = "backend")]
        index: Option<PathBuf>,
        #[arg(long, group = "backend")]
        db: Option<String>,
        #[arg(long, default_value = "./models/bge-small-en-v1.5")]
        model: PathBuf,
    },
    /// Hybrid search (BM25 + vector + RRF + MMR)
    Hybrid {
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "10")]
        top_k: usize,
        #[arg(long, default_value = "./data/sparse_index")]
        index_dir: PathBuf,
        #[arg(long)]
        vector_index: PathBuf,
        #[arg(long, default_value = "./models/bge-small-en-v1.5")]
        model: PathBuf,
    },
}

#[derive(Subcommand)]
enum EvalAction {
    /// Run benchmark on an evaluation dataset
    Run {
        /// Path to JSONL dataset file
        #[arg(long)]
        dataset: PathBuf,
        /// Output results to JSON file
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two benchmark result files
    Compare {
        /// First results file
        #[arg(long)]
        results_a: PathBuf,
        /// Second results file
        #[arg(long)]
        results_b: PathBuf,
    },
    /// Display a human-readable report from results file
    Report {
        /// Path to results JSON file
        #[arg(long)]
        results: PathBuf,
    },
    /// Create an adversarial evaluation dataset
    Adversarial {
        /// Output JSONL file path
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum MemoryAction {
    /// List all long-term memories
    List {
        #[arg(long, default_value = "./data/memory.json")]
        file: PathBuf,
    },
    /// Add a memory entry
    Add {
        #[arg(long)]
        key: String,
        #[arg(long)]
        value: String,
        #[arg(long, default_value = "fact")]
        category: String,
        #[arg(long, default_value = "./data/memory.json")]
        file: PathBuf,
    },
    /// Delete a memory entry
    Delete {
        #[arg(long)]
        key: String,
        #[arg(long, default_value = "./data/memory.json")]
        file: PathBuf,
    },
    /// Search memory entries
    Search {
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "10")]
        max: usize,
        #[arg(long, default_value = "./data/memory.json")]
        file: PathBuf,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Clear all caches
    Clear,
    /// Show cache statistics
    Stats,
}

#[derive(Subcommand)]
enum SessionAction {
    /// Optimize a conversation from a JSON file
    Optimize {
        /// Input JSON file
        input: PathBuf,
        /// Input format: openai, anthropic, raw (default: auto-detect)
        #[arg(long, default_value = "auto")]
        format: String,
        /// Model profile (default: local_large)
        #[arg(long, default_value = "local_large")]
        model: String,
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Print optimization report to stderr
        #[arg(long)]
        report: bool,
        /// Enable LLM-enhanced memory consolidation (requires LLM server)
        #[arg(long)]
        llm_consolidate: bool,
    },
    /// Show budget status for a conversation
    Status {
        /// Input JSON file
        input: PathBuf,
        /// Input format: openai, anthropic, raw
        #[arg(long, default_value = "auto")]
        format: String,
        /// Model profile
        #[arg(long, default_value = "local_large")]
        model: String,
    },
    /// Show extracted memory facts from a conversation
    Memory {
        /// Input JSON file
        input: PathBuf,
        /// Input format: openai, anthropic, raw
        #[arg(long, default_value = "auto")]
        format: String,
    },
    /// Compact tool results only (no trimming/reset)
    Compact {
        /// Input JSON file
        input: PathBuf,
        /// Input format: openai, anthropic, raw
        #[arg(long, default_value = "auto")]
        format: String,
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run session evaluation benchmarks
    Eval {
        /// Model profile (default: local_large)
        #[arg(long, default_value = "local_large")]
        model: String,
        /// Output results to JSON file
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Recommend a session config for a model
    Recommend {
        /// Model name (optional)
        #[arg(long)]
        model_name: Option<String>,
        /// Context window size
        #[arg(long)]
        context_window: u32,
        /// Max output tokens
        #[arg(long)]
        max_output: u32,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { name, endpoint, dir } => {
            match init::init_project(&dir, name.as_deref(), endpoint.as_deref()) {
                Ok(()) => {
                    println!("Project initialized in {}", dir.display());
                    println!("  Created: cwc.toml, data/, index/, models/");
                    println!("  Next: place ONNX models in models/, then run: cwc ingest --path <docs>");
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Ingest { path, incremental, full_rebuild } => {
            if path.is_empty() {
                eprintln!("Error: provide at least one --path");
                std::process::exit(1);
            }

            let config = load_config_or_default(&cli.config);
            let data_dir = PathBuf::from(&config.paths.data_dir);
            let index_dir = PathBuf::from(&config.paths.index_dir);
            let chunks_path = data_dir.join("chunks.jsonl");
            let manifest_path = data_dir.join("manifest.json");

            let _ = std::fs::create_dir_all(&data_dir);
            let _ = std::fs::create_dir_all(&index_dir);

            // Collect all input file paths (expand directories recursively)
            let mut all_file_paths: Vec<PathBuf> = Vec::new();
            for p in &path {
                if p.is_dir() {
                    collect_files_recursive(p, &mut all_file_paths);
                } else if p.is_file() {
                    if let Ok(c) = p.canonicalize() {
                        all_file_paths.push(c);
                    }
                } else {
                    eprintln!("Warning: {} is not a file or directory, skipping", p.display());
                }
            }

            // Incremental: detect changes BEFORE chunking to skip unchanged files
            let mut paths_to_process = all_file_paths.clone();
            if incremental && !full_rebuild {
                let manifest = cwc_index::incremental::Manifest::load(&manifest_path)
                    .unwrap_or_default();
                match cwc_index::incremental::detect_changes(&manifest, &all_file_paths) {
                    Ok(changes) => {
                        println!("Incremental: {} added, {} modified, {} deleted, {} unchanged",
                            changes.added.len(), changes.modified.len(),
                            changes.deleted.len(), changes.unchanged);
                        if changes.is_empty() {
                            println!("No changes detected. Index is up to date.");
                            return;
                        }
                        // Only process added + modified files
                        paths_to_process = changes.added.iter()
                            .chain(changes.modified.iter())
                            .cloned()
                            .collect();
                    }
                    Err(e) => {
                        eprintln!("Warning: incremental detection failed ({e}), doing full rebuild");
                    }
                }
            }

            let tokenizer = match cwc_core::Tokenizer::default_tokenizer() {
                Ok(t) => t,
                Err(e) => { eprintln!("Error loading tokenizer: {e}"); std::process::exit(1); }
            };
            let chunker = cwc_ingest::RecursiveChunker::new(512, 50);
            let mut all_chunks = Vec::new();
            for p in &paths_to_process {
                match cwc_ingest::load_file(p) {
                    Ok(doc) => {
                        match cwc_ingest::Chunker::chunk(&chunker, &doc, &tokenizer) {
                            Ok(chunks) => all_chunks.extend(chunks),
                            Err(e) => { eprintln!("Error chunking {}: {e}", p.display()); std::process::exit(1); }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error loading {}: {e}", p.display());
                        std::process::exit(1);
                    }
                }
            }

            println!("Processed {} files, created {} chunks", paths_to_process.len(), all_chunks.len());

            if let Err(e) = cwc_ingest::save_chunks(&all_chunks, &chunks_path) {
                eprintln!("Error saving chunks: {e}");
                std::process::exit(1);
            }
            println!("Saved chunks to {}", chunks_path.display());

            // Use update_index for incremental (preserves existing index), build_index for full
            let index_result = if incremental && !full_rebuild {
                cwc_index::update_index(&all_chunks, &index_dir)
            } else {
                cwc_index::build_index(&all_chunks, &index_dir)
            };
            match index_result {
                Ok(n) => println!("Indexed {} chunks into {}", n, index_dir.display()),
                Err(e) => {
                    eprintln!("Error building index: {e}");
                    std::process::exit(1);
                }
            }

            // Save manifest for future incremental runs — use doc_ids from chunks
            let mut manifest = if incremental && !full_rebuild {
                cwc_index::incremental::Manifest::load(&manifest_path).unwrap_or_default()
            } else {
                cwc_index::incremental::Manifest::default()
            };
            // Build doc_id map from actual chunks (first chunk per source_path)
            let mut doc_id_by_path: std::collections::HashMap<PathBuf, uuid::Uuid> = std::collections::HashMap::new();
            for chunk in &all_chunks {
                let p = PathBuf::from(&chunk.source_path);
                doc_id_by_path.entry(p).or_insert(chunk.doc_id);
            }
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
                eprintln!("Warning: failed to save manifest: {e}");
            }
        }
        Commands::Status => {
            let config = load_config_or_default(&cli.config);
            println!("CWC Status");
            println!("==========");
            println!("Config:          {} ({:?} mode)", cli.config.display(), config.mode);
            println!("LLM endpoint:    {}", config.model.llm_endpoint);
            println!("Embedding model: {}", config.model.embedding_model);
            println!("Context window:  {} tokens", config.model.context_window);
            println!("Max output:      {} tokens", config.model.max_output_tokens);
            if let Some(ref rm) = config.model.rerank_model {
                println!("Rerank model:    {}", rm);
            }
            println!();

            // Indexes
            println!("Indexes:");
            let data_dir = PathBuf::from(&config.paths.data_dir);
            let chunks_path = data_dir.join("chunks.jsonl");
            if chunks_path.exists() {
                match cwc_ingest::store::load_chunks(&chunks_path) {
                    Ok(chunks) => println!("  Chunks:        {}", chunks.len()),
                    Err(e) => println!("  Chunks:        error reading: {e}"),
                }
            } else {
                println!("  Chunks:        none (run: cwc ingest --path <docs>)");
            }

            let index_dir = PathBuf::from(&config.paths.index_dir);
            if index_dir.exists() && index_dir.read_dir().map(|mut d| d.next().is_some()).unwrap_or(false) {
                println!("  BM25 index:    exists at {}", index_dir.display());
            } else {
                println!("  BM25 index:    none");
            }
            println!();

            // Memory
            println!("Memory:");
            let memory_path = data_dir.join("memory.json");
            if memory_path.exists() {
                let mem = cwc_memory::file_memory::FileMemory::new(&memory_path);
                match mem.list_all() {
                    Ok(entries) => println!("  Long-term:     {} entries", entries.len()),
                    Err(e) => println!("  Long-term:     error: {e}"),
                }
            } else {
                println!("  Long-term:     none");
            }
            println!();

            // Config validation
            let warnings = cwc_core::validate::validate_config(&config);
            if warnings.is_empty() {
                println!("Validation:      all checks passed");
            } else {
                println!("Validation:");
                for w in &warnings {
                    println!("  {w}");
                }
            }
        }
        Commands::Query { query: _, no_verify: _, raw: _, verbose: _, task_type: _, session: _ } => {
            eprintln!("Error: full query pipeline requires running LLM server.");
            eprintln!("Use 'cwc search' for retrieval-only or 'cwc generate' for direct LLM calls.");
            eprintln!("With --session <file>, session management is applied to the conversation.");
            std::process::exit(1);
        }
        Commands::Repl { session: _ } => {
            eprintln!("Error: REPL requires running LLM server and configured indexes.");
            eprintln!("Use --session to enable session context management.");
            std::process::exit(1);
        }
        Commands::Cache { action } => {
            match action {
                CacheAction::Clear => {
                    println!("Note: Caches are in-memory and per-process.");
                    println!("This command has no effect in one-shot CLI mode.");
                    println!("Caches are active during `cwc repl` sessions and when");
                    println!("using CWC as a library in a long-running process.");
                }
                CacheAction::Stats => {
                    println!("Cache Statistics");
                    println!("================");
                    println!("Note: Caches are in-memory and per-process.");
                    println!("Stats are only meaningful during `cwc repl` sessions");
                    println!("or when using CWC as a library in a long-running process.");
                    println!();
                    println!("Configuration:");
                    println!("  Retrieval cache: TTL 5min, max 1000 entries");
                    println!("  Prompt cache:    TTL 1min, max 100 entries");
                    println!();
                    println!("Caches are automatically invalidated after ingestion.");
                }
            }
        }
        Commands::Search { kind } => match kind {
            SearchKind::Sparse { query, top_k, index_dir } => {
                let index = match cwc_index::SparseIndex::open_or_create(&index_dir) {
                    Ok(i) => i,
                    Err(e) => {
                        eprintln!("Error opening index: {e}");
                        std::process::exit(1);
                    }
                };
                match index.search(&query, top_k) {
                    Ok(hits) => print_retrieval_hits(&hits),
                    Err(e) => {
                        eprintln!("Search error: {e}");
                        std::process::exit(1);
                    }
                }
            }
            SearchKind::Dense { query, top_k, index, db, model } => {
                let model_path = model.join("model.onnx");
                let tokenizer_path = model.join("tokenizer.json");
                let config = cwc_embed::EmbedConfig {
                    model_path: model_path.clone(),
                    tokenizer_path: tokenizer_path.clone(),
                    ..Default::default()
                };
                let embedder = match cwc_embed::OnnxEmbedder::load(&model_path, &tokenizer_path, &config) {
                    Ok(e) => e,
                    Err(e) => { eprintln!("Error loading model: {e}"); std::process::exit(1); }
                };
                let query_emb = match embedder.embed_query(&query) {
                    Ok(e) => e,
                    Err(e) => { eprintln!("Error embedding query: {e}"); std::process::exit(1); }
                };
                if let Some(index_path) = index {
                    let mem_index = match cwc_index::InMemoryVectorIndex::load(&index_path) {
                        Ok(i) => i,
                        Err(e) => { eprintln!("Error loading index: {e}"); std::process::exit(1); }
                    };
                    print_dense_results(&mem_index.search(&query_emb, top_k));
                } else if let Some(db_url) = db {
                    let chunk_db = match cwc_index::ChunkDb::new(&db_url, config.dim).await {
                        Ok(d) => d,
                        Err(e) => { eprintln!("Error connecting: {e}"); std::process::exit(1); }
                    };
                    match chunk_db.search_dense(&query_emb, top_k).await {
                        Ok(results) => print_dense_results(&results),
                        Err(e) => { eprintln!("Search error: {e}"); std::process::exit(1); }
                    }
                } else {
                    eprintln!("Provide either --index or --db for dense search");
                    std::process::exit(1);
                }
            }
            SearchKind::Hybrid { query, top_k, index_dir, vector_index, model } => {
                let retrieval_config = cwc_core::config::RetrievalConfig::default();
                let decision = cwc_retrieve::classify_retrieval_need(&query, &retrieval_config);
                if decision == cwc_retrieve::RetrievalDecision::Skip {
                    println!("Retrieval skipped (procedural query)");
                    return;
                }
                let sparse_index = match cwc_index::SparseIndex::open_or_create(&index_dir) {
                    Ok(i) => i,
                    Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                };
                let sparse: Arc<dyn cwc_core::traits::Retriever> = Arc::new(cwc_index::SparseRetriever::new(sparse_index));
                let model_path = model.join("model.onnx");
                let tokenizer_path = model.join("tokenizer.json");
                let embed_config = cwc_embed::EmbedConfig {
                    model_path: model_path.clone(), tokenizer_path: tokenizer_path.clone(), ..Default::default()
                };
                let embedder = match cwc_embed::OnnxEmbedder::load(&model_path, &tokenizer_path, &embed_config) {
                    Ok(e) => e,
                    Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                };
                let embedder: Arc<dyn cwc_core::traits::Embedder> = Arc::new(embedder);
                let mem_index = match cwc_index::InMemoryVectorIndex::load(&vector_index) {
                    Ok(i) => i,
                    Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                };
                let dense: Arc<dyn cwc_core::traits::Retriever> = Arc::new(cwc_index::InMemoryDenseRetriever::new(mem_index, embedder.clone()));
                let reranker: Arc<dyn cwc_core::traits::Reranker> = Arc::new(cwc_core::noop::NoopReranker);
                let hybrid = cwc_retrieve::HybridRetriever::new(sparse, dense, embedder, reranker, retrieval_config);
                match cwc_core::traits::Retriever::retrieve(&hybrid, &query, top_k) {
                    Ok(hits) => {
                        if hits.is_empty() { println!("No results."); } else {
                            for (rank, hit) in hits.iter().enumerate() {
                                let preview: String = hit.chunk.text.chars().take(80).collect();
                                println!("{:>3}. [fused={:.4} sparse={:.4} dense={:.4}] {} — {}",
                                    rank + 1, hit.score_fused, hit.score_sparse, hit.score_dense, hit.chunk.chunk_id, preview);
                            }
                        }
                    }
                    Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                }
            }
        },
        Commands::Compile { query, chunks, model_ctx, max_output, format, task_type } => {
            use cwc_compile::scaffold::{PromptScaffold, ScaffoldConfig, debug_prompt};
            use cwc_compile::schema::schema_for_task;

            let loaded = match cwc_ingest::store::load_chunks(&chunks) {
                Ok(c) => c,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let hits: Vec<cwc_core::types::RetrievalHit> = loaded.into_iter().map(|chunk| cwc_core::types::RetrievalHit {
                score_sparse: 1.0, score_dense: 1.0, score_fused: 1.0, score_rerank: 1.0, chunk,
            }).collect();
            let tokenizer = match cwc_core::Tokenizer::default_tokenizer() {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let tc: Arc<dyn TokenCounter> = Arc::new(tokenizer);
            let schema = task_type.as_deref().map(|t| schema_for_task(parse_task_type(t)));
            let budget_config = cwc_core::config::BudgetConfig::default();
            let (selected, report, source_ids) = cwc_compile::compile_sources(cwc_compile::CompileParams {
                hits, context_window: model_ctx, max_output_tokens: max_output,
                instruction_text: "", memory_text: "", tokenizer: Arc::clone(&tc),
                config: &budget_config, strategy: cwc_compile::reorder::ReorderStrategy::EdgePlacement,
            });
            let ctx = cwc_core::types::CompiledContext {
                instruction_block: String::new(),
                sources: selected.iter().map(|h| h.chunk.clone()).collect(),
                output_schema: schema,
                budget: cwc_core::types::TokenBudget {
                    total: model_ctx, instruction: report.instruction_tokens, sources: report.sources_budget,
                    memory: report.memory_tokens, output_reserved: max_output,
                    remaining: model_ctx.saturating_sub(report.instruction_tokens + report.sources_budget + report.memory_tokens + max_output),
                },
            };
            let scaffold = PromptScaffold::new(ScaffoldConfig::default());
            match format.as_str() {
                "chat" => {
                    let msgs = scaffold.compose_chat(&ctx, &query, None, &source_ids);
                    println!("{}", serde_json::to_string_pretty(&msgs).unwrap_or_else(|e| format!("JSON error: {e}")));
                }
                "debug" => {
                    let prompt = scaffold.compose(&ctx, &query, None, &source_ids);
                    let debug = debug_prompt(&prompt, &ctx, tc.as_ref(), &source_ids);
                    println!("{debug}");
                    println!("\n--- Full Prompt ---\n{}", prompt.text);
                }
                _ => {
                    let prompt = scaffold.compose(&ctx, &query, None, &source_ids);
                    println!("{}", prompt.text);
                }
            }
            eprintln!("Budget: {}/{} sources tokens ({:.0}% util), {} chunks, {} dropped",
                report.sources_used, report.sources_budget, report.utilization * 100.0,
                report.chunks_selected, report.chunks_dropped);
        }
        Commands::Generate { query, chunks, endpoint, backend, model, model_ctx, max_output, task_type, temperature } => {
            use cwc_compile::scaffold::{PromptScaffold, ScaffoldConfig};
            use cwc_compile::schema::schema_for_task;
            use cwc_core::traits::LlmClient;
            use cwc_llm::grammar::json_schema_to_gbnf;
            use cwc_llm::parse::parse_structured_response;

            let loaded = match cwc_ingest::store::load_chunks(&chunks) {
                Ok(c) => c,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let hits: Vec<cwc_core::types::RetrievalHit> = loaded.into_iter().map(|chunk| cwc_core::types::RetrievalHit {
                score_sparse: 1.0, score_dense: 1.0, score_fused: 1.0, score_rerank: 1.0, chunk,
            }).collect();
            let tokenizer = match cwc_core::Tokenizer::default_tokenizer() {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let tc: Arc<dyn TokenCounter> = Arc::new(tokenizer);
            let schema = task_type.as_deref().map(|t| schema_for_task(parse_task_type(t)));
            let budget_config = cwc_core::config::BudgetConfig::default();
            let (selected, report, source_ids) = cwc_compile::compile_sources(cwc_compile::CompileParams {
                hits, context_window: model_ctx, max_output_tokens: max_output,
                instruction_text: "", memory_text: "", tokenizer: Arc::clone(&tc),
                config: &budget_config, strategy: cwc_compile::reorder::ReorderStrategy::EdgePlacement,
            });
            let ctx = cwc_core::types::CompiledContext {
                instruction_block: String::new(),
                sources: selected.iter().map(|h| h.chunk.clone()).collect(),
                output_schema: schema.clone(),
                budget: cwc_core::types::TokenBudget {
                    total: model_ctx, instruction: report.instruction_tokens, sources: report.sources_budget,
                    memory: report.memory_tokens, output_reserved: max_output,
                    remaining: model_ctx.saturating_sub(report.instruction_tokens + report.sources_budget + report.memory_tokens + max_output),
                },
            };
            let scaffold = PromptScaffold::new(ScaffoldConfig::default());
            let msgs = scaffold.compose_chat(&ctx, &query, None, &source_ids);
            let backend_type: cwc_llm::Backend = backend.parse().unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1); });
            let grammar_str = if backend_type == cwc_llm::Backend::LlamaCpp {
                schema.as_ref().and_then(|s| json_schema_to_gbnf(s).ok())
            } else {
                schema.as_ref().map(|s| serde_json::to_string(s).unwrap_or_default())
            };
            let params = cwc_llm::GenerateParams { temperature, ..Default::default() };
            let result = match backend_type {
                cwc_llm::Backend::LlamaCpp => {
                    let client = cwc_llm::llamacpp::LlamaCppClient::new(&endpoint, params);
                    client.generate_chat(&msgs, grammar_str.as_deref(), max_output).await
                }
                cwc_llm::Backend::Vllm => {
                    let client = cwc_llm::vllm::VllmClient::new(&endpoint, &model, params);
                    client.generate_chat(&msgs, grammar_str.as_deref(), max_output).await
                }
            };
            match result {
                Ok(response) => {
                    match parse_structured_response(&response, schema.as_ref()) {
                        Ok(p) => {
                            if let Some(json) = &p.json { println!("{}", serde_json::to_string_pretty(json).unwrap_or_else(|e| format!("JSON error: {e}"))); }
                            else { println!("{}", p.raw_text); }
                            if !p.citations.is_empty() { eprintln!("Citations: {}", p.citations.join(", ")); }
                            if p.is_abstention { eprintln!("(Model abstained: INSUFFICIENT_EVIDENCE)"); }
                        }
                        Err(e) => { eprintln!("Parse error: {e}"); println!("{response}"); }
                    }
                }
                Err(e) => { eprintln!("LLM error: {e}"); std::process::exit(1); }
            }
            eprintln!("Budget: {}/{} tokens, {} chunks, {:.0}% util", report.sources_used, report.sources_budget, report.chunks_selected, report.utilization * 100.0);
        }
        Commands::Eval { action } => {
            match action {
                EvalAction::Run { dataset, output } => {
                    let ds = match cwc_eval::dataset::EvalDataset::load_jsonl(&dataset) {
                        Ok(d) => d,
                        Err(e) => { eprintln!("Error loading dataset: {e}"); std::process::exit(1); }
                    };
                    println!("Loaded dataset '{}' with {} queries", ds.name, ds.queries.len());
                    println!("Note: Full benchmark run requires a running LLM server.");
                    println!("Dataset loaded successfully. Categories:");
                    let mut cats = std::collections::HashMap::new();
                    for q in &ds.queries {
                        *cats.entry(format!("{:?}", q.category)).or_insert(0usize) += 1;
                    }
                    for (cat, count) in &cats {
                        println!("  {cat}: {count}");
                    }
                    if let Some(out) = output {
                        match ds.save_jsonl(&out) {
                            Ok(()) => println!("Saved dataset to {}", out.display()),
                            Err(e) => { eprintln!("Error saving: {e}"); std::process::exit(1); }
                        }
                    }
                }
                EvalAction::Compare { results_a: _, results_b: _ } => {
                    eprintln!("A/B comparison requires serialized BenchmarkReport files.");
                    eprintln!("Run 'cwc eval run' with different configs to generate reports first.");
                    std::process::exit(1);
                }
                EvalAction::Report { results: _ } => {
                    eprintln!("Report display requires serialized BenchmarkReport file.");
                    eprintln!("Run 'cwc eval run' to generate results first.");
                    std::process::exit(1);
                }
                EvalAction::Adversarial { output } => {
                    let ds = cwc_eval::adversarial::create_adversarial_dataset();
                    match ds.save_jsonl(&output) {
                        Ok(()) => {
                            println!("Created adversarial dataset with {} queries", ds.queries.len());
                            println!("Saved to {}", output.display());
                        }
                        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                    }
                }
            }
        }
        Commands::Memory { action } => {
            use cwc_memory::file_memory::FileMemory;
            use cwc_memory::longterm::MemoryCategory;

            fn parse_category(s: &str) -> MemoryCategory {
                match s {
                    "preference" => MemoryCategory::UserPreference,
                    "fact" => MemoryCategory::ProjectFact,
                    "decision" => MemoryCategory::PriorDecision,
                    "correction" => MemoryCategory::Correction,
                    other => { eprintln!("Unknown category: {other}. Use: preference, fact, decision, correction"); std::process::exit(1); }
                }
            }

            match action {
                MemoryAction::List { file } => {
                    let mem = FileMemory::new(&file);
                    match mem.list_all() {
                        Ok(entries) => {
                            if entries.is_empty() { println!("No memories stored."); } else {
                                for e in &entries { println!("[{}] {} = {} (accessed {}x)", e.category.as_str(), e.key, e.value, e.access_count); }
                                println!("\n{} entries total.", entries.len());
                            }
                        }
                        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                    }
                }
                MemoryAction::Add { key, value, category, file } => {
                    let mem = FileMemory::new(&file);
                    let cat = parse_category(&category);
                    match mem.upsert(&key, &value, cat) {
                        Ok(()) => println!("Stored: {key} = {value}"),
                        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                    }
                }
                MemoryAction::Delete { key, file } => {
                    let mem = FileMemory::new(&file);
                    match mem.delete(&key) {
                        Ok(true) => println!("Deleted: {key}"),
                        Ok(false) => println!("Not found: {key}"),
                        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                    }
                }
                MemoryAction::Search { query, max, file } => {
                    let mem = FileMemory::new(&file);
                    match mem.retrieve(&query, max) {
                        Ok(entries) => {
                            if entries.is_empty() { println!("No matching memories."); } else {
                                for e in &entries { println!("[{}] {} = {}", e.category.as_str(), e.key, e.value); }
                            }
                        }
                        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                    }
                }
            }
        }
        Commands::Session { action } => {
            run_session_command(action).await;
        }
        Commands::Embed { chunks, query, model, output, cache_dir } => {
            let model_path = model.join("model.onnx");
            let tokenizer_path = model.join("tokenizer.json");
            let config = cwc_embed::EmbedConfig { model_path: model_path.clone(), tokenizer_path: tokenizer_path.clone(), ..Default::default() };
            let embedder = match cwc_embed::OnnxEmbedder::load(&model_path, &tokenizer_path, &config) {
                Ok(e) => e, Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            if let Some(query_text) = query {
                match embedder.embed_query(&query_text) {
                    Ok(emb) => {
                        println!("Query embedding (dim={}):", emb.len());
                        let preview: Vec<String> = emb.iter().take(5).map(|v| format!("{v:.6}")).collect();
                        println!("  [{}, ...]", preview.join(", "));
                    }
                    Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                }
            } else if let Some(chunks_path) = chunks {
                let loaded = match cwc_ingest::store::load_chunks(&chunks_path) {
                    Ok(c) => c, Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                };
                println!("Loaded {} chunks from {}", loaded.len(), chunks_path.display());
                let cache = cwc_embed::EmbeddingCache::new(&cache_dir);
                let embedder_trait: &dyn cwc_core::traits::Embedder = &embedder;
                match cwc_embed::embed_chunks(&loaded, embedder_trait, &cache, config.batch_size) {
                    Ok(results) => {
                        let dim = config.dim as u32;
                        let count = results.len() as u32;
                        if let Some(parent) = output.parent() { let _ = std::fs::create_dir_all(parent); }
                        let mut data = Vec::new();
                        data.extend_from_slice(&dim.to_le_bytes());
                        data.extend_from_slice(&count.to_le_bytes());
                        for (_, emb) in &results { for v in emb { data.extend_from_slice(&v.to_le_bytes()); } }
                        match std::fs::write(&output, &data) {
                            Ok(()) => println!("Wrote {count} embeddings to {}", output.display()),
                            Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                        }
                        let idx_path = output.with_extension("idx");
                        let idx_lines: Vec<String> = results.iter().enumerate().map(|(i, (cid, _))| format!("{cid}\t{}", 8 + i * (dim as usize) * 4)).collect();
                        match std::fs::write(&idx_path, idx_lines.join("\n")) {
                            Ok(()) => println!("Wrote index to {}", idx_path.display()),
                            Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                        }
                    }
                    Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
                }
            } else {
                eprintln!("Provide either --chunks or --query");
                std::process::exit(1);
            }
        }
    }
}

async fn run_session_command(action: SessionAction) {
    use cwc_session::{SessionManager, SessionManagerConfig, format as session_format};
    use cwc_session::config::ModelProfileConfig;

    struct WordTokenizer;
    impl cwc_core::traits::TokenCounter for WordTokenizer {
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

    let tokenizer: Arc<dyn cwc_core::traits::TokenCounter> = Arc::new(WordTokenizer);

    match action {
        SessionAction::Optimize { input, format, model, output, report, llm_consolidate } => {
            let json_str = match std::fs::read_to_string(&input) {
                Ok(s) => s,
                Err(e) => { eprintln!("Error reading {}: {e}", input.display()); std::process::exit(1); }
            };
            let messages: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error parsing JSON: {e}"); std::process::exit(1); }
            };

            let parsed = match parse_session_messages(&format, &messages, &*tokenizer) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let config = SessionManagerConfig {
                session: cwc_session::SessionConfig {
                    model: ModelProfileConfig::Preset(model),
                    ..Default::default()
                },
                llm_consolidation: cwc_session::LlmConsolidationConfig {
                    enabled: llm_consolidate,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut mgr = match SessionManager::new(config, tokenizer.clone()) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let (mut optimized, rpt) = match mgr.optimize_messages(parsed) {
                Ok(r) => r,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            // Run async LLM consolidation post-step if enabled
            // Note: requires set_llm() to have been called; without it,
            // llm_consolidate() returns a no-op report.
            if llm_consolidate {
                let llm_rpt = mgr.llm_consolidate().await;
                if llm_rpt.groups_refined > 0 || llm_rpt.fallback_to_heuristic > 0 {
                    eprintln!("LLM consolidation: {} refined, {} heuristic fallback, {} bytes saved",
                        llm_rpt.groups_refined, llm_rpt.fallback_to_heuristic, llm_rpt.bytes_saved);
                } else if llm_rpt.groups_refined == 0 && llm_rpt.source_facts_replaced == 0 {
                    eprintln!("Warning: --llm-consolidate had no effect (no LLM client or no groups found)");
                }
                // Refresh the memory message in the output to reflect consolidation
                mgr.refresh_memory_in_messages(&mut optimized);
            }

            let output_json = session_format::openai::to_openai(&optimized);
            let output_str = serde_json::to_string_pretty(&output_json).unwrap_or_else(|e| {
                eprintln!("JSON serialization error: {e}");
                std::process::exit(1);
            });

            if let Some(path) = output {
                std::fs::write(&path, &output_str).unwrap_or_else(|e| {
                    eprintln!("Error writing {}: {e}", path.display());
                    std::process::exit(1);
                });
            } else {
                println!("{output_str}");
            }

            if report {
                eprintln!("--- Optimization Report ---");
                eprintln!("Input tokens:  {}", rpt.input_tokens);
                eprintln!("Output tokens: {}", rpt.output_tokens);
                eprintln!("Tokens saved:  {}", rpt.tokens_saved);
                eprintln!("Trim action:   {:?}", rpt.trim);
                eprintln!("Nudge:         {}", rpt.nudge_injected);
                eprintln!("Memory facts:  {}", rpt.memory_facts);
                if let Some(ref cr) = rpt.consolidation {
                    eprintln!("Consolidation: {} groups, {} facts replaced, {} bytes saved",
                        cr.facts_consolidated, cr.source_facts_replaced, cr.bytes_saved);
                }
                if !rpt.preflight_issues.is_empty() {
                    eprintln!("Preflight issues:");
                    for issue in &rpt.preflight_issues {
                        eprintln!("  [{:?}] {} (repaired: {})", issue.kind, issue.description, issue.auto_repaired);
                    }
                }
            }
        }
        SessionAction::Status { input, format, model } => {
            let json_str = match std::fs::read_to_string(&input) {
                Ok(s) => s,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let messages: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let parsed = match parse_session_messages(&format, &messages, &*tokenizer) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let config = SessionManagerConfig {
                session: cwc_session::SessionConfig {
                    model: ModelProfileConfig::Preset(model),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mgr = match SessionManager::new(config, tokenizer.clone()) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let mut session = cwc_session::Session::new(tokenizer.clone());
            for msg in parsed {
                session.messages_mut().push(msg);
            }
            session.recalculate_total();

            let status = mgr.budget_status(&session);
            println!("Budget Status:");
            println!("  Total tokens:    {}", status.total_tokens);
            println!("  Usable budget:   {}", status.usable_budget);
            println!("  Utilization:     {:.1}%", status.utilization_percent);
            println!("  Action needed:   {:?}", status.action_needed);
            println!("  Turns:           {}", status.turns);
            println!("  Memory facts:    {}", status.memory_facts);
        }
        SessionAction::Memory { input, format } => {
            let json_str = match std::fs::read_to_string(&input) {
                Ok(s) => s,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let messages: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let parsed = match parse_session_messages(&format, &messages, &*tokenizer) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let config = SessionManagerConfig::default();
            let mut mgr = match SessionManager::new(config, tokenizer.clone()) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let mut session = cwc_session::Session::new(tokenizer.clone());
            for msg in parsed {
                session.messages_mut().push(msg);
            }
            session.recalculate_total();
            let _ = mgr.optimize(&mut session);

            println!("Extracted Memory Facts:");
            for fact in mgr.memory().all() {
                println!("  [{:?}] {}: {}", fact.priority, fact.key, fact.value);
            }
            if let Some(goal) = mgr.goal() {
                println!("Goal: {goal}");
            }
        }
        SessionAction::Compact { input, format, output } => {
            let json_str = match std::fs::read_to_string(&input) {
                Ok(s) => s,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };
            let messages: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let parsed = match parse_session_messages(&format, &messages, &*tokenizer) {
                Ok(m) => m,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            let dir = tempfile::tempdir().unwrap_or_else(|e| {
                eprintln!("Error: {e}"); std::process::exit(1);
            });
            let engine = cwc_session::CompactionEngine::new(
                cwc_session::compaction::rules::default_rules(),
                dir.path(),
                tokenizer.clone(),
            ).unwrap_or_else(|e| {
                eprintln!("Error: {e}"); std::process::exit(1);
            });

            let mut session = cwc_session::Session::new(tokenizer.clone());
            for msg in parsed {
                session.messages_mut().push(msg);
            }
            session.recalculate_total();

            let report = cwc_session::compact_session(&mut session, &engine).unwrap_or_else(|e| {
                eprintln!("Error: {e}"); std::process::exit(1);
            });

            let output_json = session_format::openai::to_openai(session.messages());
            let output_str = serde_json::to_string_pretty(&output_json).unwrap_or_else(|e| {
                eprintln!("JSON serialization error: {e}");
                std::process::exit(1);
            });

            if let Some(path) = output {
                std::fs::write(&path, &output_str).unwrap_or_else(|e| {
                    eprintln!("Error writing {}: {e}", path.display());
                    std::process::exit(1);
                });
            } else {
                println!("{output_str}");
            }

            eprintln!("Compacted {} messages, saved {} tokens", report.messages_compacted, report.tokens_saved);
        }
        SessionAction::Eval { model, output } => {
            use cwc_session::eval::{SessionEvalRunner, format_session_report};

            let config = SessionManagerConfig {
                session: cwc_session::SessionConfig {
                    model: ModelProfileConfig::Preset(model),
                    ..Default::default()
                },
                ..Default::default()
            };
            let runner = SessionEvalRunner::new(config, tokenizer.clone());
            let report = match runner.run_all() {
                Ok(r) => r,
                Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
            };

            if let Some(path) = output {
                let json = serde_json::to_string_pretty(&report).unwrap_or_else(|e| {
                    eprintln!("JSON serialization error: {e}");
                    std::process::exit(1);
                });
                std::fs::write(&path, &json).unwrap_or_else(|e| {
                    eprintln!("Error writing {}: {e}", path.display());
                    std::process::exit(1);
                });
                eprintln!("Report written to {}", path.display());
            }

            print!("{}", format_session_report(&report));
        }
        SessionAction::Recommend { model_name, context_window, max_output } => {
            use cwc_session::eval::tuning::{recommend_config, tuned_profiles};

            let config = recommend_config(model_name.as_deref(), context_window, max_output);

            println!("Recommended Session Config:");
            println!("  Model: {:?}", config.session.model);
            println!("  Sliding window fraction: {:.2}", config.session.sliding_window_fraction);
            println!("  Hard reset fraction:     {:.2}", config.session.hard_reset_fraction);
            println!("  Tail tokens:             {}", config.session.tail_tokens);
            println!("  Reinforcement enabled:   {}", config.reinforcement.enabled);
            println!("  Nudge every N results:   {}", config.reinforcement.nudge_every_n_tool_results);
            println!("  Include goal in nudge:   {}", config.reinforcement.include_goal);

            // Show which profile was matched
            let profiles = tuned_profiles();
            if let Some(name) = &model_name {
                println!("\n  Matched from model name: {name}");
            } else {
                println!("\n  Matched by nearest context window ({context_window})");
            }
            println!("\nAvailable tuned profiles:");
            for p in &profiles {
                println!("  {} (ctx={}, max_out={})", p.model.name, p.model.context_window, p.model.max_output_tokens);
            }
        }
    }
}

fn parse_session_messages(
    format: &str,
    messages: &[serde_json::Value],
    tokenizer: &dyn cwc_core::traits::TokenCounter,
) -> cwc_session::Result<Vec<cwc_session::SessionMessage>> {
    match format {
        "openai" => cwc_session::format::openai::from_openai(messages, tokenizer),
        "anthropic" => cwc_session::format::anthropic::from_anthropic(None, messages, tokenizer),
        "raw" => {
            // Native SessionMessage JSON format — re-serialize then parse
            let json_str = serde_json::to_string(messages)
                .map_err(|e| cwc_session::SessionError::InvalidFormat(e.to_string()))?;
            cwc_session::format::raw::from_json(&json_str, tokenizer)
        }
        "auto" => cwc_session::format::auto_detect_and_parse(messages, None, tokenizer),
        other => Err(cwc_session::SessionError::InvalidFormat(format!("unknown format: {other}"))),
    }
}

/// Recursively collect all files under a directory.
fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let ep = entry.path();
        if ep.is_dir() {
            collect_files_recursive(&ep, out);
        } else if ep.is_file() {
            if let Ok(c) = ep.canonicalize() {
                out.push(c);
            }
        }
    }
}

fn load_config_or_default(path: &PathBuf) -> cwc_core::config::CwcConfig {
    if path.exists() {
        cwc_core::config::CwcConfig::load(path).unwrap_or_else(|e| {
            eprintln!("Warning: failed to load {}: {e}. Using defaults.", path.display());
            cwc_core::config::CwcConfig::default()
        })
    } else {
        cwc_core::config::CwcConfig::default()
    }
}

fn parse_task_type(s: &str) -> cwc_compile::schema::TaskType {
    match s {
        "qa" => cwc_compile::schema::TaskType::QuestionAnswer,
        "summary" => cwc_compile::schema::TaskType::Summary,
        "extraction" => cwc_compile::schema::TaskType::Extraction,
        "classification" => cwc_compile::schema::TaskType::Classification,
        "freeform" => cwc_compile::schema::TaskType::FreeForm,
        other => { eprintln!("Unknown task type: {other}"); std::process::exit(1); }
    }
}

fn print_retrieval_hits(hits: &[cwc_core::types::RetrievalHit]) {
    if hits.is_empty() { println!("No results."); } else {
        for (rank, hit) in hits.iter().enumerate() {
            let preview: String = hit.chunk.text.chars().take(80).collect();
            println!("{:>3}. [{:.4}] {} — {}", rank + 1, hit.score_sparse, hit.chunk.chunk_id, preview);
        }
    }
}

fn print_dense_results(results: &[(cwc_core::types::Chunk, f32)]) {
    if results.is_empty() { println!("No results."); } else {
        for (rank, (chunk, score)) in results.iter().enumerate() {
            let preview: String = chunk.text.chars().take(80).collect();
            println!("{:>3}. [{:.4}] {} — {}", rank + 1, score, chunk.chunk_id, preview);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WordTokenizer;
    impl cwc_core::traits::TokenCounter for WordTokenizer {
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
    fn test_tc() -> &'static dyn cwc_core::traits::TokenCounter {
        &WordTokenizer
    }

    #[test]
    fn test_load_config_missing_file() {
        let config = load_config_or_default(&PathBuf::from("/nonexistent/cwc.toml"));
        assert_eq!(config.model.context_window, 4096);
    }

    #[test]
    fn test_cli_session_optimize_reads_json() {
        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("conversation.json");

        let input = serde_json::json!([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ]);
        std::fs::write(&input_path, serde_json::to_string(&input).unwrap()).unwrap();

        // Parse and optimize
        let json_str = std::fs::read_to_string(&input_path).unwrap();
        let messages: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap();
        let parsed = parse_session_messages("auto", &messages, test_tc()).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].role, cwc_session::SessionRole::System);
    }

    #[test]
    fn test_cli_session_status_shows_budget() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "hello"}),
        ];
        let parsed = parse_session_messages("openai", &messages, test_tc()).unwrap();
        let tokenizer: Arc<dyn cwc_core::traits::TokenCounter> = Arc::new(WordTokenizer);
        let config = cwc_session::SessionManagerConfig::default();
        let mgr = cwc_session::SessionManager::new(config, tokenizer.clone()).unwrap();
        let mut session = cwc_session::Session::new(tokenizer);
        for msg in parsed {
            session.messages_mut().push(msg);
        }
        session.recalculate_total();
        let status = mgr.budget_status(&session);
        assert!(status.total_tokens > 0);
        assert!(status.usable_budget > 0);
    }

    #[test]
    fn test_cli_session_memory_extracts_goal() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "Find security vulnerabilities"}),
            serde_json::json!({"role": "assistant", "content": "I will search"}),
        ];
        let parsed = parse_session_messages("openai", &messages, test_tc()).unwrap();
        let tokenizer: Arc<dyn cwc_core::traits::TokenCounter> = Arc::new(WordTokenizer);
        let config = cwc_session::SessionManagerConfig::default();
        let mut mgr = cwc_session::SessionManager::new(config, tokenizer.clone()).unwrap();
        let mut session = cwc_session::Session::new(tokenizer);
        for msg in parsed {
            session.messages_mut().push(msg);
        }
        session.recalculate_total();
        let _ = mgr.optimize(&mut session);
        assert!(mgr.goal().is_some());
    }

    #[test]
    fn test_cli_session_compact_no_trim() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "hello"}),
        ];
        let parsed = parse_session_messages("openai", &messages, test_tc()).unwrap();
        let tokenizer: Arc<dyn cwc_core::traits::TokenCounter> = Arc::new(WordTokenizer);
        let dir = tempfile::tempdir().unwrap();
        let engine = cwc_session::CompactionEngine::new(
            cwc_session::compaction::rules::default_rules(),
            dir.path(),
            tokenizer.clone(),
        ).unwrap();
        let mut session = cwc_session::Session::new(tokenizer);
        for msg in parsed {
            session.messages_mut().push(msg);
        }
        session.recalculate_total();
        let report = cwc_session::compact_session(&mut session, &engine).unwrap();
        // No tool results to compact
        assert_eq!(report.messages_compacted, 0);
    }

    #[test]
    fn test_cli_parse_raw_format_native_json() {
        // "raw" format should parse native SessionMessage JSON, not auto-detect
        // Build native messages, serialize, then parse back via "raw"
        let messages = vec![
            cwc_session::SessionMessage::system("sys"),
            cwc_session::SessionMessage::text(cwc_session::SessionRole::User, "hello"),
        ];
        let json_str = cwc_session::format::raw::to_json(&messages).unwrap();
        let raw_values: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap();
        let parsed = parse_session_messages("raw", &raw_values, test_tc()).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].role, cwc_session::SessionRole::System);
        assert_eq!(parsed[1].role, cwc_session::SessionRole::User);
    }
}
