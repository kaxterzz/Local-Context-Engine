# Architecture — Local Context Engine

## Overview

The Local Context Engine is a modular, privacy-first codebase indexing system. Every component is independently replaceable. No component ever communicates with the internet.

## Pipeline Flow

```
Source Code Repository
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  SCANNER  (Phase 1)                                    │
│  • Recursive directory traversal                        │
│  • .gitignore-aware filtering (pathspec)                │
│  • File blacklist enforcement (security.never_index)    │
│  • Language detection from extension                    │
│  • File hash computation (MD5)                          │
│  • Modification time tracking                           │
│  → Emits: FileRecord[]                                  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  PARSER  (Phase 2)                                      │
│  • Tree-sitter grammars per language                    │
│  • PHP: classes, interfaces, traits, methods, routes    │
│  • TypeScript: classes, interfaces, components, hooks   │
│  • Python: classes, functions, methods                  │
│  • Laravel-aware: Controller/Model/Service detection    │
│  • React-aware: PascalCase components, use* hooks       │
│  → Emits: Symbol[], Relationship[]                      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  SECURITY LAYER  (between Parse and Chunk)              │
│  • PIIMasker applies regex patterns to content          │
│  • Replaces: email, phone, JWT, API keys, NIC, TIN     │
│  • Produces content_masked (original preserved in DB)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  CHUNKER  (Phase 3)                                     │
│  • Symbol-boundary-aware splitting                      │
│  • Target: 50–1200 tokens per chunk                     │
│  • Adds context header: file path + symbol name         │
│  • Falls back to line-based splitting for huge classes  │
│  • Captures "gap" code between symbol declarations      │
│  → Emits: Chunk[]                                       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  EMBEDDER  (Phase 4)                                    │
│  • Sentence-transformers (local HuggingFace models)     │
│  • Supports torch and ONNX Runtime backends             │
│  • Embeds content_masked (never raw content)            │
│  • Batch processing with configurable batch_size        │
│  • GPU (CUDA/MPS) or CPU                                │
│  → Emits: ChunkVector[] (chunk_id + np.ndarray)         │
└──────────────────────────┬─────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ VECTOR STORE │  │  METADATA    │  │  SYMBOL GRAPH    │
│ (Phase 5)    │  │  STORE       │  │  (Phase 7)       │
│              │  │  (Phase 6)   │  │                  │
│ FAISS index  │  │ SQLite DB    │  │ NetworkX DiGraph  │
│ + sidecar    │  │ (SQLAlchemy) │  │                  │
│ JSON metadata│  │             │  │ Nodes: Symbols    │
│              │  │ files        │  │ Edges: Relations  │
│ .context/    │  │ symbols      │  │                  │
│ vectors/     │  │ chunks       │  │ .context/        │
│              │  │ relationships│  │ graph.pkl        │
└──────────────┘  └──────────────┘  └──────────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  HYBRID RETRIEVER  (Phase 8)                            │
│                                                         │
│  final_score = semantic_score × 0.50                    │
│              + bm25_score     × 0.20                    │
│              + symbol_score   × 0.20                    │
│              + graph_score    × 0.10                    │
│                                                         │
│  Semantic:  FAISS cosine similarity                     │
│  BM25:      rank-bm25 over chunk content                │
│  Symbol:    name-matching boost from metadata           │
│  Graph:     neighbourhood scoring via NetworkX          │
│                                                         │
│  → Returns: SearchResult[] (ranked)                     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  OUTPUT REDACTOR                                        │
│  • Second PII sweep on snippets before returning        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  MCP SERVER  (Phase 12)                                 │
│  FastMCP, stdio or HTTP transport                       │
│  Tools: search_codebase, search_symbol, read_chunk,     │
│         find_references, find_dependencies, trace_flow, │
│         explain_architecture, find_similar_code,        │
│         recent_changes, save_memory, retrieve_memory    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                    AI Coding Agent
      (Claude Code, Codex CLI, Cursor, Windsurf,
       Cline, Continue.dev, or any MCP client)
```

## Directory Structure

```
local-context-engine/
├── src/local_context_engine/
│   ├── core/               # Types, config, exceptions, logging
│   ├── security/           # Blacklist, PII masker, redactor
│   ├── metadata_store/     # SQLAlchemy ORM + repositories
│   ├── indexer/
│   │   ├── scanner/        # File traversal + gitignore
│   │   ├── parsers/        # Tree-sitter parsers per language
│   │   ├── chunkers/       # Symbol-aware chunking
│   │   ├── embedder/       # Local embedding models
│   │   └── pipeline.py     # Orchestrates full index run
│   ├── vector_store/       # FAISS + abstract interface
│   ├── symbol_graph/       # NetworkX graph + analyzers
│   ├── retrieval/          # Hybrid retriever + BM25
│   ├── memory/             # Agent memory (SQLite)
│   ├── mcp_server/         # FastMCP server + tools
│   └── cli/                # Typer CLI commands
├── config/                 # Default YAML config
├── scripts/                # MCP server startup scripts (PS1, Bash)
├── docker/                 # Dockerfile + Compose
├── tests/                  # Unit, integration, benchmarks
└── docs/                   # This documentation
```

## Key Design Decisions

### 1. Security First
The security layer runs at two points:
- **Pre-scan**: FileBlacklist checks every path before the file is opened
- **Pre-embed**: PIIMasker transforms content before it reaches the embedding model

This ensures sensitive data never enters the vector database, even if a file escapes the blacklist.

### 2. Symbol Boundaries for Chunking
Fixed-size character chunking splits code mid-function, breaking semantic coherence. The SymbolChunker uses parse tree boundaries (class/method/function) to keep related code together. This dramatically improves retrieval relevance.

### 3. Hybrid Retrieval
Pure semantic search misses exact name lookups (e.g., "UserController"). Pure BM25 misses conceptual queries (e.g., "how does payment processing work"). The weighted hybrid combines both, plus symbol-name boosting and graph-neighbourhood scoring.

### 4. Incremental Indexing
File hashes are stored in SQLite. On re-index, only files whose hash changed are re-parsed and re-embedded. This makes incremental runs fast even for 1M+ LOC repositories.

### 5. Lazy Model Loading
Embedding models are loaded on first use, not at startup. This keeps CLI commands (`stats`, `doctor`, `graph`) fast even when the model is large.

### 6. ONNX Backend
The default `torch` backend loads PyTorch and ~200 weight files on every cold start (5-15s). The `onnx` backend exports the model to ONNX format once (during indexing or first search), then loads via ONNX Runtime in under a second. This is especially important for the MCP server, where the AI client may timeout waiting for the model.

### 7. HTTP Transport for MCP
The stdio transport spawns a new process per AI session, cold-starting the model each time. The `streamable-http` transport runs a persistent HTTP server that stays warm across sessions. Any MCP-compatible client (Claude Code, Codex CLI, Cursor, Windsurf, Cline, Continue.dev) can connect via either transport. Helper scripts (`scripts/start-mcp-server.ps1` for Windows, `scripts/start-mcp-server.sh` for Linux/macOS) make it easy to start a server for any indexed repository.

## Extension Points

| Component | Interface | How to Extend |
|---|---|---|
| Parsers | `BaseParser` | Implement `parse()` for new language |
| Embedders | `BaseEmbedder` | Implement `embed_documents()` + `embed_query()` |
| Vector stores | `BaseVectorStore` | Implement 6 abstract methods |
| Chunkers | `BaseChunker` | Implement `chunk()` for new strategy |
| MCP tools | FastMCP `@mcp.tool()` | Add decorators in `server.py` |
