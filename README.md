# Local Context Engine

**Privacy-first, self-hosted codebase context engine for AI coding assistants.**

Everything runs locally. No source code, embeddings, metadata, or sensitive information ever leaves the machine. Designed for enterprise and government repositories.

---

## Features

| Feature | Detail |
|---|---|
| **100% local** | No cloud APIs. No telemetry. No analytics. |
| **Privacy-first** | PII masking before embeddings. File blacklisting before scanning. |
| **Code parsing** | Accurate symbol extraction for PHP, TypeScript, JavaScript, Python, C#, ASP.NET, and SQL |
| **Laravel-aware** | Understands Controllers, Models, Services, Repositories, Migrations, Events |
| **React-aware** | Understands components, hooks, TanStack Query, TanStack Router, Context |
| **Hybrid retrieval** | Semantic (50%) + BM25 (20%) + Symbol (20%) + Graph (10%) |
| **Symbol graph** | Cross-file dependency analysis with NetworkX |
| **Incremental indexing** | Only re-indexes changed files via hash comparison |
| **MCP server** | Works with Claude Code, Claude Desktop, Codex CLI, Cursor, Windsurf, Cline, Continue.dev |
| **Agent memory** | Persistent architectural knowledge across sessions |
| **Docker support** | Isolated, reproducible deployment |
| **GPU/CPU** | Auto-detects CUDA and Apple Silicon |

---

## Supported Languages

- **PHP 8.4+** (Laravel 12+)
- **TypeScript / TSX**
- **JavaScript / JSX**
- **Python 3.12+**
- **C# / .NET**
- **ASP.NET** (Razor and Web Forms) and classic ASP
- **SQL**
- CSS / SCSS (partial)

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

Or with uv (recommended):

```bash
uv sync
```

### 2. Index Your Repository

Each project gets its own isolated index stored inside `.context/` within the project directory.

```bash
context index /path/to/your/laravel-app
context index /path/to/your/react-frontend
```

Re-index after major changes:

```bash
context index --full /path/to/your/laravel-app
```

### 3. Search

```bash
context search "journal entry approval flow"
context search "user authentication" --lang php
context search "useQuery hook" --lang typescript

# Search a specific project when not in its directory
context search "Agreement model" --repo /path/to/backend
```

### 4. Connect to Your AI Assistant

Each indexed project runs as its own MCP server. You can connect multiple projects simultaneously.

Works with any MCP-compatible client: Claude Code, Claude Desktop, Codex CLI, Cursor, Windsurf, Cline, Continue.dev, and others.

The MCP server supports two transport modes:

- **stdio** (default) — The AI client spawns a new process per session. Simple setup, but cold-starts the embedding model on every session.
- **HTTP** (recommended) — A persistent server you start once. Stays warm across sessions, no timeout issues.

---

## MCP Integration

### Option A: HTTP Transport (Recommended)

Start a persistent MCP server for each project:

**Windows (PowerShell):**

```powershell
.\scripts\start-mcp-server.ps1 D:\path\to\backend
.\scripts\start-mcp-server.ps1 D:\path\to\frontend -Port 8766
```

**Linux / macOS:**

```bash
./scripts/start-mcp-server.sh /path/to/backend
./scripts/start-mcp-server.sh /path/to/frontend 8766
```

Each server runs in the foreground (Ctrl+C to stop). The default port is `8765`.

Set the port per project in `.context/config.yaml`:

```yaml
mcp_server:
  port: 8766
```

Then configure your AI client to connect via HTTP. The server URL is `http://127.0.0.1:<port>/mcp`.

#### Claude Code / Claude Desktop

Edit `~/.claude/settings.json` (Claude Code) or `claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "my-backend": {
      "url": "http://127.0.0.1:8765/mcp",
      "type": "streamable-http"
    },
    "my-frontend": {
      "url": "http://127.0.0.1:8766/mcp",
      "type": "streamable-http"
    }
  }
}
```

#### Codex CLI

```bash
codex --mcp-server-url http://127.0.0.1:8765/mcp
```

Or in `~/.codex/config.yaml`:

```yaml
mcp_servers:
  - url: http://127.0.0.1:8765/mcp
  - url: http://127.0.0.1:8766/mcp
```

#### Cursor

In Cursor Settings > MCP, add a new server:

| Field | Value |
|---|---|
| Transport | `http` |
| URL | `http://127.0.0.1:8765/mcp` |

#### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "my-backend": {
      "serverUrl": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

#### Cline (VS Code)

In Cline Settings > MCP Servers, add:

```json
{
  "my-backend": {
    "url": "http://127.0.0.1:8765/mcp",
    "transportType": "streamable-http"
  }
}
```

#### Any MCP Client

The server speaks the standard [Model Context Protocol](https://modelcontextprotocol.io).
Any client that supports `streamable-http` transport can connect to `http://127.0.0.1:<port>/mcp`.

---

### Option B: stdio Transport

The simpler setup — no separate server to manage. The AI client spawns the process directly. Works best when the ONNX backend is enabled (see [Configuration](#configuration)) to keep cold-start times low.

Any MCP client that supports stdio transport can use:

```
command: context
args:    ["mcp", "/path/to/your/repo"]
```

> **Windows note:** If `context` is not on PATH, use the full path to the executable:
> `C:\Users\<you>\AppData\Roaming\Python\Python313\Scripts\context.exe`

#### Claude Desktop

Config file location:

| OS | Path |
|---|---|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "my-backend": {
      "command": "context",
      "args": ["mcp", "/path/to/backend"]
    },
    "my-frontend": {
      "command": "context",
      "args": ["mcp", "/path/to/frontend"]
    }
  }
}
```

#### Claude Code

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "my-backend": {
      "command": "context",
      "args": ["mcp", "/path/to/backend"],
      "type": "stdio"
    }
  }
}
```

> **Tip:** If you experience timeouts on first tool calls, add `"timeout": 30000`
> to give the embedding model time to load, or switch to HTTP transport above.

#### Cursor / Windsurf / Cline

In your editor's MCP settings, add a new stdio server with:
- **Command:** `context`
- **Args:** `mcp /path/to/your/repo`

#### Codex CLI

```bash
codex --mcp-server-command "context mcp /path/to/repo"
```

#### Continue.dev (`.continue/config.json`)

```json
{
  "contextProviders": [
    {
      "name": "mcp",
      "params": {
        "transport": {
          "type": "stdio",
          "command": "context",
          "args": ["mcp", "/path/to/your/repo"]
        }
      }
    }
  ]
}
```

---

## CLI Reference

```
context index    <repo>         Incremental index (skips unchanged files)
context index    <repo> --full  Force full re-index
context reindex  <repo>         Alias for index --full
context search   <query>        Search the indexed codebase
context inspect  <file>         Show all symbols in a file
context stats                   Show indexing statistics
context graph    <symbol>       Show symbol graph relationships
context memory list             Browse stored memories
context memory save <content>   Save agent knowledge
context mcp      <repo>         Start the MCP server (stdio)
context mcp      <repo> -t streamable-http  Start persistent HTTP server
context doctor                  Diagnose configuration issues
context benchmark               Run performance benchmarks
```

---

## MCP Tools

| Tool | Description |
|---|---|
| `search_codebase` | Hybrid semantic + keyword search |
| `search_symbol` | Find symbols by name and type |
| `read_chunk` | Fetch full content of a code chunk |
| `find_references` | What uses this symbol? |
| `find_dependencies` | What does this symbol depend on? |
| `trace_flow` | Trace path between two symbols |
| `explain_architecture` | High-level repository summary |
| `find_similar_code` | Find semantically similar code |
| `recent_changes` | Recently modified files |
| `save_memory` | Persist agent knowledge |
| `retrieve_memory` | Recall agent knowledge |
| `get_stats` | Indexing statistics |

---

## Configuration

The engine is configured via YAML. A project-specific config lives at `.context/config.yaml`.

Key settings:

```yaml
embedding:
  model: "BAAI/bge-small-en-v1.5"   # Local model, no cloud
  device: "auto"                      # "auto" | "cpu" | "cuda" | "mps"
  backend: "onnx"                     # "torch" | "onnx" (faster cold start)

vector_store:
  backend: "faiss"
  dimension: 384

security:
  enable_pii_masking: true
  never_index_patterns:
    - "**/.env"
    - "**/*.key"
    - "**/vendor/**"
    - "**/node_modules/**"
```

> **ONNX backend**: Setting `backend: "onnx"` exports the model to ONNX format on first use,
> then loads via ONNX Runtime on subsequent starts. This reduces cold-start time significantly.
> Requires: `pip install sentence-transformers[onnx-gpu]` (or `[onnx]` for CPU-only).

See [docs/configuration.md](docs/configuration.md) for all options.

---

## Supported Embedding Models (All Local)

| Model | Dimensions | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | Default, fast |
| `BAAI/bge-base-en-v1.5` | 768 | Balanced |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Excellent for code |
| `all-MiniLM-L6-v2` | 384 | General purpose |

---

## Architecture

```
Repository
    ↓
Scanner (gitignore + blacklist aware)
    ↓
Parser (Tree-sitter: PHP / TS / JS / Python)
    ↓
Chunker (symbol-boundary aware, 50–1200 tokens)
    ↓ (PII masking before this step)
Embedder (local sentence-transformers)
    ↓
Vector Store (FAISS, local disk)
    ↓
Metadata Store (SQLite)
    ↓
Symbol Graph (NetworkX)
    ↓
Hybrid Retriever (semantic + BM25 + symbol + graph)
    ↓
MCP Server (FastMCP, stdio or HTTP)
    ↓
AI Agent
```

---

## Security Guarantees

1. **File blacklist**: `.env`, `*.key`, `*.pem`, `vendor/`, `node_modules/`, logs, sessions are never read.
2. **PII masking**: Email, phone, NIC, TIN, JWT, API keys, signed URLs are replaced before embedding.
3. **Read-only**: The engine never modifies source files.
4. **No network**: No outbound connections after model download.
5. **Local models**: All embedding models run on-device via HuggingFace.
6. **Non-root Docker**: Container runs as an unprivileged `lce` user.

---

## Docker

```bash
# Build
docker build -t local-context-engine -f docker/Dockerfile .

# Index
docker run --rm \
  -v /path/to/repo:/repo:ro \
  -v lce_data:/data \
  local-context-engine index /repo

# MCP server (stdio)
docker run -i --rm \
  -v /path/to/repo:/repo:ro \
  -v lce_data:/data \
  local-context-engine mcp /repo
```

---

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires models downloaded)
pytest tests/integration/ -m slow

# Benchmarks
pytest tests/benchmarks/ --benchmark-only
```

---

## Development

```bash
# Install with dev extras
uv sync --extra dev

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Format
ruff format src/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE).
