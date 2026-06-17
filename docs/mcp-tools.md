# MCP Tools Reference

The Local Context Engine exposes 12 tools via the Model Context Protocol (MCP).

All tools are **read-only**. No tool modifies source files.
All snippets returned have passed through the PII redactor.

The MCP server supports **stdio** and **streamable-http** transports and works
with any MCP-compatible client: Claude Code, Claude Desktop, Codex CLI, Cursor,
Windsurf, Cline, Continue.dev, and others. The HTTP transport is recommended for
persistent use to avoid cold-start timeouts.
See [configuration.md](configuration.md#mcp-server) for setup details.

---

## search_codebase

Hybrid semantic + keyword search over the indexed codebase.

```json
{
  "name": "search_codebase",
  "arguments": {
    "query": "journal entry approval flow",
    "limit": 10,
    "language": "php",
    "path_filter": "app/Http/**"
  }
}
```

**Returns:**
```json
[
  {
    "chunk_id": "abc123",
    "file_path": "app/Http/Controllers/JournalController.php",
    "symbol_name": "approve",
    "symbol_type": "method",
    "language": "php",
    "line_start": 45,
    "line_end": 72,
    "score": 0.8934,
    "score_breakdown": {
      "semantic": 0.912,
      "bm25": 0.734,
      "symbol": 0.5,
      "graph": 0.0
    },
    "snippet": "// File: app/Http/Controllers/JournalController.php\n..."
  }
]
```

---

## search_symbol

Find symbols (classes, functions, components) by name.

```json
{
  "name": "search_symbol",
  "arguments": {
    "name": "UserService",
    "symbol_type": "service",
    "language": "php"
  }
}
```

**Returns:**
```json
[
  {
    "id": "sym-001",
    "name": "UserService",
    "qualified_name": "App\\Services\\UserService",
    "symbol_type": "service",
    "language": "php",
    "file_path": "app/Services/UserService.php",
    "line_start": 10,
    "line_end": 150,
    "visibility": null,
    "docstring": "/** Handles user business logic. */"
  }
]
```

---

## read_chunk

Fetch the full source content of a chunk returned by `search_codebase`.

```json
{
  "name": "read_chunk",
  "arguments": {
    "chunk_id": "abc123"
  }
}
```

**Returns:**
```json
{
  "chunk_id": "abc123",
  "file_path": "app/Http/Controllers/JournalController.php",
  "language": "php",
  "symbol_name": "approve",
  "symbol_type": "method",
  "line_start": 45,
  "line_end": 72,
  "token_count": 234,
  "content": "// File: ...\n// Symbol: approve\n\n    public function approve(...) {...}"
}
```

---

## find_references

Find all symbols that reference the given symbol (who uses it).

```json
{
  "name": "find_references",
  "arguments": {
    "symbol_name": "UserService",
    "depth": 2
  }
}
```

**Returns:** List of referencing symbols with relationship type and file path.

---

## find_dependencies

Find all symbols that the given symbol depends on.

```json
{
  "name": "find_dependencies",
  "arguments": {
    "symbol_name": "OrderController",
    "depth": 3
  }
}
```

**Returns:** Ordered list of dependency symbols (Controller → Service → Repository → Model).

---

## trace_flow

Find the shortest dependency path between two symbols.

```json
{
  "name": "trace_flow",
  "arguments": {
    "from_symbol": "CheckoutController",
    "to_symbol": "Order"
  }
}
```

**Returns:**
```json
{
  "from": "CheckoutController",
  "to": "Order",
  "path_length": 3,
  "path": [
    {"name": "CheckoutController", "symbol_type": "controller", "file_path": "..."},
    {"name": "OrderService",       "symbol_type": "service",    "file_path": "..."},
    {"name": "Order",              "symbol_type": "model",      "file_path": "..."}
  ]
}
```

---

## explain_architecture

Return a high-level summary of the repository.

```json
{
  "name": "explain_architecture",
  "arguments": {}
}
```

**Returns:**
```json
{
  "repository": "/path/to/repo",
  "total_files": 847,
  "total_symbols": 12403,
  "total_chunks": 8921,
  "total_vectors": 8921,
  "languages": {"php": 432, "typescript": 289, "python": 126},
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "symbol_graph": {"nodes": 12403, "edges": 34201}
}
```

---

## find_similar_code

Find code that is semantically similar to a provided snippet.

```json
{
  "name": "find_similar_code",
  "arguments": {
    "code_snippet": "public function approve(Request $request, Journal $journal)\n{\n    $journal->update(['status' => 'approved']);\n}",
    "limit": 5
  }
}
```

Useful for finding duplication, existing patterns, or related implementations.

---

## recent_changes

List the most recently indexed files.

```json
{
  "name": "recent_changes",
  "arguments": {
    "limit": 20
  }
}
```

---

## save_memory

Persist a piece of architectural knowledge for future sessions.

```json
{
  "name": "save_memory",
  "arguments": {
    "content": "All API responses use Laravel API Resources and are wrapped in a data key. Use JsonResource for single objects, ResourceCollection for lists.",
    "category": "convention",
    "tags": ["laravel", "api", "response-format"]
  }
}
```

**Categories:** `architecture`, `convention`, `decision`, `bug`, `api`, `domain`, `tooling`, `general`

---

## retrieve_memory

Recall previously saved knowledge.

```json
{
  "name": "retrieve_memory",
  "arguments": {
    "query": "API response format",
    "category": "convention",
    "limit": 5
  }
}
```

---

## get_stats

Return indexing statistics (alias for `explain_architecture`).

```json
{
  "name": "get_stats",
  "arguments": {}
}
```

---

## Typical Agent Workflow

```
1. explain_architecture()              → Understand what's in the repo
2. search_symbol("UserController")     → Locate the entry point
3. find_dependencies("UserController") → Trace downstream dependencies
4. search_codebase("payment processing flow") → Find relevant code
5. read_chunk("<chunk_id>")            → Read the full implementation
6. trace_flow("CheckoutController", "Payment") → Understand the flow
7. save_memory("Payment uses Stripe...")  → Remember for next session
```
