# Configuration Reference

Configuration is loaded in this priority order (highest wins):

1. Environment variables (`LCE_EMBEDDING__MODEL=...`)
2. Project config: `.context/config.yaml`
3. User config: `~/.config/local-context-engine/config.yaml`
4. Bundled defaults: `config/default.yaml`

---

## Embedding

```yaml
embedding:
  # Local HuggingFace model. Downloaded once, cached locally.
  model: "BAAI/bge-small-en-v1.5"

  # Compute device
  device: "cpu"        # "cpu" | "cuda" | "mps"

  # Model backend
  backend: "onnx"      # "torch" | "onnx"

  # Batch size for embedding generation
  batch_size: 32       # Lower if OOM on GPU

  # Max sequence length (tokens)
  max_seq_length: 512  # Max supported by most models

  # L2-normalize embeddings (required for cosine similarity with FAISS IP index)
  normalize_embeddings: true

  # Local cache directory (null = ~/.cache/huggingface)
  cache_dir: null

  # Query prefix (for asymmetric models like BGE)
  query_prefix: "Represent this sentence for searching relevant passages: "
  document_prefix: ""
```

### ONNX Backend

The `onnx` backend exports the model to ONNX format on first use (one-time, ~30-40s), then loads via ONNX Runtime on subsequent starts. This reduces cold-start time significantly compared to the default `torch` backend, which must load PyTorch + ~200 weight files on every start.

**Install:**

```bash
# GPU (CUDA)
pip install sentence-transformers[onnx-gpu]

# CPU only
pip install sentence-transformers[onnx]
```

> **Note:** After switching backends, run `context index <repo> --full` to rebuild embeddings
> with the new backend. Search results will be identical since the underlying model is the same.

### Recommended Models

| Use Case | Model | Dimensions |
|---|---|---|
| Fastest, smallest RAM | `BAAI/bge-small-en-v1.5` | 384 |
| Best balance | `BAAI/bge-base-en-v1.5` | 768 |
| Best quality (code) | `nomic-ai/nomic-embed-text-v1.5` | 768 |
| General purpose | `all-MiniLM-L6-v2` | 384 |

> **Important**: When changing models, run `context index . --full` to rebuild embeddings.

---

## Vector Store

```yaml
vector_store:
  backend: "faiss"    # Currently only "faiss" supported

  # Must match embedding model output dimension
  dimension: 384

  # FAISS index type
  index_type: "hnsw"  # "flat" | "hnsw" | "ivf"
  # flat: exact search, any size, slower for large sets
  # hnsw: approximate, fastest, recommended for >10k chunks
  # ivf:  approximate, memory-efficient for very large sets

  # HNSW parameters (only when index_type = "hnsw")
  hnsw_m: 16                  # Connections per node (higher = better recall, more RAM)
  hnsw_ef_construction: 200   # Build quality (higher = slower build, better recall)
  hnsw_ef_search: 50          # Search quality (higher = slower, better recall)

  # Storage path (relative to .context/ directory)
  storage_path: ".context/vectors"
```

---

## Security

```yaml
security:
  enable_pii_masking: true

  # Files matching these patterns are NEVER indexed
  # (in addition to built-in immutable blocklist)
  never_index_patterns:
    - "**/.env"
    - "**/*.key"
    - "**/vendor/**"
    - "**/node_modules/**"
    # Add custom patterns:
    # - "**/internal/secret/**"

  # Files larger than this are skipped (bytes)
  max_file_size_bytes: 2097152  # 2 MB

  # Binary extensions to skip
  binary_extensions:
    - ".exe"
    - ".dll"
    # ... (see default.yaml for full list)
```

---

## PII Masking

```yaml
pii_masking:
  mask_email: true         # user@example.com → [REDACTED_EMAIL]
  mask_phone: true         # +1 555 123 4567 → [REDACTED_PHONE]
  mask_nic: true           # NIC numbers → [REDACTED_NIC]
  mask_tin: true           # Tax IDs → [REDACTED_TIN]
  mask_vat: true           # VAT numbers → [REDACTED_VAT]
  mask_jwt: true           # eyJ... → [REDACTED_JWT]
  mask_api_keys: true      # api_key = 'sk-...' → [REDACTED_API_KEY]
  mask_access_tokens: true # access_token = '...' → [REDACTED_TOKEN]
  mask_signed_urls: true   # AWS/GCS/Azure signed URLs → [REDACTED_SIGNED_URL]
  mask_ip_addresses: false # Keep IPs (useful for network code)
  mask_credit_cards: true  # Card numbers → [REDACTED_CC]
```

---

## Chunking

```yaml
chunking:
  min_tokens: 50       # Chunks smaller than this are merged
  target_tokens: 500   # Ideal chunk size
  max_tokens: 1200     # Hard upper limit
  overlap_tokens: 50   # Context overlap between adjacent chunks
  strategy: "symbol"   # "symbol" (preferred) | "line" (fallback)
  tokenizer: "approx"  # "approx" (fast) | "tiktoken" (accurate)
```

---

## Retrieval

```yaml
retrieval:
  default_limit: 10    # Default results per search
  max_limit: 50        # Maximum results per search

  # Hybrid scoring weights (must sum to 1.0)
  semantic_weight: 0.50
  bm25_weight: 0.20
  symbol_weight: 0.20
  graph_weight: 0.10

  # Minimum score to include in results (0.0 = include all)
  min_score: 0.05

  # Candidates fetched from each retriever before reranking
  candidate_multiplier: 5
```

---

## MCP Server

```yaml
mcp_server:
  transport: "stdio"     # "stdio" | "streamable-http"
  host: "127.0.0.1"     # HTTP transport host (NEVER 0.0.0.0)
  port: 8765             # HTTP transport port
  name: "local-context-engine"
  version: "0.1.0"
```

### Transport Modes

| Mode | How it works | Best for |
|---|---|---|
| `stdio` | AI client spawns a new process per session | Simple setups, single-session use |
| `streamable-http` | Persistent HTTP server you start manually | Multi-session use, any MCP client |

**HTTP transport** keeps the embedding model warm in memory across sessions, eliminating
the 5-15s cold-start delay. Start the server with:

```bash
# Windows
.\scripts\start-mcp-server.ps1 /path/to/repo -Port 8765

# Linux / macOS
./scripts/start-mcp-server.sh /path/to/repo 8765
```

Then point your AI client to `http://127.0.0.1:8765/mcp`.

**Per-project port assignment:** When running multiple repos, give each a unique port
via `.context/config.yaml`:

```yaml
# In project-a/.context/config.yaml
mcp_server:
  port: 8765

# In project-b/.context/config.yaml
mcp_server:
  port: 8766
```

---

## Performance

```yaml
performance:
  scan_workers: 4         # Parallel file scanning threads
  parse_workers: 4        # Parallel file parsing threads
  embedding_batch_size: 32
  incremental: true       # false = always full re-index
  hash_algorithm: "md5"   # "md5" | "sha256"
```

---

## Environment Variables

All settings are overridable via environment variables using the `LCE_` prefix and `__` as separator:

```bash
# Use GPU
LCE_EMBEDDING__DEVICE=cuda

# Use a different model
LCE_EMBEDDING__MODEL=nomic-ai/nomic-embed-text-v1.5
LCE_VECTOR_STORE__DIMENSION=768

# Disable PII masking (not recommended)
LCE_SECURITY__ENABLE_PII_MASKING=false

# Custom data directory
LCE_METADATA_STORE__DB_PATH=/data/metadata.db
LCE_VECTOR_STORE__STORAGE_PATH=/data/vectors
```
