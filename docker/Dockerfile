# ============================================================
# Local Context Engine — Docker Image
# ============================================================
# Multi-stage build to keep the final image lean.
#
# Build:
#   docker build -t local-context-engine -f docker/Dockerfile .
#
# Run (MCP server via stdio):
#   docker run -i --rm \
#     -v /path/to/your/repo:/repo:ro \
#     -v lce_data:/data \
#     local-context-engine mcp /repo
#
# Run (index first):
#   docker run --rm \
#     -v /path/to/your/repo:/repo:ro \
#     -v lce_data:/data \
#     local-context-engine index /repo
# ============================================================

# ── Stage 1: Build dependencies ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy only dependency files first (layer caching)
COPY pyproject.toml ./
COPY src/ ./src/

# Install to a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir -e ".[dev]"

# ── Stage 2: Runtime image ────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Local Context Engine"
LABEL org.opencontainers.image.description="Privacy-first AI codebase context engine"
LABEL org.opencontainers.image.source="https://github.com/your-org/local-context-engine"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime OS dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --from=builder /app/src /app/src
COPY config/ /app/config/

WORKDIR /app

# Create data directory for index persistence
RUN mkdir -p /data

# ── Security: non-root user ───────────────────────────────────
RUN groupadd -r lce && useradd -r -g lce -s /bin/false lce
RUN chown -R lce:lce /data /app
USER lce

# ── Default environment ───────────────────────────────────────
ENV LCE_METADATA_STORE__DB_PATH=/data/metadata.db
ENV LCE_VECTOR_STORE__STORAGE_PATH=/data/vectors
ENV LCE_SYMBOL_GRAPH__STORAGE_PATH=/data/graph.pkl
ENV LCE_MEMORY__STORAGE_PATH=/data/memory.db

# Source repo is mounted read-only at /repo
VOLUME ["/repo", "/data"]

ENTRYPOINT ["context"]
CMD ["--help"]
