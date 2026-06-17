#!/usr/bin/env bash
#
# Start a Local Context Engine MCP server for a repository.
#
# Usage:
#   ./start-mcp-server.sh /path/to/repo [port]
#
# Examples:
#   ./start-mcp-server.sh ~/projects/mis-admin
#   ./start-mcp-server.sh ~/projects/mis-backend 8766

set -euo pipefail

# -- Args ---------------------------------------------------------
REPO_PATH="${1:?Usage: $0 <repo-path> [port]}"
PORT="${2:-8765}"

# -- Resolve repo path --------------------------------------------
REPO_PATH="$(cd "$REPO_PATH" && pwd)"

if [ ! -d "$REPO_PATH/.context" ]; then
    echo "[ERROR] No .context/ folder in $REPO_PATH - run 'context index' first." >&2
    exit 1
fi

# -- Locate context executable ------------------------------------
if command -v context &>/dev/null; then
    CONTEXT_EXE="$(command -v context)"
else
    CONTEXT_EXE="$HOME/.local/bin/context"
fi

if [ ! -x "$CONTEXT_EXE" ]; then
    echo "[ERROR] context executable not found. Install with: pip install local-context-engine" >&2
    exit 1
fi

# -- Check if port is already taken -------------------------------
if command -v ss &>/dev/null; then
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "[OK] Port $PORT is already in use - server may already be running."
        exit 0
    fi
elif command -v lsof &>/dev/null; then
    if lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n &>/dev/null; then
        echo "[OK] Port $PORT is already in use - server may already be running."
        exit 0
    fi
fi

# -- Environment --------------------------------------------------
export PYTHONUTF8=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# -- Launch -------------------------------------------------------
REPO_NAME="$(basename "$REPO_PATH")"

echo ""
echo "  Local Context Engine - MCP Server"
echo "  Repo : $REPO_PATH"
echo "  Port : $PORT"
echo "  URL  : http://127.0.0.1:$PORT/mcp"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

exec "$CONTEXT_EXE" mcp "$REPO_PATH" --transport streamable-http
