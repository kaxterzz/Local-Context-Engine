"""Launch Local Context Engine MCP from the workspace source tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("LCE_DISABLE_SEMANTIC_SEARCH", "1")

from local_context_engine.cli.main import main  # noqa: E402


if __name__ == "__main__":
    main()
