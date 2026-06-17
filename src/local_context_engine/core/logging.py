"""
Structured logging configuration for Local Context Engine.

Supports Rich console output (default), plain text, and JSON.
Uses standard Python ``logging`` module so all third-party libraries
participate in the same log hierarchy.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def configure_logging(
    level: str = "INFO",
    format: Literal["rich", "plain", "json"] = "rich",
    log_file: Path | None = None,
) -> None:
    """
    Configure the root logger for the engine.

    Args:
        level:    Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format:   Output format. ``rich`` renders coloured output in terminals;
                  ``json`` emits one JSON object per line for log aggregators.
        log_file: Optional file path to write logs alongside console output.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    if format == "rich":
        rich_handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(numeric_level)
        handlers.append(rich_handler)
    elif format == "json":
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore[import]

            json_handler = logging.StreamHandler(sys.stderr)
            json_formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
            json_handler.setFormatter(json_formatter)
            json_handler.setLevel(numeric_level)
            handlers.append(json_handler)
        except ImportError:
            # Fallback to plain if python-json-logger not installed
            plain_handler = logging.StreamHandler(sys.stderr)
            plain_handler.setLevel(numeric_level)
            handlers.append(plain_handler)
    else:
        plain_handler = logging.StreamHandler(sys.stderr)
        plain_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        plain_handler.setFormatter(plain_formatter)
        plain_handler.setLevel(numeric_level)
        handlers.append(plain_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger namespaced under ``local_context_engine``."""
    return logging.getLogger(f"local_context_engine.{name}")
