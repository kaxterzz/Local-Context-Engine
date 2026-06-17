"""
Local Context Engine CLI.

Commands:
  context index   <repo_path>   — Incremental index of a repository
  context reindex <repo_path>   — Force full re-index (ignore hashes)
  context search  <query>       — Search the indexed codebase
  context stats                 — Show indexing statistics
  context inspect <file>        — Inspect symbols in a file
  context graph   <symbol>      — Show symbol graph relationships
  context memory                — Browse/manage agent memory
  context mcp                   — Start the MCP server
  context benchmark             — Run performance benchmarks
  context doctor                — Diagnose configuration issues
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 I/O on Windows (equivalent to PYTHONUTF8=1).
# Must run before Rich Console objects are created so they inherit the
# correct encoding.  No-op on platforms that already default to UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from local_context_engine.core.config import load_config
from local_context_engine.core.logging import configure_logging

app = typer.Typer(
    name="context",
    help="Local Context Engine — Privacy-first AI codebase indexer",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)

# ─────────────────────────────────────────────────────────────
# Shared options
# ─────────────────────────────────────────────────────────────

_config_option = typer.Option(
    None, "--config", "-c", help="Path to a YAML configuration override."
)
_verbose_option = typer.Option(
    False, "--verbose", "-v", help="Enable verbose (DEBUG) logging."
)


# ─────────────────────────────────────────────────────────────
# context index
# ─────────────────────────────────────────────────────────────

@app.command("index")
def cmd_index(
    repo_path: Path = typer.Argument(
        Path("."), help="Path to the repository root to index."
    ),
    full: bool = typer.Option(False, "--full", help="Force full re-index (ignore hashes)."),
    config_path: Optional[Path] = _config_option,
    verbose: bool = _verbose_option,
) -> None:
    """
    Index a repository for AI context retrieval.

    On first run, performs a full index. Subsequent runs only re-index
    modified files unless --full is specified.

    [bold green]Examples:[/bold green]
      context index .
      context index /projects/my-laravel-app
      context index . --full
    """
    configure_logging("DEBUG" if verbose else "INFO")

    repo_path = repo_path.resolve()
    if not repo_path.exists():
        err_console.print(f"[red]Repository path not found: {repo_path}[/red]")
        raise typer.Exit(1)

    config = load_config(project_root=repo_path, config_override=config_path)

    console.print(
        Panel(
            f"[bold]Indexing repository[/bold]\n\n"
            f"  Path:  [cyan]{repo_path}[/cyan]\n"
            f"  Model: [cyan]{config.embedding.model}[/cyan]\n"
            f"  Mode:  [cyan]{'full' if full else 'incremental'}[/cyan]",
            title="[bold green]Local Context Engine[/bold green]",
            expand=False,
        )
    )

    from local_context_engine.indexer.pipeline import IndexingPipeline

    pipeline = IndexingPipeline.from_config(config)

    with console.status("[bold green]Indexing…[/bold green]") as status:
        def _progress(p):
            status.update(
                f"[bold green]{p.phase.title()}[/bold green] "
                + (f"[{p.processed}/{p.total}]" if p.total else "")
                + (f" — [dim]{p.current_file}[/dim]" if p.current_file else "")
            )

        stats = asyncio.run(_run_index(pipeline, repo_path, not full, _progress))

    # Summary table
    table = Table(title="Indexing Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Files scanned", str(stats.total_files_scanned))
    table.add_row("Files indexed", str(stats.files_indexed))
    table.add_row("Files skipped", str(stats.files_skipped))
    table.add_row("Files failed", str(stats.files_failed))
    table.add_row("Symbols extracted", str(stats.total_symbols))
    table.add_row("Chunks created", str(stats.total_chunks))
    table.add_row("Vectors stored", str(stats.total_vectors))
    table.add_row("Scan time", f"{stats.scan_time_seconds:.2f}s")
    table.add_row("Embed time", f"{stats.embedding_time_seconds:.2f}s")
    table.add_row("Total time", f"{stats.total_time_seconds:.2f}s")

    console.print(table)

    if stats.errors:
        console.print(f"\n[yellow]{len(stats.errors)} error(s) during indexing.[/yellow]")
        if verbose:
            for err in stats.errors[:10]:
                console.print(f"  [dim]{err}[/dim]")

    if stats.files_failed == 0:
        console.print("\n[bold green]Indexing complete![/bold green]")
    else:
        console.print(f"\n[yellow]Completed with {stats.files_failed} failure(s).[/yellow]")


async def _run_index(pipeline, repo_path, incremental, progress_callback):
    await pipeline.initialize()
    try:
        stats = await pipeline.index_repository(
            repo_root=repo_path,
            incremental=incremental,
            progress_callback=progress_callback,
        )
    finally:
        await pipeline.shutdown()
    return stats


# ─────────────────────────────────────────────────────────────
# context search
# ─────────────────────────────────────────────────────────────

@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Search query (natural language or code)."),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results."),
    language: Optional[str] = typer.Option(None, "--lang", "-l", help="Language filter."),
    repo_path: Path = typer.Option(Path("."), "--repo", help="Repository root."),
    config_path: Optional[Path] = _config_option,
    verbose: bool = _verbose_option,
) -> None:
    """
    Search the indexed codebase.

    [bold green]Examples:[/bold green]
      context search "journal entry approval flow"
      context search "user authentication" --lang php
      context search "useQuery hook" --lang typescript
    """
    configure_logging("DEBUG" if verbose else "WARNING")

    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    results = asyncio.run(_run_search(config, query, limit, language))

    if not results:
        console.print("[yellow]No results found. Make sure the repository is indexed.[/yellow]")
        console.print("[dim]Run: context index <repo_path>[/dim]")
        return

    console.print(f"\n[bold]Search:[/bold] [cyan]{query}[/cyan]")
    console.print(f"[dim]{len(results)} result(s)[/dim]\n")

    for i, result in enumerate(results, 1):
        score_bar = "█" * int(result["score"] * 20)
        console.print(
            f"[bold cyan]{i}.[/bold cyan] "
            f"[green]{result['file_path']}[/green]"
            f"[dim]:{result['line_start']}-{result['line_end']}[/dim]"
        )
        if result.get("symbol_name"):
            console.print(
                f"   [yellow]{result['symbol_type']}[/yellow] "
                f"[bold]{result['symbol_name']}[/bold]"
            )
        console.print(
            f"   Score: [cyan]{result['score']:.3f}[/cyan] "
            f"[dim]{score_bar}[/dim]"
        )
        if result.get("snippet"):
            snippet_lines = result["snippet"].strip().split("\n")[:5]
            for line in snippet_lines:
                console.print(f"   [dim]{line}[/dim]")
        console.print()


async def _run_search(config, query, limit, language):
    from local_context_engine.core.types import Language, SearchQuery
    from local_context_engine.indexer.embedder.factory import EmbedderFactory
    from local_context_engine.metadata_store.database import Database
    from local_context_engine.retrieval.bm25_retriever import BM25Retriever
    from local_context_engine.retrieval.hybrid_retriever import HybridRetriever
    from local_context_engine.security.pii_masker import PIIMasker
    from local_context_engine.security.redactor import ContentRedactor
    from local_context_engine.symbol_graph.graph import SymbolGraph
    from local_context_engine.vector_store.factory import VectorStoreFactory

    db = Database.from_config(config.metadata_store)
    await db.init()
    vs = VectorStoreFactory.create(config.vector_store)
    vs.load()
    embedder = EmbedderFactory.create(config.embedding)
    bm25 = BM25Retriever()
    masker = PIIMasker.from_config(config.pii_masking)
    redactor = ContentRedactor.from_masker(masker)
    graph = SymbolGraph()

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vs,
        bm25=bm25,
        database=db,
        config=config.retrieval,
        redactor=redactor,
        symbol_graph=graph,
    )
    await retriever.initialize()

    lang_filter = None
    if language:
        try:
            lang_filter = Language(language.lower())
        except ValueError:
            pass

    sq = SearchQuery(text=query, limit=limit, language_filter=lang_filter)
    results = await retriever.search(sq)
    await db.close()

    return [
        {
            "file_path": r.file_path,
            "symbol_name": r.symbol_name,
            "symbol_type": r.symbol_type,
            "line_start": r.line_start,
            "line_end": r.line_end,
            "score": r.score,
            "snippet": r.snippet,
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────
# context stats
# ─────────────────────────────────────────────────────────────

@app.command("stats")
def cmd_stats(
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """Show repository indexing statistics."""
    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    async def _get_stats():
        from local_context_engine.metadata_store.database import Database
        from local_context_engine.metadata_store.repositories import (
            ChunkRepository, FileRepository, SymbolRepository,
        )
        from local_context_engine.vector_store.factory import VectorStoreFactory

        db = Database.from_config(config.metadata_store)
        await db.init()
        vs = VectorStoreFactory.create(config.vector_store)
        vs.load()

        async with db.session() as session:
            file_repo = FileRepository(session)
            sym_repo = SymbolRepository(session)
            chunk_repo = ChunkRepository(session)

            total_files = await file_repo.count()
            lang_counts = await file_repo.language_counts()
            total_symbols = await sym_repo.count()
            total_chunks = await chunk_repo.count()

        await db.close()
        return {
            "total_files": total_files,
            "languages": lang_counts,
            "total_symbols": total_symbols,
            "total_chunks": total_chunks,
            "total_vectors": vs.total_vectors,
        }

    stats = asyncio.run(_get_stats())

    table = Table(title="Repository Statistics", header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total files", str(stats["total_files"]))
    table.add_row("Total symbols", str(stats["total_symbols"]))
    table.add_row("Total chunks", str(stats["total_chunks"]))
    table.add_row("Total vectors", str(stats["total_vectors"]))

    console.print(table)

    if stats["languages"]:
        lang_table = Table(title="Languages", header_style="bold")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Files", style="green", justify="right")
        for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
            lang_table.add_row(lang, str(count))
        console.print(lang_table)


# ─────────────────────────────────────────────────────────────
# context reindex
# ─────────────────────────────────────────────────────────────

@app.command("reindex")
def cmd_reindex(
    repo_path: Path = typer.Argument(
        Path("."), help="Path to the repository root."
    ),
    config_path: Optional[Path] = _config_option,
    verbose: bool = _verbose_option,
) -> None:
    """
    Force a full re-index of the repository, ignoring all cached hashes.

    Equivalent to: context index <path> --full

    [bold green]Examples:[/bold green]
      context reindex .
      context reindex /projects/my-laravel-app
    """
    cmd_index(repo_path=repo_path, full=True, config_path=config_path, verbose=verbose)


# ─────────────────────────────────────────────────────────────
# context inspect
# ─────────────────────────────────────────────────────────────

@app.command("inspect")
def cmd_inspect(
    file_path: str = typer.Argument(..., help="File path to inspect (relative to repo root)."),
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
    verbose: bool = _verbose_option,
) -> None:
    """
    Inspect all indexed symbols in a file.

    [bold green]Examples:[/bold green]
      context inspect app/Http/Controllers/UserController.php
      context inspect src/components/UserProfile.tsx
      context inspect app/Services/PaymentService.php
    """
    configure_logging("DEBUG" if verbose else "WARNING")

    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    async def _get_file_symbols():
        from local_context_engine.metadata_store.database import Database
        from local_context_engine.metadata_store.repositories import (
            FileRepository,
            SymbolRepository,
        )

        db = Database.from_config(config.metadata_store)
        await db.init()

        file_record = None
        symbols = []

        async with db.session() as session:
            file_repo = FileRepository(session)
            sym_repo = SymbolRepository(session)

            # Try the path as provided first (relative), then normalised
            file_record = await file_repo.get_by_path(file_path)
            if file_record is None:
                # Try computing relative path from repo root
                try:
                    rel = str(Path(file_path).resolve().relative_to(repo_path))
                    file_record = await file_repo.get_by_path(rel)
                except ValueError:
                    pass

            if file_record is not None:
                symbols = await sym_repo.get_by_file_id(file_record.id)

        await db.close()
        return file_record, symbols

    file_record, symbols = asyncio.run(_get_file_symbols())

    if file_record is None:
        console.print(f"[red]File not found in index: {file_path}[/red]")
        console.print("[dim]Run 'context index <repo>' first, then try again.[/dim]")
        raise typer.Exit(1)

    console.print(f"\n[bold]File:[/bold] [cyan]{file_record.path}[/cyan]")
    console.print(
        f"[dim]{file_record.language.value}"
        + (f" · {file_record.framework}" if file_record.framework else "")
        + f" · {file_record.line_count:,} lines"
        + f" · {file_record.size_bytes:,} bytes[/dim]\n"
    )

    if not symbols:
        console.print("[yellow]No symbols found in this file.[/yellow]")
        return

    table = Table(title=f"Symbols — {file_record.path}", header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Lines", justify="right")
    table.add_column("Visibility", style="dim")
    table.add_column("Parent", style="dim")

    for sym in symbols:
        table.add_row(
            sym.name,
            sym.symbol_type.value,
            f"{sym.line_start}–{sym.line_end}",
            sym.visibility or "",
            sym.parent_name or "",
        )

    console.print(table)
    console.print(f"\n[dim]{len(symbols)} symbol(s)[/dim]")


# ─────────────────────────────────────────────────────────────
# context graph
# ─────────────────────────────────────────────────────────────

@app.command("graph")
def cmd_graph(
    symbol: str = typer.Argument(..., help="Symbol name to inspect."),
    direction: str = typer.Option("both", help="'deps', 'refs', or 'both'."),
    depth: int = typer.Option(2, "--depth", "-d"),
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """Show symbol graph relationships."""
    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    graph = __import__(
        "local_context_engine.symbol_graph.graph", fromlist=["SymbolGraph"]
    ).SymbolGraph()
    loaded = graph.load(config.symbol_graph.storage_path)

    if not loaded:
        console.print("[yellow]Symbol graph not found. Run 'context index' first.[/yellow]")
        raise typer.Exit(1)

    symbols = graph.lookup_by_name(symbol)
    if not symbols:
        console.print(f"[red]Symbol '{symbol}' not found.[/red]")
        raise typer.Exit(1)

    target = symbols[0]
    console.print(f"\n[bold]Symbol:[/bold] [green]{target.qualified_name}[/green]")
    console.print(f"[dim]{target.symbol_type.value} in {target.file_path}:{target.line_start}[/dim]\n")

    if direction in ("deps", "both"):
        deps = graph.find_dependencies(target.id, depth=depth)
        if deps:
            table = Table(title="Dependencies (what this depends on)", header_style="bold")
            table.add_column("Symbol", style="cyan")
            table.add_column("Type")
            table.add_column("Relationship", style="yellow")
            table.add_column("File", style="dim")
            for dep, rel_type, confidence in deps:
                table.add_row(dep.name, dep.symbol_type.value, rel_type, dep.file_path)
            console.print(table)

    if direction in ("refs", "both"):
        refs = graph.find_references(target.id, depth=depth)
        if refs:
            table = Table(title="References (what uses this)", header_style="bold")
            table.add_column("Symbol", style="cyan")
            table.add_column("Type")
            table.add_column("Relationship", style="yellow")
            table.add_column("File", style="dim")
            for ref, rel_type, confidence in refs:
                table.add_row(ref.name, ref.symbol_type.value, rel_type, ref.file_path)
            console.print(table)


# ─────────────────────────────────────────────────────────────
# context memory
# ─────────────────────────────────────────────────────────────

memory_app = typer.Typer(name="memory", help="Manage agent memory.")
app.add_typer(memory_app)


@memory_app.command("list")
def cmd_memory_list(
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """List all stored memories."""
    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    from local_context_engine.memory.agent_memory import AgentMemory, MemoryCategory

    memory = AgentMemory.from_config(config.memory)
    memory.init()

    cat = None
    if category:
        try:
            cat = MemoryCategory(category)
        except ValueError:
            pass

    entries = memory.list_all(category=cat, limit=50)
    memory.close()

    if not entries:
        console.print("[yellow]No memories stored yet.[/yellow]")
        return

    table = Table(title="Agent Memory", header_style="bold")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Category", style="yellow")
    table.add_column("Content", style="cyan", max_width=80)
    table.add_column("Tags", style="dim")
    table.add_column("Updated", style="dim")

    for entry in entries:
        table.add_row(
            entry.id[:8],
            entry.category.value,
            entry.content[:100] + ("…" if len(entry.content) > 100 else ""),
            ", ".join(entry.tags[:3]),
            entry.updated_at.strftime("%Y-%m-%d"),
        )

    console.print(table)


@memory_app.command("save")
def cmd_memory_save(
    content: str = typer.Argument(..., help="Knowledge to remember."),
    category: str = typer.Option("general", "--category", "-c"),
    tags: str = typer.Option("", "--tags", "-t", help="Comma-separated tags."),
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """Save a piece of knowledge to agent memory."""
    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    from local_context_engine.memory.agent_memory import AgentMemory, MemoryCategory

    memory = AgentMemory.from_config(config.memory)
    memory.init()

    try:
        cat = MemoryCategory(category.lower())
    except ValueError:
        cat = MemoryCategory.GENERAL

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    memory_id = memory.save_memory(category=cat, content=content, tags=tag_list)
    memory.close()

    console.print(f"[green]Memory saved:[/green] [dim]{memory_id}[/dim]")


# ─────────────────────────────────────────────────────────────
# context mcp
# ─────────────────────────────────────────────────────────────

@app.command("mcp")
def cmd_mcp(
    repo_path: Path = typer.Argument(Path("."), help="Repository root."),
    transport: str = typer.Option("stdio", "--transport", "-t"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """
    Start the MCP server for AI assistant integration.

    In stdio mode (default), reads JSON-RPC from stdin and writes to stdout.
    Compatible with Claude Desktop, Cursor, Continue.dev, and any MCP client.

    [bold green]Claude Desktop config:[/bold green]
      Add to mcpServers in claude_desktop_config.json:
      {
        "local-context": {
          "command": "context",
          "args": ["mcp", "/path/to/your/repo"]
        }
      }
    """
    repo_path = repo_path.resolve()

    # Use plain (no-Rich) logging in MCP mode so nothing Unicode-heavy is
    # written to stderr while the stdio JSON-RPC channel is active.
    configure_logging("WARNING", format="plain")

    from local_context_engine.mcp_server.server import create_mcp_server

    mcp = create_mcp_server(repo_root=repo_path, config_path=config_path)
    # In stdio mode stdout is the JSON-RPC channel — suppress the FastMCP
    # startup banner so no non-JSON bytes are written before the first
    # JSON-RPC response.
    show_banner = transport != "stdio"

    transport_kwargs: dict = {}
    if transport != "stdio":
        config = load_config(project_root=repo_path, config_override=config_path)
        transport_kwargs["host"] = config.mcp_server.host
        transport_kwargs["port"] = config.mcp_server.port

    mcp.run(transport=transport, show_banner=show_banner, **transport_kwargs)


# ─────────────────────────────────────────────────────────────
# context doctor
# ─────────────────────────────────────────────────────────────

@app.command("doctor")
def cmd_doctor(
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """
    Diagnose configuration and environment issues.

    Checks:
      - Python version
      - Required dependencies
      - Tree-sitter grammar availability
      - GPU/CUDA availability
      - Index health
    """
    from rich.markdown import Markdown

    console.print(Panel("[bold]Local Context Engine — Doctor[/bold]", expand=False))
    all_ok = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {icon}  {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))
        if not ok:
            all_ok = False

    # Python version
    check(
        "Python version",
        sys.version_info >= (3, 12),
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    # Core dependencies
    deps = [
        ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("tree_sitter", "tree-sitter"),
        ("sqlalchemy", "sqlalchemy"),
        ("fastmcp", "fastmcp"),
        ("networkx", "networkx"),
        ("rank_bm25", "rank-bm25"),
        ("pathspec", "pathspec"),
        ("typer", "typer"),
        ("rich", "rich"),
    ]
    for module, package in deps:
        try:
            __import__(module)
            check(f"Package: {package}", True)
        except ImportError:
            check(f"Package: {package}", False, f"pip install {package}")

    # Tree-sitter grammars
    ts_langs = [
        ("tree_sitter_php", "tree-sitter-php"),
        ("tree_sitter_typescript", "tree-sitter-typescript"),
        ("tree_sitter_javascript", "tree-sitter-javascript"),
        ("tree_sitter_python", "tree-sitter-python"),
    ]
    for module, package in ts_langs:
        try:
            __import__(module)
            check(f"Grammar: {package}", True)
        except ImportError:
            check(f"Grammar: {package}", False, f"pip install {package}")

    # GPU
    try:
        import torch

        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        check("CUDA GPU", cuda, "torch.cuda.is_available()")
        check("Apple Silicon (MPS)", mps, "torch.backends.mps.is_available()")
    except ImportError:
        check("PyTorch", False, "pip install torch")

    # Index health
    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)
    db_path = config.metadata_store.db_path
    vector_path = config.vector_store.storage_path / "faiss.index"

    check("Metadata database", db_path.exists(), str(db_path))
    check("Vector index", vector_path.exists(), str(vector_path))

    console.print()
    if all_ok:
        console.print("[bold green]All checks passed![/bold green]")
    else:
        console.print("[bold yellow]Some issues found. Fix the items marked ✗ above.[/bold yellow]")


# ─────────────────────────────────────────────────────────────
# context benchmark
# ─────────────────────────────────────────────────────────────

@app.command("benchmark")
def cmd_benchmark(
    repo_path: Path = typer.Option(Path("."), "--repo"),
    config_path: Optional[Path] = _config_option,
) -> None:
    """Run performance benchmarks on the indexed codebase."""
    import time

    repo_path = repo_path.resolve()
    config = load_config(project_root=repo_path, config_override=config_path)

    console.print("[bold]Running benchmarks…[/bold]\n")

    queries = [
        "user authentication",
        "database migration",
        "API endpoint",
        "React component",
        "error handling",
    ]

    async def _run_benchmarks():
        from local_context_engine.core.types import SearchQuery
        from local_context_engine.indexer.embedder.factory import EmbedderFactory
        from local_context_engine.metadata_store.database import Database
        from local_context_engine.retrieval.bm25_retriever import BM25Retriever
        from local_context_engine.retrieval.hybrid_retriever import HybridRetriever
        from local_context_engine.vector_store.factory import VectorStoreFactory

        db = Database.from_config(config.metadata_store)
        await db.init()
        vs = VectorStoreFactory.create(config.vector_store)
        vs.load()
        embedder = EmbedderFactory.create(config.embedding)
        bm25 = BM25Retriever()

        retriever = HybridRetriever(
            embedder=embedder, vector_store=vs, bm25=bm25,
            database=db, config=config.retrieval,
        )
        await retriever.initialize()

        timings = []
        for q in queries:
            t0 = time.monotonic()
            results = await retriever.search(SearchQuery(text=q, limit=10))
            elapsed = (time.monotonic() - t0) * 1000  # ms
            timings.append((q, elapsed, len(results)))

        await db.close()
        return timings

    timings = asyncio.run(_run_benchmarks())

    table = Table(title="Search Benchmark", header_style="bold")
    table.add_column("Query", style="cyan")
    table.add_column("Time (ms)", style="yellow", justify="right")
    table.add_column("Results", justify="right")

    for q, elapsed, count in timings:
        table.add_row(q, f"{elapsed:.1f}", str(count))

    avg_ms = sum(t for _, t, _ in timings) / len(timings)
    table.add_row("[bold]Average[/bold]", f"[bold]{avg_ms:.1f}[/bold]", "")

    console.print(table)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
