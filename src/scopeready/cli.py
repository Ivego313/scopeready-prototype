"""Command line entry point.

Commands are added by the step that makes them work: a command that prints a
plausible answer built from nothing is worse than a missing one, because only
the first kind gets believed.

Status output goes to stderr and data to stdout, so a report can be piped
without ANSI escapes landing in the file.
"""

import asyncio
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from scopeready import __description__, __title__, __version__
from scopeready.applicability import profile_from_flags
from scopeready.chunking import HeuristicBudget, chunk_document, chunk_documents
from scopeready.config import (
    DEFAULT_DB,
    DEFAULT_EMBEDDER,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    FIXTURE_CORPUS_DIR,
    TAXONOMY_DIR,
)
from scopeready.corpus import CorpusError, SqliteCorpus, fts5_available
from scopeready.embeddings import build_embedder
from scopeready.ingest import IngestError, read_directory
from scopeready.llm import ModelError, OllamaBackend
from scopeready.models import Granularity, ProbeResult, Refutation, SkipReason
from scopeready.retrieval import HybridRetriever
from scopeready.store import IndexedChunk
from scopeready.taxonomy import (
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
    select_categories,
)
from scopeready.text import build_context_text, normalize_for_index

app = typer.Typer(name=__title__, help=__description__, no_args_is_help=True)
taxonomy_app = typer.Typer(help="Inspect and check the rubric.", no_args_is_help=True)
app.add_typer(taxonomy_app, name="taxonomy")

err = Console(stderr=True)
out = Console()

TaxonomyDir = Annotated[
    Path, typer.Option("--taxonomy-dir", help="Directory holding the rubric files.")
]


@app.callback()
def main() -> None:
    """Group the commands under one help screen."""


@app.command()
def version() -> None:
    """Print the installed ScopeReady version."""
    typer.echo(__version__)


def _load(directory: Path) -> Taxonomy:
    try:
        return load_taxonomy(directory)
    except TaxonomyError as error:
        err.print(f"[red]taxonomy is not loadable[/red]\n{error}")
        raise typer.Exit(code=1) from error


@taxonomy_app.command("check")
def taxonomy_check(taxonomy_dir: TaxonomyDir = TAXONOMY_DIR) -> None:
    """Validate the rubric files and print their fingerprint."""
    taxonomy = _load(taxonomy_dir)
    err.print(
        f"[green]ok[/green] {len(taxonomy)} categories, "
        f"{len(taxonomy.feature_ids)} profile features, "
        f"version {taxonomy.version}, digest {taxonomy.digest}"
    )


@taxonomy_app.command("list")
def taxonomy_list(taxonomy_dir: TaxonomyDir = TAXONOMY_DIR) -> None:
    """List the rubric: weight, level and applicability of every category."""
    taxonomy = _load(taxonomy_dir)
    table = Table(title=f"Taxonomy {taxonomy.version} ({taxonomy.digest})")
    table.add_column("category")
    table.add_column("weight", justify="right")
    table.add_column("source")
    table.add_column("granularity")
    table.add_column("applies when")
    for category in taxonomy.categories:
        table.add_row(
            category.id,
            f"{category.weight:.2f}",
            category.weight_source.value,
            ", ".join(level.value for level in category.granularity),
            category.applies_when,
        )
    out.print(table)


@taxonomy_app.command("select")
def taxonomy_select(
    granularity: Annotated[
        Granularity, typer.Option("--granularity", help="Scope level of the run.")
    ] = Granularity.EPIC,
    feature: Annotated[
        list[str] | None,
        typer.Option("--feature", help="Profile feature as name=true|false."),
    ] = None,
    taxonomy_dir: TaxonomyDir = TAXONOMY_DIR,
) -> None:
    """Show the applicability gate as a table, with no corpus and no model."""
    taxonomy = _load(taxonomy_dir)
    flags: dict[str, bool] = {}
    for item in feature or []:
        name, _, raw = item.partition("=")
        if raw.strip().lower() not in {"true", "false"}:
            err.print(f"[red]expected name=true or name=false, got {item!r}[/red]")
            raise typer.Exit(code=2)
        flags[name.strip()] = raw.strip().lower() == "true"

    unknown = sorted(set(flags) - set(taxonomy.feature_ids))
    if unknown:
        err.print(f"[red]unknown profile features: {', '.join(unknown)}[/red]")
        raise typer.Exit(code=2)

    selection = select_categories(taxonomy, profile_from_flags(flags), granularity)
    table = Table(title=f"Applicability gate at {granularity.value} level")
    table.add_column("category")
    table.add_column("asked", justify="center")
    table.add_column("why")
    for category in selection.applicable:
        table.add_row(category.id, "[green]yes[/green]", "")
    for skipped in selection.skipped:
        colour = "yellow" if skipped.reason is SkipReason.NOT_APPLICABLE else "dim"
        table.add_row(
            skipped.category_id, f"[{colour}]no[/{colour}]", skipped.explanation
        )
    out.print(table)
    err.print(
        f"{len(selection.applicable)} of {len(taxonomy)} categories would be asked"
    )


@app.command()
def ingest(
    corpus_dir: Annotated[
        Path, typer.Argument(help="Directory of Markdown files with front matter.")
    ] = FIXTURE_CORPUS_DIR,
    show_chunks: Annotated[
        bool, typer.Option("--show-chunks", help="Print every chunk with its path.")
    ] = False,
) -> None:
    """Read a corpus directory and report what it contains.

    Nothing is stored yet: this is the command that answers whether the files
    parse and whether the corpus is shaped the way the run assumes.
    """
    try:
        result = read_directory(corpus_dir)
    except IngestError as error:
        err.print(f"[red]cannot read the corpus[/red]\n{error}")
        raise typer.Exit(code=1) from error

    budget = HeuristicBudget()
    chunks, report = chunk_documents(result.documents, budget)

    table = Table(title=f"Corpus at {corpus_dir}")
    table.add_column("document")
    table.add_column("kind")
    table.add_column("role")
    table.add_column("parent")
    table.add_column("chunks", justify="right")
    for document in result.documents:
        count = sum(1 for chunk in chunks if chunk.doc_id == document.doc_id)
        table.add_row(
            document.doc_id,
            document.provenance.source_kind.value,
            document.corpus_role.value,
            document.parent_doc_id or "",
            str(count),
        )
    out.print(table)

    if show_chunks:
        for chunk in chunks:
            path = " / ".join(chunk.heading_path) or "—"
            out.print(f"[bold]{chunk.chunk_id}[/bold]  [dim]{path}[/dim]")
            out.print(chunk.text)
            out.print()

    err.print(
        f"{result.requirements} requirement and {result.contexts} context documents, "
        f"{report.chunks} chunks "
        f"({report.oversplit_chunks} split below block level, "
        f"{report.title_only_documents} title-only)"
    )


DbPath = Annotated[Path, typer.Option("--db", help="SQLite corpus file.")]
EmbedderKey = Annotated[
    str, typer.Option("--embedder", help="stub, bge-small or gte-modernbert.")
]


@app.command()
def index(
    corpus_dir: Annotated[
        Path, typer.Argument(help="Directory of Markdown files with front matter.")
    ] = FIXTURE_CORPUS_DIR,
    db: DbPath = DEFAULT_DB,
    embedder_key: EmbedderKey = DEFAULT_EMBEDDER,
    reset: Annotated[
        bool, typer.Option("--reset", help="Rebuild the file from scratch.")
    ] = True,
    lexical_only: Annotated[
        bool, typer.Option("--lexical-only", help="Skip embedding the chunks.")
    ] = False,
) -> None:
    """Chunk a corpus directory, write it to SQLite and embed the chunks."""
    try:
        result = read_directory(corpus_dir)
    except IngestError as error:
        err.print(f"[red]cannot read the corpus[/red]\n{error}")
        raise typer.Exit(code=1) from error

    try:
        embedder = build_embedder(embedder_key)
    except (ValueError, ImportError) as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    # The chunker measures the window of the model that will actually encode the
    # text. A heuristic here would let a chunk arrive over the window and be
    # truncated silently, which shows up only as retrieval getting worse.
    budget = embedder
    try:
        with SqliteCorpus.open(db, reset=reset) as store:
            store.declare_embedding_space(
                model_name=embedder.name,
                dimensions=embedder.dimensions,
                max_tokens=embedder.max_tokens,
            )
            store.add_documents(result.documents)
            oversplit = 0
            for document in result.documents:
                chunks, report = chunk_document(document, budget)
                oversplit += report.oversplit_chunks
                store.add_chunks(
                    [
                        IndexedChunk(
                            chunk=chunk,
                            context_text=build_context_text(
                                document.provenance.title, chunk.heading_path
                            ),
                            index_text=normalize_for_index(chunk.text),
                            token_count=budget.count_tokens(chunk.text),
                            oversplit=report.oversplit_chunks > 0,
                        )
                        for chunk in chunks
                    ]
                )
            store.integrity_check()

            embedded = 0
            if not lexical_only:
                pending = list(store.iter_unembedded())
                texts = [text for _, text in pending]
                vectors = embedder.encode_documents(texts)
                store.add_vectors(
                    [(chunk_pk, vectors[i]) for i, (chunk_pk, _) in enumerate(pending)]
                )
                embedded = len(pending)
            stats = store.stats()
    except CorpusError as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    err.print(
        f"indexed {stats.documents} documents "
        f"({stats.requirement_documents} requirement, "
        f"{stats.context_documents} context), {stats.chunks} chunks, "
        f"{embedded} embedded with {embedder.name} "
        f"({oversplit} split below block level) into {db}"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    db: DbPath = DEFAULT_DB,
    embedder_key: EmbedderKey = DEFAULT_EMBEDDER,
    limit: Annotated[int, typer.Option("--limit", help="How many results.")] = 5,
    channels: Annotated[
        bool,
        typer.Option("--channels", help="Show lexical, vector and fused side by side."),
    ] = False,
) -> None:
    """Search the indexed corpus and show what each channel returns."""
    try:
        embedder = build_embedder(embedder_key)
        with SqliteCorpus.open(db) as store:
            store.declare_embedding_space(
                model_name=embedder.name,
                dimensions=embedder.dimensions,
                max_tokens=embedder.max_tokens,
            )
            results = HybridRetriever(store, embedder).search_channels(
                query, limit=limit
            )
    except (CorpusError, ValueError, ImportError) as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if channels:
        table = Table(title=f"Channels for {query!r}")
        table.add_column("rank", justify="right")
        table.add_column("lexical")
        table.add_column("vector")
        table.add_column("fused")
        columns = (results.lexical, results.vector, results.fused)
        for rank in range(limit):
            table.add_row(
                str(rank + 1),
                *(
                    column[rank].chunk.chunk_id if rank < len(column) else "—"
                    for column in columns
                ),
            )
        out.print(table)
        return

    if not results.fused:
        err.print("[yellow]nothing matched[/yellow]")
        return
    table = Table(title=f"Search: {query!r}")
    table.add_column("chunk")
    table.add_column("score", justify="right")
    table.add_column("section")
    table.add_column("text")
    for hit in results.fused:
        table.add_row(
            hit.chunk.chunk_id,
            f"{hit.score:.4f}",
            " / ".join(hit.chunk.heading_path) or "—",
            hit.chunk.text[:90].replace("\n", " "),
        )
    out.print(table)


@app.command()
def doctor(
    model: Annotated[str, typer.Option("--model", help="Ollama model tag.")] = (
        DEFAULT_MODEL
    ),
    url: Annotated[str, typer.Option("--ollama-url")] = DEFAULT_OLLAMA_URL,
    num_ctx: Annotated[int, typer.Option("--num-ctx")] = DEFAULT_NUM_CTX,
    embedder_key: EmbedderKey = DEFAULT_EMBEDDER,
) -> None:
    """Check that this machine can actually run an analysis.

    Everything checked here fails late and confusingly otherwise: a SQLite built
    without FTS5 loses half of retrieval at the first query, a missing model tag
    fails after the corpus is indexed, and an answer schema the server cannot
    compile into a grammar fails on the first probe of a long run.
    """
    table = Table(title="Environment")
    table.add_column("check")
    table.add_column("result")
    problems = 0

    def record(name: str, ok: bool, detail: str) -> None:
        nonlocal problems
        problems += 0 if ok else 1
        mark = "[green]ok[/green]" if ok else "[red]no[/red]"
        table.add_row(name, f"{mark}  {detail}")

    record("sqlite FTS5", fts5_available(), "lexical retrieval needs it")

    try:
        embedder = build_embedder(embedder_key)
        record(
            f"embedder {embedder_key}",
            True,
            f"{embedder.name}, {embedder.dimensions} dims, "
            f"{embedder.max_tokens} token window",
        )
    except (ValueError, ImportError, OSError) as error:
        record(f"embedder {embedder_key}", False, str(error)[:120])

    try:
        taxonomy = load_taxonomy(TAXONOMY_DIR)
        record(
            "taxonomy",
            True,
            f"{len(taxonomy)} categories, digest {taxonomy.digest}",
        )
    except TaxonomyError as error:
        record("taxonomy", False, str(error)[:120])

    backend = OllamaBackend(model=model, url=url, num_ctx=num_ctx)
    try:
        tags = asyncio.run(backend.tags())
    except ModelError as error:
        record("ollama", False, str(error)[:120])
        tags = []
    else:
        record("ollama", True, f"{len(tags)} models available")
        present = any(tag == model or tag.startswith(f"{model}:") for tag in tags)
        record(
            f"model {model}",
            present,
            "installed" if present else f"not pulled; available: {', '.join(tags)}",
        )
        if present:
            # A schema the server cannot turn into a grammar returns a 400. That
            # is worth discovering now rather than on the first probe of a run.
            for schema in (ProbeResult, Refutation):
                try:
                    verdict = asyncio.run(backend.check(schema))
                except (ModelError, httpx.HTTPError) as error:
                    verdict = f"rejected: {error}"[:120]
                record(f"schema {schema.__name__}", verdict == "ok", verdict)

    out.print(table)
    if problems:
        err.print(f"[yellow]{problems} check(s) failed[/yellow]")
        raise typer.Exit(code=1)
    err.print("[green]ready[/green]")
