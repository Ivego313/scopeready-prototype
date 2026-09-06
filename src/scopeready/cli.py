"""Command line entry point.

Commands are added by the step that makes them work: a command that prints a
plausible answer built from nothing is worse than a missing one, because only
the first kind gets believed.

Status output goes to stderr and data to stdout, so a report can be piped
without ANSI escapes landing in the file.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from scopeready import __description__, __title__, __version__
from scopeready.analyze import (
    AnalysisInputs,
    RunLog,
    RunSettings,
    ThreadedRetriever,
    build_unit,
    determine_profile,
    run_probes,
)
from scopeready.analyze import (
    analyze as run_analysis,
)
from scopeready.applicability import profile_from_flags
from scopeready.chunking import HeuristicBudget, chunk_document, chunk_documents
from scopeready.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONCURRENCY,
    DEFAULT_DB,
    DEFAULT_EMBEDDER,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUT_DIR,
    FIXTURE_CORPUS_DIR,
    TAXONOMY_DIR,
)
from scopeready.corpus import CorpusError, SqliteCorpus, fts5_available
from scopeready.embeddings import build_embedder
from scopeready.ingest import IngestError, read_directory
from scopeready.llm import (
    CachingBackend,
    ModelError,
    ModelRequest,
    OllamaBackend,
    StubPlan,
    build_backend,
)
from scopeready.models import Granularity, ProbeResult, Refutation, SkipReason
from scopeready.prompts import build_prefix, probe_tail
from scopeready.report import prompt_eval_table, render_json, render_markdown
from scopeready.retrieval import HybridRetriever
from scopeready.store import IndexedChunk
from scopeready.taxonomy import (
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
    select_categories,
)
from scopeready.text import build_context_text, normalize_for_index
from scopeready.verification import check_probe

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


Backend = Annotated[str, typer.Option("--backend", help="ollama or stub.")]
Unit = Annotated[str, typer.Option("--unit", help="Document id of the unit root.")]
GranularityOption = Annotated[
    Granularity, typer.Option("--granularity", help="Scope level of the run.")
]


@dataclass(frozen=True, slots=True)
class _Run:
    """Everything a pipeline command needs, opened once."""

    inputs: AnalysisInputs
    taxonomy: Taxonomy
    backend: Any
    retriever: ThreadedRetriever
    settings: RunSettings


def _prepare(
    db: Path,
    unit_id: str,
    granularity: Granularity,
    embedder_key: str,
    backend_kind: str,
    model: str,
    url: str,
    num_ctx: int,
    concurrency: int,
    taxonomy_dir: Path,
    cache_dir: Path | None,
    refresh_cache: bool,
) -> _Run:
    taxonomy = _load(taxonomy_dir)
    embedder = build_embedder(embedder_key)

    store = SqliteCorpus.open(db)
    store.declare_embedding_space(
        model_name=embedder.name,
        dimensions=embedder.dimensions,
        max_tokens=embedder.max_tokens,
    )
    documents = store.documents()
    all_chunks = store.chunks()
    stats = store.stats()
    store.close()

    if not any(document.doc_id == unit_id for document in documents):
        err.print(f"[red]no document {unit_id!r} in {db}[/red]")
        raise typer.Exit(code=2)

    unit = build_unit(documents, unit_id, granularity.value)
    unit_documents = tuple(
        document for document in documents if unit.contains(document.doc_id)
    )
    unit_chunks = tuple(chunk for chunk in all_chunks if unit.contains(chunk.doc_id))

    backend = build_backend(
        backend_kind,
        model=model,
        url=url,
        num_ctx=num_ctx,
        chunks=all_chunks,
        stub_plan=StubPlan.for_fixture(),
    )
    if cache_dir is not None:
        backend = CachingBackend(backend, cache_dir, refresh=refresh_cache)

    # Opened inside the retrieval worker, because a SQLite connection belongs to
    # the thread that made it.
    def open_retriever() -> Any:
        worker_store = SqliteCorpus.open(db)
        return HybridRetriever(worker_store, build_embedder(embedder_key))

    return _Run(
        inputs=AnalysisInputs(
            unit=unit,
            documents=unit_documents,
            unit_chunks=unit_chunks,
            all_chunks=all_chunks,
            stats=stats,
        ),
        taxonomy=taxonomy,
        backend=backend,
        retriever=ThreadedRetriever(open_retriever),
        settings=RunSettings(concurrency=concurrency, embedder_name=embedder.name),
    )


@app.command()
def profile(
    unit_id: Unit = "jira:SCOPE-1",
    granularity: GranularityOption = Granularity.EPIC,
    db: DbPath = DEFAULT_DB,
    embedder_key: EmbedderKey = DEFAULT_EMBEDDER,
    backend_kind: Backend = "stub",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    url: Annotated[str, typer.Option("--ollama-url")] = DEFAULT_OLLAMA_URL,
    num_ctx: Annotated[int, typer.Option("--num-ctx")] = DEFAULT_NUM_CTX,
    taxonomy_dir: TaxonomyDir = TAXONOMY_DIR,
) -> None:
    """Determine the project profile and show which categories it unlocks."""
    run = _prepare(
        db,
        unit_id,
        granularity,
        embedder_key,
        backend_kind,
        model,
        url,
        num_ctx,
        DEFAULT_CONCURRENCY,
        taxonomy_dir,
        None,
        False,
    )
    log = RunLog()
    prefix = build_prefix(run.inputs.unit, run.inputs.documents, run.inputs.unit_chunks)
    try:
        determined = asyncio.run(
            determine_profile(run.backend, prefix, run.taxonomy, log)
        )
    except ModelError as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    finally:
        run.retriever.close()

    table = Table(title=f"Profile of {unit_id}")
    table.add_column("feature")
    table.add_column("value", justify="center")
    table.add_column("rationale")
    for feature in run.taxonomy.profile.features:
        determined_feature = determined.features.get(feature.id)
        value = "—" if determined_feature is None else str(determined_feature.value)
        colour = "green" if determined_feature and determined_feature.value else "dim"
        table.add_row(
            feature.id,
            f"[{colour}]{value}[/{colour}]",
            determined_feature.rationale if determined_feature else "not answered",
        )
    out.print(table)

    selection = select_categories(run.taxonomy, determined, granularity)
    err.print(
        f"{len(selection.applicable)} of {len(run.taxonomy)} categories apply; "
        f"prefix digest {prefix.digest}, about {prefix.approximate_tokens} tokens"
    )
    for warning in log.warnings:
        err.print(f"[yellow]{warning}[/yellow]")


@app.command()
def probe(
    category_id: Annotated[str, typer.Argument(help="Category to ask about.")],
    unit_id: Unit = "jira:SCOPE-1",
    granularity: GranularityOption = Granularity.EPIC,
    db: DbPath = DEFAULT_DB,
    embedder_key: EmbedderKey = DEFAULT_EMBEDDER,
    backend_kind: Backend = "stub",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    url: Annotated[str, typer.Option("--ollama-url")] = DEFAULT_OLLAMA_URL,
    num_ctx: Annotated[int, typer.Option("--num-ctx")] = DEFAULT_NUM_CTX,
    taxonomy_dir: TaxonomyDir = TAXONOMY_DIR,
    show_prompt: Annotated[
        bool, typer.Option("--show-prompt", help="Print the exact request sent.")
    ] = False,
) -> None:
    """Run one category probe, and optionally show the exact bytes sent.

    This is the loop for working on probe wording, which is where most of the
    quality of the whole thing lives.
    """
    run = _prepare(
        db,
        unit_id,
        granularity,
        embedder_key,
        backend_kind,
        model,
        url,
        num_ctx,
        DEFAULT_CONCURRENCY,
        taxonomy_dir,
        None,
        False,
    )
    categories = {item.id: item for item in run.taxonomy.categories}
    category = categories.get(category_id)
    if category is None:
        run.retriever.close()
        err.print(f"[red]no category {category_id!r} in the taxonomy[/red]")
        raise typer.Exit(code=2)

    prefix = build_prefix(run.inputs.unit, run.inputs.documents, run.inputs.unit_chunks)
    log = RunLog()

    async def once() -> None:
        fragments = await run.retriever.search_many(
            category.retrieval_queries, limit=run.settings.probe_fragments
        )
        if show_prompt:
            request = ModelRequest(
                system=prefix.system,
                prefix=prefix.corpus,
                tail=probe_tail(category, fragments, ProbeResult),
                label=f"probe:{category.id}",
            )
            out.print(request.system)
            out.print(request.content)
        answers = await run_probes(
            run.backend,
            prefix,
            [category],
            {category.id: fragments},
            run.settings,
            log,
        )
        answer = answers.get(category.id)
        if answer is None:
            err.print("[red]the probe returned nothing usable[/red]")
            # The reason lives in the log, and swallowing it here would leave a
            # failed probe with no way to find out why.
            for reason in log.warnings:
                err.print(f"[yellow]{reason}[/yellow]")
            raise typer.Exit(code=1)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in run.inputs.all_chunks}
        checked = check_probe(answer, category.id, chunks_by_id, run.inputs.unit)
        err.print(
            f"verdict [bold]{checked.probe.verdict.value}[/bold] "
            f"at confidence {checked.probe.confidence:.2f}"
            + (f", coverage {checked.location.value}" if checked.location else "")
        )
        for item in checked.probe.evidence:
            out.print(f"  [{item.chunk_id}] {item.quote}")
        if checked.probe.missing:
            err.print(f"missing: {checked.probe.missing}")
        err.print(f"reasoning: {checked.probe.reasoning}")
        for warning in (*checked.warnings, *log.warnings):
            err.print(f"[yellow]{warning}[/yellow]")

    try:
        asyncio.run(once())
    finally:
        run.retriever.close()


@app.command()
def analyze(
    unit_id: Unit = "jira:SCOPE-1",
    granularity: GranularityOption = Granularity.EPIC,
    db: DbPath = DEFAULT_DB,
    embedder_key: EmbedderKey = DEFAULT_EMBEDDER,
    backend_kind: Backend = "stub",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    url: Annotated[str, typer.Option("--ollama-url")] = DEFAULT_OLLAMA_URL,
    num_ctx: Annotated[int, typer.Option("--num-ctx")] = DEFAULT_NUM_CTX,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Probes in flight after the first.")
    ] = DEFAULT_CONCURRENCY,
    taxonomy_dir: TaxonomyDir = TAXONOMY_DIR,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    refresh_cache: Annotated[bool, typer.Option("--refresh-cache")] = False,
    out_dir: Annotated[Path, typer.Option("--out")] = DEFAULT_OUT_DIR,
    output_format: Annotated[
        str, typer.Option("--format", help="md, json or both.")
    ] = "md",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Assemble every prompt, call no model."),
    ] = False,
) -> None:
    """Run the full audit and write the report."""
    run = _prepare(
        db,
        unit_id,
        granularity,
        embedder_key,
        backend_kind,
        model,
        url,
        num_ctx,
        concurrency,
        taxonomy_dir,
        None if no_cache else cache_dir,
        refresh_cache,
    )
    try:
        if dry_run:
            _dry_run(run)
            return
        report = asyncio.run(
            run_analysis(
                run.inputs, run.taxonomy, run.backend, run.retriever, run.settings
            )
        )
    except ModelError as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    finally:
        run.retriever.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = unit_id.replace(":", "-")
    written: list[Path] = []
    if output_format in {"md", "both"}:
        path = out_dir / f"{stem}.md"
        path.write_text(render_markdown(report))
        written.append(path)
    if output_format in {"json", "both"}:
        path = out_dir / f"{stem}.json"
        path.write_text(render_json(report))
        written.append(path)

    table = Table(title=f"Gaps in {unit_id}")
    table.add_column("severity", justify="right")
    table.add_column("category")
    table.add_column("verdict")
    table.add_column("missing")
    for finding in report.ranked_gaps():
        candidate = finding.candidate
        table.add_row(
            f"{candidate.severity:.2f}",
            candidate.category_id,
            candidate.probe.verdict.value,
            (candidate.probe.missing or "")[:60],
        )
    out.print(table)

    err.print(
        f"{len(report.gaps)} gaps, {len(report.suppressed)} suppressed by the "
        f"refutation pass, {len(report.covered)} covered, "
        f"{len(report.skipped)} not asked"
    )
    for warning in report.warnings:
        err.print(f"[yellow]{warning}[/yellow]")
    err.print(
        f"{report.usage.calls} calls, {report.usage.total_tokens} tokens, "
        f"{report.wall_clock_seconds:.1f}s wall clock"
    )
    for path in written:
        err.print(f"wrote {path}")


def _dry_run(run: _Run) -> None:
    """Assemble everything and call nothing.

    The fastest way to answer whether a prompt edit blew the context window, and
    the only way to work on prompts without a loaded model.
    """
    prefix = build_prefix(run.inputs.unit, run.inputs.documents, run.inputs.unit_chunks)
    profile_flags = StubPlan.for_fixture().features
    determined = profile_from_flags(dict(profile_flags))
    selection = select_categories(run.taxonomy, determined, run.inputs.unit.granularity)

    async def measure() -> list[tuple[str, int]]:
        sizes: list[tuple[str, int]] = []
        for category in selection.applicable:
            fragments = await run.retriever.search_many(
                category.retrieval_queries, limit=run.settings.probe_fragments
            )
            tail = probe_tail(category, fragments, ProbeResult)
            sizes.append((category.id, len(tail) // 4))
        return sizes

    tails = asyncio.run(measure())
    table = Table(title="Dry run: request sizes in approximate tokens")
    table.add_column("part")
    table.add_column("tokens", justify="right")
    table.add_row("system prompt", str(len(prefix.system) // 4))
    table.add_row("corpus prefix (shared)", str(len(prefix.corpus) // 4))
    for category_id, size in tails:
        table.add_row(f"tail {category_id}", str(size))
    out.print(table)
    total = len(prefix.system) // 4 + len(prefix.corpus) // 4
    err.print(
        f"prefix digest {prefix.digest}; "
        f"largest request about {total + max((size for _, size in tails), default=0)} "
        f"tokens against a window of {DEFAULT_NUM_CTX}"
    )
    err.print(prompt_eval_table([]).splitlines()[0])
