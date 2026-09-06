"""Command line entry point.

Commands are added by the step that makes them work: a command that prints a
plausible answer built from nothing is worse than a missing one, because only
the first kind gets believed.

Status output goes to stderr and data to stdout, so a report can be piped
without ANSI escapes landing in the file.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from scopeready import __description__, __title__, __version__
from scopeready.applicability import profile_from_flags
from scopeready.chunking import HeuristicBudget, chunk_documents
from scopeready.config import FIXTURE_CORPUS_DIR, TAXONOMY_DIR
from scopeready.ingest import IngestError, read_directory
from scopeready.models import Granularity, SkipReason
from scopeready.taxonomy import (
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
    select_categories,
)

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
