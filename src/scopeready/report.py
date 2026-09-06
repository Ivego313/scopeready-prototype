"""Rendering a finished analysis, in Markdown and in JSON.

Pure functions over a finished report. Two sections here are not decoration.

The category selection table exists because the applicability gate works by
removing categories, and a removal nobody can see is indistinguishable from a
category missing from the rubric. The suppressed section exists because a gap
that was raised and then closed by evidence found elsewhere is the one thing a
single prompt to a chat model cannot produce — it has no access to the corpus —
so hiding it would hide the mechanism.

The renderer is one-way. `AnalysisReport` carries computed fields and forbids
extra keys, so it cannot be validated back from its own dump; nothing in the
prototype reads a report, and the properties that make it unreadable are the
ones that turn a renamed field into an error instead of a silently missing
section.
"""

from scopeready.models import (
    AnalysisReport,
    CoverageLocation,
    GapFinding,
)


def render_json(report: AnalysisReport) -> str:
    return report.model_dump_json(indent=2)


def render_markdown(report: AnalysisReport) -> str:
    parts = [
        _header(report),
        _profile(report),
        _selection(report),
        _gaps(report),
        _covered_elsewhere(report),
        _suppressed(report),
        _covered_here(report),
        _warnings(report),
        _cost(report),
    ]
    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def _header(report: AnalysisReport) -> str:
    meta = report.meta
    if meta is None:
        return "# Requirements gap analysis"
    return (
        f"# Requirements gap analysis of {meta.unit.root_doc_id}\n\n"
        f"| | |\n|---|---|\n"
        f"| unit | {meta.unit.granularity.value}, "
        f"{len(meta.unit.doc_ids)} documents |\n"
        f"| taxonomy | {meta.taxonomy_digest} |\n"
        f"| prompts | v{meta.prompt_version} |\n"
        f"| backend | {meta.backend} / {meta.model} |\n"
        f"| embedder | {meta.embedder} |\n"
        f"| prefix digest | `{meta.prefix_digest}` |\n"
        f"| started | {meta.started_at.isoformat(timespec='seconds')} |\n"
        f"| corpus | {report.corpus.requirement_documents} requirement and "
        f"{report.corpus.context_documents} context documents, "
        f"{report.corpus.chunks} chunks |"
    )


def _profile(report: AnalysisReport) -> str:
    if not report.profile.features:
        return ""
    rows = "\n".join(
        f"| `{name}` | {'yes' if feature.value else 'no'} | {feature.rationale} |"
        for name, feature in sorted(report.profile.features.items())
    )
    return (
        "## Project profile\n\n"
        "A wrong feature here silently removes checklist categories, so this "
        "comes before the findings.\n\n"
        f"| feature | value | why |\n|---|---|---|\n{rows}"
    )


def _selection(report: AnalysisReport) -> str:
    if not report.skipped:
        return ""
    rows = "\n".join(
        f"| `{item.category_id}` | {item.reason.value} | {item.explanation} |"
        for item in sorted(report.skipped, key=lambda item: item.category_id)
    )
    asked = len(report.gaps) + len(report.suppressed) + len(report.covered)
    return (
        "## Categories not asked\n\n"
        f"{asked} categories were judged; the following were not, and why.\n\n"
        f"| category | reason | detail |\n|---|---|---|\n{rows}"
    )


def _gaps(report: AnalysisReport) -> str:
    ranked = report.ranked_gaps()
    if not ranked:
        return "## Gaps\n\nNone survived the refutation pass."
    return "## Gaps\n\n" + "\n\n".join(_gap(finding) for finding in ranked)


def _gap(finding: GapFinding) -> str:
    candidate = finding.candidate
    lines = [
        f"### {candidate.category_title} "
        f"(`{candidate.category_id}`, severity {candidate.severity:.2f})",
        "",
        f"- verdict: **{candidate.probe.verdict.value}** at confidence "
        f"{candidate.probe.confidence:.2f}",
        f"- weight: {candidate.weight:.2f} ({candidate.weight_source.value})",
    ]
    if candidate.probe.missing:
        lines.append(f"- missing: {candidate.probe.missing}")
    lines.append(f"- why it costs: {candidate.why_it_costs}")
    if finding.refutation is None:
        lines.append("- refutation: not run")
    else:
        lines.append(
            f"- refutation: searched and found nothing that settles it "
            f"({finding.refutation.reasoning})"
        )
    if finding.question:
        lines.extend(["", f"> **Ask the customer:** {finding.question}"])
    return "\n".join(lines)


def _covered_elsewhere(report: AnalysisReport) -> str:
    elsewhere = [
        item for item in report.covered if item.location is CoverageLocation.ELSEWHERE
    ]
    if not elsewhere:
        return ""
    blocks = []
    for item in sorted(elsewhere, key=lambda entry: entry.category_id):
        quotes = "\n".join(
            f"  - `{quote.chunk_id}`: {quote.quote}" for quote in item.probe.evidence
        )
        blocks.append(f"- **{item.category_title}** (`{item.category_id}`)\n{quotes}")
    return (
        "## The requirement exists, but not here\n\n"
        "Stated somewhere in the corpus, and not in the documents under audit. "
        "Whether that is enough depends on who will read what.\n\n" + "\n".join(blocks)
    )


def _suppressed(report: AnalysisReport) -> str:
    if not report.suppressed:
        return ""
    blocks = []
    for item in sorted(
        report.suppressed, key=lambda entry: entry.candidate.category_id
    ):
        quotes = "\n".join(
            f"  - `{quote.chunk_id}`: {quote.quote}"
            for quote in item.refutation.evidence
        )
        blocks.append(
            f"- **{item.candidate.category_title}** "
            f"(`{item.candidate.category_id}`, severity would have been "
            f"{item.candidate.severity:.2f})\n"
            f"  {item.refutation.reasoning}\n{quotes}"
        )
    return (
        "## Raised, then closed by the corpus\n\n"
        "The probe found nothing in the unit, a search of the whole corpus was "
        "then run, and it settled the question. Dropping a heavy candidate and "
        "dropping a light one say different things about this pass, so both are "
        "shown.\n\n" + "\n".join(blocks)
    )


def _covered_here(report: AnalysisReport) -> str:
    inside = [
        item for item in report.covered if item.location is CoverageLocation.INSIDE
    ]
    if not inside:
        return ""
    rows = "\n".join(
        f"- `{item.category_id}` — {item.category_title} "
        f"({item.probe.evidence[0].chunk_id})"
        for item in sorted(inside, key=lambda entry: entry.category_id)
    )
    return (
        "## Covered in the unit\n\n"
        "Listed without quotes, so that a category which was checked and passed "
        "is not mistaken for one that crashed.\n\n" + rows
    )


def _warnings(report: AnalysisReport) -> str:
    if not report.warnings:
        return ""
    listed = "\n".join(f"- {warning}" for warning in report.warnings)
    return f"## Warnings\n\n{listed}"


def _cost(report: AnalysisReport) -> str:
    usage = report.usage
    share = "—" if usage.cache_share is None else f"{usage.cache_share:.0%}"
    return (
        "## Cost\n\n"
        f"| | |\n|---|---|\n"
        f"| calls | {usage.calls} ({usage.cached_calls} from cache, {share}) |\n"
        f"| prompt tokens | {usage.prompt_tokens} |\n"
        f"| completion tokens | {usage.completion_tokens} |\n"
        f"| wall clock | {report.wall_clock_seconds:.1f}s |"
    )
