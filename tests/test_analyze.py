"""End-to-end tests of the pipeline, on the fixture corpus with no model.

These are the tests that would notice if the mechanism stopped being the
mechanism: a gap closed by a wiki page must end up suppressed rather than
reported, a gate removal must stay visible, and the first call of a fan-out must
be the only one that pays for the corpus.
"""

import asyncio
from pathlib import Path

import pytest

from scopeready.analyze import (
    AnalysisInputs,
    RunSettings,
    ThreadedRetriever,
    analyze,
    build_unit,
    warmed_fan_out,
)
from scopeready.chunking import chunk_document
from scopeready.config import FIXTURE_CORPUS_DIR, TAXONOMY_DIR
from scopeready.corpus import SqliteCorpus
from scopeready.embeddings import StubEmbedder
from scopeready.ingest import read_directory
from scopeready.llm import CachingBackend, CallRecord, StubBackend, StubPlan
from scopeready.models import (
    AnalysisReport,
    CoverageLocation,
    ProbeResult,
    SkipReason,
    Verdict,
)
from scopeready.report import render_json, render_markdown
from scopeready.retrieval import HybridRetriever
from scopeready.store import IndexedChunk
from scopeready.taxonomy import load_taxonomy
from scopeready.text import build_context_text, normalize_for_index


def _build_corpus(path: Path) -> SqliteCorpus:
    embedder = StubEmbedder()
    store = SqliteCorpus.open(path, reset=True)
    store.declare_embedding_space(
        model_name=embedder.name,
        dimensions=embedder.dimensions,
        max_tokens=embedder.max_tokens,
    )
    documents = read_directory(FIXTURE_CORPUS_DIR).documents
    store.add_documents(documents)
    for document in documents:
        chunks, _ = chunk_document(document, embedder)
        store.add_chunks(
            [
                IndexedChunk(
                    chunk=chunk,
                    context_text=build_context_text(
                        document.provenance.title, chunk.heading_path
                    ),
                    index_text=normalize_for_index(chunk.text),
                    token_count=embedder.count_tokens(chunk.text),
                )
                for chunk in chunks
            ]
        )
    pending = list(store.iter_unembedded())
    vectors = embedder.encode_documents([text for _, text in pending])
    store.add_vectors([(pk, vectors[i]) for i, (pk, _) in enumerate(pending)])
    return store


def _run(tmp_path: Path, *, cache: Path | None = None) -> AnalysisReport:
    db = tmp_path / "corpus.db"
    store = _build_corpus(db)
    documents = store.documents()
    all_chunks = store.chunks()
    stats = store.stats()
    store.close()

    unit = build_unit(documents, "jira:SCOPE-1", "epic")
    inputs = AnalysisInputs(
        unit=unit,
        documents=tuple(d for d in documents if unit.contains(d.doc_id)),
        unit_chunks=tuple(c for c in all_chunks if unit.contains(c.doc_id)),
        all_chunks=all_chunks,
        stats=stats,
    )
    plan = StubPlan.for_fixture()
    backend = StubBackend(
        all_chunks,
        verdicts=plan.verdicts,
        closes=plan.closes,
        features=plan.features,
    )
    wrapped = CachingBackend(backend, cache) if cache else backend

    def open_retriever() -> HybridRetriever:
        return HybridRetriever(SqliteCorpus.open(db), StubEmbedder())

    with ThreadedRetriever(open_retriever) as retriever:
        return asyncio.run(
            analyze(
                inputs,
                load_taxonomy(TAXONOMY_DIR),
                wrapped,
                retriever,
                RunSettings(concurrency=2, embedder_name=StubEmbedder().name),
            )
        )


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> AnalysisReport:
    return _run(tmp_path_factory.mktemp("run"))


# --- the mechanisms ---------------------------------------------------------


def test_the_gate_removes_categories_and_the_report_says_so(
    report: AnalysisReport,
) -> None:
    # A removal nobody can see is indistinguishable from a category missing from
    # the rubric file.
    skipped = {item.category_id: item for item in report.skipped}
    assert skipped["payments_pci"].reason is SkipReason.NOT_APPLICABLE
    assert "has_payments=false" in skipped["payments_pci"].explanation
    assert skipped["warranty_period"].reason is SkipReason.WRONG_GRANULARITY


def test_a_gap_closed_elsewhere_in_the_corpus_is_suppressed(
    report: AnalysisReport,
) -> None:
    # This is the mechanic a single prompt to a chat model cannot reproduce: it
    # has no access to the corpus.
    suppressed = {item.candidate.category_id for item in report.suppressed}
    assert "data_lifecycle_gdpr" in suppressed
    reported = {finding.candidate.category_id for finding in report.gaps}
    assert "data_lifecycle_gdpr" not in reported


def test_a_suppressed_gap_carries_the_quote_that_closed_it(
    report: AnalysisReport,
) -> None:
    for item in report.suppressed:
        assert item.refutation.closes_gap
        assert item.refutation.evidence


def test_a_gap_with_nothing_to_close_it_survives(report: AnalysisReport) -> None:
    reported = {finding.candidate.category_id for finding in report.gaps}
    assert "out_of_scope" in reported


def test_every_quote_in_the_report_is_in_the_corpus(tmp_path: Path) -> None:
    # Checked against the corpus rather than trusted: a coverage claim resting
    # on a paraphrase is the failure the schema cannot catch.
    db = tmp_path / "corpus.db"
    store = _build_corpus(db)
    chunks = {chunk.chunk_id: chunk for chunk in store.chunks()}
    store.close()
    generated = _run(tmp_path)
    quotes = [
        (item.chunk_id, item.quote)
        for entry in generated.covered
        for item in entry.probe.evidence
    ] + [
        (item.chunk_id, item.quote)
        for entry in generated.suppressed
        for item in entry.refutation.evidence
    ]
    assert quotes
    for chunk_id, quote in quotes:
        assert quote in chunks[chunk_id].text


def test_coverage_is_placed_inside_or_outside_the_unit(
    report: AnalysisReport,
) -> None:
    assert report.covered
    for item in report.covered:
        assert item.location in {CoverageLocation.INSIDE, CoverageLocation.ELSEWHERE}


def test_every_category_lands_in_exactly_one_section(
    report: AnalysisReport,
) -> None:
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    seen = (
        [finding.candidate.category_id for finding in report.gaps]
        + [item.candidate.category_id for item in report.suppressed]
        + [item.category_id for item in report.covered]
        + [item.category_id for item in report.skipped]
    )
    assert len(seen) == len(set(seen)) == len(taxonomy)


def test_gaps_are_ranked_by_severity(report: AnalysisReport) -> None:
    severities = [finding.candidate.severity for finding in report.ranked_gaps()]
    assert severities == sorted(severities, reverse=True)


def test_a_surviving_gap_carries_a_drafted_question(report: AnalysisReport) -> None:
    assert all(finding.question for finding in report.gaps)


def test_the_run_records_what_produced_it(report: AnalysisReport) -> None:
    # Two runs differing only in model version are two different measurements,
    # and nothing in the numbers would otherwise say so.
    assert report.meta is not None
    assert report.meta.taxonomy_digest and report.meta.prefix_digest
    assert report.meta.unit.root_doc_id == "jira:SCOPE-1"


# --- the fan-out ------------------------------------------------------------


async def test_the_first_call_runs_alone_then_the_rest_are_bounded() -> None:
    in_flight = 0
    peak = 0
    order: list[int] = []

    async def run(key: int) -> CallRecord[ProbeResult]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        order.append(key)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return CallRecord(
            answer=ProbeResult(verdict=Verdict.ABSENT, confidence=0.5, reasoning="x"),
            label=f"probe:{key}",
            prompt_eval_count=0,
            eval_count=0,
            duration_seconds=0.0,
            cached=False,
        )

    results = await warmed_fan_out(list(range(6)), run, concurrency=2)
    assert len(results) == 6 and all(results)
    # The first call must be alone: parallel slots hold independent caches, so
    # firing everything at once makes every slot encode the prefix from scratch.
    assert order[0] == 0
    assert peak <= 2


async def test_a_cached_first_call_does_not_count_as_a_warm_up() -> None:
    calls: list[int] = []

    async def run(key: int) -> CallRecord[ProbeResult]:
        calls.append(key)
        return CallRecord(
            answer=ProbeResult(verdict=Verdict.ABSENT, confidence=0.5, reasoning="x"),
            label=f"probe:{key}",
            prompt_eval_count=0,
            eval_count=0,
            duration_seconds=0.0,
            cached=key < 2,
        )

    await warmed_fan_out(list(range(4)), run, concurrency=2)
    # Keys 0 and 1 came from disk and warmed nothing, so key 2 also had to run
    # on its own before the rest fanned out.
    assert calls[:3] == [0, 1, 2]


async def test_a_failing_call_does_not_cancel_its_siblings() -> None:
    # A TaskGroup cancels siblings on the first exception, and losing eighteen
    # good verdicts to one malformed answer is not a trade a long run can make.
    async def run(key: int) -> CallRecord[ProbeResult] | None:
        if key == 2:
            return None
        return CallRecord(
            answer=ProbeResult(verdict=Verdict.ABSENT, confidence=0.5, reasoning="x"),
            label=f"probe:{key}",
            prompt_eval_count=0,
            eval_count=0,
            duration_seconds=0.0,
            cached=False,
        )

    results = await warmed_fan_out(list(range(5)), run, concurrency=2)
    assert results[2] is None
    assert sum(1 for item in results if item is not None) == 4


async def test_fanning_out_over_nothing_is_not_an_error() -> None:
    async def run(key: int) -> CallRecord[ProbeResult] | None:  # pragma: no cover
        raise AssertionError("must not be called")

    assert await warmed_fan_out([], run, concurrency=2) == []


# --- rerunning is free ------------------------------------------------------


def test_a_second_run_is_served_from_cache_and_is_identical(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = _run(tmp_path / "a", cache=cache)
    second = _run(tmp_path / "b", cache=cache)
    assert second.usage.cache_share == 1.0
    assert second.usage.total_tokens == 0
    # Byte stability is the property the disk cache actually buys; the model
    # itself does not guarantee it even at temperature zero.
    assert _without_timing(render_json(first)) == _without_timing(render_json(second))


def _without_timing(rendered: str) -> str:
    import json
    import re

    payload = json.loads(rendered)
    payload["wall_clock_seconds"] = 0.0
    payload["usage"] = {}
    if payload.get("meta"):
        payload["meta"]["started_at"] = ""
    return re.sub(r"\s+", " ", json.dumps(payload, sort_keys=True))


# --- rendering --------------------------------------------------------------


def test_the_markdown_report_shows_the_mechanisms(report: AnalysisReport) -> None:
    rendered = render_markdown(report)
    assert "## Project profile" in rendered
    assert "## Categories not asked" in rendered
    assert "## Raised, then closed by the corpus" in rendered
    assert "## Gaps" in rendered
    assert "## Cost" in rendered


def test_the_report_names_what_produced_it(report: AnalysisReport) -> None:
    assert report.meta is not None
    assert report.meta.prefix_digest in render_markdown(report)


def test_a_report_with_nothing_in_it_still_renders() -> None:
    from scopeready.models import ProjectProfile

    rendered = render_markdown(AnalysisReport(profile=ProjectProfile()))
    assert "None survived the refutation pass." in rendered
