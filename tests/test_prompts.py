"""Tests for prompt assembly and for holding a verdict to its evidence.

Two properties are load-bearing. The prefix must be byte-identical across every
probe of a run, or the cache saves nothing and the run costs nineteen times what
it should. And a coverage claim must not survive without a quote the engine can
find in the corpus, because that is the failure the schema cannot catch.
"""

import os
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from scopeready.config import FIXTURE_CORPUS_DIR, TAXONOMY_DIR
from scopeready.ingest import read_directory
from scopeready.models import (
    AnalysisUnit,
    Chunk,
    CoverageLocation,
    Evidence,
    Granularity,
    ProbeResult,
    Refutation,
    Verdict,
)
from scopeready.prompts import (
    FRAGMENT_CLOSE,
    FRAGMENT_OPEN,
    build_prefix,
    build_profile_schema,
    describe_schema,
    probe_tail,
    render_chunk,
)
from scopeready.store import ScoredChunk
from scopeready.taxonomy import load_taxonomy
from scopeready.verification import (
    RejectionReason,
    check_probe,
    locate_coverage,
    verify_evidence,
)

UNIT = AnalysisUnit(
    granularity=Granularity.EPIC,
    root_doc_id="jira:SCOPE-1",
    doc_ids=("jira:SCOPE-1", "jira:SCOPE-2"),
)

INSIDE = Chunk(
    doc_id="jira:SCOPE-2",
    ordinal=1,
    heading_path=("Levels",),
    text="Only the owner may delete a workspace, and ownership can be transferred.",
)
OUTSIDE = Chunk(
    doc_id="confluence:1001",
    ordinal=0,
    heading_path=("Deletion",),
    text="Deleted accounts are removed within 30 days of the request being made.",
)
CHUNKS = {chunk.chunk_id: chunk for chunk in (INSIDE, OUTSIDE)}


def _probe(
    verdict: Verdict, *, evidence: tuple[Evidence, ...] = (), confidence: float = 0.8
) -> ProbeResult:
    return ProbeResult(
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        reasoning="Because of what the documents say.",
    )


def _quote(chunk: Chunk) -> Evidence:
    return Evidence(chunk_id=chunk.chunk_id, quote=chunk.text)


# --- the prefix is invariant ------------------------------------------------


def test_the_prefix_is_identical_for_every_category() -> None:
    documents = read_directory(FIXTURE_CORPUS_DIR).documents
    unit_documents = [
        document for document in documents if document.doc_id.startswith("jira:SCOPE-")
    ]
    chunks = [INSIDE]
    unit = AnalysisUnit(
        granularity=Granularity.EPIC,
        root_doc_id="jira:SCOPE-1",
        doc_ids=tuple(sorted(document.doc_id for document in unit_documents)),
    )
    prefix = build_prefix(unit, unit_documents, chunks)
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    requests = [
        prefix.system + prefix.corpus + probe_tail(category, (), ProbeResult)
        for category in taxonomy.categories
    ]
    head = len(prefix.system) + len(prefix.corpus)
    assert len({request[:head] for request in requests}) == 1


def test_the_prefix_carries_nothing_that_varies() -> None:
    documents = read_directory(FIXTURE_CORPUS_DIR).documents
    first = build_prefix(UNIT, documents, [INSIDE, OUTSIDE])
    second = build_prefix(UNIT, list(reversed(documents)), [OUTSIDE, INSIDE])
    # Document order comes from the caller and must not reach the bytes.
    assert first.corpus == second.corpus
    assert first.digest == second.digest
    assert str(datetime.now(UTC).year) not in first.corpus


def test_the_prefix_is_stable_across_processes() -> None:
    # A set anywhere in the serialization would make the digest depend on the
    # hash seed, and the loss would show up only as a cache that never hits.
    script = (
        "from scopeready.config import FIXTURE_CORPUS_DIR;"
        "from scopeready.ingest import read_directory;"
        "from scopeready.prompts import build_prefix;"
        "from scopeready.models import AnalysisUnit, Granularity;"
        "d=read_directory(FIXTURE_CORPUS_DIR).documents;"
        "u=AnalysisUnit(granularity=Granularity.EPIC, root_doc_id='jira:SCOPE-1',"
        " doc_ids=tuple(sorted(x.doc_id for x in d)));"
        "print(build_prefix(u, d, []).digest)"
    )
    digests = set()
    for seed in ("0", "1", "random"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1


def test_a_fragment_is_fenced_and_identified() -> None:
    rendered = render_chunk(INSIDE, title="Access levels")
    assert rendered.startswith(f"[{INSIDE.chunk_id}]")
    assert FRAGMENT_OPEN in rendered and FRAGMENT_CLOSE in rendered
    # The heading must stay outside the quotable body, or the model will quote
    # it and the verbatim check will reject a correct answer.
    assert "Levels" not in rendered.split(FRAGMENT_OPEN)[1]


# --- the answer format has to be written out --------------------------------


def test_the_answer_format_is_described_in_words() -> None:
    # Measured: stripping every description out of the schema left the reported
    # prompt token count unchanged, so a field explained only in a docstring is
    # a field the model has to guess.
    described = describe_schema(ProbeResult)
    assert "verdict" in described and "covered" in described
    assert "evidence" in described and "confidence" in described


def test_the_profile_schema_names_every_feature() -> None:
    # A free-form mapping survives into the grammar as "any object", leaving the
    # model free to answer about features that do not exist.
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    schema = build_profile_schema(taxonomy.profile)
    assert set(schema.model_fields) == set(taxonomy.feature_ids)


def test_the_probe_tail_carries_the_rubric_not_just_the_question() -> None:
    taxonomy = load_taxonomy(TAXONOMY_DIR)
    category = next(c for c in taxonomy.categories if c.id == "acceptance_criteria")
    tail = probe_tail(category, (ScoredChunk(chunk=INSIDE, score=1.0),), ProbeResult)
    assert category.probe in tail
    assert category.covered_when in tail
    # The near-miss wording is what separates a useful `partial` from a rubric
    # that only knows yes and no.
    assert category.insufficient_when in tail
    assert INSIDE.chunk_id in tail


# --- evidence verification --------------------------------------------------


def test_a_verbatim_quote_is_accepted() -> None:
    verified = verify_evidence([_quote(INSIDE)], CHUNKS)
    assert verified.kept and not verified.rejected


def test_a_paraphrase_is_rejected() -> None:
    item = Evidence(chunk_id=INSIDE.chunk_id, quote="only owners can remove workspaces")
    assert (
        verify_evidence([item], CHUNKS).rejected[0][1] is RejectionReason.NOT_VERBATIM
    )


def test_an_invented_identifier_is_told_apart_from_a_paraphrase() -> None:
    # They say different things about the model: one is not reading the corpus,
    # the other is reading it and rewriting.
    item = Evidence(chunk_id="jira:MADE-UP#3", quote=INSIDE.text)
    assert (
        verify_evidence([item], CHUNKS).rejected[0][1] is RejectionReason.UNKNOWN_CHUNK
    )


def test_an_elided_quote_is_rejected_rather_than_stitched() -> None:
    # Honouring an ellipsis would accept a quote assembled from two distant
    # paragraphs, which is the fabrication the rule exists to stop.
    item = Evidence(
        chunk_id=INSIDE.chunk_id, quote="Only the owner ... can be transferred."
    )
    assert (
        verify_evidence([item], CHUNKS).rejected[0][1] is RejectionReason.USES_ELLIPSIS
    )


def test_a_quote_too_short_to_state_anything_is_rejected() -> None:
    item = Evidence(chunk_id=INSIDE.chunk_id, quote="owner")
    assert verify_evidence([item], CHUNKS).rejected[0][1] is RejectionReason.TOO_SHORT


def test_presentation_differences_are_forgiven() -> None:
    chunk = Chunk(
        doc_id="jira:SCOPE-2", ordinal=9, text="The **owner** — alone — may act."
    )
    item = Evidence(chunk_id=chunk.chunk_id, quote="The owner - alone - may act.")
    assert verify_evidence([item], {chunk.chunk_id: chunk}).kept


# --- holding a verdict to its evidence --------------------------------------


def test_coverage_without_a_surviving_quote_becomes_absence() -> None:
    # Hallucinated coverage is worse than a false flag: it silently removes a
    # real risk, and nothing downstream ever looks at it again.
    bad = Evidence(chunk_id=INSIDE.chunk_id, quote="a sentence that is not there")
    checked = check_probe(
        _probe(Verdict.COVERED, evidence=(bad,)), "roles", CHUNKS, UNIT
    )
    assert checked.probe.verdict is Verdict.ABSENT
    assert checked.downgraded and checked.probe.evidence == ()


def test_a_downgrade_does_not_damp_confidence() -> None:
    # The downgrade asserts there is no proof of coverage, not that we are less
    # sure a gap exists; damping would push a real gap under a threshold.
    bad = Evidence(chunk_id="nope#0", quote="x" * 40)
    checked = check_probe(
        _probe(Verdict.COVERED, evidence=(bad,), confidence=0.9), "roles", CHUNKS, UNIT
    )
    assert checked.probe.confidence == pytest.approx(0.9)


def test_one_good_quote_is_enough_to_keep_the_verdict() -> None:
    bad = Evidence(chunk_id=INSIDE.chunk_id, quote="not present at all")
    checked = check_probe(
        _probe(Verdict.COVERED, evidence=(_quote(INSIDE), bad)), "roles", CHUNKS, UNIT
    )
    assert checked.probe.verdict is Verdict.COVERED
    assert len(checked.probe.evidence) == 1
    assert checked.warnings


def test_a_quoted_absence_has_its_quotes_dropped() -> None:
    checked = check_probe(
        _probe(Verdict.ABSENT, evidence=(_quote(INSIDE),)), "roles", CHUNKS, UNIT
    )
    assert checked.probe.evidence == ()
    assert any("citations attached" in warning for warning in checked.warnings)


def test_coverage_from_two_documents_is_flagged_for_contradiction() -> None:
    # A probe answers an existence question, so two places disagreeing read to
    # it as coverage twice over.
    checked = check_probe(
        _probe(Verdict.COVERED, evidence=(_quote(INSIDE), _quote(OUTSIDE))),
        "nfr_performance",
        CHUNKS,
        UNIT,
    )
    assert any("contradict" in warning for warning in checked.warnings)


# --- where the coverage lives -----------------------------------------------


def test_coverage_inside_the_unit_is_told_from_coverage_outside_it() -> None:
    assert locate_coverage([_quote(INSIDE)], CHUNKS, UNIT) is CoverageLocation.INSIDE
    assert (
        locate_coverage([_quote(OUTSIDE)], CHUNKS, UNIT) is CoverageLocation.ELSEWHERE
    )


def test_mixed_evidence_counts_as_inside() -> None:
    # The requirement is stated here and corroborated elsewhere, which is not
    # the "it exists, but not here" case.
    where = locate_coverage([_quote(OUTSIDE), _quote(INSIDE)], CHUNKS, UNIT)
    assert where is CoverageLocation.INSIDE


def test_the_engine_decides_the_location_not_the_model() -> None:
    checked = check_probe(
        _probe(Verdict.COVERED, evidence=(_quote(OUTSIDE),)), "gdpr", CHUNKS, UNIT
    )
    assert checked.location is CoverageLocation.ELSEWHERE


def test_a_refutation_still_has_to_cite_something_real() -> None:
    bad = Evidence(chunk_id="invented#1", quote="the policy covers this entirely")
    refutation = Refutation(closes_gap=True, evidence=(bad,), reasoning="Covered.")
    assert verify_evidence(refutation.evidence, CHUNKS).kept == ()
