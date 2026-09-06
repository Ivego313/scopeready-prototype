"""Tests for the models carried over from the reviewed core.

They restate the invariants rather than the implementation: what a record
refuses to be is the part other modules are allowed to rely on.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from scopeready.models import (
    AnalysisReport,
    AnalysisUnit,
    AuthorSide,
    Chunk,
    CorpusRole,
    CorpusStats,
    CoverageLocation,
    CoveredCategory,
    Document,
    Evidence,
    GapCandidate,
    GapFinding,
    Granularity,
    ProbeResult,
    ProfileFeature,
    ProjectProfile,
    Provenance,
    Refutation,
    RunMeta,
    SkippedCategory,
    SkipReason,
    SourceKind,
    SuppressedGap,
    Usage,
    Verdict,
    WeightSource,
)


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(
        source_kind=SourceKind.TRACKER,
        source_name="jira",
        source_id="SCOPE-1",
        title="Checkout redesign",
    )


@pytest.fixture
def document(provenance: Provenance) -> Document:
    return Document(
        provenance=provenance,
        corpus_role=CorpusRole.REQUIREMENT,
        text="## Roles\n\nOnly the owner may delete a workspace.",
    )


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        doc_id="jira:SCOPE-1",
        ordinal=0,
        heading_path=("Roles",),
        text="Only the owner may delete a workspace.",
    )


@pytest.fixture
def evidence(chunk: Chunk) -> Evidence:
    return Evidence.from_chunk(chunk, "Only the owner may delete a workspace.")


def _probe(
    verdict: Verdict = Verdict.ABSENT,
    confidence: float = 0.8,
    evidence: tuple[Evidence, ...] = (),
) -> ProbeResult:
    return ProbeResult(
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        reasoning="Nothing in the unit states it.",
    )


def _candidate(
    category_id: str = "acceptance_criteria",
    weight: float = 0.9,
    probe: ProbeResult | None = None,
) -> GapCandidate:
    return GapCandidate(
        category_id=category_id,
        category_title="Acceptance criteria",
        weight=weight,
        weight_source=WeightSource.SEED,
        why_it_costs="Disputes at handover are settled by whoever wrote them down.",
        probe=probe if probe is not None else _probe(),
    )


# --- identifiers ------------------------------------------------------------


def test_doc_id_is_derived_from_provenance(document: Document) -> None:
    assert document.doc_id == "jira:SCOPE-1"
    assert document.parent_doc_id is None


def test_chunk_id_is_derived_from_position(chunk: Chunk) -> None:
    assert chunk.chunk_id == "jira:SCOPE-1#0"


def test_source_name_must_be_canonical() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            source_kind=SourceKind.WIKI,
            source_name="Confluence",
            source_id="42",
            title="Data retention",
        )


def test_author_side_defaults_to_unknown(provenance: Provenance) -> None:
    assert provenance.author_side is AuthorSide.UNKNOWN


def test_evidence_refuses_a_quote_that_is_not_verbatim(chunk: Chunk) -> None:
    with pytest.raises(ValueError, match="verbatim"):
        Evidence.from_chunk(chunk, "only owners can delete workspaces")


# --- analysis unit ----------------------------------------------------------


def test_unit_documents_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        AnalysisUnit(
            granularity=Granularity.EPIC,
            root_doc_id="jira:SCOPE-1",
            doc_ids=("jira:SCOPE-2", "jira:SCOPE-1"),
        )


def test_unit_must_contain_its_own_root() -> None:
    with pytest.raises(ValidationError, match="not among the unit documents"):
        AnalysisUnit(
            granularity=Granularity.EPIC,
            root_doc_id="jira:SCOPE-1",
            doc_ids=("jira:SCOPE-2",),
        )


def test_unit_membership_is_a_lookup() -> None:
    unit = AnalysisUnit(
        granularity=Granularity.EPIC,
        root_doc_id="jira:SCOPE-1",
        doc_ids=("jira:SCOPE-1", "jira:SCOPE-2"),
    )
    assert unit.contains("jira:SCOPE-2")
    assert not unit.contains("confluence:42")


# --- what a model is allowed to answer badly --------------------------------


@pytest.mark.parametrize(
    ("reported", "parsed"),
    [(95, 0.95), (30, 0.3), (1.2, 1.0), (-0.3, 0.0), (140, 1.0), (0.62, 0.62)],
)
def test_confidence_off_the_unit_interval_is_repaired(
    reported: float, parsed: float
) -> None:
    assert _probe(confidence=reported).confidence == pytest.approx(parsed)


@pytest.mark.parametrize("spelled", ["null", "None", " N/A ", "", "unknown"])
def test_a_spelled_absence_becomes_none(spelled: str) -> None:
    probe = ProbeResult(
        verdict=Verdict.COVERED,
        confidence=0.9,
        missing=spelled,
        reasoning="Stated in the wiki.",
    )
    assert probe.missing is None


def test_a_real_missing_note_survives() -> None:
    assert _probe().model_copy(update={"missing": "No SLA is stated"}).missing


def test_a_coverage_claim_without_a_quote_is_accepted_by_the_schema() -> None:
    # Deliberate: the engine downgrades it to `absent` with a warning, while a
    # validation error here would abort the run and lose the finding.
    assert _probe(verdict=Verdict.COVERED).evidence == ()


# --- gap arithmetic ---------------------------------------------------------


def test_severity_multiplies_weight_verdict_and_confidence() -> None:
    assert _candidate().severity == pytest.approx(0.72)


def test_a_partial_verdict_is_discounted() -> None:
    candidate = _candidate(probe=_probe(verdict=Verdict.PARTIAL))
    assert candidate.severity == pytest.approx(0.396)


@pytest.mark.parametrize("verdict", [Verdict.COVERED, Verdict.NOT_APPLICABLE])
def test_a_non_gap_verdict_cannot_become_a_candidate(verdict: Verdict) -> None:
    with pytest.raises(ValidationError, match="not a gap verdict"):
        _candidate(probe=_probe(verdict=verdict))


def test_weight_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _candidate(weight=1.5)


# --- refutation asymmetry ---------------------------------------------------


def test_a_closing_refutation_must_cite(evidence: Evidence) -> None:
    with pytest.raises(ValidationError, match="must cite"):
        Refutation(closes_gap=True, reasoning="The wiki covers it.")
    assert Refutation(
        closes_gap=True, evidence=(evidence,), reasoning="The wiki covers it."
    ).closes_gap


def test_a_non_closing_refutation_needs_nothing() -> None:
    assert not Refutation(closes_gap=False, reasoning="Nothing on topic.").closes_gap


def test_a_closed_candidate_may_not_stay_a_finding(evidence: Evidence) -> None:
    closing = Refutation(closes_gap=True, evidence=(evidence,), reasoning="Covered.")
    with pytest.raises(ValidationError, match="suppressed section"):
        GapFinding(candidate=_candidate(), refutation=closing)


def test_a_suppressed_gap_must_carry_what_closed_it() -> None:
    with pytest.raises(ValidationError, match="must carry the refutation"):
        SuppressedGap(
            candidate=_candidate(),
            refutation=Refutation(closes_gap=False, reasoning="Nothing found."),
        )


def test_a_finding_without_a_refutation_means_the_pass_did_not_run() -> None:
    assert GapFinding(candidate=_candidate()).refutation is None


# --- coverage and skipping --------------------------------------------------


def test_confirmed_coverage_must_be_quoted(evidence: Evidence) -> None:
    with pytest.raises(ValidationError, match="must cite"):
        CoveredCategory(
            category_id="roles_and_permissions",
            category_title="Roles and access rights",
            location=CoverageLocation.INSIDE,
            probe=_probe(verdict=Verdict.COVERED),
        )
    confirmed = CoveredCategory(
        category_id="roles_and_permissions",
        category_title="Roles and access rights",
        location=CoverageLocation.ELSEWHERE,
        probe=_probe(verdict=Verdict.COVERED, evidence=(evidence,)),
    )
    assert confirmed.location is CoverageLocation.ELSEWHERE


def test_a_gap_verdict_is_not_coverage(evidence: Evidence) -> None:
    with pytest.raises(ValidationError, match="does not confirm coverage"):
        CoveredCategory(
            category_id="roles_and_permissions",
            category_title="Roles and access rights",
            location=CoverageLocation.INSIDE,
            probe=_probe(verdict=Verdict.PARTIAL, evidence=(evidence,)),
        )


def test_a_skipped_category_states_why() -> None:
    skipped = SkippedCategory(
        category_id="payments_pci",
        category_title="Payments and PCI",
        reason=SkipReason.NOT_APPLICABLE,
        explanation="has_payments is false",
    )
    assert skipped.reason is SkipReason.NOT_APPLICABLE


# --- profile ----------------------------------------------------------------


def test_an_undetermined_feature_reads_as_false() -> None:
    profile = ProjectProfile(
        features={"has_payments": ProfileFeature(value=True, rationale="Stripe.")}
    )
    assert profile.is_set("has_payments")
    assert not profile.is_set("is_game_project")


# --- usage ------------------------------------------------------------------


def test_cached_calls_cannot_exceed_calls() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        Usage(calls=1, cached_calls=2)


def test_cache_share_of_a_run_that_called_nothing_is_unknown() -> None:
    assert Usage().cache_share is None
    assert Usage(calls=4, cached_calls=1).cache_share == pytest.approx(0.25)


def test_usage_merges_across_calls() -> None:
    merged = Usage.merged(
        [Usage(calls=1, prompt_tokens=10), Usage(calls=1, completion_tokens=5)]
    )
    assert (merged.calls, merged.total_tokens) == (2, 15)


# --- report -----------------------------------------------------------------


def test_a_category_may_not_appear_in_two_sections(evidence: Evidence) -> None:
    closing = Refutation(closes_gap=True, evidence=(evidence,), reasoning="Covered.")
    with pytest.raises(ValidationError, match="reported more than once"):
        AnalysisReport(
            profile=ProjectProfile(),
            gaps=(GapFinding(candidate=_candidate()),),
            suppressed=(SuppressedGap(candidate=_candidate(), refutation=closing),),
        )


def test_gaps_rank_by_severity_then_by_category_id() -> None:
    report = AnalysisReport(
        profile=ProjectProfile(),
        gaps=(
            GapFinding(candidate=_candidate("localization", weight=0.5)),
            GapFinding(candidate=_candidate("acceptance_criteria", weight=0.9)),
            GapFinding(candidate=_candidate("a_tied_category", weight=0.9)),
        ),
    )
    assert [finding.candidate.category_id for finding in report.ranked_gaps()] == [
        "a_tied_category",
        "acceptance_criteria",
        "localization",
    ]


def test_corpus_stats_are_split_by_role() -> None:
    stats = CorpusStats(requirement_documents=6, context_documents=3, chunks=41)
    assert stats.documents == 9


def test_run_meta_pins_what_produced_the_report() -> None:
    meta = RunMeta(
        unit=AnalysisUnit(
            granularity=Granularity.EPIC,
            root_doc_id="jira:SCOPE-1",
            doc_ids=("jira:SCOPE-1",),
        ),
        taxonomy_digest="a1b2c3",
        prompt_version="1",
        backend="stub",
        model="stub",
        embedder="stub-hash-64",
        prefix_digest="d4e5f6",
        started_at=datetime(2026, 9, 7, tzinfo=UTC),
    )
    assert AnalysisReport(profile=ProjectProfile(), meta=meta).meta is meta


def test_the_report_does_not_read_its_own_dump() -> None:
    # Computed fields plus extra="forbid" make the dump a strictly wider shape
    # than the input. The rendering path is one-way on purpose; the test exists
    # so the property is a decision rather than a surprise.
    report = AnalysisReport(profile=ProjectProfile())
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(report.model_dump())
