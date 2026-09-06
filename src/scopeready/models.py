"""Canonical models: the corpus a run reads and the result it produces.

Every record carries where it came from. Provenance is not decoration here:
without it a verdict cannot be shown to the user, and adding it later means
reindexing the whole corpus (design decision, section 2).

A document is described along two independent axes: the kind of source it came
from, and the role it plays in the analysis (design decision, section 4). One
enum mixing them would disguise a decision about the role as a property of the
source, and a wiki page differs from a requirements document by role only.

The result half follows the pipeline in order: profile, probe verdict, gap
candidate, refutation, report. Some of these are the schemas the model answers
by. Their docstrings are written for the model and their rationale lives in
comments instead, because a docstring reaches the JSON Schema `description` and
a comment does not — but note that a backend is free to use the schema only as
a decoding grammar and never show it to the model, which is why `prompts.py`
renders the answer format into the prompt text itself rather than relying on
the schema to carry it.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    computed_field,
    model_validator,
)

# Identifiers and titles arrive from exports, where an empty cell is common and
# means "field missing", not "field empty". Such a value must fail on ingest.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# The source name prefixes every document identifier, so its spelling has to be
# canonical: "Jira" and "jira" would split one corpus into two disjoint halves,
# and nothing downstream would report that as an error.
SourceName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z0-9][a-z0-9_-]*$"),
]

# A document identifier as `Document.doc_id` derives it. Checked wherever it
# is passed in rather than computed, so that a bare source id ("SCOPE-1")
# cannot slip in and silently point at nothing.
DocId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z0-9][a-z0-9_-]*:.+$"),
]

# Identifiers declared in the taxonomy YAML. Lowercase snake_case because they
# are not only keys: a category id becomes part of a metric name and of a file
# name in the eval report, and one that needs escaping there gets renamed —
# after the baseline was recorded under the old spelling.
TaxonomySlug = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z][a-z0-9_]*$"),
]
CategoryId = TaxonomySlug
FeatureId = TaxonomySlug

# What the model reports about its own answer. Bounded to keep severity
# comparable across categories; the bound says nothing about the number being
# trustworthy (see ML-07, step 16).
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

# Rubric weight of a category, normalized: severity then lands in (0, 1] and a
# threshold means the same for every category. Step 4's taxonomy is bound by it.
CategoryWeight = Annotated[float, Field(gt=0.0, le=1.0)]


# A JSON Schema travels into Ollama as a decoding grammar, and a grammar carries
# structure but not bounds: `minimum`/`maximum` are dropped, so a small model
# routinely answers 95 or 1.2 where the schema says [0, 1]. Rejecting the whole
# probe over the scale of a number loses a real finding, so the two mistakes a
# model actually makes are repaired here and the raw value is compared against
# the parsed one by the caller, which is what turns a silent repair into a
# warning in the report.
def _as_unit_interval(value: object) -> object:
    # The two repairs are not symmetric in cost, and the boundary between them
    # is where that shows. A model answering on a 0-100 scale (30, 70, 95) loses
    # its whole signal if every answer is clamped to 1.0, so [2, 100] is read as
    # a percentage. A model answering 1.2 means "certain, sloppily", and reading
    # that as 1.2% would drive a real gap's severity to nearly zero and drop it
    # under any threshold — so (1, 2) is clamped instead. Nobody reports a
    # confidence of 1.5 meaning 150%, and nobody reports 1.5%.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    number = float(value)
    if 2.0 <= number <= 100.0:
        number /= 100.0
    return min(max(number, 0.0), 1.0)


ReportedConfidence = Annotated[
    float, BeforeValidator(_as_unit_interval), Field(ge=0.0, le=1.0)
]

# The same grammar drops `minLength`, so an optional string comes back as the
# word for absence rather than as JSON null. This is the single most common
# junk value in a structured answer.
_SPELLED_ABSENCE = frozenset(
    {"", "null", "none", "n/a", "na", "nil", "-", "unknown", "not applicable"}
)


def _spelled_absence_is_none(value: object) -> object:
    if isinstance(value, str) and value.strip().casefold() in _SPELLED_ABSENCE:
        return None
    return value


ReportedText = Annotated[NonEmptyStr | None, BeforeValidator(_spelled_absence_is_none)]


class SourceKind(StrEnum):
    """What kind of system a document came from, not which one.

    The pipeline never branches on "Jira versus Linear": a new tracker is a new
    adapter, not a new member here. Parsing a particular export format stays
    inside the adapter as well.
    """

    TRACKER = "tracker"
    WIKI = "wiki"
    CHAT = "chat"
    FILE = "file"


class AuthorSide(StrEnum):
    """Which side of the contract the author speaks for.

    A fact about the export, resolved from a list of the executor's own
    accounts. It carries the weight of a fragment: a word from the customer
    changes the agreed scope, a word from the executor does not. `UNKNOWN` is
    an honest admission, and whatever reads it has to stay conservative.
    """

    CUSTOMER = "customer"
    EXECUTOR = "executor"
    UNKNOWN = "unknown"


class CorpusRole(StrEnum):
    """Whether the document is stated scope, or everything else around it.

    `REQUIREMENT` is scope as stated, whatever it is written in: a PRD page
    from the customer and a ticket in their tracker are both this. `CONTEXT`
    is everything else in the corpus — company knowledge, a later
    clarification, a question from the team.

    The role does not say what is under audit. That is the analysis unit of a
    run, and a run may audit the PRD while the tickets serve as evidence, or
    the other way round (design decision, section 3, step 0). Whether a
    fragment actually closes a gap is likewise not this field's call but the
    refutation pass's: a question about a topic mentions it without settling
    it.
    """

    REQUIREMENT = "requirement"
    CONTEXT = "context"


class CorpusModel(BaseModel):
    """Shared configuration of every corpus record.

    Frozen because a record is a snapshot of an export: editing it after
    indexing would desynchronize it from the index built over it. Extra fields
    are forbidden because a misspelled key in an adapter or a fixture is a
    silently dropped field otherwise, and the field most likely to be dropped
    is an optional one from provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Provenance(CorpusModel):
    """Origin of a document: everything needed to cite it and to audit it."""

    source_kind: SourceKind
    # Which system exactly: "jira", "confluence", "linear". Free-form on
    # purpose, see SourceKind.
    source_name: SourceName
    source_id: NonEmptyStr
    # Always present, because every citation in the report is captioned by it.
    # A chat message has no title of its own: the adapter derives one from the
    # thread subject, or from the first line of the message when there is none.
    title: NonEmptyStr
    url: HttpUrl | None = None
    author: NonEmptyStr | None = None
    author_side: AuthorSide = AuthorSide.UNKNOWN
    # Dates must be timezone-aware: exports of one project come from systems
    # configured for different timezones, and a naive timestamp makes the
    # ordering between them a guess.
    created_at: AwareDatetime | None = None
    updated_at: AwareDatetime | None = None
    thread_id: NonEmptyStr | None = None


class Document(CorpusModel):
    """A normalized document from any source, with its origin attached."""

    provenance: Provenance
    # Set explicitly on ingest and never inferred from the item type: the same
    # Confluence page can be the agreed scope or an internal how-to, and the
    # same comment thread is the scope in one mode of work and a discussion of
    # it in another (design decision, section 3).
    corpus_role: CorpusRole
    # How the source itself named the item: "Epic", "Story", "Bug", "Page".
    # A caption for the report; the engine takes no decision on it.
    item_label: NonEmptyStr | None = None
    # Workflow state as the source reports it: "Done", "Won't Do", "Cancelled".
    # A fact from the export, kept rather than filtered on ingest: a cancelled
    # story still names what was dropped from scope, and whether its text may
    # count as coverage is the pipeline's call, not the adapter's.
    status: NonEmptyStr | None = None
    # Normalized Markdown, whatever the export format was: the chunker reads
    # headings from it and must not branch on the source kind. An empty body is
    # legitimate: a ticket may carry its whole content in the title, and
    # dropping it would hide a real gap.
    text: str = ""
    # Identifier of the parent inside the same source system: the epic key of a
    # story, the ticket key of a comment. Cross-source linking is not automatic
    # (design decision, section 13).
    parent_source_id: NonEmptyStr | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def doc_id(self) -> str:
        """Corpus-wide identifier: an id is only unique within its source."""
        return f"{self.provenance.source_name}:{self.provenance.source_id}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def parent_doc_id(self) -> str | None:
        return (
            None
            if self.parent_source_id is None
            else f"{self.provenance.source_name}:{self.parent_source_id}"
        )


class Chunk(CorpusModel):
    """A fragment of a document: the unit of both retrieval and citation."""

    doc_id: DocId
    ordinal: int = Field(ge=0)
    # Path of headings the fragment sits under, outermost first. Empty for a
    # document without headings, such as a chat message.
    heading_path: tuple[NonEmptyStr, ...] = ()
    text: NonEmptyStr

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_id(self) -> str:
        """Derived from position, so it stays the same across runs.

        The prompt prefix repeats these identifiers byte for byte and is only
        reused while it does not change (design decision, section 8).
        """
        return f"{self.doc_id}#{self.ordinal}"


# Nested in both answer schemas, so this docstring is prompt text as well (see
# `Verdict`). The rule it enforces: a report contains no free-standing text —
# anything shown to the user is traceable back to a fragment of the corpus.
class Evidence(CorpusModel):
    """A quote, with the id of the corpus fragment you took it from."""

    chunk_id: NonEmptyStr
    quote: NonEmptyStr

    @classmethod
    def from_chunk(cls, chunk: Chunk, quote: str) -> Self:
        """Build evidence, refusing a quote that is not in the chunk verbatim."""
        normalized = quote.strip()
        if normalized not in chunk.text:
            msg = f"quote is not present verbatim in chunk {chunk.chunk_id!r}"
            raise ValueError(msg)
        return cls(chunk_id=chunk.chunk_id, quote=normalized)


class Granularity(StrEnum):
    """How much scope one run audits.

    A property of the run, not of the source: both a wiki page and a tracker
    ticket can belong to an epic-level unit. There is deliberately no `wiki`
    member — that would confuse where a document came from with how much of the
    project is under audit.
    """

    PROJECT = "project"
    EPIC = "epic"
    TICKET = "ticket"


class AnalysisUnit(CorpusModel):
    """The documents one run treats as the declared scope under audit.

    Everything outside the unit is still searched — that is what makes a
    refutation possible — but coverage found there is reported separately,
    because the question "is this written down anywhere" and the question "is
    this written down here" have different answers and different actions.
    """

    granularity: Granularity
    root_doc_id: DocId
    # Sorted and deduplicated by the validator rather than by convention: this
    # tuple is rendered into the prompt prefix, and a prefix that reorders
    # between runs silently costs the whole KV cache.
    doc_ids: tuple[DocId, ...]

    @model_validator(mode="after")
    def _doc_ids_are_sorted_and_contain_the_root(self) -> Self:
        if not self.doc_ids:
            msg = "an analysis unit must contain at least one document"
            raise ValueError(msg)
        if self.root_doc_id not in self.doc_ids:
            msg = f"root {self.root_doc_id!r} is not among the unit documents"
            raise ValueError(msg)
        expected = tuple(sorted(set(self.doc_ids)))
        if self.doc_ids != expected:
            msg = "unit documents must be sorted and unique"
            raise ValueError(msg)
        return self

    def contains(self, doc_id: str) -> bool:
        return doc_id in self.doc_ids


class ResultModel(BaseModel):
    """Shared configuration of every record a run produces.

    Frozen because a result is the audit trail of one run: editable, it stops
    being evidence. `extra="forbid"` both catches a misspelled key and puts
    `additionalProperties: false` into the schemas the model answers by.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# The docstring below is prompt text: Pydantic puts it into the JSON Schema
# that step 13 passes to the model, so it holds the verdict definitions the
# model needs and nothing addressed to a reader of this file. Two such notes:
# `covered` is deliberately blind to whether the citation sits inside the unit
# under audit — the engine classifies that off the citation's document, and
# coverage from outside is a finding of its own (design decision, section 3,
# step 3). And a `not_applicable` verdict is the probe overruling the
# applicability gate, which the gate cannot see coming: it selects on profile
# features, the probe reads the text.
class Verdict(StrEnum):
    """Answer of a coverage probe about one category.

    - `covered`: the corpus states it, and the quote is attached.
    - `partial`: the topic is mentioned but not settled.
    - `absent`: the corpus does not state it.
    - `not_applicable`: the category is meaningless for this unit.
    """

    COVERED = "covered"
    PARTIAL = "partial"
    ABSENT = "absent"
    NOT_APPLICABLE = "not_applicable"


class WeightSource(StrEnum):
    """Where the weight of a category came from.

    Printed next to severity, because a seed weight orders gaps without
    measuring them and must not be quoted as a result (design decision,
    section 3, step 5).
    """

    SEED = "seed"
    CALIBRATED = "calibrated"


# Multipliers of the severity formula (design decision, section 3, step 5).
# Doubles as the definition of "a gap verdict": the two absent members have no
# multiplier, so a candidate carrying one is a pipeline bug.
GAP_VERDICT_MULTIPLIERS: Mapping[Verdict, float] = MappingProxyType(
    {Verdict.ABSENT: 1.0, Verdict.PARTIAL: 0.55}
)


# Part of the profile schema built from the taxonomy at step 15, so the
# docstring is prompt text (see `Verdict`).
class ProfileFeature(ResultModel):
    """Whether the feature holds for this project, and why you decided so."""

    value: bool
    # Required, because a feature opens or closes whole groups of categories and
    # the rationale is the only way a wrong one gets noticed. Not `Evidence`:
    # the absence of a topic has no fragment to quote.
    rationale: NonEmptyStr


class ProjectProfile(ResultModel):
    """Applicability features of the analyzed unit, keyed by feature id.

    A mapping and not named fields: the feature list is taxonomy data, and the
    engine knows no feature by name (ARCH-01, step 4). The schema the profile
    call answers by is built from that list, not from this class.
    """

    features: Mapping[FeatureId, ProfileFeature] = Field(default_factory=dict)

    def is_set(self, feature_id: str) -> bool:
        """Read a feature, treating an undetermined one as false.

        The conservative direction: an undetermined feature must not open a
        category, or the false positives the gate exists to prevent return
        through the hole in the profile (design decision, section 2).
        """
        feature = self.features.get(feature_id)
        return feature is not None and feature.value


# Answer schema of a probe, so the docstring is prompt text (see `Verdict`).
# It carries no category id on purpose: the pipeline knows which category it
# asked about, and an echoed id would spend tokens on a known value and add one
# more thing the model can get wrong.
class ProbeResult(ResultModel):
    """Your verdict on one category of the requirements checklist."""

    verdict: Verdict
    # Not a probability of being right: severity uses it for being monotone,
    # not for being calibrated. `ReportedConfidence` rather than `Confidence`
    # because this field is filled by a model, not by the engine.
    confidence: ReportedConfidence
    # Not required by the schema even though the design demands a quote for
    # `covered`: such a verdict is downgraded to `absent` with a warning (design
    # decision, section 3, step 3), and a validation error would abort the run
    # instead — losing the finding.
    evidence: tuple[Evidence, ...] = ()
    # The drafted question is built from this, so an absence without it is not
    # actionable.
    missing: ReportedText = None
    reasoning: NonEmptyStr


# Answer schema of the refutation pass, so the docstring is prompt text (see
# `Verdict`).
class Refutation(ResultModel):
    """Whether the fragments you were shown close the gap, and which ones do."""

    # Reads as absolute and is not: the flag answers for scope agreement, and
    # where the evidence lies decides who else will ever see it (design
    # decision, section 2). Widening it later costs nothing — a result record
    # has no index behind it.
    closes_gap: bool
    evidence: tuple[Evidence, ...] = ()
    reasoning: NonEmptyStr

    @model_validator(mode="after")
    def _closing_requires_evidence(self) -> Self:
        """Reject a refutation that closes a gap without citing anything.

        The opposite choice from `ProbeResult`, on the same grounds: a
        quoteless coverage claim is recoverable by downgrading it, a quoteless
        refutation removes a real gap and nothing notices the loss.
        """
        if self.closes_gap and not self.evidence:
            msg = "a refutation that closes a gap must cite what closes it"
            raise ValueError(msg)
        return self


class GapCandidate(ResultModel):
    """A probe verdict that did not confirm coverage, with its category weight.

    Category fields are copied in rather than referenced by id: taxonomy is
    versioned separately from the engine (design decision, section 5), so a
    report reading today's weight would restate yesterday's run in today's
    terms.
    """

    category_id: CategoryId
    category_title: NonEmptyStr
    weight: CategoryWeight
    weight_source: WeightSource
    # Commercial risk of the gap, from the taxonomy record: it goes into the
    # report and into the drafted question.
    why_it_costs: NonEmptyStr
    probe: ProbeResult

    @model_validator(mode="after")
    def _verdict_is_a_gap(self) -> Self:
        if self.probe.verdict not in GAP_VERDICT_MULTIPLIERS:
            msg = f"verdict {self.probe.verdict.value!r} is not a gap verdict"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity(self) -> float:
        """Rubric weight × verdict multiplier × confidence.

        On the candidate, not on the finding: it needs candidate data only, and
        the suppressed section needs it too — "a heavy candidate was dropped" is
        a different statement from "a light one was". Rounded because four
        decimals already exceed the precision a seed weight and a self-reported
        confidence carry between them.
        """
        return round(
            self.weight
            * GAP_VERDICT_MULTIPLIERS[self.probe.verdict]
            * self.probe.confidence,
            4,
        )


class GapFinding(ResultModel):
    """A candidate that survived the refutation pass: a gap for the report."""

    candidate: GapCandidate
    # The attempt that failed. `None` means the pass has not run, which is not
    # the claim "nothing was found" — and the report calls a gap real precisely
    # because a refutation was looked for.
    refutation: Refutation | None = None
    # Drafted once for all surviving gaps together (design decision, section 3,
    # step 6), so a finding exists before its question. Always a draft.
    question: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _refutation_did_not_close_the_gap(self) -> Self:
        if self.refutation is not None and self.refutation.closes_gap:
            msg = "a closed candidate belongs in the suppressed section"
            raise ValueError(msg)
        return self


class SuppressedGap(ResultModel):
    """A candidate the refutation pass dropped, kept for the report.

    Kept because nothing else carries the proof that the mechanism ran, or the
    answer to "why was this not flagged" (design decision, section 2).
    """

    candidate: GapCandidate
    refutation: Refutation

    @model_validator(mode="after")
    def _refutation_closes_the_gap(self) -> Self:
        if not self.refutation.closes_gap:
            msg = "a suppressed gap must carry the refutation that closed it"
            raise ValueError(msg)
        return self


class CoverageLocation(StrEnum):
    """Where the quote that confirms a category was found."""

    INSIDE = "inside"
    ELSEWHERE = "elsewhere"


class CoveredCategory(ResultModel):
    """A category the probe confirmed, and where the confirmation lives.

    The location is decided by the engine from `Evidence.chunk_id`, never by the
    model: whether a document belongs to the unit under audit is a fact the run
    already knows, and asking the model to restate it adds a way to be wrong
    without adding information.
    """

    category_id: CategoryId
    category_title: NonEmptyStr
    location: CoverageLocation
    probe: ProbeResult

    @model_validator(mode="after")
    def _confirmed_coverage_is_quoted(self) -> Self:
        if self.probe.verdict is not Verdict.COVERED:
            msg = f"verdict {self.probe.verdict.value!r} does not confirm coverage"
            raise ValueError(msg)
        # A coverage claim with no surviving quote is downgraded to `absent`
        # before it ever reaches this model, so an empty tuple here is a bug in
        # the engine rather than a bad answer from the model.
        if not self.probe.evidence:
            msg = "confirmed coverage must cite what confirms it"
            raise ValueError(msg)
        return self


class SkipReason(StrEnum):
    """Why a category from the taxonomy produced no verdict."""

    NOT_APPLICABLE = "not_applicable"
    WRONG_GRANULARITY = "wrong_granularity"
    PROBE_FAILED = "probe_failed"


class SkippedCategory(ResultModel):
    """A category that was never judged, and the reason it was not.

    The applicability gate exists to remove false positives, which means it
    removes categories — and a removal nobody can see is indistinguishable from
    a category missing from the taxonomy file. Recording the reason is what
    keeps a wrong profile feature debuggable instead of invisible.
    """

    category_id: CategoryId
    category_title: NonEmptyStr
    reason: SkipReason
    explanation: NonEmptyStr


class CorpusStats(ResultModel):
    """Size of the corpus one run saw, split by the role documents played.

    Split rather than totalled: zero context documents explains a long gap list
    and an empty suppressed section without further digging — there was nothing
    to close a gap with.
    """

    requirement_documents: int = Field(default=0, ge=0)
    context_documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def documents(self) -> int:
        return self.requirement_documents + self.context_documents


class Usage(ResultModel):
    """What a run spent on model calls, and how much of it the cache absorbed.

    Counted tokens are tokens actually computed, so a fully cached rerun reports
    zero (design decision, section 8) — not a free run but the same run
    replayed, which is what `cached_calls` says.
    """

    calls: int = Field(default=0, ge=0)
    cached_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _cached_calls_are_calls(self) -> Self:
        if self.cached_calls > self.calls:
            msg = "cached calls cannot exceed the number of calls"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_share(self) -> float | None:
        """Share of calls served from disk, or `None` when nothing was called.

        `None` rather than zero, which would read as a cache that missed every
        time.
        """
        return None if self.calls == 0 else round(self.cached_calls / self.calls, 4)

    @classmethod
    def merged(cls, parts: Iterable[Self]) -> Self:
        """Add up the usage of individual calls into the usage of a run."""
        collected = tuple(parts)
        return cls(
            calls=sum(part.calls for part in collected),
            cached_calls=sum(part.cached_calls for part in collected),
            prompt_tokens=sum(part.prompt_tokens for part in collected),
            completion_tokens=sum(part.completion_tokens for part in collected),
        )


class RunMeta(ResultModel):
    """What produced a report, in enough detail to reproduce or refuse it.

    A report without this is not a baseline: two runs differing only in model
    version are two different measurements, and nothing in the numbers says so.
    `prefix_digest` is here for a narrower reason — it is the evidence that the
    invariant part of every prompt really was invariant.
    """

    unit: AnalysisUnit
    taxonomy_digest: NonEmptyStr
    prompt_version: NonEmptyStr
    backend: NonEmptyStr
    model: NonEmptyStr
    embedder: NonEmptyStr
    prefix_digest: NonEmptyStr
    started_at: AwareDatetime


class AnalysisReport(ResultModel):
    """The full result of one run: what was found, what was dropped, what it cost."""

    profile: ProjectProfile
    corpus: CorpusStats = Field(default_factory=CorpusStats)
    # Optional so that an empty report stays constructible in tests; every real
    # run fills it.
    meta: RunMeta | None = None
    gaps: tuple[GapFinding, ...] = ()
    suppressed: tuple[SuppressedGap, ...] = ()
    covered: tuple[CoveredCategory, ...] = ()
    skipped: tuple[SkippedCategory, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    # Deliberately not a field of `Usage`: with probes running concurrently the
    # sum of per-call times and the time a human waited differ, and only the
    # second is the metric of a local run (design decision, section 8).
    wall_clock_seconds: float = Field(default=0.0, ge=0.0)
    # Everything the run had to work around: a coverage claim without a quote, a
    # category retrieval found nothing for, a category confirmed by fragments
    # that contradict each other. The last one is why this is not optional — to
    # a probe answering the existence question, a contradiction reads as double
    # coverage (design decision, section 3, step 3), and dropping the warning
    # would make the report look cleaner than the run was.
    warnings: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _each_category_is_reported_once(self) -> Self:
        """Reject a category appearing twice across the report sections.

        A category is probed once and lands in exactly one section, so a
        duplicate is a pipeline bug — invisible in a rendered report, and
        double-counted in the harness, where one category is one measurement.
        """
        reported = [finding.candidate.category_id for finding in self.gaps]
        reported += [dropped.candidate.category_id for dropped in self.suppressed]
        reported += [confirmed.category_id for confirmed in self.covered]
        reported += [skipped.category_id for skipped in self.skipped]
        duplicates = sorted({item for item in reported if reported.count(item) > 1})
        if duplicates:
            msg = f"categories reported more than once: {', '.join(duplicates)}"
            raise ValueError(msg)
        return self

    def ranked_gaps(self) -> tuple[GapFinding, ...]:
        """Gaps by descending severity, ties broken by category id.

        A method, not a computed field, which would serialize a second copy of
        every gap. The tie-break is explicit so that rows do not reshuffle
        between runs of a report that is read next to the previous one.
        """
        return tuple(
            sorted(
                self.gaps,
                key=lambda finding: (
                    -finding.candidate.severity,
                    finding.candidate.category_id,
                ),
            )
        )
