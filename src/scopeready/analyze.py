"""The pipeline: profile, gate, probe, refute, score, draft, assemble.

Deterministic fan-out over a fixed number of steps, so plain asyncio is the
whole orchestration story. Two things here are subtler than they look.

The first probe runs alone. Parallel model slots hold independent KV caches, so
firing all nineteen at once makes every slot encode the same fifteen thousand
token prefix from scratch — the fan-out costs more than the sequence it
replaced. One call goes first and warms the cache; the rest follow with bounded
concurrency. The proof is in the report: the first probe's prompt token count is
an order of magnitude larger than the rest.

The second is that all retrieval happens before each fan-out, in one worker
thread. A SQLite connection belongs to the thread that opened it, and the
embedder is synchronous. Batching the searches up front means that during the
fan-out the pipeline touches nothing blocking at all, which is a stronger
guarantee than remembering to wrap each call.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from types import TracebackType
from typing import Any, Protocol, Self

from pydantic import BaseModel, ValidationError

from scopeready.config import DEFAULT_CONCURRENCY
from scopeready.llm import CallRecord, ModelBackend, ModelError, ModelRequest
from scopeready.models import (
    AnalysisReport,
    AnalysisUnit,
    Chunk,
    CorpusStats,
    CoverageLocation,
    CoveredCategory,
    Document,
    GapCandidate,
    GapFinding,
    Granularity,
    ProbeResult,
    ProfileFeature,
    ProjectProfile,
    Refutation,
    RunMeta,
    SkippedCategory,
    SkipReason,
    SuppressedGap,
    Usage,
    Verdict,
)
from scopeready.prompts import (
    PROMPT_VERSION,
    DraftedQuestions,
    PromptPrefix,
    build_prefix,
    build_profile_schema,
    probe_tail,
    profile_tail,
    questions_tail,
    refutation_tail,
)
from scopeready.retrieval import reciprocal_rank_fusion
from scopeready.store import ScoredChunk
from scopeready.taxonomy import CategorySpec, Taxonomy, select_categories
from scopeready.verification import check_probe, verify_evidence

PROBE_FRAGMENTS = 6
REFUTATION_FRAGMENTS = 8


class SyncRetriever(Protocol):
    def search(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]: ...


@dataclass(frozen=True, slots=True)
class RunSettings:
    concurrency: int = DEFAULT_CONCURRENCY
    probe_fragments: int = PROBE_FRAGMENTS
    refutation_fragments: int = REFUTATION_FRAGMENTS
    embedder_name: str = "unknown"


class ThreadedRetriever:
    """An async face on a synchronous corpus, with exactly one worker.

    One worker, not a pool: a SQLite connection may only be used from the thread
    that created it, and the embedder is not thread-safe either. Serializing
    retrieval costs milliseconds next to a model call, and it makes the
    threading model provable instead of argued.
    """

    def __init__(self, open_retriever: Callable[[], SyncRetriever]) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval")
        # Opened inside the worker, so the connection and its thread are born
        # together and can never be separated.
        self._inner = self._pool.submit(open_retriever).result()

    async def search(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, partial(self._inner.search, query, limit=limit)
        )

    async def search_many(
        self, queries: Sequence[str], *, limit: int
    ) -> tuple[ScoredChunk, ...]:
        """One ranked list for a category, from several phrasings of its topic.

        A topic is named differently in a ticket, a wiki page and a chat, so a
        category carries several phrasings. Their results are fused by rank, the
        same way the two retrieval channels are.

        Merging by best rank instead looks equivalent and is not: phrasings tie
        at rank one constantly, and an alphabetical tie-break then hands every
        tie to whichever source name sorts first. On this corpus that quietly
        promoted a wiki page over the ticket that actually answered the
        question, for every category, in one direction.
        """
        channels: dict[str, list[str]] = {}
        found: dict[str, ScoredChunk] = {}
        for query in queries:
            hits = await self.search(query, limit=limit)
            channels[query] = [hit.chunk.chunk_id for hit in hits]
            found.update({hit.chunk.chunk_id: hit for hit in hits})
        fused = reciprocal_rank_fusion(channels)
        return tuple(
            ScoredChunk(chunk=found[chunk_id].chunk, score=score)
            for chunk_id, score in fused[:limit]
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass
class RunLog:
    """Everything a run wants to say afterwards, collected as it goes."""

    warnings: list[str] = field(default_factory=list)
    records: list[CallRecord[Any]] = field(default_factory=list)
    prompt_evals: list[tuple[str, int]] = field(default_factory=list)

    def add(self, record: CallRecord[Any]) -> None:
        self.records.append(record)
        self.prompt_evals.append((record.label, record.prompt_eval_count))

    @property
    def usage(self) -> Usage:
        return Usage.merged(record.usage for record in self.records)


async def warmed_fan_out[K, T: BaseModel](
    keys: Sequence[K],
    run: Callable[[K], Awaitable[CallRecord[T] | None]],
    *,
    concurrency: int,
) -> list[CallRecord[T] | None]:
    """Run one call alone to warm the cache, then the rest with a bound.

    The semaphore wraps the await, not the task creation: bounding creation
    bounds nothing, since creating a task is free. It is also built inside the
    coroutine rather than at module scope, so it belongs to the running loop.
    """
    results: list[CallRecord[T] | None] = [None] * len(keys)
    if not keys:
        return results

    warmed = False
    index = 0
    while index < len(keys) and not warmed:
        record = await run(keys[index])
        results[index] = record
        # A first call served from disk warmed nothing, so keep pulling from the
        # head until a real call has happened. On a fully cached rerun this
        # degenerates into a sequential loop over disk reads, which costs
        # microseconds and preserves the promise that a rerun is free.
        warmed = record is not None and not record.cached
        index += 1

    limit = asyncio.Semaphore(max(concurrency, 1))

    async def guarded(position: int, key: K) -> None:
        async with limit:
            results[position] = await run(key)

    async with asyncio.TaskGroup() as group:
        for position in range(index, len(keys)):
            group.create_task(guarded(position, keys[position]))
    return results


async def determine_profile(
    backend: ModelBackend,
    prefix: PromptPrefix,
    taxonomy: Taxonomy,
    log: RunLog,
) -> ProjectProfile:
    schema = build_profile_schema(taxonomy.profile)
    request = ModelRequest(
        system=prefix.system,
        prefix=prefix.corpus,
        tail=profile_tail(taxonomy.profile, schema),
        label="profile",
    )
    record = await backend.complete(request, schema)
    log.add(record)
    answered = record.answer.model_dump()
    features: dict[str, ProfileFeature] = {}
    for feature in taxonomy.profile.features:
        value = answered.get(feature.id)
        if isinstance(value, dict) and "value" in value:
            features[feature.id] = ProfileFeature.model_validate(value)
        else:
            # An undetermined feature reads as false everywhere else, so the
            # warning is what keeps that from being invisible.
            log.warnings.append(
                f"the profile call did not answer {feature.id!r}; reading it as false"
            )
    return ProjectProfile(features=features)


async def run_probes(
    backend: ModelBackend,
    prefix: PromptPrefix,
    categories: Sequence[CategorySpec],
    fragments: Mapping[str, tuple[ScoredChunk, ...]],
    settings: RunSettings,
    log: RunLog,
) -> dict[str, ProbeResult]:
    async def probe(category: CategorySpec) -> CallRecord[ProbeResult] | None:
        request = ModelRequest(
            system=prefix.system,
            prefix=prefix.corpus,
            tail=probe_tail(category, fragments.get(category.id, ()), ProbeResult),
            label=f"probe:{category.id}",
        )
        try:
            record = await backend.complete(request, ProbeResult)
        except (ModelError, ValidationError) as error:
            # Never raised out of the task: a TaskGroup cancels its siblings on
            # the first exception, and losing eighteen good verdicts to one
            # malformed answer is not a trade a ten-minute run can make.
            log.warnings.append(f"[{category.id}] the probe failed: {error}")
            return None
        log.add(record)
        _warn_about_repairs(category.id, record, log)
        return record

    outcomes = await warmed_fan_out(
        list(categories), probe, concurrency=settings.concurrency
    )
    return {
        category.id: outcome.answer
        for category, outcome in zip(categories, outcomes, strict=True)
        if outcome is not None
    }


async def run_refutations(
    backend: ModelBackend,
    prefix: PromptPrefix,
    candidates: Sequence[GapCandidate],
    categories: Mapping[str, CategorySpec],
    fragments: Mapping[str, tuple[ScoredChunk, ...]],
    settings: RunSettings,
    log: RunLog,
) -> dict[str, Refutation]:
    async def refute(candidate: GapCandidate) -> CallRecord[Refutation] | None:
        category = categories[candidate.category_id]
        request = ModelRequest(
            system=prefix.system,
            prefix=prefix.corpus,
            tail=refutation_tail(
                candidate,
                category,
                fragments.get(candidate.category_id, ()),
                Refutation,
            ),
            label=f"refute:{candidate.category_id}",
        )
        try:
            record = await backend.complete(request, Refutation)
        except (ModelError, ValidationError) as error:
            log.warnings.append(
                f"[{candidate.category_id}] the refutation pass failed: {error}"
            )
            return None
        log.add(record)
        return record

    outcomes = await warmed_fan_out(
        list(candidates), refute, concurrency=settings.concurrency
    )
    return {
        candidate.category_id: outcome.answer
        for candidate, outcome in zip(candidates, outcomes, strict=True)
        if outcome is not None
    }


async def draft_questions(
    backend: ModelBackend,
    prefix: PromptPrefix,
    candidates: Sequence[GapCandidate],
    log: RunLog,
) -> dict[str, str]:
    if not candidates:
        return {}
    request = ModelRequest(
        system=prefix.system,
        prefix=prefix.corpus,
        tail=questions_tail(candidates, DraftedQuestions),
        label="questions",
    )
    try:
        record = await backend.complete(request, DraftedQuestions)
    except (ModelError, ValidationError) as error:
        log.warnings.append(f"drafting the questions failed: {error}")
        return {}
    log.add(record)
    return {item.category_id: item.question for item in record.answer.questions}


def _warn_about_repairs(
    category_id: str, record: CallRecord[ProbeResult], log: RunLog
) -> None:
    """Say when a value had to be repaired on the way in.

    A silent repair is indistinguishable from a good answer, and a model that
    systematically answers confidence on a 0-100 scale is worth knowing about.
    """
    reported = record.raw.get("confidence")
    if isinstance(reported, int | float) and not isinstance(reported, bool):
        if abs(float(reported) - record.answer.confidence) > 1e-6:
            log.warnings.append(
                f"[{category_id}] the model reported a confidence of {reported}, "
                f"read as {record.answer.confidence}"
            )


def build_unit(
    documents: Sequence[Document], root_doc_id: str, granularity: str
) -> AnalysisUnit:
    """The root document plus its direct children, as the declared scope."""
    members = {root_doc_id} | {
        document.doc_id
        for document in documents
        if document.parent_doc_id == root_doc_id
    }
    return AnalysisUnit(
        granularity=Granularity(granularity),
        root_doc_id=root_doc_id,
        doc_ids=tuple(sorted(members)),
    )


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Everything a run reads, resolved before the first model call."""

    unit: AnalysisUnit
    documents: tuple[Document, ...]
    unit_chunks: tuple[Chunk, ...]
    all_chunks: tuple[Chunk, ...]
    stats: CorpusStats


async def analyze(
    inputs: AnalysisInputs,
    taxonomy: Taxonomy,
    backend: ModelBackend,
    retriever: ThreadedRetriever,
    settings: RunSettings,
) -> AnalysisReport:
    """One full audit, from profile to drafted questions."""
    started = time.perf_counter()
    log = RunLog()
    chunks_by_id = {chunk.chunk_id: chunk for chunk in inputs.all_chunks}
    categories = {category.id: category for category in taxonomy.categories}

    prefix = build_prefix(inputs.unit, inputs.documents, inputs.unit_chunks)
    profile = await determine_profile(backend, prefix, taxonomy, log)
    selection = select_categories(taxonomy, profile, inputs.unit.granularity)
    skipped = list(selection.skipped)

    # All retrieval happens here, before any model call. During the fan-out the
    # pipeline then touches nothing synchronous at all.
    probe_fragments = {
        category.id: await retriever.search_many(
            category.retrieval_queries, limit=settings.probe_fragments
        )
        for category in selection.applicable
    }

    answers = await run_probes(
        backend, prefix, selection.applicable, probe_fragments, settings, log
    )
    for category in selection.applicable:
        if category.id not in answers:
            skipped.append(
                SkippedCategory(
                    category_id=category.id,
                    category_title=category.title,
                    reason=SkipReason.PROBE_FAILED,
                    explanation="the model did not return a usable verdict",
                )
            )

    covered: list[CoveredCategory] = []
    candidates: list[GapCandidate] = []
    for category in selection.applicable:
        answer = answers.get(category.id)
        if answer is None:
            continue
        checked = check_probe(answer, category.id, chunks_by_id, inputs.unit)
        log.warnings.extend(checked.warnings)
        verdict = checked.probe.verdict
        if verdict is Verdict.COVERED:
            covered.append(
                CoveredCategory(
                    category_id=category.id,
                    category_title=category.title,
                    location=checked.location or CoverageLocation.INSIDE,
                    probe=checked.probe,
                )
            )
            continue
        if verdict is Verdict.NOT_APPLICABLE:
            skipped.append(
                SkippedCategory(
                    category_id=category.id,
                    category_title=category.title,
                    reason=SkipReason.NOT_APPLICABLE,
                    explanation="the model judged the category inapplicable here",
                )
            )
            continue
        candidates.append(
            GapCandidate(
                category_id=category.id,
                category_title=category.title,
                weight=category.weight,
                weight_source=category.weight_source,
                why_it_costs=category.why_it_costs,
                probe=checked.probe,
            )
        )

    refutation_fragments = {
        candidate.category_id: await retriever.search_many(
            categories[candidate.category_id].retrieval_queries,
            limit=settings.refutation_fragments,
        )
        for candidate in candidates
    }
    refutations = await run_refutations(
        backend, prefix, candidates, categories, refutation_fragments, settings, log
    )

    gaps: list[GapCandidate] = []
    suppressed: list[SuppressedGap] = []
    kept_refutations: dict[str, Refutation] = {}
    for candidate in candidates:
        refutation = refutations.get(candidate.category_id)
        if refutation is None:
            gaps.append(candidate)
            continue
        verified = verify_evidence(refutation.evidence, chunks_by_id)
        for item, reason in verified.rejected:
            log.warnings.append(
                f"[{candidate.category_id}] the refutation cited "
                f"{item.chunk_id!r} unusably: {reason.value}"
            )
        if refutation.closes_gap and not verified.kept:
            # A refutation that closes a gap must cite what closes it. A quoteless
            # coverage claim can be downgraded and the finding survives; a
            # quoteless refutation removes a real gap and nothing notices.
            log.warnings.append(
                f"[{candidate.category_id}] a refutation claimed to close the gap "
                "without a usable quote, so the gap stands"
            )
            gaps.append(candidate)
            continue
        checked_refutation = refutation.model_copy(update={"evidence": verified.kept})
        kept_refutations[candidate.category_id] = checked_refutation
        if checked_refutation.closes_gap:
            suppressed.append(
                SuppressedGap(candidate=candidate, refutation=checked_refutation)
            )
        else:
            gaps.append(candidate)

    questions = await draft_questions(backend, prefix, gaps, log)
    findings = tuple(
        GapFinding(
            candidate=candidate,
            refutation=kept_refutations.get(candidate.category_id),
            question=questions.get(candidate.category_id),
        )
        for candidate in gaps
    )

    return AnalysisReport(
        profile=profile,
        corpus=inputs.stats,
        meta=RunMeta(
            unit=inputs.unit,
            taxonomy_digest=taxonomy.digest,
            prompt_version=PROMPT_VERSION,
            backend=backend.name,
            model=backend.model,
            embedder=settings.embedder_name,
            prefix_digest=prefix.digest,
            started_at=datetime.now(UTC),
        ),
        gaps=findings,
        suppressed=tuple(suppressed),
        covered=tuple(covered),
        skipped=tuple(skipped),
        usage=log.usage,
        wall_clock_seconds=round(time.perf_counter() - started, 3),
        warnings=tuple(log.warnings),
    )
