"""Tests for the model layer.

The Ollama adapter is tested through its request body and its parsing rather
than against a running server: the two things that actually break are a request
that quietly omits `num_ctx` and an answer that is accepted without being
validated. Both are checkable offline.
"""

import json
from pathlib import Path

import pytest

from scopeready.config import DEFAULT_NUM_CTX
from scopeready.llm import (
    CachingBackend,
    CallRecord,
    ModelError,
    ModelRequest,
    OllamaBackend,
    StubBackend,
    StubPlan,
    build_backend,
    parse_answer,
)
from scopeready.models import Chunk, ProbeResult, Refutation, Verdict


@pytest.fixture
def chunks() -> tuple[Chunk, ...]:
    return (
        Chunk(
            doc_id="jira:SCOPE-2",
            ordinal=0,
            heading_path=("Levels",),
            text="Only the owner may delete a workspace or remove its members.",
        ),
        Chunk(
            doc_id="confluence:1001",
            ordinal=0,
            heading_path=("Deletion",),
            text="Deleted accounts are removed within 30 days of the request.",
        ),
    )


def _fragments(*chunks: Chunk) -> str:
    """A tail shaped like the one `prompts.py` builds.

    The stub cites whatever retrieval surfaced, so a tail with no fragments in
    it means there is nothing honest to quote — which is itself a case worth
    testing.
    """
    rendered = "\n\n".join(
        f"[{chunk.chunk_id}]\n<<<\n{chunk.text}\n>>>" for chunk in chunks
    )
    return f"## Retrieved fragments\n{rendered}\n"


def _request(
    label: str = "probe:roles_and_permissions", tail: str = ""
) -> ModelRequest:
    return ModelRequest(
        system="You audit requirements.", prefix="CORPUS", tail=tail, label=label
    )


# --- the request body -------------------------------------------------------


def test_the_context_window_is_always_stated() -> None:
    # Ollama defaults to 2-4k and truncates from the front, silently, starting
    # with the system prompt. An inherited default is the bug this prevents.
    backend = OllamaBackend(model="m", url="http://x", num_ctx=DEFAULT_NUM_CTX)
    body = backend.body(_request(), ProbeResult)
    assert body["options"]["num_ctx"] == DEFAULT_NUM_CTX
    assert body["options"]["temperature"] == 0.0
    assert body["keep_alive"] == "30m"
    assert body["stream"] is False


def test_the_answer_schema_travels_with_the_request() -> None:
    body = OllamaBackend(model="m", url="http://x", num_ctx=1024).body(
        _request(tail="ASK"), ProbeResult
    )
    assert body["format"]["properties"]["verdict"]
    assert body["messages"][0]["role"] == "system"
    # The invariant prefix and the varying tail reach the model as one message,
    # in that order: everything before the tail is what the cache can reuse.
    assert body["messages"][1]["content"] == "CORPUSASK"


# --- parsing ----------------------------------------------------------------


def test_a_reasoning_preamble_is_stripped() -> None:
    text = (
        "<think>The wiki mentions it.</think>"
        '{"verdict": "covered", "confidence": 0.8, "reasoning": "Stated."}'
    )
    assert parse_answer(ProbeResult, text).verdict is Verdict.COVERED


def test_a_fenced_answer_is_accepted() -> None:
    text = '```json\n{"verdict": "absent", "confidence": 0.4, "reasoning": "No."}\n```'
    assert parse_answer(ProbeResult, text).verdict is Verdict.ABSENT


def test_an_answer_outside_the_schema_is_an_error_not_a_shrug() -> None:
    with pytest.raises(ModelError, match="did not answer in the ProbeResult schema"):
        parse_answer(ProbeResult, '{"verdict": "probably", "confidence": 0.4}')


def test_bounds_the_grammar_drops_are_repaired_by_the_validator() -> None:
    # `minimum`/`maximum` do not survive into the decoding grammar, so a model
    # answering on a 0-100 scale is routine rather than exceptional.
    text = (
        '{"verdict": "partial", "confidence": 80, "missing": "N/A", "reasoning": "x"}'
    )
    answer = parse_answer(ProbeResult, text)
    assert answer.confidence == pytest.approx(0.8)
    assert answer.missing is None


# --- the stub ---------------------------------------------------------------


async def test_the_stub_quotes_the_corpus_verbatim(chunks: tuple[Chunk, ...]) -> None:
    # The engine rejects any quote it cannot find in the named chunk, so a stub
    # that invented text would make every offline run look broken.
    backend = StubBackend(chunks, verdicts={"roles_and_permissions": "covered"})
    record = await backend.complete(_request(tail=_fragments(chunks[0])), ProbeResult)
    quoted = record.answer.evidence[0]
    source = next(chunk for chunk in chunks if chunk.chunk_id == quoted.chunk_id)
    assert quoted.quote in source.text


async def test_the_stub_is_deterministic(chunks: tuple[Chunk, ...]) -> None:
    backend = StubBackend(chunks, verdicts={"roles_and_permissions": "covered"})
    request = _request(tail=_fragments(*chunks))
    first = await backend.complete(request, ProbeResult)
    second = await backend.complete(request, ProbeResult)
    assert first.answer == second.answer


async def test_the_stub_reaches_every_branch(chunks: tuple[Chunk, ...]) -> None:
    backend = StubBackend(
        chunks, verdicts={"a_covered": "covered"}, closes=("a_closed",)
    )
    tail = _fragments(*chunks)
    covered = await backend.complete(_request("probe:a_covered", tail), ProbeResult)
    absent = await backend.complete(_request("probe:a_gap", tail), ProbeResult)
    closed = await backend.complete(_request("refute:a_closed", tail), Refutation)
    open_gap = await backend.complete(_request("refute:a_gap", tail), Refutation)

    assert covered.answer.verdict is Verdict.COVERED
    assert absent.answer.verdict is Verdict.ABSENT
    assert closed.answer.closes_gap and closed.answer.evidence
    assert not open_gap.answer.closes_gap


async def test_a_coverage_claim_with_nothing_retrieved_carries_no_quote(
    chunks: tuple[Chunk, ...],
) -> None:
    # Nothing was retrieved, so there is nothing honest to cite. The engine then
    # downgrades the claim, which is exactly the path worth exercising offline.
    backend = StubBackend(chunks, verdicts={"roles_and_permissions": "covered"})
    record = await backend.complete(_request(), ProbeResult)
    assert record.answer.evidence == ()


async def test_the_stub_refuses_a_call_it_was_not_built_for() -> None:
    with pytest.raises(ModelError, match="no answer for"):
        await StubBackend().complete(_request("summarize"), ProbeResult)


def test_the_fixture_plan_exercises_the_interesting_cases() -> None:
    plan = StubPlan.for_fixture()
    assert "covered" in plan.verdicts.values()
    assert "partial" in plan.verdicts.values()
    assert plan.closes


# --- the cache --------------------------------------------------------------


async def test_a_repeated_call_is_served_from_disk(
    tmp_path: Path, chunks: tuple[Chunk, ...]
) -> None:
    inner = StubBackend(chunks, verdicts={"roles_and_permissions": "covered"})
    cached = CachingBackend(inner, tmp_path)
    request = _request(tail=_fragments(*chunks))
    first = await cached.complete(request, ProbeResult)
    second = await cached.complete(request, ProbeResult)

    assert not first.cached and second.cached
    assert first.answer == second.answer
    # A replayed call computed no tokens; it is the same run again, which is
    # what `cached_calls` says instead.
    assert second.usage.prompt_tokens == 0
    assert second.usage.cached_calls == 1


async def test_the_model_name_is_part_of_the_key(tmp_path: Path) -> None:
    # Changing the model is a new baseline by construction, never a quiet change
    # of results served from an old cache.
    stub = StubBackend(verdicts={"x": "absent"})
    first = CachingBackend(stub, tmp_path).key(_request(), ProbeResult)
    ollama = OllamaBackend(model="gpt-oss:20b", url="http://x", num_ctx=1024)
    second = CachingBackend(ollama, tmp_path).key(_request(), ProbeResult)
    assert first != second


async def test_refresh_bypasses_the_cache(
    tmp_path: Path, chunks: tuple[Chunk, ...]
) -> None:
    request = _request(tail=_fragments(*chunks))
    stub = StubBackend(chunks, verdicts={"roles_and_permissions": "covered"})
    await CachingBackend(stub, tmp_path).complete(request, ProbeResult)
    again = await CachingBackend(stub, tmp_path, refresh=True).complete(
        request, ProbeResult
    )
    assert not again.cached


async def test_a_cached_entry_is_readable(
    tmp_path: Path, chunks: tuple[Chunk, ...]
) -> None:
    # The cache doubles as the structural log of a run, so an entry has to be
    # something a person can open and read.
    cached = CachingBackend(StubBackend(chunks), tmp_path)
    await cached.complete(_request(tail=_fragments(*chunks)), ProbeResult)
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["label"] == "probe:roles_and_permissions"
    assert stored["answer"]["verdict"] == "absent"


# --- construction -----------------------------------------------------------


def test_an_unknown_backend_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="known backends are"):
        build_backend("openai", model="x", url="y", num_ctx=1)


def test_usage_of_a_fresh_call_counts_tokens() -> None:
    record = CallRecord(
        answer=ProbeResult(verdict=Verdict.ABSENT, confidence=0.5, reasoning="x"),
        label="probe:x",
        prompt_eval_count=1200,
        eval_count=300,
        duration_seconds=1.0,
        cached=False,
    )
    assert record.usage.total_tokens == 1500
    assert record.usage.cached_calls == 0
