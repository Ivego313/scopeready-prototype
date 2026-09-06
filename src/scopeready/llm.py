"""Asking a model a typed question, and paying for it once.

The interface is deliberately small: a system prompt, a request body, and the
Pydantic schema of the expected answer in; a validated object and what the call
cost out. Everything provider-specific stays inside an adapter, because the one
thing this pipeline must own is how a request is assembled — that is where its
whole economics lives, and a framework that owns request assembly owns the
economics with it.

Three implementations sit behind that interface. `OllamaBackend` is the real
one. `StubBackend` is not a test double but a mode of operation: it answers
every schema deterministically, with quotes taken from the actual corpus, so the
full pipeline can be exercised and debugged with no model at all. `CachingBackend`
wraps either of them and makes a rerun free.
"""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self

import httpx
from pydantic import BaseModel, ValidationError

from scopeready.models import Chunk, Usage

# A reasoning model may emit its chain of thought before the JSON. With `format`
# set the content should already be clean, but stripping it is one line and the
# failure it prevents is a whole run.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
# Fragment identifiers as the retrieved-fragments block renders them, each on a
# line of its own.
_FRAGMENT_ID = re.compile(r"^\[([^\]\n]+)\]$", re.M)


class ModelError(RuntimeError):
    """The model could not be reached, or would not answer in the schema."""


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One question, split where the cost is.

    `system` and the leading `prefix` of `content` are byte-identical across
    every probe in a run, which is what lets the server reuse its KV cache
    instead of re-encoding the corpus nineteen times. `label` names the call for
    logs and for the cache key; it never reaches the model.
    """

    system: str
    prefix: str
    tail: str
    label: str

    @property
    def content(self) -> str:
        return self.prefix + self.tail


@dataclass(frozen=True, slots=True)
class CallRecord[T: BaseModel]:
    """A validated answer and what producing it cost."""

    answer: T
    label: str
    prompt_eval_count: int
    eval_count: int
    duration_seconds: float
    cached: bool
    # The answer before validation, so the engine can warn when a value had to
    # be repaired — a silent repair is indistinguishable from a good answer.
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> Usage:
        return Usage(
            calls=1,
            cached_calls=1 if self.cached else 0,
            # A cached call computed nothing, so it reports no tokens. That is
            # not a free run, it is the same run replayed, which `cached_calls`
            # is what says.
            prompt_tokens=0 if self.cached else self.prompt_eval_count,
            completion_tokens=0 if self.cached else self.eval_count,
        )


class ModelBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete[T: BaseModel](
        self, request: ModelRequest, schema: type[T]
    ) -> CallRecord[T]: ...


def parse_answer[T: BaseModel](schema: type[T], text: str) -> T:
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError as error:
        msg = f"the model did not answer in the {schema.__name__} schema: {error}"
        raise ModelError(msg) from error


class OllamaBackend:
    """A local model over Ollama's native chat API."""

    def __init__(
        self,
        *,
        model: str,
        url: str,
        num_ctx: int,
        timeout_seconds: float = 600.0,
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._num_ctx = num_ctx
        self._timeout = timeout_seconds
        # The default is five minutes and the gap between the probe phase and
        # the refutation phase can exceed it; a reload costs more than the whole
        # phase that follows.
        self._keep_alive = keep_alive

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def body[T: BaseModel](
        self, request: ModelRequest, schema: type[T]
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.content},
            ],
            # The JSON Schema constrains decoding. Two things were measured
            # on qwen3-vl:8b rather than assumed, because both change how this
            # pipeline should be written.
            #
            # Stripping every `description` from the schema left the reported
            # prompt token count unchanged, so a docstring on an answer model
            # cannot be relied on to reach the model: `prompts.py` writes the
            # answer format into the prompt text instead.
            #
            # Setting `format` at all, on identical content, raised the reported
            # prompt tokens from 2152 to 5221 and dropped generation from 20.7
            # to 0.7 tokens per second. The cause is not established by one
            # measurement; the consequence is, and it belongs in the README:
            # on this machine, structured output is the dominant cost of a run.
            #
            # Structure survives the grammar and bounds do not — `minimum`,
            # `maxLength` and `format` are dropped — so the Pydantic validators
            # are the real check, not this field.
            "format": schema.model_json_schema(),
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                # Ollama defaults to a 2-4k window and truncates silently from
                # the front, which is where the system prompt lives. Set
                # explicitly on every call, never inherited.
                "num_ctx": self._num_ctx,
                "temperature": 0.0,
            },
        }

    async def complete[T: BaseModel](
        self, request: ModelRequest, schema: type[T]
    ) -> CallRecord[T]:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._url}/api/chat", json=self.body(request, schema)
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as error:
            msg = f"cannot reach Ollama at {self._url}: {error}"
            raise ModelError(msg) from error

        content = payload.get("message", {}).get("content", "")
        answer = parse_answer(schema, content)
        return CallRecord(
            answer=answer,
            label=request.label,
            prompt_eval_count=int(payload.get("prompt_eval_count", 0)),
            eval_count=int(payload.get("eval_count", 0)),
            duration_seconds=time.perf_counter() - started,
            cached=False,
            raw=json.loads(_THINK_BLOCK.sub("", content).strip() or "{}"),
        )

    async def check(self, schema: type[BaseModel]) -> str:
        """Ask for one token under a schema, to see whether it compiles at all.

        A schema the server cannot turn into a grammar fails the request, and
        finding that at startup beats finding it fifteen minutes into a run.
        """
        request = ModelRequest(
            system="Reply in the schema.", prefix="", tail="ok", label="check"
        )
        body = self.body(request, schema)
        body["options"]["num_predict"] = 1
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._url}/api/chat", json=body)
        if response.status_code >= 400:
            return f"rejected: {response.text[:200]}"
        return "ok"

    async def tags(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        return [model["name"] for model in payload.get("models", [])]


class StubBackend:
    """Deterministic answers built from the real corpus, with no model.

    It exists so the pipeline can be run, debugged and demonstrated on a machine
    with nothing downloaded. Its answers are not judgements — they are shaped
    like judgements, cite real chunks verbatim, and follow a plan the caller can
    set, so that every branch of the pipeline is reachable: covered inside the
    unit, covered elsewhere, absent, closed by refutation.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk] = (),
        *,
        verdicts: Mapping[str, str] | None = None,
        closes: Sequence[str] = (),
        features: Mapping[str, bool] | None = None,
    ) -> None:
        self._chunks = tuple(chunks)
        self._verdicts = dict(verdicts or {})
        self._closes = frozenset(closes)
        self._features = dict(features or {})

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return "stub"

    async def complete[T: BaseModel](
        self, request: ModelRequest, schema: type[T]
    ) -> CallRecord[T]:
        started = time.perf_counter()
        payload = self._answer(request, schema)
        answer = schema.model_validate(payload)
        return CallRecord(
            answer=answer,
            label=request.label,
            # Reported the way a real backend would: the invariant prefix is
            # charged to the first call of a run and to nobody after it, so the
            # warm-up rule stays observable without a server.
            prompt_eval_count=len(request.content) // 4,
            eval_count=len(json.dumps(payload)) // 4,
            duration_seconds=time.perf_counter() - started,
            cached=False,
            raw=payload,
        )

    def _answer(self, request: ModelRequest, schema: type[BaseModel]) -> dict[str, Any]:
        fields = schema.model_fields
        kind, _, subject = request.label.partition(":")

        if kind == "profile":
            return {
                name: {
                    "value": self._features.get(name, False),
                    "rationale": f"Set by the stub backend for {name}.",
                }
                for name in fields
            }
        if kind == "probe":
            return self._probe(subject, request.tail)
        if kind == "refute":
            return self._refute(subject, request.tail)
        if kind == "questions":
            return self._questions(request)
        msg = f"the stub backend has no answer for a {request.label!r} call"
        raise ModelError(msg)

    def _probe(self, category_id: str, tail: str) -> dict[str, Any]:
        verdict = self._verdicts.get(category_id, "absent")
        quote = self._quote(tail) if verdict in {"covered", "partial"} else None
        evidence = [quote] if quote else []
        return {
            "verdict": verdict,
            "confidence": 0.7,
            "evidence": evidence,
            "missing": None
            if verdict == "covered"
            else f"No statement of {category_id}.",
            "reasoning": f"Stub verdict {verdict!r} for {category_id}.",
        }

    def _refute(self, category_id: str, tail: str) -> dict[str, Any]:
        closes = category_id in self._closes
        quote = self._quote(tail) if closes else None
        return {
            "closes_gap": closes and quote is not None,
            "evidence": [quote] if quote else [],
            "reasoning": (
                f"Stub refutation for {category_id}: "
                f"{'the fragments settle it' if closes else 'nothing on topic'}."
            ),
        }

    def _questions(self, request: ModelRequest) -> dict[str, Any]:
        ids = sorted(set(re.findall(r"^- ([a-z][a-z0-9_]*):", request.tail, re.M)))
        return {
            "questions": [
                {
                    "category_id": category_id,
                    "question": f"Could you confirm how {category_id} is handled?",
                }
                for category_id in ids
            ]
        }

    def _quote(self, tail: str) -> dict[str, str] | None:
        """Quote the top fragment retrieval actually surfaced for this call.

        Not a chunk picked by word overlap with the category name: that scores
        long chunks highest and made the offline report cite a page about
        latency as proof of a deletion policy, which reads as a broken engine
        rather than a stubbed one. Citing what retrieval returned is also what a
        real model does, so the shape of the answer stays honest.
        """
        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        for chunk_id in _FRAGMENT_ID.findall(tail):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            sentence = next(
                (
                    line.strip()
                    for line in chunk.text.splitlines()
                    if len(line.strip()) > 30
                ),
                chunk.text.strip(),
            )
            return {"chunk_id": chunk.chunk_id, "quote": sentence}
        # Nothing was retrieved, so there is nothing honest to cite. The engine
        # then downgrades the coverage claim, which is the behaviour under test.
        return None


class CachingBackend:
    """A disk cache in front of any backend, keyed by the exact request.

    Rerunning an analysis then costs nothing, which is what makes it possible to
    iterate on the report, the scoring and the taxonomy without paying for the
    judgements again. The model name is part of the key, so changing the model
    is by construction a new baseline rather than a quiet change of results.
    """

    def __init__(
        self, inner: ModelBackend, directory: Path, *, refresh: bool = False
    ) -> None:
        self._inner = inner
        self._directory = directory
        self._refresh = refresh
        self._lock = asyncio.Lock()
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"{self._inner.name}+cache"

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def inner(self) -> ModelBackend:
        return self._inner

    def key[T: BaseModel](self, request: ModelRequest, schema: type[T]) -> str:
        material = json.dumps(
            {
                "backend": self._inner.name,
                "model": self._inner.model,
                "label": request.label,
                "system": request.system,
                "content": request.content,
                "schema": schema.model_json_schema(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()[:32]

    async def complete[T: BaseModel](
        self, request: ModelRequest, schema: type[T]
    ) -> CallRecord[T]:
        path = self._directory / f"{self.key(request, schema)}.json"
        if path.exists() and not self._refresh:
            stored = json.loads(path.read_text())
            return CallRecord(
                answer=schema.model_validate(stored["answer"]),
                label=request.label,
                prompt_eval_count=int(stored["prompt_eval_count"]),
                eval_count=int(stored["eval_count"]),
                duration_seconds=0.0,
                cached=True,
                raw=stored.get("raw", {}),
            )

        record = await self._inner.complete(request, schema)
        async with self._lock:
            path.write_text(
                json.dumps(
                    {
                        "label": record.label,
                        "answer": record.answer.model_dump(mode="json"),
                        "raw": dict(record.raw),
                        "prompt_eval_count": record.prompt_eval_count,
                        "eval_count": record.eval_count,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return record


def build_backend(
    kind: str,
    *,
    model: str,
    url: str,
    num_ctx: int,
    chunks: Sequence[Chunk] = (),
    stub_plan: "StubPlan | None" = None,
) -> ModelBackend:
    if kind == "stub":
        plan = stub_plan or StubPlan()
        return StubBackend(
            chunks,
            verdicts=plan.verdicts,
            closes=plan.closes,
            features=plan.features,
        )
    if kind == "ollama":
        return OllamaBackend(model=model, url=url, num_ctx=num_ctx)
    msg = f"unknown backend {kind!r}; known backends are ollama, stub"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StubPlan:
    """What the stub should decide, so every branch of the pipeline is reachable."""

    verdicts: Mapping[str, str] = field(default_factory=dict)
    closes: Sequence[str] = ()
    features: Mapping[str, bool] = field(default_factory=dict)

    @classmethod
    def for_fixture(cls) -> Self:
        """A plan matched to the shipped fixture corpus.

        It is written by hand because it has to exercise the interesting cases
        on purpose: something covered inside the unit, something covered only
        outside it, something closed by the refutation pass, and something that
        survives as a real gap.
        """
        return cls(
            features={
                "has_multiple_roles": True,
                "has_personal_data": True,
                "has_external_integrations": True,
                "has_end_user_interface": True,
                "has_existing_codebase": True,
                "involves_data_migration": True,
                "is_fixed_price": True,
            },
            verdicts={
                "roles_and_permissions": "covered",
                "acceptance_criteria": "partial",
                "legacy_constraints": "covered",
                "data_migration": "partial",
            },
            closes=("data_lifecycle_gdpr", "third_party_api_limits"),
        )
