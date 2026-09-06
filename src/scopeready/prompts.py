"""Assembling every request, and keeping the expensive half of it identical.

A run asks the same model about nineteen categories over the same corpus. Done
naively that is nineteen re-encodings of the same fifteen thousand tokens. The
layout here prevents that: a request is a system prompt plus a corpus — byte for
byte the same for every probe — followed by a short tail that differs.

The invariance is structural rather than promised. The prefix is built once into
`PromptPrefix`, probes receive it already finished, and nothing in the tail path
can reach into it. Anything varying — a timestamp, a run id, an unsorted set —
would silently cost the whole cache, so the prefix is also fingerprinted and the
digest goes into the report.

One measured fact shapes this module. Stripping every description out of the
answer schema left Ollama's reported prompt token count unchanged, so a
docstring on an answer model cannot be relied on to reach the model at all. The
answer format therefore has to be written into the prompt text, which is what
`describe_schema` does.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, Field, create_model

from scopeready.models import (
    AnalysisUnit,
    Chunk,
    Document,
    GapCandidate,
    ProfileFeature,
)
from scopeready.store import ScoredChunk
from scopeready.taxonomy import CategorySpec, ProfileSpec

# Part of the cache key and of the report: a reworded prompt is a different
# measurement, and nothing in the numbers would otherwise say so.
PROMPT_VERSION: Final = "1"

SYSTEM_PROMPT: Final = """\
You audit software project requirements against a checklist.

You are given the documents of one analysis unit — the scope currently under \
audit — and, for a single checklist category, some fragments retrieved from the \
wider corpus. You answer about that one category only.

Rules you must follow:

1. Every claim that something is covered must quote the corpus. Copy the quote \
character for character from a fragment you were shown, and name the fragment \
identifier it came from. Do not paraphrase, do not join two places with an \
ellipsis, and do not invent an identifier.
2. If you cannot find a quote, say the category is absent. An absence is a \
useful answer; a coverage claim without a quote silently removes a real risk.
3. Judge what the documents say, not what a competent team would probably do.
4. If two places contradict each other, that is not coverage. Say so in your \
reasoning and do not answer "covered".
5. Answer with JSON only, in the shape described at the end of the request.\
"""

FRAGMENT_OPEN: Final = "<<<"
FRAGMENT_CLOSE: Final = ">>>"


@dataclass(frozen=True, slots=True)
class PromptPrefix:
    """The invariant half of every request in a run.

    Built once and handed to the probes finished. The type exists so that
    "the prefix does not vary" is enforced by there being no way to append to
    it, rather than by everyone remembering not to.
    """

    system: str
    corpus: str

    @property
    def digest(self) -> str:
        material = f"{self.system}\n{self.corpus}".encode()
        return hashlib.sha256(material).hexdigest()[:16]

    @property
    def approximate_tokens(self) -> int:
        return (len(self.system) + len(self.corpus)) // 4


def render_chunk(chunk: Chunk, *, title: str) -> str:
    """One fragment, in the form the model is told to quote from.

    The identifier is on its own line and the body is fenced, so the model has
    an unambiguous thing to copy. The heading path is shown as context and kept
    outside the fence — a heading inside the quotable body would eventually be
    quoted, and the verbatim check against `Chunk.text` would then reject a
    correct answer.
    """
    section = " / ".join(chunk.heading_path)
    where = f"{title} — {section}" if section else title
    return (
        f"[{chunk.chunk_id}] {where}\n{FRAGMENT_OPEN}\n{chunk.text}\n{FRAGMENT_CLOSE}"
    )


def render_corpus(documents: Sequence[Document], chunks: Sequence[Chunk]) -> str:
    """The analysis unit, in a fixed order that does not depend on a dict or a set."""
    titles = {document.doc_id: document.provenance.title for document in documents}
    parts: list[str] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        head = (
            f"### {document.doc_id} — {document.provenance.title}\n"
            f"kind: {document.provenance.source_kind.value}, "
            f"role: {document.corpus_role.value}"
        )
        if document.item_label:
            head += f", type: {document.item_label}"
        if document.status:
            head += f", status: {document.status}"
        parts.append(head)
        for chunk in sorted(
            (chunk for chunk in chunks if chunk.doc_id == document.doc_id),
            key=lambda item: item.ordinal,
        ):
            parts.append(render_chunk(chunk, title=titles[chunk.doc_id]))
    return "\n\n".join(parts)


def build_prefix(
    unit: AnalysisUnit, documents: Sequence[Document], chunks: Sequence[Chunk]
) -> PromptPrefix:
    header = (
        "## Analysis unit\n"
        f"granularity: {unit.granularity.value}\n"
        f"root: {unit.root_doc_id}\n"
        f"documents: {', '.join(unit.doc_ids)}\n\n"
        "## Documents under audit\n"
    )
    return PromptPrefix(
        system=SYSTEM_PROMPT, corpus=header + render_corpus(documents, chunks)
    )


def describe_schema(schema: type[BaseModel]) -> str:
    """Write the answer format into the prompt, because the schema does not.

    Measured, not assumed: with Ollama's `format` the schema constrains decoding
    and never enters the context, so a field whose meaning lives only in a
    docstring is a field the model has to guess.
    """
    definition = schema.model_json_schema()
    definitions = definition.get("$defs", {})
    lines = ["Answer with a JSON object with these fields:"]
    for name, spec in definition["properties"].items():
        lines.append(f"- {name}: {_describe_field(spec, definitions)}")
    return "\n".join(lines)


def _describe_field(spec: Mapping[str, Any], definitions: Mapping[str, Any]) -> str:
    described = _type_of(spec, definitions)
    note = spec.get("description")
    return f"{described}. {note}" if note else described


def _type_of(spec: Mapping[str, Any], definitions: Mapping[str, Any]) -> str:
    if "$ref" in spec:
        name = str(spec["$ref"]).rsplit("/", 1)[-1]
        target = definitions.get(name, {})
        if "enum" in target:
            return "one of " + ", ".join(json.dumps(value) for value in target["enum"])
        fields = ", ".join(target.get("properties", {}))
        return f"an object with {fields}"
    if "anyOf" in spec:
        options = [
            _type_of(option, definitions)
            for option in spec["anyOf"]
            if option.get("type") != "null"
        ]
        return f"{' or '.join(options)}, or null"
    if spec.get("type") == "array":
        return f"a list of {_type_of(spec.get('items', {}), definitions)}"
    return {
        "string": "a string",
        "number": "a number between 0 and 1",
        "integer": "an integer",
        "boolean": "true or false",
    }.get(str(spec.get("type")), "a value")


def build_profile_schema(profile: ProfileSpec) -> type[BaseModel]:
    """One field per feature, rather than a free-form object.

    A mapping with arbitrary keys survives into the grammar as "any object", so
    the model is free to answer about features that do not exist and to omit the
    ones that do. Naming the fields makes the grammar do the work.
    """
    fields: dict[str, Any] = {
        feature.id: (
            ProfileFeature,
            Field(description=feature.question),
        )
        for feature in profile.features
    }
    return create_model("ProjectProfileAnswer", **fields)


def profile_tail(profile: ProfileSpec, schema: type[BaseModel]) -> str:
    questions = "\n".join(
        f"- {feature.id}: {feature.question}" for feature in profile.features
    )
    return (
        "\n\n## Task\n"
        "Decide each property of this project from the documents above. "
        "Answer false when the documents do not say; a guess here silently "
        "removes checklist categories later.\n\n"
        f"{questions}\n\n"
        "For each property give the value and a one-sentence rationale.\n\n"
        f"{describe_schema(schema)}\n"
    )


def probe_tail(
    category: CategorySpec, fragments: Sequence[ScoredChunk], schema: type[BaseModel]
) -> str:
    return (
        "\n\n## Category under audit\n"
        f"id: {category.id}\n"
        f"name: {category.title}\n\n"
        f"Question: {category.probe}\n\n"
        f"Count it as covered when: {category.covered_when}\n\n"
        f"Count it as only partial when: {category.insufficient_when}\n\n"
        f"{_fragments_block(fragments)}"
        f"\n{describe_schema(schema)}\n"
    )


def refutation_tail(
    candidate: GapCandidate,
    category: CategorySpec,
    fragments: Sequence[ScoredChunk],
    schema: type[BaseModel],
) -> str:
    """One narrow question, with no freedom to restate the gap.

    The gap was decided by the probe. This step may only answer whether the
    fragments in front of it settle that same gap, which is what stops the pass
    from quietly turning a missing requirement into a different, smaller one.
    """
    missing = candidate.probe.missing or category.covered_when
    return (
        "\n\n## A gap was raised elsewhere in this project\n"
        f"category: {category.id} — {category.title}\n"
        f"what is missing: {missing}\n\n"
        "Below are fragments found by searching the whole corpus, including "
        "documents outside the unit under audit.\n\n"
        f"{_fragments_block(fragments)}"
        "\nAnswer one question only: do these fragments state what is missing? "
        "Do not restate the gap, do not widen it, and do not answer about a "
        "related topic. Quote character for character if they do.\n\n"
        f"{describe_schema(schema)}\n"
    )


def questions_tail(candidates: Sequence[GapCandidate], schema: type[BaseModel]) -> str:
    """One call for the whole surviving set, not one per gap.

    Asked separately, the model writes three questions about the same
    underlying omission and the reader has to merge them by hand.
    """
    listed = "\n".join(
        f"- {candidate.category_id}: {candidate.category_title} — "
        f"{candidate.probe.missing or 'not stated'} "
        f"(why it matters: {candidate.why_it_costs})"
        for candidate in candidates
    )
    return (
        "\n\n## Gaps that survived\n"
        f"{listed}\n\n"
        "Draft one question per gap, addressed to the customer. Each question "
        "must be answerable in a sentence or two and must not repeat another "
        "question in the set. Use the project's own vocabulary.\n\n"
        f"{describe_schema(schema)}\n"
    )


def _fragments_block(fragments: Sequence[ScoredChunk]) -> str:
    if not fragments:
        return "## Retrieved fragments\n(nothing was retrieved for this category)\n"
    # Ordered by the retriever, which orders deterministically. Re-sorting here
    # would discard the ranking; leaving it to a set would discard repeatability.
    rendered = "\n\n".join(
        f"[{hit.chunk.chunk_id}]\n{FRAGMENT_OPEN}\n{hit.chunk.text}\n{FRAGMENT_CLOSE}"
        for hit in fragments
    )
    return f"## Retrieved fragments\n{rendered}\n"


class DraftedQuestion(BaseModel):
    """One question to put to the customer about one missing requirement."""

    category_id: str
    question: str


class DraftedQuestions(BaseModel):
    """The questions to put to the customer, one per gap."""

    questions: tuple[DraftedQuestion, ...] = ()
