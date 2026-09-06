"""Checking that a claimed quote is really in the corpus, and where it is.

This is the load-bearing safety net of the whole pipeline, and the schema is
not. A model that answers in perfect JSON and cites a paraphrase has produced a
well-formed claim that a requirement is covered when it is not — and a false
sense of coverage is worse than a false gap, because a gap gets argued about
while a silent coverage claim removes a real risk and nobody notices.

Where the quote sits is decided here too, by the engine rather than by the
model. Whether a document belongs to the unit under audit is a fact the run
already knows; asking the model to restate it would add a way to be wrong
without adding anything.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from scopeready.models import (
    AnalysisUnit,
    Chunk,
    CoverageLocation,
    Evidence,
    ProbeResult,
    Verdict,
)
from scopeready.text import ELLIPSIS, normalize_for_match

# A quote of "GDPR" passes a verbatim check and proves nothing. Twenty-four
# normalized characters is a defensible floor for "a statement rather than a
# word", and it is a parameter so a gold set can move it.
MIN_QUOTE_CHARS: Final = 24


class RejectionReason(StrEnum):
    """Why a citation was not accepted.

    The reasons are kept apart because they say different things about the
    model. An invented identifier means it is not reading the corpus at all; a
    paraphrase means it is reading and rewriting. The first is the faster signal
    that a model is too small for this task, which is why the report counts it.
    """

    UNKNOWN_CHUNK = "unknown_chunk"
    NOT_VERBATIM = "not_verbatim"
    USES_ELLIPSIS = "uses_ellipsis"
    TOO_SHORT = "too_short"


@dataclass(frozen=True, slots=True)
class VerifiedEvidence:
    kept: tuple[Evidence, ...]
    rejected: tuple[tuple[Evidence, RejectionReason], ...]


def verify_evidence(
    items: Sequence[Evidence],
    chunks: Mapping[str, Chunk],
    *,
    min_quote_chars: int = MIN_QUOTE_CHARS,
) -> VerifiedEvidence:
    kept: list[Evidence] = []
    rejected: list[tuple[Evidence, RejectionReason]] = []
    for item in items:
        chunk = chunks.get(item.chunk_id)
        if chunk is None:
            rejected.append((item, RejectionReason.UNKNOWN_CHUNK))
            continue
        quote = normalize_for_match(item.quote)
        if ELLIPSIS in quote:
            # Never supported, only detected. Honouring an elision means
            # matching ordered substrings, which would accept a "quote" stitched
            # together from two distant paragraphs — exactly the fabrication the
            # verbatim rule exists to stop.
            rejected.append((item, RejectionReason.USES_ELLIPSIS))
            continue
        if len(quote) < min_quote_chars:
            rejected.append((item, RejectionReason.TOO_SHORT))
            continue
        if quote not in normalize_for_match(chunk.text):
            rejected.append((item, RejectionReason.NOT_VERBATIM))
            continue
        kept.append(item)
    return VerifiedEvidence(kept=tuple(kept), rejected=tuple(rejected))


@dataclass(frozen=True, slots=True)
class CheckedProbe:
    """A probe answer after the engine has held it to its evidence."""

    probe: ProbeResult
    location: CoverageLocation | None
    warnings: tuple[str, ...]
    downgraded: bool


def check_probe(
    probe: ProbeResult,
    category_id: str,
    chunks: Mapping[str, Chunk],
    unit: AnalysisUnit,
    *,
    min_quote_chars: int = MIN_QUOTE_CHARS,
) -> CheckedProbe:
    """Hold a verdict to the quotes that were supposed to license it."""
    verified = verify_evidence(probe.evidence, chunks, min_quote_chars=min_quote_chars)
    warnings = [
        f"[{category_id}] dropped a citation of {item.chunk_id!r}: {reason.value}"
        for item, reason in verified.rejected
    ]

    if probe.verdict in {Verdict.ABSENT, Verdict.NOT_APPLICABLE}:
        if probe.evidence:
            # A quoted absence is incoherent and usually means the model
            # answered a different question than the one it was asked.
            warnings.append(
                f"[{category_id}] a verdict of {probe.verdict.value!r} arrived "
                "with citations attached; they were dropped"
            )
        return CheckedProbe(
            probe=probe.model_copy(update={"evidence": ()}),
            location=None,
            warnings=tuple(warnings),
            downgraded=False,
        )

    if not verified.kept:
        warnings.append(
            f"[{category_id}] a verdict of {probe.verdict.value!r} had no quote "
            "that survived checking, so it was read as absent"
        )
        return CheckedProbe(
            # Confidence is deliberately not damped. The downgrade asserts that
            # there is no proof of coverage, not that we are less sure a gap
            # exists — and damping would quietly push a real gap under a
            # severity threshold.
            probe=probe.model_copy(update={"verdict": Verdict.ABSENT, "evidence": ()}),
            location=None,
            warnings=tuple(warnings),
            downgraded=True,
        )

    location = locate_coverage(verified.kept, chunks, unit)
    if probe.verdict is Verdict.COVERED and _spans_documents(verified.kept, chunks):
        # A probe answers an existence question, so two places contradicting
        # each other read to it as coverage twice over. Detecting the
        # contradiction itself is a different mechanism; noticing that the
        # claim rests on more than one document is the cheap observable.
        warnings.append(
            f"[{category_id}] coverage is claimed from more than one document; "
            "check that they agree rather than contradict"
        )
    return CheckedProbe(
        probe=probe.model_copy(update={"evidence": verified.kept}),
        location=location,
        warnings=tuple(warnings),
        downgraded=False,
    )


def locate_coverage(
    evidence: Sequence[Evidence], chunks: Mapping[str, Chunk], unit: AnalysisUnit
) -> CoverageLocation:
    """Inside the unit under audit, or somewhere else in the corpus.

    Resolved through the chunk map rather than by splitting the identifier. The
    split happens to work today and stops working the day a source id contains a
    "#", and the map is already in hand from verification.
    """
    documents = {
        chunks[item.chunk_id].doc_id for item in evidence if item.chunk_id in chunks
    }
    # Mixed evidence counts as inside: the requirement is stated here and
    # corroborated elsewhere, which is not the "it exists, but not here" case.
    inside = any(unit.contains(doc_id) for doc_id in documents)
    return CoverageLocation.INSIDE if inside else CoverageLocation.ELSEWHERE


def _spans_documents(evidence: Sequence[Evidence], chunks: Mapping[str, Chunk]) -> bool:
    documents = {
        chunks[item.chunk_id].doc_id for item in evidence if item.chunk_id in chunks
    }
    return len(documents) > 1
