"""Fusing the two search channels into one ordered list of fragments.

Lexical search and vector search fail in opposite directions. The first cannot
bridge vocabulary — a corpus that says "access levels" is invisible to a query
about a "permissions matrix" no matter how good the stemmer is. The second
cannot be trusted on exact tokens: an identifier, an error code or a version
number is exactly what a dense model smooths away.

They are combined by rank rather than by score, because their scores are on
unrelated scales and calibrating them would need the labelled data this
prototype does not have yet. Reciprocal rank fusion needs no such calibration,
which is why it is here and not a weighted sum.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from scopeready.embeddings import Embedder, Vector
from scopeready.models import Chunk
from scopeready.store import ScoredChunk

# From Cormack, Clarke and Buettcher (2009), and the default in most engines
# since. What it controls is how flat the head is: at k=1 a single channel's top
# hit wins outright and fusing achieves nothing, at k=1000 rank stops mattering
# and fusion degenerates into "appeared in both lists". At 60, agreement across
# both channels outranks a strong hit in one — which is the behaviour the
# refutation pass needs, since that is where a wrong fragment is most expensive.
RRF_K: Final = 60

# How deep each channel is read before fusing. At 50 a tail hit still counts for
# something but cannot outrank a head hit in the other channel.
CHANNEL_DEPTH: Final = 50

LEXICAL: Final = "lexical"
VECTOR: Final = "vector"


class Retriever(Protocol):
    def search(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]: ...


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[str]], *, k: int = RRF_K
) -> tuple[tuple[str, float], ...]:
    """score(d) = sum over channels of 1 / (k + rank), ranks starting at 1.

    Channels are accumulated in sorted order. With two channels that is free,
    since float addition is commutative — but it is not associative, so the
    moment a reranker adds a third channel an unordered iteration would make
    scores depend on dictionary order. The insurance costs one `sorted`.
    """
    scores: dict[str, float] = {}
    for name in sorted(channels):
        for rank, chunk_id in enumerate(channels[name], start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # Ties are the normal case rather than an edge: a chunk ranked first in one
    # channel only and a chunk ranked first in the other score an identical
    # 1/61. The tie-break is by chunk id so that the order reaching the prompt
    # is the same on every machine.
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


class VectorStore(Protocol):
    """The part of the corpus store that retrieval needs, and no more."""

    def search_lexical(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]: ...

    def search_vector(
        self, query: Vector, *, limit: int
    ) -> tuple[ScoredChunk, ...]: ...


@dataclass(frozen=True, slots=True)
class ChannelResults:
    """What each channel returned, kept apart so a run can be explained."""

    lexical: tuple[ScoredChunk, ...]
    vector: tuple[ScoredChunk, ...]
    fused: tuple[ScoredChunk, ...]


class HybridRetriever:
    """Lexical and vector search over one corpus, fused by rank."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        depth: int = CHANNEL_DEPTH,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._depth = depth

    def search(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]:
        return self.search_channels(query, limit=limit).fused

    def search_channels(self, query: str, *, limit: int) -> ChannelResults:
        lexical = self._store.search_lexical(query, limit=self._depth)
        vector = self._store.search_vector(
            self._embedder.encode_query(query), limit=self._depth
        )
        fused_ids = reciprocal_rank_fusion(
            {
                LEXICAL: [hit.chunk.chunk_id for hit in lexical],
                VECTOR: [hit.chunk.chunk_id for hit in vector],
            }
        )
        by_id: dict[str, Chunk] = {
            hit.chunk.chunk_id: hit.chunk for hit in (*lexical, *vector)
        }
        fused = tuple(
            ScoredChunk(chunk=by_id[chunk_id], score=score)
            for chunk_id, score in fused_ids[:limit]
        )
        return ChannelResults(lexical=lexical, vector=vector, fused=fused)
