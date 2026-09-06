"""What the pipeline is allowed to ask of a corpus, and nothing more.

The interface is narrow on purpose. A prototype runs on one SQLite file, but the
production path is Postgres with pgvector and row-level isolation per tenant,
and that swap is only cheap while nothing above this line knows which one it is
talking to. The rule is checkable rather than aspirational: no module except
`corpus.py` imports `sqlite3`, and a test asserts it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from scopeready.models import Chunk, CorpusStats, Document


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A chunk together with the two derived texts the index is built from.

    They are computed once, at write time, because the lexical index and the
    embedder must see the same normalization the retrieval side assumes, and
    recomputing it at read time is how the two drift apart.
    """

    chunk: Chunk
    context_text: str
    index_text: str
    token_count: int
    oversplit: bool = False


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class CorpusStore(Protocol):
    """Add, read, search lexically, search by vector, count."""

    def add_documents(self, documents: Sequence[Document]) -> None: ...

    def add_chunks(self, chunks: Sequence[IndexedChunk]) -> None: ...

    def documents(self) -> tuple[Document, ...]: ...

    def chunks(self) -> tuple[Chunk, ...]: ...

    def chunk(self, chunk_id: str) -> Chunk | None: ...

    def search_lexical(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]: ...

    def stats(self) -> CorpusStats: ...
