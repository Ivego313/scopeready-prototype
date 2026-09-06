"""Turning text into vectors, behind an interface narrow enough to swap.

Two implementations ship. The neural one is the point; the stub is not a test
double but a full mode of operation — it needs no torch, runs everywhere, and is
what lets the entire pipeline be exercised on a laptop with no downloads. Tests
that must not pull 130 MB run on it, and so does anyone who wants to see the
mechanism before deciding whether the mechanism is worth 2.5 GB of dependencies.

`max_tokens` and `count_tokens` are part of the interface because the chunker
enforces the window, and it can only do that honestly if the number comes from
the object that will actually encode the text. A table keyed by model name drifts
from the model; the model does not drift from itself.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]

# float32, C-contiguous, L2-normalized. Fixed here because the stored bytes are
# read back by length, and a float64 write would reshape into plausible garbage.
DTYPE: Final = np.float32

_WORDS = re.compile(r"\w+|[^\w\s]")


class Embedder(Protocol):
    """Encodes documents and queries into one comparable vector space."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def encode_documents(self, texts: list[str]) -> Vector: ...

    def encode_query(self, text: str) -> Vector: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What a model is, before it is loaded."""

    key: str
    repo: str
    dimensions: int
    max_tokens: int
    # bge asks for an instruction on the query side and nothing on the passage
    # side. Encoding a query as if it were a passage costs recall silently, so
    # the asymmetry lives inside the embedder rather than at every call site.
    query_prefix: str = ""
    note: str = ""


MODEL_SPECS: Final[dict[str, ModelSpec]] = {
    "bge-small": ModelSpec(
        key="bge-small",
        repo="BAAI/bge-small-en-v1.5",
        dimensions=384,
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
        note="33M parameters, fast on CPU, ~130 MB of weights",
    ),
    "gte-modernbert": ModelSpec(
        key="gte-modernbert",
        repo="Alibaba-NLP/gte-modernbert-base",
        dimensions=768,
        max_tokens=8192,
        note="149M parameters, 8k window, ~600 MB of weights",
    ),
}

STUB_NAME: Final = "stub-hash-256"
STUB_DIMENSIONS: Final = 256
# Deliberately the same window as bge-small so that switching between the stub
# and the small model does not change how documents were chunked.
STUB_MAX_TOKENS: Final = 512


class StubEmbedder:
    """A deterministic hash embedder: character n-grams into a fixed space.

    It measures surface overlap, not meaning. That is a limitation and also the
    reason it is useful: it makes the difference a real model buys visible on
    the same corpus and the same query, instead of asserted.
    """

    @property
    def name(self) -> str:
        return STUB_NAME

    @property
    def dimensions(self) -> int:
        return STUB_DIMENSIONS

    @property
    def max_tokens(self) -> int:
        return STUB_MAX_TOKENS

    def count_tokens(self, text: str) -> int:
        # Words and standalone punctuation, which tracks a wordpiece count far
        # better than characters divided by four does.
        return len(_WORDS.findall(text))

    def encode_documents(self, texts: list[str]) -> Vector:
        if not texts:
            return np.zeros((0, STUB_DIMENSIONS), dtype=DTYPE)
        stacked = np.vstack([self._encode(text) for text in texts])
        matrix: Vector = np.ascontiguousarray(stacked, dtype=DTYPE)
        return matrix

    def encode_query(self, text: str) -> Vector:
        return self._encode(text)

    def _encode(self, text: str) -> Vector:
        vector = np.zeros(STUB_DIMENSIONS, dtype=DTYPE)
        folded = " ".join(text.lower().split())
        for size in (3, 4, 5):
            for start in range(max(len(folded) - size + 1, 0)):
                gram = folded[start : start + size]
                # blake2b rather than hash(): the built-in is salted per process
                # and the same text would embed differently on the next run.
                digest = hashlib.blake2b(gram.encode(), digest_size=4).digest()
                vector[int.from_bytes(digest) % STUB_DIMENSIONS] += 1.0
        return _l2_normalize(vector)


class SentenceTransformerEmbedder:
    """A local sentence-transformers model, loaded once and reused."""

    def __init__(self, spec: ModelSpec) -> None:
        # Imported here rather than at module scope so that the stub path never
        # needs torch installed.
        from sentence_transformers import SentenceTransformer

        self._spec = spec
        self._model = SentenceTransformer(spec.repo)
        # Renamed in sentence-transformers 6; the old name still works and
        # warns. Preferring the new one keeps the install quiet on both.
        report_dimension = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        actual = int(report_dimension() or 0)
        if actual != spec.dimensions:
            msg = (
                f"{spec.repo} reports {actual} dimensions, the spec says "
                f"{spec.dimensions}; the spec is what the stored vectors were "
                "written against"
            )
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return self._spec.repo

    @property
    def dimensions(self) -> int:
        return self._spec.dimensions

    @property
    def max_tokens(self) -> int:
        return self._spec.max_tokens

    def count_tokens(self, text: str) -> int:
        # With the special tokens: they are two of the window, and counting
        # without them lets a 512-token chunk arrive as 514 and be truncated
        # silently, which is the failure this whole check exists to prevent.
        return len(self._model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def encode_documents(self, texts: list[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=DTYPE)
        encoded = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        matrix: Vector = np.ascontiguousarray(encoded, dtype=DTYPE)
        return matrix

    def encode_query(self, text: str) -> Vector:
        encoded = self._model.encode(
            [self._spec.query_prefix + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector: Vector = np.ascontiguousarray(encoded[0], dtype=DTYPE)
        return vector


def build_embedder(key: str) -> Embedder:
    if key == "stub":
        return StubEmbedder()
    spec = MODEL_SPECS.get(key)
    if spec is None:
        known = ", ".join(["stub", *sorted(MODEL_SPECS)])
        msg = f"unknown embedder {key!r}; known embedders are {known}"
        raise ValueError(msg)
    return SentenceTransformerEmbedder(spec)


def _l2_normalize(vector: Vector) -> Vector:
    norm = float(np.linalg.norm(vector))
    # A zero vector has no direction; leaving it at zero makes its cosine
    # similarity to everything zero, which is the honest answer.
    return vector if norm == 0.0 else (vector / norm).astype(DTYPE)


def cosine_similarity(matrix: Vector, query: Vector) -> NDArray[np.float32]:
    """Similarity of every row to the query.

    A plain dot product, because every vector this module produces is already
    L2-normalized. Normalizing twice is harmless; normalizing neither is a
    ranking that silently favours long text.
    """
    return np.asarray(matrix @ query, dtype=DTYPE)
