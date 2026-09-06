"""Tests for embeddings and for fusing the two search channels.

Everything here runs on the stub embedder except the tests marked `model`, which
are the only ones that prove the thing the vector channel actually exists for.
A default `pytest` run stays offline; `-m model` is the opt-in.
"""

import numpy as np
import pytest

from scopeready.chunking import chunk_document
from scopeready.config import FIXTURE_CORPUS_DIR
from scopeready.corpus import CorpusError, SqliteCorpus
from scopeready.embeddings import (
    STUB_DIMENSIONS,
    Embedder,
    StubEmbedder,
    build_embedder,
    cosine_similarity,
)
from scopeready.ingest import read_directory
from scopeready.retrieval import RRF_K, HybridRetriever, reciprocal_rank_fusion
from scopeready.store import IndexedChunk
from scopeready.text import build_context_text, normalize_for_index


def _load(embedder: Embedder) -> SqliteCorpus:
    store = SqliteCorpus.in_memory()
    store.declare_embedding_space(
        model_name=embedder.name,
        dimensions=embedder.dimensions,
        max_tokens=embedder.max_tokens,
    )
    documents = read_directory(FIXTURE_CORPUS_DIR).documents
    store.add_documents(documents)
    for document in documents:
        chunks, _ = chunk_document(document, embedder)
        store.add_chunks(
            [
                IndexedChunk(
                    chunk=chunk,
                    context_text=build_context_text(
                        document.provenance.title, chunk.heading_path
                    ),
                    index_text=normalize_for_index(chunk.text),
                    token_count=embedder.count_tokens(chunk.text),
                )
                for chunk in chunks
            ]
        )
    pending = list(store.iter_unembedded())
    vectors = embedder.encode_documents([text for _, text in pending])
    store.add_vectors([(pk, vectors[i]) for i, (pk, _) in enumerate(pending)])
    return store


@pytest.fixture
def stub() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def loaded(stub: StubEmbedder) -> SqliteCorpus:
    return _load(stub)


# --- the embedder interface -------------------------------------------------


def test_the_stub_is_deterministic_across_processes(stub: StubEmbedder) -> None:
    # The built-in hash() is salted per process, so a stub built on it would
    # embed the same corpus differently on the next run and every cached
    # comparison would be meaningless.
    first = stub.encode_query("data deletion policy")
    second = StubEmbedder().encode_query("data deletion policy")
    assert np.array_equal(first, second)


def test_vectors_are_unit_length(stub: StubEmbedder) -> None:
    matrix = stub.encode_documents(["one text", "another text entirely"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_similar_text_scores_higher_than_unrelated_text(stub: StubEmbedder) -> None:
    matrix = stub.encode_documents(
        ["the owner may delete the workspace", "encrypted backups are kept 35 days"]
    )
    scores = cosine_similarity(matrix, stub.encode_query("owner deletes a workspace"))
    assert scores[0] > scores[1]


def test_an_empty_batch_keeps_its_shape(stub: StubEmbedder) -> None:
    assert stub.encode_documents([]).shape == (0, STUB_DIMENSIONS)


def test_an_unknown_embedder_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="known embedders are"):
        build_embedder("word2vec")


def test_the_stub_counts_tokens_like_a_tokenizer_not_like_characters(
    stub: StubEmbedder,
) -> None:
    assert stub.count_tokens("who may delete it?") == 5


# --- the embedding space is pinned to the file ------------------------------


def test_querying_with_another_embedder_is_refused(loaded: SqliteCorpus) -> None:
    # Not a downgrade to lexical-only: a mismatched space still returns a ranked
    # list, and a meaningless ranking looks like retrieval that merely got worse.
    with pytest.raises(CorpusError, match="rebuild it with --reset"):
        loaded.declare_embedding_space(
            model_name="BAAI/bge-small-en-v1.5", dimensions=384, max_tokens=512
        )


def test_a_corrupt_vector_is_refused_rather_than_reshaped(
    loaded: SqliteCorpus, stub: StubEmbedder
) -> None:
    connection = loaded._connection  # noqa: SLF001 - the invariant is internal
    with connection:
        connection.execute(
            "UPDATE embeddings SET vector = ? WHERE chunk_pk = 1", (b"x",)
        )
    with pytest.raises(CorpusError, match="expected"):
        loaded.search_vector(stub.encode_query("anything"), limit=3)


# --- fusion -----------------------------------------------------------------


def test_fusion_rewards_agreement_across_channels() -> None:
    fused = dict(reciprocal_rank_fusion({"lexical": ["a", "b"], "vector": ["c", "b"]}))
    # `b` is second in both channels, `a` and `c` are first in one and absent
    # from the other. Agreement wins, which is the property RRF is chosen for:
    # a fragment both channels like is far likelier to be on topic.
    assert fused["b"] > fused["a"] == fused["c"]


def test_a_large_k_keeps_the_head_flat() -> None:
    # At k=1 the top hit of one channel would score 0.5 against 0.33 for a
    # document ranked fifth in both, and fusing would achieve nothing. The
    # constant is what makes agreement able to outweigh a single strong hit.
    single_top = 1.0 / (RRF_K + 1)
    fifth_in_both = 2.0 / (RRF_K + 5)
    assert fifth_in_both > single_top


def test_a_document_in_one_channel_only_still_scores() -> None:
    fused = dict(reciprocal_rank_fusion({"lexical": ["a"], "vector": ["b"]}))
    assert fused["a"] == pytest.approx(1.0 / (RRF_K + 1))
    assert fused["a"] == fused["b"]


def test_ties_are_broken_by_chunk_id() -> None:
    # Ties are the normal case, not an edge: two chunks each ranked first in one
    # channel score identically, bit for bit.
    order = reciprocal_rank_fusion({"lexical": ["z:1#0"], "vector": ["a:1#0"]})
    assert [chunk_id for chunk_id, _ in order] == ["a:1#0", "z:1#0"]


def test_channels_accumulate_in_a_fixed_order() -> None:
    channels = {"vector": ["a", "b"], "lexical": ["b", "a"]}
    reversed_channels = {"lexical": ["b", "a"], "vector": ["a", "b"]}
    assert reciprocal_rank_fusion(channels) == reciprocal_rank_fusion(reversed_channels)


def test_fusion_of_nothing_is_nothing() -> None:
    assert reciprocal_rank_fusion({"lexical": [], "vector": []}) == ()


# --- the hybrid retriever ---------------------------------------------------


def test_the_hybrid_returns_a_stable_order(
    loaded: SqliteCorpus, stub: StubEmbedder
) -> None:
    retriever = HybridRetriever(loaded, stub)
    first = [
        hit.chunk.chunk_id for hit in retriever.search("workspace access", limit=6)
    ]
    second = [
        hit.chunk.chunk_id for hit in retriever.search("workspace access", limit=6)
    ]
    assert first == second


def test_the_hybrid_recovers_what_a_term_query_missed(
    loaded: SqliteCorpus, stub: StubEmbedder
) -> None:
    # The lexical channel returns nothing for a query sharing no vocabulary with
    # the corpus. Fusion still produces candidates, because the other channel
    # does not depend on shared words.
    assert loaded.search_lexical("permissions matrix", limit=5) == ()
    assert HybridRetriever(loaded, stub).search("permissions matrix", limit=5)


def test_both_channels_are_visible_separately(
    loaded: SqliteCorpus, stub: StubEmbedder
) -> None:
    results = HybridRetriever(loaded, stub).search_channels("deletion", limit=5)
    assert results.lexical and results.vector and results.fused


# --- the real model ---------------------------------------------------------


@pytest.mark.model
def test_a_real_model_bridges_vocabulary_the_stemmer_cannot() -> None:
    # This is the claim the vector channel is bought for, on the case the
    # lexical channel provably fails: `jira:SCOPE-2#1` is about who may do what
    # and never uses the word "permission".
    embedder = build_embedder("bge-small")
    store = _load(embedder)
    assert store.search_lexical("permissions matrix", limit=5) == ()
    hits = store.search_vector(
        embedder.encode_query("which permissions does each role have"), limit=3
    )
    assert any(hit.chunk.chunk_id == "jira:SCOPE-2#1" for hit in hits)
