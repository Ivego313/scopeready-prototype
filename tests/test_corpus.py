"""Tests for the SQLite corpus and the lexical channel.

Two properties matter more than the rest. Records must come back as the models
that went in, because a store built on serialized dumps would work until the
first read. And the full-text index must stay in step with the table it shadows,
because a stale index does not fail — it quietly returns less.
"""

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import HttpUrl

from scopeready.chunking import HeuristicBudget, chunk_document
from scopeready.config import FIXTURE_CORPUS_DIR
from scopeready.corpus import SqliteCorpus, fts5_available
from scopeready.ingest import read_directory
from scopeready.models import (
    AuthorSide,
    Chunk,
    CorpusRole,
    Document,
    Provenance,
    SourceKind,
)
from scopeready.store import IndexedChunk
from scopeready.text import build_context_text, normalize_for_index


@pytest.fixture
def store() -> SqliteCorpus:
    return SqliteCorpus.in_memory()


@pytest.fixture
def loaded(store: SqliteCorpus) -> SqliteCorpus:
    budget = HeuristicBudget()
    documents = read_directory(FIXTURE_CORPUS_DIR).documents
    store.add_documents(documents)
    for document in documents:
        chunks, report = chunk_document(document, budget)
        store.add_chunks(
            [
                IndexedChunk(
                    chunk=chunk,
                    context_text=build_context_text(
                        document.provenance.title, chunk.heading_path
                    ),
                    index_text=normalize_for_index(chunk.text),
                    token_count=budget.count_tokens(chunk.text),
                    oversplit=report.oversplit_chunks > 0,
                )
                for chunk in chunks
            ]
        )
    return store


def _document() -> Document:
    return Document(
        provenance=Provenance(
            source_kind=SourceKind.WIKI,
            source_name="confluence",
            source_id="1001",
            title="Data retention policy",
            url=HttpUrl("https://example.invalid/wiki/1001"),
            author="Dana Reyes",
            author_side=AuthorSide.CUSTOMER,
            created_at=datetime(2025, 11, 20, 12, tzinfo=UTC),
            thread_id=None,
        ),
        corpus_role=CorpusRole.CONTEXT,
        item_label="Page",
        status="Published",
        text="Deleted accounts are removed within 30 days.",
        parent_source_id=None,
    )


# --- the architectural boundary ---------------------------------------------


def test_only_the_sqlite_module_knows_about_sqlite() -> None:
    # The production path is Postgres with pgvector and per-tenant isolation,
    # and that swap stays cheap only while the dependency is confined.
    offenders = []
    for path in sorted(Path("src/scopeready").glob("*.py")):
        if path.name == "corpus.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imported = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.split(".")[0] == "sqlite3" for name in imported):
                offenders.append(path.name)
    assert offenders == []


def test_fts5_is_available() -> None:
    # Without it the lexical half of retrieval is missing entirely, so this is
    # checked at startup rather than at the first query.
    assert fts5_available()


# --- records survive the round trip -----------------------------------------


def test_a_document_comes_back_as_the_model_that_went_in(
    store: SqliteCorpus,
) -> None:
    original = _document()
    store.add_documents([original])
    assert store.document("confluence:1001") == original


def test_a_chunk_keeps_its_heading_path_and_verbatim_text(
    store: SqliteCorpus,
) -> None:
    store.add_documents([_document()])
    chunk = Chunk(
        doc_id="confluence:1001",
        ordinal=0,
        heading_path=("Deletion", "On request"),
        text="Removed  within **30** days.",
    )
    store.add_chunks(
        [IndexedChunk(chunk=chunk, context_text="ctx", index_text="idx", token_count=7)]
    )
    stored = store.chunk("confluence:1001#0")
    assert stored == chunk
    # Verbatim, not normalized: `Evidence.from_chunk` checks quotes against this.
    assert stored is not None and stored.text == "Removed  within **30** days."


def test_a_document_id_may_not_be_written_twice(store: SqliteCorpus) -> None:
    store.add_documents([_document()])
    with pytest.raises(Exception, match="UNIQUE"):
        store.add_documents([_document()])


def test_stats_are_split_by_role(loaded: SqliteCorpus) -> None:
    stats = loaded.stats()
    assert (stats.requirement_documents, stats.context_documents) == (7, 3)
    assert stats.documents == 10 and stats.chunks == 23


# --- the lexical channel ----------------------------------------------------


def test_the_stemmer_matches_across_word_forms(loaded: SqliteCorpus) -> None:
    hits = loaded.search_lexical("deleting", limit=5)
    assert any("deletion" in hit.chunk.text.lower() for hit in hits)


def test_a_higher_score_means_more_relevant(loaded: SqliteCorpus) -> None:
    hits = loaded.search_lexical("invitation", limit=5)
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )


def test_a_paraphrase_is_missed_and_that_is_the_point(loaded: SqliteCorpus) -> None:
    # `jira:SCOPE-2#1` is precisely about who may do what, but it says "access
    # levels" where the query says "permissions matrix". No amount of stemming
    # bridges that, which is what the vector channel is for.
    assert loaded.search_lexical("permissions matrix", limit=5) == ()
    direct = loaded.search_lexical("access levels", limit=5)
    assert any(hit.chunk.doc_id == "jira:SCOPE-2" for hit in direct)


@pytest.mark.parametrize(
    "query",
    ['who owns the "key"', "NEAR(a b)", "rate-limit^2", "a AND OR b", "*", '""'],
)
def test_a_query_cannot_be_read_as_syntax(loaded: SqliteCorpus, query: str) -> None:
    loaded.search_lexical(query, limit=5)


def test_a_query_with_no_terms_returns_nothing(loaded: SqliteCorpus) -> None:
    # Nothing, not everything: an empty filter that matches the whole corpus
    # would look like a very successful search.
    assert loaded.search_lexical("!!!", limit=5) == ()


def test_a_title_only_document_is_still_findable(store: SqliteCorpus) -> None:
    store.add_documents([_document()])
    store.add_chunks(
        [
            IndexedChunk(
                chunk=Chunk(doc_id="confluence:1001", ordinal=0, text="Data retention"),
                context_text=build_context_text("Data retention policy", ()),
                index_text="",
                token_count=3,
            )
        ]
    )
    assert store.search_lexical("retention", limit=5)


def test_results_are_ordered_deterministically(loaded: SqliteCorpus) -> None:
    first = [hit.chunk.chunk_id for hit in loaded.search_lexical("workspace", limit=8)]
    second = [hit.chunk.chunk_id for hit in loaded.search_lexical("workspace", limit=8)]
    assert first == second


# --- the index stays in step ------------------------------------------------


def test_the_index_matches_the_table(loaded: SqliteCorpus) -> None:
    loaded.integrity_check()


def test_deleting_a_document_leaves_no_orphan_index_rows(
    loaded: SqliteCorpus,
) -> None:
    connection = loaded._connection  # noqa: SLF001 - the invariant is internal
    with connection:
        connection.execute("DELETE FROM documents WHERE doc_id = 'jira:SCOPE-2'")
    loaded.integrity_check()
    assert all(
        hit.chunk.doc_id != "jira:SCOPE-2"
        for hit in loaded.search_lexical("access levels", limit=10)
    )


def test_unembedded_chunks_are_the_work_queue(loaded: SqliteCorpus) -> None:
    pending = list(loaded.iter_unembedded())
    assert len(pending) == 23
    assert all(text.strip() for _, text in pending)
