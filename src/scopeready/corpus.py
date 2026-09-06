"""The SQLite implementation of the corpus store.

One file holds the documents, the chunks, a full-text index over them and, from
the next layer up, their vectors — small enough to attach to a ticket, which is
what makes a disagreement about a verdict reproducible.

Three things here are decisions rather than plumbing:

Records are written column by column, never as a serialized model. `Document`
and `Chunk` carry computed fields and forbid extra keys, so a dump of one cannot
be validated back into one; a store built on `model_dump` would work until the
first read.

The full-text index is an external-content FTS5 table kept in step by triggers,
so the database enforces the invariant instead of every future write path
remembering a second statement.

The lexical index reads two columns. The body is indexed in its normalized form,
and the document title with its heading path is indexed separately — a ticket
whose whole content is its title is legitimate, and without the second column it
would be invisible to retrieval and therefore uncitable.
"""

import json
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from scopeready.models import (
    AuthorSide,
    Chunk,
    CorpusRole,
    CorpusStats,
    Document,
    Provenance,
    SourceKind,
)
from scopeready.store import IndexedChunk, ScoredChunk
from scopeready.text import to_fts_match

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Exactly one embedding space per file. Mixing two would make the stored
-- vectors incomparable while every query still returned a ranked list.
CREATE TABLE embedding_space (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    model_name  TEXT    NOT NULL,
    dimensions  INTEGER NOT NULL CHECK (dimensions > 0),
    max_tokens  INTEGER NOT NULL CHECK (max_tokens > 0),
    created_at  TEXT    NOT NULL
);

CREATE TABLE documents (
    doc_pk           INTEGER PRIMARY KEY,
    source_kind      TEXT    NOT NULL,
    source_name      TEXT    NOT NULL,
    source_id        TEXT    NOT NULL,
    title            TEXT    NOT NULL,
    url              TEXT,
    author           TEXT,
    author_side      TEXT    NOT NULL,
    created_at       TEXT,
    updated_at       TEXT,
    thread_id        TEXT,
    corpus_role      TEXT    NOT NULL,
    item_label       TEXT,
    status           TEXT,
    text             TEXT    NOT NULL,
    parent_source_id TEXT,
    doc_id TEXT GENERATED ALWAYS AS (source_name || ':' || source_id) STORED
);
CREATE UNIQUE INDEX documents_doc_id ON documents(doc_id);
CREATE INDEX documents_role ON documents(corpus_role);

CREATE TABLE chunks (
    chunk_pk     INTEGER PRIMARY KEY,
    doc_id       TEXT    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL CHECK (ordinal >= 0),
    chunk_id     TEXT    NOT NULL UNIQUE,
    heading_path TEXT    NOT NULL CHECK (json_valid(heading_path)),
    text         TEXT    NOT NULL,
    context_text TEXT    NOT NULL,
    index_text   TEXT    NOT NULL,
    token_count  INTEGER NOT NULL,
    oversplit    INTEGER NOT NULL DEFAULT 0 CHECK (oversplit IN (0, 1)),
    UNIQUE (doc_id, ordinal)
);
CREATE INDEX chunks_doc ON chunks(doc_id, ordinal);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    context_text,
    index_text,
    content       = 'chunks',
    content_rowid = 'chunk_pk',
    tokenize      = 'porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, context_text, index_text)
    VALUES (new.chunk_pk, new.context_text, new.index_text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, context_text, index_text)
    VALUES ('delete', old.chunk_pk, old.context_text, old.index_text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, context_text, index_text)
    VALUES ('delete', old.chunk_pk, old.context_text, old.index_text);
    INSERT INTO chunks_fts(rowid, context_text, index_text)
    VALUES (new.chunk_pk, new.context_text, new.index_text);
END;

CREATE TABLE embeddings (
    chunk_pk INTEGER PRIMARY KEY REFERENCES chunks(chunk_pk) ON DELETE CASCADE,
    vector   BLOB NOT NULL
);
"""


class CorpusError(RuntimeError):
    """A corpus file that cannot answer a question correctly."""


def fts5_available() -> bool:
    """Whether the interpreter's SQLite was built with FTS5.

    Worth checking rather than assuming: the whole lexical channel is missing
    without it, and a build without FTS5 fails at the first query rather than at
    startup.
    """
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    except sqlite3.OperationalError:
        return False
    else:
        return True
    finally:
        connection.close()


class SqliteCorpus:
    """A corpus in one SQLite file."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path, *, reset: bool = False) -> Self:
        if reset and path.exists():
            for suffix in ("", "-wal", "-shm"):
                path.with_name(path.name + suffix).unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        if not fts5_available():
            msg = (
                "this Python's SQLite was built without FTS5, so the lexical "
                "half of retrieval cannot work"
            )
            raise CorpusError(msg)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        store = cls(connection)
        if fresh:
            store._create_schema()
        return store

    @classmethod
    def in_memory(cls) -> Self:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        store = cls(connection)
        store._create_schema()
        return store

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- writing ------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        rows = [
            (
                document.provenance.source_kind.value,
                document.provenance.source_name,
                document.provenance.source_id,
                document.provenance.title,
                str(document.provenance.url) if document.provenance.url else None,
                document.provenance.author,
                document.provenance.author_side.value,
                _to_iso(document.provenance.created_at),
                _to_iso(document.provenance.updated_at),
                document.provenance.thread_id,
                document.corpus_role.value,
                document.item_label,
                document.status,
                document.text,
                document.parent_source_id,
            )
            for document in documents
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO documents(
                    source_kind, source_name, source_id, title, url, author,
                    author_side, created_at, updated_at, thread_id,
                    corpus_role, item_label, status, text, parent_source_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )

    def add_chunks(self, chunks: Sequence[IndexedChunk]) -> None:
        rows = [
            (
                item.chunk.doc_id,
                item.chunk.ordinal,
                item.chunk.chunk_id,
                json.dumps(list(item.chunk.heading_path)),
                item.chunk.text,
                item.context_text,
                item.index_text,
                item.token_count,
                int(item.oversplit),
            )
            for item in chunks
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO chunks(
                    doc_id, ordinal, chunk_id, heading_path, text,
                    context_text, index_text, token_count, oversplit)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )

    # --- reading ------------------------------------------------------------

    def documents(self) -> tuple[Document, ...]:
        rows = self._connection.execute(
            "SELECT * FROM documents ORDER BY doc_id"
        ).fetchall()
        return tuple(_document_of(row) for row in rows)

    def document(self, doc_id: str) -> Document | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return None if row is None else _document_of(row)

    def chunks(self) -> tuple[Chunk, ...]:
        rows = self._connection.execute(
            "SELECT * FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
        return tuple(_chunk_of(row) for row in rows)

    def chunk(self, chunk_id: str) -> Chunk | None:
        row = self._connection.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return None if row is None else _chunk_of(row)

    def iter_unembedded(self) -> Iterator[tuple[int, str]]:
        """Chunk primary keys with no vector yet, with the text to encode.

        A chunk without a vector is a legitimate state between `ingest` and
        `index`, which is why the vectors live in their own table.
        """
        rows = self._connection.execute(
            """
            SELECT c.chunk_pk, c.context_text, c.index_text
            FROM chunks c LEFT JOIN embeddings e ON e.chunk_pk = c.chunk_pk
            WHERE e.chunk_pk IS NULL
            ORDER BY c.chunk_pk
            """
        ).fetchall()
        for row in rows:
            yield int(row["chunk_pk"]), f"{row['context_text']}\n\n{row['index_text']}"

    def stats(self) -> CorpusStats:
        counts = self._connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM documents WHERE corpus_role = 'requirement'),
              (SELECT COUNT(*) FROM documents WHERE corpus_role = 'context'),
              (SELECT COUNT(*) FROM chunks)
            """
        ).fetchone()
        return CorpusStats(
            requirement_documents=counts[0],
            context_documents=counts[1],
            chunks=counts[2],
        )

    # --- lexical search -----------------------------------------------------

    def search_lexical(self, query: str, *, limit: int) -> tuple[ScoredChunk, ...]:
        """Rank chunks by BM25 over the two indexed columns.

        The query is rewritten into an expression whose operators are ours, so
        a phrasing containing a quote, a hyphen or the word NEAR is a search and
        not a syntax error. A query with no searchable term returns nothing
        rather than everything.
        """
        match = to_fts_match(query)
        if match is None:
            return ()
        rows = self._connection.execute(
            """
            SELECT c.*, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_pk = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank, c.chunk_id
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        # bm25() returns a negative number, smaller meaning better. It is turned
        # around here so that every channel in this codebase means "higher is
        # more relevant", which is what the fusion step assumes.
        return tuple(
            ScoredChunk(chunk=_chunk_of(row), score=-float(row["rank"])) for row in rows
        )

    def integrity_check(self) -> None:
        """Assert the full-text index still matches the table behind it."""
        try:
            self._connection.execute(
                "INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')"
            )
        except sqlite3.DatabaseError as error:
            msg = f"the full-text index is out of step with the chunks: {error}"
            raise CorpusError(msg) from error


def _to_iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _from_iso(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _document_of(row: Any) -> Document:
    return Document(
        provenance=Provenance(
            source_kind=SourceKind(row["source_kind"]),
            source_name=row["source_name"],
            source_id=row["source_id"],
            title=row["title"],
            url=row["url"],
            author=row["author"],
            author_side=AuthorSide(row["author_side"]),
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
            thread_id=row["thread_id"],
        ),
        corpus_role=CorpusRole(row["corpus_role"]),
        item_label=row["item_label"],
        status=row["status"],
        text=row["text"],
        parent_source_id=row["parent_source_id"],
    )


def _chunk_of(row: Any) -> Chunk:
    return Chunk(
        doc_id=row["doc_id"],
        ordinal=row["ordinal"],
        heading_path=tuple(json.loads(row["heading_path"])),
        text=row["text"],
    )
