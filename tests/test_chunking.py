"""Tests for text normalization, the chunker and the ingest adapter.

The chunker is where a document can be silently lost or silently truncated, so
most of these assert that nothing disappears rather than that the output looks
tidy.
"""

from pathlib import Path

import pytest

from scopeready.chunking import HeuristicBudget, chunk_document, chunk_documents
from scopeready.config import FIXTURE_CORPUS_DIR
from scopeready.ingest import IngestError, read_directory, read_file
from scopeready.models import CorpusRole, Document, Provenance, SourceKind
from scopeready.text import (
    build_context_text,
    normalize_for_index,
    normalize_for_match,
    to_fts_match,
)


def _document(text: str, title: str = "Access levels") -> Document:
    return Document(
        provenance=Provenance(
            source_kind=SourceKind.TRACKER,
            source_name="jira",
            source_id="SCOPE-2",
            title=title,
        ),
        corpus_role=CorpusRole.REQUIREMENT,
        text=text,
    )


# --- normalization ----------------------------------------------------------


def test_link_targets_do_not_reach_the_index() -> None:
    # Otherwise every wiki chunk carries the host and the page id into the term
    # statistics and distorts what BM25 considers rare.
    indexed = normalize_for_index("See [export policy](https://wiki.corp/pages/12345)")
    assert "wiki.corp" not in indexed
    assert "export policy" in indexed


def test_context_text_carries_the_title_and_the_section() -> None:
    assert build_context_text("Access levels", ("Levels",)) == "Access levels / Levels"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("data deletion", '"data" OR "deletion"'),
        ('who owns the "key"', '"owns" OR "key"'),
        ("NEAR(a b)", '"NEAR" OR "b"'),
        ("rate-limit", '"rate" OR "limit"'),
        ("   ", None),
        ("!!!", None),
        # Only function words: no lexical signal at all, so the lexical channel
        # stands aside rather than matching the whole corpus through "the".
        ("what does it do", None),
    ],
)
def test_a_query_cannot_carry_fts_syntax(query: str, expected: str | None) -> None:
    assert to_fts_match(query) == expected


def test_function_words_do_not_reach_the_lexical_index() -> None:
    # They match nearly every chunk, and the fusion step reads ranks rather than
    # scores — so a list of near-zero hits arrives looking as authoritative as a
    # list of real ones.
    match = to_fts_match("which permissions does each role have")
    assert match == '"permissions" OR "role"'


def test_presentation_differences_do_not_break_a_quote() -> None:
    chunk = "The owner — and only the owner — may **delete** it."
    quote = "The owner - and only the owner - may delete it."
    assert normalize_for_match(quote) in normalize_for_match(chunk)


# --- chunking ---------------------------------------------------------------


def test_a_heading_change_closes_a_chunk() -> None:
    chunks, _ = chunk_document(
        _document("## Levels\n\nThree.\n\n## Acceptance criteria\n\nOne."),
        HeuristicBudget(),
    )
    assert [chunk.heading_path for chunk in chunks] == [
        ("Levels",),
        ("Acceptance criteria",),
    ]


def test_sibling_sections_do_not_nest_when_there_is_no_top_heading() -> None:
    # Real exports usually have no `#`, because the title lives in a field.
    chunks, _ = chunk_document(
        _document("## A\n\none\n\n### A1\n\ntwo\n\n## B\n\nthree"), HeuristicBudget()
    )
    assert [chunk.heading_path for chunk in chunks] == [("A",), ("A", "A1"), ("B",)]


def test_heading_text_is_not_quotable() -> None:
    # A heading inside the body would eventually be quoted by the model, and the
    # verbatim check on `Chunk.text` would then reject a correct answer.
    chunks, _ = chunk_document(
        _document("## Levels\n\nThree levels."), HeuristicBudget()
    )
    assert chunks[0].text == "Three levels."


def test_a_list_is_never_cut_through_an_item() -> None:
    items = "\n".join(f"- item number {index} with some words" for index in range(40))
    chunks, report = chunk_document(_document(items), HeuristicBudget(max_tokens=64))
    assert len(chunks) > 1
    for chunk in chunks:
        assert all(line.startswith("- ") for line in chunk.text.splitlines())
    assert report.oversplit_chunks == 0
    rejoined = "\n".join(chunk.text for chunk in chunks)
    assert rejoined.count("item number") == 40


def test_a_table_fragment_keeps_its_header() -> None:
    rows = "\n".join(f"| role{index} | may edit | may delete |" for index in range(30))
    body = "| role | edit | delete |\n| --- | --- | --- |\n" + rows
    chunks, _ = chunk_document(_document(body), HeuristicBudget(max_tokens=48))
    assert len(chunks) > 1
    assert all(chunk.text.startswith("| role | edit | delete |") for chunk in chunks)


def test_a_code_block_stays_valid_markdown_after_splitting() -> None:
    body = "```json\n" + "\n".join(f'  "field{i}": {i},' for i in range(40)) + "\n```"
    chunks, _ = chunk_document(_document(body), HeuristicBudget(max_tokens=48))
    assert len(chunks) > 1
    assert all(chunk.text.startswith("```json") for chunk in chunks)


def test_an_indivisible_unit_is_split_rather_than_truncated() -> None:
    # Truncation would discard text that may be the only place a requirement is
    # stated, and the resulting gap could never be found.
    body = "- " + " ".join(f"word{index}" for index in range(400))
    chunks, report = chunk_document(_document(body), HeuristicBudget(max_tokens=64))
    assert report.oversplit_chunks >= 1
    rejoined = " ".join(chunk.text for chunk in chunks)
    assert "word0" in rejoined and "word399" in rejoined


def test_no_chunk_exceeds_the_window() -> None:
    budget = HeuristicBudget(max_tokens=80)
    documents = read_directory(FIXTURE_CORPUS_DIR).documents
    chunks, _ = chunk_documents(documents, budget)
    by_id = {document.doc_id: document for document in documents}
    for chunk in chunks:
        encoded = build_context_text(
            by_id[chunk.doc_id].provenance.title, chunk.heading_path
        )
        assert budget.count_tokens(f"{encoded}\n\n{chunk.text}") <= budget.max_tokens


def test_a_document_with_no_body_is_still_citable() -> None:
    chunks, report = chunk_document(_document("", title="Ship it"), HeuristicBudget())
    assert [chunk.text for chunk in chunks] == ["Ship it"]
    assert report.title_only_documents == 1


def test_a_short_statement_is_not_dropped() -> None:
    chunks, _ = chunk_document(
        _document("## Out of scope\n\nNothing."), HeuristicBudget()
    )
    assert any("Nothing." in chunk.text for chunk in chunks)


def test_chunk_ids_are_stable_across_runs() -> None:
    body = "## A\n\none\n\n## B\n\ntwo"
    first, _ = chunk_document(_document(body), HeuristicBudget())
    second, _ = chunk_document(_document(body), HeuristicBudget())
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


# --- ingest -----------------------------------------------------------------


def test_the_fixture_corpus_reads() -> None:
    result = read_directory(FIXTURE_CORPUS_DIR)
    assert (result.requirements, result.contexts) == (7, 3)
    assert {document.doc_id for document in result.documents} >= {
        "jira:SCOPE-1",
        "confluence:1001",
        "teams:thread-88",
    }


def test_the_fixture_corpus_reads_in_a_stable_order() -> None:
    first = [
        document.doc_id for document in read_directory(FIXTURE_CORPUS_DIR).documents
    ]
    second = [
        document.doc_id for document in read_directory(FIXTURE_CORPUS_DIR).documents
    ]
    assert first == second == sorted(first)


def test_the_role_is_never_inferred(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text(
        "---\nsource_kind: wiki\nsource_name: confluence\nsource_id: '7'\n"
        "title: Policy\n---\n\nBody.\n"
    )
    with pytest.raises(IngestError, match="corpus_role is required"):
        read_file(path)


def test_front_matter_must_be_present(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("Just a body.\n")
    with pytest.raises(IngestError, match="front matter"):
        read_file(path)


def test_an_unclosed_front_matter_block_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("---\ntitle: Policy\n\nBody.\n")
    with pytest.raises(IngestError, match="not closed"):
        read_file(path)


def test_a_bad_source_name_is_refused_on_ingest(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text(
        "---\nsource_kind: wiki\nsource_name: Confluence\nsource_id: '7'\n"
        "title: Policy\ncorpus_role: context\n---\n\nBody.\n"
    )
    with pytest.raises(IngestError):
        read_file(path)


def test_two_documents_may_not_share_an_id(tmp_path: Path) -> None:
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(
            "---\nsource_kind: wiki\nsource_name: confluence\nsource_id: '7'\n"
            "title: Policy\ncorpus_role: context\n---\n\nBody.\n"
        )
    with pytest.raises(IngestError, match="duplicate document id"):
        read_directory(tmp_path)


def test_an_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no .md files"):
        read_directory(tmp_path)
