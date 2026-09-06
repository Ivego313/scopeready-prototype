"""Splitting a document into pieces that stay quotable and stay encodable.

Two constraints pull against each other. A chunk has to be quotable, so it keeps
the heading path it sits under and is never cut through the middle of a list
item. A chunk also has to fit the embedder's token window, because text over the
window is truncated at encode time with no error and no warning — the retrieval
quality drops and nothing says why. The window is therefore checked here, on the
text that will actually be encoded, rather than trusted to be large enough.

The scanner is line-based rather than a Markdown parser. Everything it needs is
recognizable from a line prefix plus one bit of state (inside a fence or not),
and the adapters are what produce this Markdown, so the input dialect is ours.
The upgrade trigger is concrete: the first real export that contains setext
headings or nested lists.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from scopeready.models import Chunk, Document
from scopeready.text import build_context_text, normalize_for_index

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^ {0,3}([-*+]|\d{1,9}[.)])\s+")
_TABLE_ROW = re.compile(r"^ {0,3}\|")
_QUOTE_LINE = re.compile(r"^ {0,3}>")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# How the context line and the body are joined when the embedder encodes them.
# Anything that changes here changes the budget arithmetic.
CONTEXT_JOIN = "\n\n"


class TokenBudget(Protocol):
    """How many tokens the thing that will encode this text can accept."""

    @property
    def max_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class HeuristicBudget:
    """A token budget for the paths where no encoder is present.

    Four characters per token is wrong exactly where it matters — dense
    technical text, tables and identifiers like `SCOPE-1234` cost more tokens
    per character than prose — so this is never used for indexing. It exists for
    the ingest preview, which runs before an embedder is chosen, and for chunker
    tests, which must not download a model. The indexing path asks the embedder
    that will actually encode the text.
    """

    max_tokens: int = 512
    chars_per_token: int = 4

    def count_tokens(self, text: str) -> int:
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    QUOTE = "quote"


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    level: int
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")


@dataclass(frozen=True, slots=True)
class ChunkingReport:
    """What the chunker had to do that a reader would want to know about."""

    chunks: int = 0
    # Split below block granularity: a list item or a code block that alone
    # exceeded the window. Not an error, but a place where one requirement may
    # now be quotable only in halves.
    oversplit_chunks: int = 0
    # Documents whose whole content was their title.
    title_only_documents: int = 0

    def merged(self, other: "ChunkingReport") -> "ChunkingReport":
        return ChunkingReport(
            chunks=self.chunks + other.chunks,
            oversplit_chunks=self.oversplit_chunks + other.oversplit_chunks,
            title_only_documents=(
                self.title_only_documents + other.title_only_documents
            ),
        )


def chunk_document(
    document: Document, budget: TokenBudget, *, min_tokens: int = 24
) -> tuple[tuple[Chunk, ...], ChunkingReport]:
    """Split one document, keeping every piece quotable and inside the window."""
    title = document.provenance.title
    body = document.text.strip()
    if not body:
        # `Chunk.text` cannot be empty, so a naive chunker emits nothing here and
        # the document becomes invisible to retrieval and uncitable — while a
        # ticket carrying everything in its title is entirely ordinary.
        chunk = Chunk(doc_id=document.doc_id, ordinal=0, text=title)
        return (chunk,), ChunkingReport(chunks=1, title_only_documents=1)

    blocks = _scan(body)
    groups = _group_by_heading(blocks)

    texts: list[tuple[tuple[str, ...], str]] = []
    body_budgets: dict[tuple[str, ...], int] = {}
    oversplit = 0
    for path, block_group in groups:
        # The budget belongs to the text the embedder actually sees, which is
        # the context line, the separator and the body. Measuring the body alone
        # reintroduces exactly the silent truncation this check exists to
        # prevent, and forgetting the separator makes it off-by-two.
        overhead = budget.count_tokens(build_context_text(title, path) + CONTEXT_JOIN)
        body_budget = max(budget.max_tokens - overhead, 32)
        body_budgets[path] = body_budget
        packed, split_here = _pack(block_group, budget, body_budget)
        oversplit += split_here
        texts.extend((path, piece) for piece in packed)

    texts = _merge_runts(texts, budget, body_budgets, min_tokens)

    chunks = tuple(
        Chunk(
            doc_id=document.doc_id,
            ordinal=ordinal,
            heading_path=path,
            text=text,
        )
        for ordinal, (path, text) in enumerate(texts)
    )
    if not chunks:  # a body made only of headings
        chunks = (Chunk(doc_id=document.doc_id, ordinal=0, text=title),)
    return chunks, ChunkingReport(chunks=len(chunks), oversplit_chunks=oversplit)


def chunk_documents(
    documents: Sequence[Document], budget: TokenBudget
) -> tuple[tuple[Chunk, ...], ChunkingReport]:
    chunks: list[Chunk] = []
    report = ChunkingReport()
    for document in documents:
        produced, produced_report = chunk_document(document, budget)
        chunks.extend(produced)
        report = report.merged(produced_report)
    return tuple(chunks), report


def _scan(body: str) -> tuple[Block, ...]:
    blocks: list[Block] = []
    pending: list[str] = []
    kind = BlockKind.PARAGRAPH
    fence: str | None = None

    def flush() -> None:
        nonlocal pending
        if pending:
            blocks.append(Block(kind=kind, level=0, lines=tuple(pending)))
            pending = []

    for line in body.splitlines():
        if fence is not None:
            pending.append(line)
            if line.strip().startswith(fence):
                flush()
                fence = None
            continue

        opening = _FENCE.match(line)
        if opening:
            flush()
            kind = BlockKind.CODE
            fence = opening.group(1)
            pending.append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            blocks.append(
                Block(
                    kind=BlockKind.HEADING,
                    level=len(heading.group(1)),
                    lines=(heading.group(2),),
                )
            )
            kind = BlockKind.PARAGRAPH
            continue

        if not line.strip():
            flush()
            kind = BlockKind.PARAGRAPH
            continue

        line_kind = _line_kind(line)
        # A continuation line of a list item is indented and belongs to the item,
        # so it must not close the list and start a paragraph.
        continues_list = (
            kind is BlockKind.LIST
            and line_kind is BlockKind.PARAGRAPH
            and line.startswith(("  ", "\t"))
        )
        if pending and line_kind is not kind and not continues_list:
            flush()
            kind = line_kind
        elif not pending:
            kind = line_kind
        pending.append(line)

    if fence is not None:  # an unterminated fence still has to be kept
        flush()
    flush()
    return tuple(blocks)


def _line_kind(line: str) -> BlockKind:
    if _LIST_ITEM.match(line):
        return BlockKind.LIST
    if _TABLE_ROW.match(line):
        return BlockKind.TABLE
    if _QUOTE_LINE.match(line):
        return BlockKind.QUOTE
    return BlockKind.PARAGRAPH


def _group_by_heading(
    blocks: Sequence[Block],
) -> tuple[tuple[tuple[str, ...], tuple[Block, ...]], ...]:
    groups: list[tuple[tuple[str, ...], list[Block]]] = []
    # Levels are kept next to titles rather than implied by position. Slicing by
    # `level - 1` looks equivalent and is not: a document whose top heading is
    # `##` puts that heading at index 0, and every sibling `##` after it then
    # nests under the first one instead of replacing it. Real exports routinely
    # have no `#` at all, because the title lives in a field.
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            # Heading text stays out of the chunk body: a quotable heading is a
            # quote the verbatim check on `Chunk.text` would later reject.
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.lines[0]))
            continue
        current = tuple(title for _, title in stack)
        if groups and groups[-1][0] == current:
            groups[-1][1].append(block)
        else:
            groups.append((current, [block]))
    return tuple((path_key, tuple(group)) for path_key, group in groups)


def _pack(
    blocks: Sequence[Block], budget: TokenBudget, body_budget: int
) -> tuple[list[str], int]:
    pieces: list[str] = []
    current: list[str] = []
    oversplit = 0

    def close() -> None:
        if current:
            pieces.append("\n\n".join(current))
            current.clear()

    for block in blocks:
        text = block.text
        if not text.strip():
            continue
        if budget.count_tokens(text) > body_budget:
            close()
            parts, split = _split_oversized(block, budget, body_budget)
            oversplit += split
            pieces.extend(parts)
            continue
        candidate = "\n\n".join([*current, text])
        if current and budget.count_tokens(candidate) > body_budget:
            close()
        current.append(text)
    close()
    return pieces, oversplit


def _split_oversized(
    block: Block, budget: TokenBudget, body_budget: int
) -> tuple[list[str], int]:
    """Break a block that alone exceeds the window, never truncating it.

    Truncation would discard text that may be the only place a requirement is
    stated, producing a false gap the system can never detect. Splitting keeps
    everything retrievable; the price is that one statement may be quotable only
    in halves, which is why the fact is counted and reported.
    """
    units, prefix = _splittable_units(block)
    pieces = _greedy_pack(units, prefix, budget, body_budget)
    hard_split = 0
    settled: list[str] = []
    for piece in pieces:
        if budget.count_tokens(piece) <= body_budget:
            settled.append(piece)
            continue
        hard_split += 1
        settled.extend(_split_by_characters(piece, budget, body_budget))
    return settled, hard_split


def _splittable_units(block: Block) -> tuple[list[str], str]:
    """The pieces a block may be cut between, plus a prefix each piece repeats."""
    match block.kind:
        case BlockKind.LIST:
            units: list[str] = []
            for line in block.lines:
                if _LIST_ITEM.match(line) or not units:
                    units.append(line)
                else:  # a continuation line stays with its item
                    units[-1] += "\n" + line
            return units, ""
        case BlockKind.TABLE:
            # A table fragment without its header row is unreadable and, worse,
            # unquotable in a way a reader would notice.
            header = "\n".join(block.lines[:2])
            return list(block.lines[2:]) or list(block.lines), header
        case BlockKind.CODE:
            fence = block.lines[0]
            inner = block.lines[1:-1] if len(block.lines) > 2 else block.lines[1:]
            return list(inner), fence
        case _:
            return _SENTENCE_END.split(block.text), ""


def _greedy_pack(
    units: Sequence[str], prefix: str, budget: TokenBudget, body_budget: int
) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []

    def rendered(parts: Sequence[str]) -> str:
        joined = "\n".join(parts)
        return f"{prefix}\n{joined}" if prefix else joined

    for unit in units:
        candidate = [*current, unit]
        if current and budget.count_tokens(rendered(candidate)) > body_budget:
            pieces.append(rendered(current))
            current = [unit]
        else:
            current = candidate
    if current:
        pieces.append(rendered(current))
    return pieces


def _split_by_characters(text: str, budget: TokenBudget, body_budget: int) -> list[str]:
    """Last resort for one indivisible unit: cut on whitespace, keep everything."""
    pieces: list[str] = []
    remaining = text
    while remaining and budget.count_tokens(remaining) > body_budget:
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if budget.count_tokens(remaining[:middle]) <= body_budget:
                low = middle
            else:
                high = middle - 1
        cut = remaining.rfind(" ", 1, low + 1)
        cut = cut if cut > 0 else low
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [piece for piece in pieces if piece]


def _merge_runts(
    texts: Sequence[tuple[tuple[str, ...], str]],
    budget: TokenBudget,
    body_budgets: Mapping[tuple[str, ...], int],
    min_tokens: int,
) -> list[tuple[tuple[str, ...], str]]:
    """Fold a tiny chunk into its neighbour, but never drop one.

    "## Out of scope\\n\\nNothing." is short, real and citable, and dropping it
    would hide the very statement the out-of-scope category looks for.
    """
    merged: list[tuple[tuple[str, ...], str]] = []
    for path, text in texts:
        joined = f"{merged[-1][1]}\n\n{text}" if merged else text
        # Against the section's own budget, not the window: the context line is
        # encoded alongside the body and has already claimed part of it.
        fits = budget.count_tokens(joined) <= body_budgets[path]
        if (
            merged
            and merged[-1][0] == path
            and budget.count_tokens(text) < min_tokens
            and fits
        ):
            merged[-1] = (path, joined)
            continue
        merged.append((path, text))
    return merged


def index_text_of(chunk: Chunk) -> str:
    return normalize_for_index(chunk.text)
