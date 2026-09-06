"""Every place text is transformed before being compared to other text.

Three transformations live here and nowhere else, because each one is a seam
that has to move as a unit. `normalize_for_index` is the multilingual seam: an
English corpus lets the lexical index lean on a Porter stemmer, and a corpus in
another language would replace this function and the tokenizer together.
`to_fts_match` is the injection seam. `normalize_for_match` is the seam where a
model's quote meets the corpus it claims to have quoted.
"""

import re
import unicodedata
from collections.abc import Iterable

# Markdown link targets are indexed as words otherwise, so every wiki chunk
# carries `confluence`, `pages` and a page id into the term statistics and
# distorts what BM25 thinks is rare.
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MARKDOWN_MARKUP = re.compile(r"[*_`~#>|]+")
_WHITESPACE = re.compile(r"\s+")
_FTS_TERM = re.compile(r"[0-9A-Za-z_]+")

# Characters a model rewrites without meaning to, and that a Markdown export
# rewrites on the way in. Both sides of a quote comparison get the same
# treatment, so a quote that genuinely carries the fancy character still matches.
_PRESENTATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        " ": " ",
        " ": " ",
        "​": "",
        "*": "",
        "_": "",
        "`": "",
    }
)

ELLIPSIS = "..."


def normalize_for_index(text: str) -> str:
    """Reduce Markdown to the words a lexical or vector index should see."""
    without_links = _MARKDOWN_LINK.sub(r"\1", text)
    without_markup = _MARKDOWN_MARKUP.sub(" ", without_links)
    return _WHITESPACE.sub(" ", without_markup).strip()


def build_context_text(title: str, heading_path: Iterable[str]) -> str:
    """The document title and section path a chunk sits under.

    Indexed as its own column because an empty document body is legitimate — a
    ticket may carry everything in its title — and without this such a document
    is invisible to retrieval and therefore uncitable.
    """
    parts = [title, *heading_path]
    return normalize_for_index(" / ".join(part for part in parts if part))


def to_fts_match(query: str) -> str | None:
    """Turn free text into an FTS5 MATCH expression, or `None` if it has no terms.

    Every term is quoted and the operators are ours, so nothing the caller
    supplies can be read as syntax. A raw query containing a quote, a hyphen or
    the word NEAR is otherwise either a syntax error or, worse, a different
    query than the one that was asked.

    The terms are joined with OR rather than AND: a probe's retrieval query is a
    phrasing, not a filter, and requiring every word would drop the chunk that
    says the same thing in three words instead of six.
    """
    terms = _FTS_TERM.findall(query)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def normalize_for_match(text: str) -> str:
    """Fold away the differences a quote is allowed to have from its source.

    The order is load-bearing: NFKC folds ligatures and some spaces but leaves
    curly quotes alone, so the explicit table has to run after it.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(_PRESENTATION)
    folded = folded.replace("…", ELLIPSIS)
    return _WHITESPACE.sub(" ", folded).strip().casefold()
