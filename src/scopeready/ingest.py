"""Reading a corpus from a directory of Markdown files with YAML front matter.

This is the only adapter in the prototype, and it is deliberately the one a
person can write by hand, because it doubles as the contract for adding real
data. Everything a live Jira or Confluence connector would have to fill in is
visible here as a named field, so the shape of the corpus is arguable before any
OAuth application exists.

A missing required field is an error rather than a default. A document whose
role or source was guessed is a document whose verdict cannot be explained
later, and the guess is invisible in the report.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from scopeready.models import CorpusRole, Document, Provenance

FRONT_MATTER_FENCE = "---"


class IngestError(ValueError):
    """A source file that cannot be turned into a document without guessing."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    documents: tuple[Document, ...]

    @property
    def requirements(self) -> int:
        return sum(
            1 for doc in self.documents if doc.corpus_role is CorpusRole.REQUIREMENT
        )

    @property
    def contexts(self) -> int:
        return len(self.documents) - self.requirements


def read_directory(directory: Path) -> IngestResult:
    """Read every `.md` file under a directory, in a stable order.

    Sorted by path rather than by directory order: the corpus is rendered into
    the prompt prefix, and a prefix whose order depends on the filesystem is a
    prefix that stops matching the cache on another machine.
    """
    if not directory.is_dir():
        msg = f"corpus directory does not exist: {directory}"
        raise IngestError(msg)
    documents = tuple(
        read_file(path) for path in sorted(directory.rglob("*.md"), key=str)
    )
    if not documents:
        msg = f"no .md files found under {directory}"
        raise IngestError(msg)
    _reject_duplicate_ids(documents)
    return IngestResult(documents=documents)


def read_file(path: Path) -> Document:
    front_matter, body = split_front_matter(path.read_text(encoding="utf-8"), path)
    try:
        role = front_matter.pop("corpus_role")
    except KeyError as error:
        msg = (
            f"{path}: corpus_role is required and is never inferred — a document "
            "is either the declared scope under audit or context that may close "
            "a gap, and the two lead to different reports"
        )
        raise IngestError(msg) from error

    item_label = front_matter.pop("item_label", None)
    status = front_matter.pop("status", None)
    parent_source_id = front_matter.pop("parent_source_id", None)

    try:
        provenance = Provenance.model_validate(front_matter)
        return Document(
            provenance=provenance,
            corpus_role=CorpusRole(role),
            item_label=item_label,
            status=status,
            text=body,
            parent_source_id=parent_source_id,
        )
    except (ValidationError, ValueError) as error:
        msg = f"{path}: {error}"
        raise IngestError(msg) from error


def split_front_matter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_FENCE:
        msg = f"{path}: must start with a '{FRONT_MATTER_FENCE}' front matter block"
        raise IngestError(msg)
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONT_MATTER_FENCE:
            head, body = "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
            break
    else:
        msg = f"{path}: front matter block is not closed"
        raise IngestError(msg)

    try:
        parsed: Any = yaml.safe_load(head)
    except yaml.YAMLError as error:
        msg = f"{path}: front matter is not valid YAML: {error}"
        raise IngestError(msg) from error
    if not isinstance(parsed, dict):
        msg = f"{path}: front matter must be a mapping"
        raise IngestError(msg)
    return dict(parsed), body.strip()


def _reject_duplicate_ids(documents: Sequence[Document]) -> None:
    seen: dict[str, str] = {}
    for document in documents:
        if document.doc_id in seen:
            msg = (
                f"duplicate document id {document.doc_id!r}; a chunk id is derived "
                "from it, so two documents sharing one would overwrite each "
                "other's evidence"
            )
            raise IngestError(msg)
        seen[document.doc_id] = document.provenance.title


def iter_markdown(directory: Path) -> Iterator[Path]:
    yield from sorted(directory.rglob("*.md"), key=str)
