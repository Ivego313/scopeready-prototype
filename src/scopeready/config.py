"""Defaults every command shares, in one place so they cannot drift apart.

The values that end up in a report — model name, embedder name, prompt version —
are settings rather than constants because changing any of them makes the next
run a different measurement, and a run has to be able to say which one it was.
"""

from pathlib import Path
from typing import Final

TAXONOMY_DIR: Final = Path("taxonomy")
FIXTURE_CORPUS_DIR: Final = Path("fixtures/corpus")
DEFAULT_DB: Final = Path("runs/corpus.db")
DEFAULT_CACHE_DIR: Final = Path(".cache/model")
DEFAULT_OUT_DIR: Final = Path("runs")

# Ollama defaults to a 2-4k context and truncates from the front, silently and
# starting with the system prompt. This is set explicitly on every call.
DEFAULT_NUM_CTX: Final = 32768
DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"
DEFAULT_MODEL: Final = "gpt-oss:20b"
# Two, not the CPU count: each Ollama slot holds its own KV cache, so more
# parallelism means more copies of a 15k-token prefix in memory next to a 13 GB
# model. On a 32 GB machine that trades speed for swap.
DEFAULT_CONCURRENCY: Final = 2

# A real model by default, because the paraphrase case is the whole reason the
# vector channel exists and the stub cannot demonstrate it. `--embedder stub`
# is the offline path and the one the test suite uses.
DEFAULT_EMBEDDER: Final = "bge-small"
