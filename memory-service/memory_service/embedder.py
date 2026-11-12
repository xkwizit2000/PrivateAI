"""In-process CPU embedding backends.

`FastEmbedder` wraps fastembed (ONNX Runtime, no torch). `HashEmbedder` is a
dependency-free deterministic embedder used by unit tests (hashed bag of words).
Both return L2-normalised float32 matrices, so cosine similarity == dot product.
"""

from __future__ import annotations

import re
import time
import zlib
from typing import Protocol, Sequence

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Per-model query-side prefix (asymmetric retrieval models).
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": BGE_QUERY_PREFIX,
    "snowflake/snowflake-arctic-embed-xs": BGE_QUERY_PREFIX,
    "snowflake/snowflake-arctic-embed-s": BGE_QUERY_PREFIX,
    "nomic-ai/nomic-embed-text-v1.5-Q": "search_query: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
}
PASSAGE_PREFIXES: dict[str, str] = {
    "nomic-ai/nomic-embed-text-v1.5-Q": "search_document: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_document: ",
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FastEmbedder:
    """fastembed/ONNX embedder with lazy load and basic timing stats."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
        threads: int = 2,
        local_only: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.local_only = local_only
        self.batch_size = batch_size
        self.query_prefix = QUERY_PREFIXES.get(model, "")
        self.passage_prefix = PASSAGE_PREFIXES.get(model, "")
        self._model = None
        self.load_seconds: float | None = None
        t0 = time.perf_counter()
        self._load()
        self.load_seconds = time.perf_counter() - t0
        self.dim = int(self._model.embed(["dim probe"]).__next__().shape[0])

    def _load(self) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy

        kwargs: dict = {"model_name": self.name, "threads": self.threads}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.local_only:
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(**kwargs)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        # Sort by length so each ONNX batch has little padding, then restore order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = list(self._model.embed([texts[i] for i in order], batch_size=self.batch_size))
        out = np.zeros((len(texts), len(vecs[0])), dtype=np.float32)
        for pos, i in enumerate(order):
            out[i] = vecs[pos]
        return _normalise(out)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic hashed bag-of-words embedder (tests / no-model fallback)."""

    name = "hash-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = zlib.crc32(tok.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            v[(h >> 8) % self.dim] += 0.5
        return v

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vec(text)[None, :])[0]


def make_embedder(settings) -> Embedder:
    if settings.embed_backend == "hash":
        return HashEmbedder()
    return FastEmbedder(
        settings.embed_model,
        cache_dir=settings.embed_cache_dir or None,
        threads=settings.embed_threads,
        local_only=settings.embed_local_only,
    )
