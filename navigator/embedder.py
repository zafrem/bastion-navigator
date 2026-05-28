"""Text embedder: local sentence-transformers or remote BGE HTTP service."""
from __future__ import annotations

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from .config import EmbedderConfig
from . import metrics

log = logging.getLogger(__name__)


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder(Embedder):
    """Runs the BGE-M3 model (or any SentenceTransformer) in-process."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        log.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model.max_seq_length = max_length

    def embed(self, text: str) -> list[float]:
        start = time.perf_counter()
        vec = self._model.encode(text, normalize_embeddings=True)
        metrics.embedding_duration_seconds.observe(time.perf_counter() - start)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        metrics.embedding_duration_seconds.observe(time.perf_counter() - start)
        return [v.tolist() for v in vecs]


class BGEHttpEmbedder(Embedder):
    """Calls a remote BGE embedding service over HTTP."""

    def __init__(self, endpoint: str, timeout: float = 10.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        start = time.perf_counter()
        resp = httpx.post(
            f"{self._endpoint}/embed",
            json={"text": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        metrics.embedding_duration_seconds.observe(time.perf_counter() - start)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        resp = httpx.post(
            f"{self._endpoint}/embed_batch",
            json={"texts": texts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        vecs = resp.json()["embeddings"]
        metrics.embedding_duration_seconds.observe(time.perf_counter() - start)
        return vecs


class CachedEmbedder(Embedder):
    """LRU-cached wrapper around any Embedder."""

    def __init__(self, inner: Embedder, max_size: int = 1024) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self._max = max_size

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            metrics.cache_hits.inc()
            return self._cache[text]
        metrics.cache_misses.inc()
        vec = self._inner.embed(text)
        if len(self._cache) >= self._max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class MockEmbedder(Embedder):
    """Deterministic mock embedder for tests."""

    def __init__(self, dims: int = 768) -> None:
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        rng = random.Random(hash(text))
        raw = [rng.gauss(0, 1) for _ in range(self._dims)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def build(cfg: EmbedderConfig) -> Embedder:
    """Construct an Embedder from config."""
    if cfg.type == "local":
        inner: Embedder = LocalEmbedder(cfg.model_name, cfg.max_length)
    elif cfg.type == "bge_http":
        inner = BGEHttpEmbedder(cfg.endpoint)
    else:
        inner = MockEmbedder()
    if cfg.cache.enabled and cfg.cache.max_size > 0:
        return CachedEmbedder(inner, cfg.cache.max_size)
    return inner
