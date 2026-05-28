"""Cross-encoder reranker: local BGE-reranker or remote HTTP service."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import httpx

from .models import SearchResult
from .config import RerankerConfig
from . import metrics

log = logging.getLogger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]: ...


class LocalReranker(Reranker):
    """Runs the BGE-reranker-v2-m3 cross-encoder in-process."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore
        log.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        start = time.perf_counter()
        pairs = [[query, c.content] for c in candidates]
        scores = self._model.predict(pairs)
        metrics.rerank_duration_seconds.observe(time.perf_counter() - start)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]


class BGEHttpReranker(Reranker):
    """Calls a remote reranker service over HTTP."""

    def __init__(self, endpoint: str, timeout: float = 10.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        start = time.perf_counter()
        resp = httpx.post(
            f"{self._endpoint}/rerank",
            json={"query": query, "candidates": [c.model_dump() for c in candidates], "top_k": top_k},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        metrics.rerank_duration_seconds.observe(time.perf_counter() - start)
        ids = {c.document_id: c for c in candidates}
        return [ids[r["document_id"]] for r in resp.json()["results"] if r["document_id"] in ids]


class MockReranker(Reranker):
    """Mock reranker that returns candidates unchanged (for tests)."""

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        return candidates[:top_k]


def build(cfg: RerankerConfig) -> Reranker:
    if cfg.type == "local":
        return LocalReranker(cfg.model_name, cfg.max_length)
    elif cfg.type == "bge_http":
        return BGEHttpReranker(cfg.endpoint)
    return MockReranker()
