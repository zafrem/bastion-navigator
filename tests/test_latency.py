"""NFR latency tests for Navigator — all operations measured against SRS targets.

NFR-PE-001: Vector search p95 < 50 ms
NFR-PE-002: Hybrid search p95 < 80 ms
NFR-PE-003: Hybrid+Rerank p95 < 150 ms

Tests drive the Orchestrator directly with mock dependencies (no Qdrant required)
so they can run in CI without external services.
"""
from __future__ import annotations

import time
from typing import Callable
from unittest.mock import MagicMock

import pytest

from navigator.config import Config
from navigator.models import (
    SearchMetadata,
    SearchOptions,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from navigator.orchestrator import Orchestrator
from navigator.searcher import MockSearcher

ITERATIONS = 200
_DIM = 64


def _percentile(durations: list[float], n: int) -> float:
    sorted_d = sorted(durations)
    idx = (n * len(sorted_d)) // 100
    if idx >= len(sorted_d):
        idx = len(sorted_d) - 1
    return sorted_d[idx]


def _measure(fn: Callable[[], None], n: int = ITERATIONS) -> list[float]:
    results = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        results.append((time.perf_counter() - t0) * 1000.0)
    return results


def _make_results(n: int = 5) -> list[SearchResult]:
    return [SearchResult(document_id=f"doc{i}", content=f"content {i}", score=0.9 - i * 0.05) for i in range(n)]


def _make_orchestrator(use_reranking: bool = False) -> Orchestrator:
    cfg = Config()
    cfg.search_defaults.use_reranking = use_reranking

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * _DIM
    mock_embedder.embed_batch.return_value = [[0.1] * _DIM]

    mock_searcher = MockSearcher()
    mock_searcher.vector_search = MagicMock(return_value=_make_results())
    mock_searcher.sparse_search = MagicMock(return_value=_make_results())

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _make_results(3)

    mock_vault = MagicMock()
    mock_vault.deanonymize.return_value = {"records": []}

    return Orchestrator(cfg, mock_embedder, mock_searcher, mock_reranker, mock_vault)


def _make_req(use_hybrid: bool = False, use_reranking: bool = False) -> SearchRequest:
    opts = SearchOptions(use_hybrid=use_hybrid, use_reranking=use_reranking, top_k=5)
    return SearchRequest(
        query="What is retrieval-augmented generation?",
        tenant_id="nfr-tenant",
        options=opts,
    )


# ── NFR-PE-001: Vector search ─────────────────────────────────────────────────

class TestNFR_VectorSearch_P95_Under50ms:
    def test_search_p95(self):
        orch = _make_orchestrator()
        req = _make_req()
        durations = _measure(lambda: orch.search(req))
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        print(f"\nVector search: p50={p50:.3f}ms p95={p95:.3f}ms (target <50ms)")
        assert p95 < 50.0, f"NFR-PE-001 FAIL: vector search p95={p95:.3f}ms exceeds 50ms"


# ── NFR-PE-002: Hybrid search ─────────────────────────────────────────────────

class TestNFR_HybridSearch_P95_Under80ms:
    def test_hybrid_search_p95(self):
        orch = _make_orchestrator()
        req = _make_req(use_hybrid=True)
        durations = _measure(lambda: orch.search(req))
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        print(f"\nHybrid search: p50={p50:.3f}ms p95={p95:.3f}ms (target <80ms)")
        assert p95 < 80.0, f"NFR-PE-002 FAIL: hybrid search p95={p95:.3f}ms exceeds 80ms"


# ── NFR-PE-003: Hybrid + Rerank ───────────────────────────────────────────────

class TestNFR_HybridRerank_P95_Under150ms:
    def test_hybrid_rerank_p95(self):
        orch = _make_orchestrator(use_reranking=True)
        req = _make_req(use_hybrid=True, use_reranking=True)
        durations = _measure(lambda: orch.search(req))
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        print(f"\nHybrid+Rerank: p50={p50:.3f}ms p95={p95:.3f}ms (target <150ms)")
        assert p95 < 150.0, f"NFR-PE-003 FAIL: hybrid+rerank p95={p95:.3f}ms exceeds 150ms"
