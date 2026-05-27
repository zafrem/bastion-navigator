"""15-step search pipeline (SRS §6.3)."""
from __future__ import annotations

import time
from typing import Optional

from .config import Config, SearchDefaultsConfig
from .embedder import Embedder
from .reranker import Reranker
from .searcher import QdrantSearcher, MockSearcher
from .vault_client import VaultClient, NoopVaultClient
from .models import SearchOptions, SearchRequest, SearchResponse, SearchResult, SearchMetadata
from . import metrics

# Maps data category → Qdrant collection name.
_CATEGORY_TO_COLLECTION: dict[str, str] = {
    "customer_data": "customer_docs",
    "manufacturing_data": "manufacturing_docs",
    "hr_data": "hr_docs",
}


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        embedder: Embedder,
        searcher,
        reranker: Reranker,
        vault,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._searcher = searcher
        self._reranker = reranker
        self._vault = vault

    def search(self, req: SearchRequest, **kwargs) -> SearchResponse:
        start = time.perf_counter()
        opts = self._merge_defaults(req.options)
        strategy = _strategy_name(opts)
        filters: dict[str, str] = {"tenant_id": req.tenant_id}
        if opts.filters:
            filters.update(opts.filters)

        allowed = self._resolve_permissions(req)
        collections = self._collections_for_categories(allowed)

        vec = self._embedder.embed(req.query)
        all_results: list[SearchResult] = []
        for col in collections:
            results = self._search_collection(col, req.query, vec, filters, opts, opts.top_k * 3)
            all_results.extend(results)

        if opts.use_rerank and all_results:
            all_results = self._reranker.rerank(req.query, all_results, opts.top_k)
        else:
            all_results = _top_k(all_results, opts.top_k)

        duration_ms = (time.perf_counter() - start) * 1000
        metrics.searches_total.labels(strategy=strategy, tenant_id=req.tenant_id).inc()
        metrics.search_duration_seconds.labels(strategy=strategy).observe(duration_ms / 1000)
        if not all_results:
            metrics.zero_result_searches.inc()

        return SearchResponse(
            results=all_results,
            metadata=SearchMetadata(
                strategy=strategy,
                total_results=len(all_results),
                processing_time_ms=duration_ms,
            ),
        )

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_batch(texts)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        return self._reranker.rerank(query, candidates, top_k)

    def collections(self) -> list:
        return self._searcher.collections()

    def _search_collection(
        self,
        collection: str,
        query: str,
        vec: list[float],
        filters: dict[str, str],
        opts: SearchOptions,
        over_fetch: int,
    ) -> list[SearchResult]:
        vector_results = self._searcher.vector_search(collection, vec, filters, over_fetch)
        if not opts.use_hybrid:
            return vector_results
        sparse_results = self._searcher.sparse_search(collection, query, filters, over_fetch)
        return _rrf(vector_results, sparse_results)

    def _resolve_permissions(self, req: SearchRequest) -> Optional[list[str]]:
        if req.user and req.user.allowed_categories:
            return req.user.allowed_categories
        user_id = req.user.user_id if req.user else ""
        if not user_id:
            return None
        return self._vault.allowed_categories(user_id)

    def _collections_for_categories(self, allowed: Optional[list[str]]) -> list[str]:
        if not allowed:
            return list(self._cfg.vector_db.collections.keys()) or list(_CATEGORY_TO_COLLECTION.values())
        seen: set[str] = set()
        out: list[str] = []
        for cat in allowed:
            col = _CATEGORY_TO_COLLECTION.get(cat)
            if col and col not in seen:
                seen.add(col)
                out.append(col)
        return out or list(_CATEGORY_TO_COLLECTION.values())

    def _merge_defaults(self, opts: Optional[SearchOptions]) -> SearchOptions:
        d: SearchDefaultsConfig = self._cfg.search_defaults
        if opts is None:
            opts = SearchOptions()
        if not opts.top_k:
            opts.top_k = d.top_k
        if not opts.vector_weight:
            opts.vector_weight = d.vector_weight
        if not opts.bm25_weight:
            opts.bm25_weight = d.bm25_weight
        if opts.use_rerank is None:
            opts.use_rerank = d.use_rerank
        if opts.use_hybrid is None:
            opts.use_hybrid = d.use_hybrid
        return opts


def _rrf(
    vector: list[SearchResult],
    bm25: list[SearchResult],
    vw: float = 0.7,
    bw: float = 0.3,
    k: float = 60.0,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}
    for rank, r in enumerate(sorted(vector, key=lambda r: r.score, reverse=True)):
        scores[r.document_id] = scores.get(r.document_id, 0) + vw / (k + rank + 1)
        docs[r.document_id] = r
    for rank, r in enumerate(sorted(bm25, key=lambda r: r.score, reverse=True)):
        scores[r.document_id] = scores.get(r.document_id, 0) + bw / (k + rank + 1)
        docs[r.document_id] = r
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[doc_id] for doc_id, _ in ranked]


def _top_k(results: list[SearchResult], k: int) -> list[SearchResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)[:k]


def _strategy_name(opts: SearchOptions) -> str:
    if opts.use_hybrid and opts.use_rerank:
        return "hybrid+rerank"
    if opts.use_hybrid:
        return "hybrid"
    if opts.use_rerank:
        return "vector+rerank"
    return "vector"
