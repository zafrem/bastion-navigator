"""15-step search pipeline (SRS §6.3)."""
from __future__ import annotations

import time
import uuid
from typing import Optional

from .config import Config, SearchDefaultsConfig
from .embedder import Embedder
from .reranker import Reranker
from .searcher import QdrantSearcher, MockSearcher
from .vault_client import VaultClient, NoopVaultClient
from .token_rewriter import TokenRewriter
from .router import Router, RoutingDecision
from .evaluator import Evaluator, EvaluatorConfig, QualityVerdict
from .models import (
    IndexRequest, IndexResponse,
    SearchOptions, SearchRequest, SearchResponse, SearchResult, SearchMetadata,
)
from .events import (
    Publisher, TraceContext,
    event_chunk_retrieved, event_query_routed,
    event_search_iteration, event_loop_completed,
)
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
        rewriter: Optional[TokenRewriter] = None,
        publisher: Optional[Publisher] = None,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._searcher = searcher
        self._reranker = reranker
        self._vault = vault
        self._rewriter = rewriter
        self._publisher = publisher
        self._router: Optional[Router] = None
        self._evaluator: Optional[Evaluator] = None

    def configure_modular_rag(self, router: Router, evaluator: Evaluator) -> None:
        """Enable Modular RAG components (MR-01/03). Called from main.py when enabled."""
        self._router = router
        self._evaluator = evaluator

    def search(self, req: SearchRequest, trace_context: Optional[TraceContext] = None, **kwargs) -> SearchResponse:
        start = time.perf_counter()
        opts = self._merge_defaults(req.options)
        filters: dict[str, str] = {"tenant_id": req.tenant_id}
        if opts.filters:
            filters.update(opts.filters)

        tc = trace_context or TraceContext(
            trace_id=req.request_id or str(uuid.uuid4()),
            tenant_id=req.tenant_id,
            user_id=req.user.user_id if req.user else "",
            request_id=req.request_id,
        )

        allowed = self._resolve_permissions(req)
        permission_collections = self._collections_for_categories(allowed)

        # MR-01: Adaptive routing — select strategy and collections.
        collections, opts = self._do_route(req.query, permission_collections, opts, tc)

        over_fetch = opts.top_k * self._cfg.search_defaults.over_fetch_multiplier
        mr_cfg = self._cfg.modular_rag
        use_loop = (
            self._evaluator is not None
            and mr_cfg.enabled
            and mr_cfg.loop.max_iterations > 1
        )

        # MR-02-001 applied to query: rewrite tokens before embedding.
        query_for_embed = (
            self._rewriter.rewrite_text(req.query) if self._rewriter else req.query
        )

        # MR-03: Re-search loop (single pass when loop is disabled).
        best_results: list[SearchResult] = []
        prev_doc_ids: Optional[list[str]] = None
        termination = "quality_met"
        current_query = req.query          # refinement operates on original query
        current_embedded = query_for_embed
        total_candidates = 0
        total_iterations = 0
        loop_start = time.perf_counter()
        max_iters = mr_cfg.loop.max_iterations if use_loop else 1

        for iteration in range(1, max_iters + 1):
            total_iterations = iteration

            if iteration > 1:
                elapsed_ms = (time.perf_counter() - loop_start) * 1000
                if elapsed_ms > mr_cfg.loop.loop_timeout_ms:
                    termination = "timeout"
                    break
                current_embedded = (
                    self._rewriter.rewrite_text(current_query) if self._rewriter
                    else current_query
                )

            vec = self._embedder.embed(current_embedded)

            iter_results: list[SearchResult] = []
            for col in collections:
                results = self._search_collection(
                    col, current_embedded, vec, filters, opts, over_fetch
                )
                iter_results.extend(results)

            total_candidates = len(iter_results)
            if opts.use_reranking and iter_results:
                iter_results = self._reranker.rerank(current_embedded, iter_results, opts.top_k)
            else:
                iter_results = _top_k(iter_results, opts.top_k)

            # Circuit breaker: identical result set as previous iteration.
            doc_ids = sorted(r.document_id for r in iter_results)
            if prev_doc_ids is not None and doc_ids == prev_doc_ids:
                termination = "duplicate"
                break
            prev_doc_ids = doc_ids

            # Union: keep best score per document across iterations.
            best_results = _merge_best(best_results, iter_results, opts.top_k)

            if not use_loop:
                break  # linear path

            iter_ms = (time.perf_counter() - loop_start) * 1000
            eval_cfg = EvaluatorConfig(
                quality_threshold=mr_cfg.loop.quality_threshold,
                coverage_threshold=mr_cfg.loop.coverage_threshold,
                uncertain_low=mr_cfg.loop.uncertain_low,
            )
            verdict, signals = self._evaluator.evaluate(current_query, iter_results, eval_cfg)

            if self._publisher:
                self._publisher.publish(event_search_iteration(
                    tc, iteration, verdict.value,
                    signals.top_score, signals.coverage,
                    None, iter_ms,
                ))

            if verdict == QualityVerdict.SUFFICIENT:
                termination = "quality_met"
                break

            if iteration == max_iters:
                termination = "max_iterations"
                break

            if verdict == QualityVerdict.UNCERTAIN and iteration > 1:
                termination = "quality_met"
                break

            # MR-03-002: refine query for next iteration.
            current_query, _ = self._evaluator.refine_query(current_query, signals)

        if use_loop and self._publisher:
            loop_ms = (time.perf_counter() - loop_start) * 1000
            self._publisher.publish(event_loop_completed(
                tc, total_iterations, termination, len(best_results), loop_ms
            ))

        all_results = best_results
        strategy = _strategy_name(opts)

        # Apply min_score filter after ranking.
        if opts.min_score > 0:
            all_results = [r for r in all_results if r.score >= opts.min_score]

        # MR-02-001: rewrite Vault tokens in result content.
        if self._rewriter and all_results:
            all_results = [
                r.model_copy(update={"content": self._rewriter.rewrite_text(r.content)})
                for r in all_results
            ]

        # MR-05-001: per-chunk lineage events.
        if self._publisher and all_results:
            for rank, r in enumerate(all_results):
                chunk_id = r.chunk_id or r.metadata.get("chunk_id", r.document_id)
                self._publisher.publish(
                    event_chunk_retrieved(tc, chunk_id, r.document_id, r.score, rank)
                )

        duration_ms = (time.perf_counter() - start) * 1000
        metrics.searches_total.labels(strategy=strategy, tenant_id=req.tenant_id).inc()
        metrics.search_duration_seconds.labels(strategy=strategy).observe(duration_ms / 1000)
        if not all_results:
            metrics.zero_result_searches.inc()

        return SearchResponse(
            request_id=req.request_id,
            results=all_results,
            metadata=SearchMetadata(
                strategy=strategy,
                total_candidates=total_candidates,
                filtered_out=total_candidates - len(all_results),
                final_count=len(all_results),
            ),
            processing_time_ms=duration_ms,
        )

    def _do_route(
        self,
        query: str,
        permission_collections: list[str],
        opts: SearchOptions,
        tc: TraceContext,
    ) -> tuple[list[str], SearchOptions]:
        """Run MR-01 routing; fall back to permission-based selection on error."""
        if self._router is None:
            return permission_collections, opts
        try:
            routing = self._router.route(
                query,
                permission_collections,
                routing_threshold=self._cfg.modular_rag.router.routing_threshold,
                strategy_override=opts.strategy,
            )
            if self._publisher:
                self._publisher.publish(event_query_routed(
                    tc, routing.intent.value, routing.strategy,
                    routing.collections, routing.excluded,
                    routing.confidence, routing.routing_ms,
                ))
            updated_opts = _apply_routing_strategy(opts, routing.strategy)
            return routing.collections or permission_collections, updated_opts
        except Exception:
            return permission_collections, opts

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_batch(texts)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        return self._reranker.rerank(query, candidates, top_k)

    def collections(self) -> list:
        return self._searcher.collections()

    def index_document(self, req: IndexRequest) -> IndexResponse:
        """Chunk, embed, and upsert a document into the vector store."""
        from .chunker import chunk_document, ChunkerConfig
        chunks = chunk_document(
            req.document_id,
            req.content,
            ChunkerConfig(),
            dict(req.metadata),
        )
        if not chunks:
            return IndexResponse(document_id=req.document_id, chunk_count=0)

        texts = [c.embed_text() for c in chunks]
        vectors = self.embed_batch(texts)

        collection = req.category or "default"
        self._searcher.ensure_collection(collection, vector_size=len(vectors[0]))

        import datetime as _dt
        last_indexed = _dt.datetime.now(_dt.timezone.utc).isoformat()
        points = []
        for chunk, vec in zip(chunks, vectors):
            points.append({
                "id": chunk.stable_uuid(),
                "vector": vec,
                "payload": {
                    "document_id": req.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "tenant_id": req.tenant_id,
                    "category": req.category,
                    "title": req.title,
                    "heading_path": " > ".join(chunk.heading_path),
                    "contains_table": str(chunk.contains_table).lower(),
                    "contains_link": str(chunk.contains_link).lower(),
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "last_indexed": last_indexed,
                    "content": chunk.content,
                    **{k: v for k, v in req.metadata.items()},
                },
            })
        self._searcher.upsert(collection, points)

        return IndexResponse(
            document_id=req.document_id,
            chunk_count=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
        )

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
        if not opts.use_reranking:
            opts.use_reranking = d.use_reranking
        if not opts.use_hybrid:
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


def _merge_best(
    existing: list[SearchResult],
    new: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Union of two result lists, deduped by document_id, limited to top_k by score."""
    seen: dict[str, SearchResult] = {r.document_id: r for r in existing}
    for r in new:
        if r.document_id not in seen or r.score > seen[r.document_id].score:
            seen[r.document_id] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]


def _apply_routing_strategy(opts: SearchOptions, strategy: str) -> SearchOptions:
    """Return a copy of opts with use_hybrid/use_reranking set per router strategy."""
    if strategy == "vector_only":
        return opts.model_copy(update={"use_hybrid": False, "use_reranking": False})
    if strategy == "hybrid":
        return opts.model_copy(update={"use_hybrid": True, "use_reranking": False})
    if strategy == "hybrid+rerank":
        return opts.model_copy(update={"use_hybrid": True, "use_reranking": True})
    return opts


def _strategy_name(opts: SearchOptions) -> str:
    if opts.use_hybrid and opts.use_reranking:
        return "hybrid+rerank"
    if opts.use_hybrid:
        return "hybrid"
    if opts.use_reranking:
        return "vector+rerank"
    return "vector"
