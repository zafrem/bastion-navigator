"""15-step search pipeline (SRS §6.3)."""
from __future__ import annotations

import time
import uuid
from typing import Optional

import concurrent.futures
import hashlib

from .config import Config, SearchDefaultsConfig
from .embedder import Embedder
from .reranker import Reranker
from .searcher import QdrantSearcher, MockSearcher
from .vault_client import VaultClient, NoopVaultClient
from .token_rewriter import TokenRewriter
from .router import Router, RoutingDecision, QueryIntent
from .evaluator import Evaluator, EvaluatorConfig, QualityVerdict
from .hyde import HyDETransformer
from .decomposer import QueryDecomposer, DecomposedQuery
from .models import (
    IndexRequest, IndexResponse,
    DeltaIndexRequest, DeltaIndexResponse,
    UpdatePurposesRequest, UpdatePurposesResponse,
    SearchOptions, SearchRequest, SearchResponse, SearchResult, SearchMetadata,
)
from .events import (
    Publisher, TraceContext,
    event_chunk_retrieved, event_query_routed,
    event_search_iteration, event_loop_completed,
    event_purpose_filtered, event_chunk_stale, event_query_transformed,
    event_document_reindexed, event_sub_queries_decomposed,
    event_steward_purposes_updated,
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
        self._hyde: Optional[HyDETransformer] = None
        self._decomposer: Optional[QueryDecomposer] = None

    def configure_modular_rag(self, router: Router, evaluator: Evaluator) -> None:
        """Enable Modular RAG components (MR-01/03/02/04/05). Called from main.py when enabled."""
        self._router = router
        self._evaluator = evaluator
        hyde_cfg = self._cfg.modular_rag.hyde
        if hyde_cfg.enabled:
            self._hyde = HyDETransformer(
                llm_endpoint=hyde_cfg.llm_endpoint,
                timeout_ms=hyde_cfg.timeout_ms,
            )
        self._decomposer = QueryDecomposer(self._cfg.decomposer)

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

        # MR-02-003: Sub-query decomposition for multi-hop queries.
        # Each sub-query is searched independently; results are merged with RRF.
        if (
            self._decomposer is not None
            and self._cfg.decomposer.enabled
            and self._router is not None
        ):
            routing = self._router.route(req.query, collections)
            if routing.intent == QueryIntent.MULTI_HOP:
                t_decomp = time.perf_counter()
                decomposed = self._decomposer.decompose(req.query, routing.intent.value)
                decomp_ms = (time.perf_counter() - t_decomp) * 1000
                if len(decomposed.sub_queries) > 1:
                    if self._publisher:
                        self._publisher.publish(event_sub_queries_decomposed(
                            tc, len(req.query), len(decomposed.sub_queries),
                            decomposed.strategy, decomp_ms,
                        ))
                    return self._search_decomposed(req, decomposed, collections, opts, filters, tc, start)

        over_fetch = opts.top_k * self._cfg.search_defaults.over_fetch_multiplier
        mr_cfg = self._cfg.modular_rag
        use_loop = (
            self._evaluator is not None
            and mr_cfg.enabled
            and mr_cfg.loop.max_iterations > 1
        )

        # MR-04-002: raise CRITICAL event if purpose=training_data (requires human review).
        if req.purpose == "training_data" and self._publisher:
            from .events import _new_event
            ev = _new_event(tc, "training_data_access_requested", "critical", "security", {
                "declared_purpose": req.purpose,
                "tenant_id": req.tenant_id,
            })
            ev.status = "alert"
            ev.action_taken = "alert_raised"
            self._publisher.publish(ev)

        # MR-02-001 applied to query: rewrite tokens before embedding.
        query_for_embed = (
            self._rewriter.rewrite_text(req.query) if self._rewriter else req.query
        )

        # MR-02-002: HyDE — replace query embedding with hypothetical document embedding.
        hyde_used = False
        if self._hyde is not None and opts.use_hyde or (
            self._hyde is not None
            and self._cfg.modular_rag.enabled
            and self._cfg.modular_rag.hyde.enabled
        ):
            routing_intent = "ambiguous"
            if self._router:
                _rd = self._router.route(query_for_embed, [], strategy_override="")
                routing_intent = _rd.intent.value
            hyde_cfg = self._cfg.modular_rag.hyde
            if self._hyde.should_apply(query_for_embed, routing_intent, hyde_cfg.max_words):
                t_hyde = time.perf_counter()
                hyp = self._hyde.generate(query_for_embed, routing_intent)
                hyde_ms = (time.perf_counter() - t_hyde) * 1000
                if hyp != query_for_embed:
                    hyde_used = True
                    if self._publisher:
                        self._publisher.publish(event_query_transformed(
                            tc, "hyde",
                            len(query_for_embed), len(hyp), hyde_ms,
                        ))
                    query_for_embed = hyp

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

            # MR-04-002: purpose pre-filter — exclude results whose permitted_purposes
            # does not include the declared purpose.
            if req.purpose:
                iter_results, purpose_excluded = _filter_by_purpose(iter_results, req.purpose)
                if purpose_excluded and self._publisher:
                    for r in purpose_excluded:
                        pp = r.metadata.get("permitted_purposes", "").split(",")
                        self._publisher.publish(
                            event_purpose_filtered(tc, r.document_id, req.purpose, pp)
                        )

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

        # MR-05-004: staleness check — flag chunks older than staleness_threshold_days.
        stale_cfg = self._cfg.modular_rag.staleness
        if stale_cfg.enabled and all_results:
            all_results = _check_staleness(all_results, stale_cfg.threshold_days, tc, self._publisher)

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
            router_cfg = self._cfg.modular_rag.router
            threshold = router_cfg.tenant_thresholds.get(
                tc.tenant_id, router_cfg.routing_threshold
            )
            # FR-MR-01-003: embedding-based domain affinity (opt-in).
            query_vector = None
            topic_vectors = None
            if router_cfg.use_embedding_affinity:
                get_topics = getattr(self._searcher, "get_topic_vectors", None)
                if callable(get_topics):
                    topic_vectors = get_topics(permission_collections)
                    if topic_vectors:
                        query_vector = self._embedder.embed(query)
            routing = self._router.route(
                query,
                permission_collections,
                routing_threshold=threshold,
                strategy_override=opts.strategy,
                query_vector=query_vector,
                topic_vectors=topic_vectors,
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
        import hashlib as _hashlib
        from .chunker import (
            chunk_by_profile, profile_for_mime,
            ChunkerConfig, _default_pii_patterns,
        )
        profile = profile_for_mime(req.mime_type)
        pii_patterns = _default_pii_patterns()
        ck = self._cfg.chunking
        if profile in ("structured_csv", "json_record", "html"):
            chunks = chunk_by_profile(req.document_id, req.content, profile, dict(req.metadata))
        else:
            from .chunker import chunk_document
            chunks = chunk_document(
                req.document_id,
                req.content,
                ChunkerConfig(
                    pii_patterns=pii_patterns,
                    heading_pattern=ck.heading_pattern,
                    table_row_pattern=ck.table_row_pattern,
                    fence_pattern=ck.fence_pattern,
                    link_pattern=ck.link_pattern,
                ),
                dict(req.metadata),
            )
        content_hash = _hashlib.sha256(req.content.encode("utf-8")).hexdigest()
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
                    "permitted_purposes": ",".join(req.permitted_purposes),
                    "content_hash": content_hash,          # MR-06-003
                    "source_version": req.source_version,  # MR-06-003
                    "mime_type": req.mime_type,             # MR-06-004
                    "content": chunk.content,
                    **{k: v for k, v in req.metadata.items()},
                },
            })
        self._searcher.upsert(collection, points)

        # FR-MR-01-003: fold the new chunk vectors into the collection topic
        # centroid used for embedding-based domain routing. Best-effort.
        update_topic = getattr(self._searcher, "update_topic_vector", None)
        if callable(update_topic):
            update_topic(collection, vectors)

        return IndexResponse(
            document_id=req.document_id,
            chunk_count=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
            content_hash=content_hash,
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

    # ── sub-query decomposition (MR-02-003) ──────────────────────────────────

    def _search_decomposed(
        self,
        req: SearchRequest,
        decomposed: DecomposedQuery,
        collections: list[str],
        opts: SearchOptions,
        filters: dict[str, str],
        tc: TraceContext,
        start: float,
    ) -> SearchResponse:
        """Execute N sub-queries in parallel and merge results with RRF."""
        max_workers = min(len(decomposed.sub_queries), self._cfg.decomposer.max_sub_queries)

        def _run_sub(sub_query: str) -> list[SearchResult]:
            vec = self._embedder.embed(
                self._rewriter.rewrite_text(sub_query) if self._rewriter else sub_query
            )
            results: list[SearchResult] = []
            for col in collections:
                results.extend(self._search_collection(col, sub_query, vec, filters, opts,
                                                        opts.top_k * self._cfg.search_defaults.over_fetch_multiplier))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_sub, sq): sq for sq in decomposed.sub_queries}
            all_sets: list[list[SearchResult]] = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    all_sets.append(fut.result())
                except Exception:
                    pass

        # Merge all result sets with equal-weight RRF.
        if not all_sets:
            merged: list[SearchResult] = []
        elif len(all_sets) == 1:
            merged = all_sets[0]
        else:
            merged = all_sets[0]
            for result_set in all_sets[1:]:
                merged = _rrf(merged, result_set)

        merged = _top_k(merged, opts.top_k)

        if opts.use_reranking and merged:
            merged = self._reranker.rerank(req.query, merged, opts.top_k)

        duration_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(
            request_id=req.request_id,
            results=merged,
            metadata=SearchMetadata(
                strategy="decompose+rrf",
                total_candidates=sum(len(s) for s in all_sets),
                filtered_out=0,
                final_count=len(merged),
            ),
            processing_time_ms=duration_ms,
        )

    # ── delta indexing (MR-06-002/003) ───────────────────────────────────────

    def delta_index_document(self, req: DeltaIndexRequest) -> DeltaIndexResponse:
        """Re-index a document only when its content hash has changed (MR-06-002).

        Steps (SC-10: deletion is sequential with insertion):
          1. Compute SHA-256 of new content.
          2. Check stored hash in Qdrant — skip if identical (unless req.force=True).
          3. Count + delete old chunks.
          4. Index new content.
        """
        new_hash = hashlib.sha256(req.content.encode("utf-8")).hexdigest()
        collection = req.category or "default"

        # Attempt to retrieve stored hash from an existing chunk's payload.
        stored_hash = self._get_stored_hash(collection, req.document_id)

        if not req.force and stored_hash and stored_hash == new_hash:
            return DeltaIndexResponse(
                document_id=req.document_id,
                indexed=False,
                content_hash=new_hash,
            )

        # Count old chunks before deletion (for the reindex event).
        old_count = self._searcher.count_by_document(collection, req.document_id)

        # Delete all existing chunks (SC-10: must complete before upsert).
        self._searcher.delete_by_document(collection, req.document_id)

        # Index the new content using the standard path.
        index_req = IndexRequest(
            document_id=req.document_id,
            tenant_id=req.tenant_id,
            category=req.category,
            title=req.title,
            content=req.content,
            metadata=req.metadata,
            permitted_purposes=req.permitted_purposes,
            mime_type=req.mime_type,
            source_version=req.source_version,
        )
        index_resp = self.index_document(index_req)

        if self._publisher:
            tc = TraceContext(tenant_id=req.tenant_id)
            self._publisher.publish(event_document_reindexed(
                tc,
                document_id=req.document_id,
                collection=collection,
                old_chunk_count=old_count,
                new_chunk_count=index_resp.chunk_count,
                changed_sections=[],
                reindex_ms=0.0,
            ))

        return DeltaIndexResponse(
            document_id=req.document_id,
            indexed=True,
            chunk_count=index_resp.chunk_count,
            old_chunk_count=old_count,
            content_hash=new_hash,
        )

    def _get_stored_hash(self, collection: str, document_id: str) -> str:
        """Retrieve the content_hash from a stored chunk payload, or '' if not found."""
        # Use sparse_search as a payload-filter scroll to get one chunk.
        try:
            results = self._searcher.vector_search(
                collection, [0.0] * 1024,
                {"document_id": document_id}, top_k=1
            )
            if results:
                return results[0].metadata.get("content_hash", "")
        except Exception:
            pass
        return ""

    # ── data steward — update purposes (MR-04-004) ───────────────────────────

    def update_document_purposes(self, req: UpdatePurposesRequest) -> UpdatePurposesResponse:
        """Steward updates permitted_purposes on all chunks of a document.

        Uses Qdrant set_payload() so no re-embedding is required.
        """
        collection = req.collection or req.tenant_id and "default" or "default"
        pp_str = ",".join(req.permitted_purposes)
        updated = self._searcher.set_payload(
            collection, req.document_id, {"permitted_purposes": pp_str}
        )
        if self._publisher:
            tc = TraceContext(tenant_id=req.tenant_id, user_id=req.steward_user_id)
            self._publisher.publish(event_steward_purposes_updated(
                tc,
                document_id=req.document_id,
                collection=collection,
                steward_user_id=req.steward_user_id,
                old_purposes=[],
                new_purposes=req.permitted_purposes,
            ))
        return UpdatePurposesResponse(
            document_id=req.document_id,
            collection=collection,
            chunks_updated=updated,
            permitted_purposes=req.permitted_purposes,
        )

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


def _filter_by_purpose(
    results: list[SearchResult],
    purpose: str,
) -> tuple[list[SearchResult], list[SearchResult]]:
    """Split results into (allowed, excluded) by purpose pre-filter (MR-04-002).

    A result is allowed when:
    - purpose is empty (no declared purpose → allow all), OR
    - no permitted_purposes set on the document (backwards-compatible → allow all), OR
    - the declared purpose appears in the comma-separated permitted_purposes metadata.
    """
    if not purpose:
        return list(results), []
    allowed: list[SearchResult] = []
    excluded: list[SearchResult] = []
    for r in results:
        pp_str = r.metadata.get("permitted_purposes", "")
        if not pp_str or purpose in pp_str.split(","):
            allowed.append(r)
        else:
            excluded.append(r)
    return allowed, excluded


def _check_staleness(
    results: list[SearchResult],
    threshold_days: int,
    tc,
    publisher,
) -> list[SearchResult]:
    """Flag results whose last_indexed exceeds threshold_days; return updated list (MR-05-004)."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    threshold = timedelta(days=threshold_days)
    updated: list[SearchResult] = []
    for r in results:
        if not r.last_indexed:
            updated.append(r)
            continue
        try:
            li = datetime.fromisoformat(r.last_indexed.replace("Z", "+00:00"))
            age = now - li
            if age > threshold:
                days_stale = age.days
                new_meta = dict(r.metadata)
                new_meta["stale"] = "true"
                new_meta["days_stale"] = str(days_stale)
                r = r.model_copy(update={"metadata": new_meta})
                if publisher:
                    from .events import event_chunk_stale
                    publisher.publish(
                        event_chunk_stale(tc, r.chunk_id, r.document_id, r.last_indexed, days_stale)
                    )
        except (ValueError, TypeError, AttributeError):
            pass
        updated.append(r)
    return updated


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
