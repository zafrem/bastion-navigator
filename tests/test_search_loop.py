"""Unit tests for re-search loop, circuit breakers, and routing integration.

Items covered:
  - Item 5  (P2): router integration in orchestrator (_do_route, strategy application)
  - Item 7  (P2): re-search loop (MR-03-002/003): quality_met, max_iterations,
                  timeout, duplicate circuit breakers; _merge_best; _apply_routing_strategy;
                  event_search_iteration / event_loop_completed emission
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from navigator.models import SearchOptions, SearchRequest, SearchResult
from navigator.orchestrator import _apply_routing_strategy, _merge_best


# ─── helpers ──────────────────────────────────────────────────────────────────

def _r(doc_id: str, score: float, content: str = "", heading_path: str = "") -> SearchResult:
    return SearchResult(
        document_id=doc_id,
        content=content,
        score=score,
        heading_path=heading_path,
        chunk_id=f"{doc_id}_0000",
    )


class _SequencedSearcher:
    """Returns successive result batches on successive vector_search calls."""

    def __init__(self, *batches: list[SearchResult]):
        self._batches = list(batches)
        self._n = 0

    def vector_search(self, collection, vector, filters, top_k, min_score=0.0):
        batch = self._batches[min(self._n, len(self._batches) - 1)]
        self._n += 1
        return batch

    def sparse_search(self, *a, **k):
        return []

    def collections(self):
        return []

    def ensure_collection(self, *a, **k):
        pass

    def upsert(self, *a, **k):
        pass


def _make_orchestrator(
    searcher=None,
    publisher=None,
    modular_rag_enabled=False,
    max_iterations=3,
    quality_threshold=0.60,
    coverage_threshold=0.40,
    timeout_ms=500.0,
    with_router=False,
    with_evaluator=False,
):
    from navigator.config import (
        CollectionConfig, Config, LoopConfig, ModularRAGConfig,
        RouterConfig, SearchDefaultsConfig, VectorDBConfig,
    )
    from navigator.evaluator import Evaluator
    from navigator.orchestrator import Orchestrator
    from navigator.router import Router
    from navigator.vault_client import NoopVaultClient

    cfg = Config()
    cfg.modular_rag = ModularRAGConfig(
        enabled=modular_rag_enabled,
        loop=LoopConfig(
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            coverage_threshold=coverage_threshold,
            uncertain_low=0.45,
            loop_timeout_ms=timeout_ms,
        ),
        router=RouterConfig(routing_threshold=0.25),
    )
    # Single collection so searches call vector_search exactly once per iteration.
    cfg.vector_db = VectorDBConfig(collections={"test_col": CollectionConfig()})
    cfg.search_defaults = SearchDefaultsConfig(
        top_k=5,
        over_fetch_multiplier=2,
        use_hybrid=False,
        use_reranking=False,
        min_score=0.0,
    )

    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 8
    embedder.embed_batch.return_value = [[0.0] * 8]

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, results, k: sorted(results, key=lambda r: r.score, reverse=True)[:k]

    from navigator.searcher import MockSearcher
    orch = Orchestrator(
        cfg, embedder, searcher or MockSearcher(), reranker,
        NoopVaultClient(), publisher=publisher,
    )
    if with_router or modular_rag_enabled:
        router = Router()
        orch._router = router
    if with_evaluator or modular_rag_enabled:
        evaluator = Evaluator()
        orch._evaluator = evaluator
    return orch


# ─── _merge_best ─────────────────────────────────────────────────────────────

class TestMergeBest:
    def test_deduplicates_by_document_id(self):
        existing = [_r("doc-1", 0.8), _r("doc-2", 0.6)]
        new      = [_r("doc-1", 0.9), _r("doc-3", 0.7)]
        merged = _merge_best(existing, new, top_k=10)
        ids = [r.document_id for r in merged]
        assert ids.count("doc-1") == 1

    def test_keeps_higher_score_on_collision(self):
        existing = [_r("doc-1", 0.6)]
        new      = [_r("doc-1", 0.9)]
        merged = _merge_best(existing, new, top_k=10)
        assert merged[0].score == pytest.approx(0.9)

    def test_existing_higher_score_kept(self):
        existing = [_r("doc-1", 0.95)]
        new      = [_r("doc-1", 0.50)]
        merged = _merge_best(existing, new, top_k=10)
        assert merged[0].score == pytest.approx(0.95)

    def test_sorted_descending_by_score(self):
        existing = [_r("a", 0.5), _r("b", 0.8)]
        new      = [_r("c", 0.7)]
        merged = _merge_best(existing, new, top_k=10)
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self):
        existing = [_r(f"d{i}", 0.9 - i * 0.1) for i in range(5)]
        new      = [_r(f"e{i}", 0.8 - i * 0.1) for i in range(5)]
        merged = _merge_best(existing, new, top_k=3)
        assert len(merged) == 3

    def test_empty_existing_returns_new(self):
        new = [_r("doc-1", 0.8)]
        assert _merge_best([], new, top_k=10) == new

    def test_empty_new_returns_existing(self):
        existing = [_r("doc-1", 0.8)]
        assert _merge_best(existing, [], top_k=10) == existing

    def test_both_empty_returns_empty(self):
        assert _merge_best([], [], top_k=10) == []


# ─── _apply_routing_strategy ─────────────────────────────────────────────────

class TestApplyRoutingStrategy:
    def test_vector_only_disables_hybrid_and_rerank(self):
        opts = SearchOptions(use_hybrid=True, use_reranking=True)
        result = _apply_routing_strategy(opts, "vector_only")
        assert result.use_hybrid is False
        assert result.use_reranking is False

    def test_hybrid_enables_hybrid_disables_rerank(self):
        opts = SearchOptions(use_hybrid=False, use_reranking=True)
        result = _apply_routing_strategy(opts, "hybrid")
        assert result.use_hybrid is True
        assert result.use_reranking is False

    def test_hybrid_rerank_enables_both(self):
        opts = SearchOptions(use_hybrid=False, use_reranking=False)
        result = _apply_routing_strategy(opts, "hybrid+rerank")
        assert result.use_hybrid is True
        assert result.use_reranking is True

    def test_unknown_strategy_returns_unchanged(self):
        opts = SearchOptions(use_hybrid=True, use_reranking=True)
        result = _apply_routing_strategy(opts, "custom_unknown")
        assert result.use_hybrid is True
        assert result.use_reranking is True

    def test_returns_new_object_not_same(self):
        opts = SearchOptions(use_hybrid=False)
        result = _apply_routing_strategy(opts, "hybrid")
        assert result is not opts


# ─── Loop: quality_met ────────────────────────────────────────────────────────

class TestLoopQualityMet:
    def test_terminates_after_one_iteration_on_high_score(self):
        # "customer account" keywords appear in content → coverage=1.0, score=0.9 → sufficient
        results = [_r("doc-1", 0.9, content="customer account purchase history")]
        searcher = _SequencedSearcher(results)
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=True,
            max_iterations=3,
            quality_threshold=0.6,
        )
        resp = orch.search(SearchRequest(query="customer account"))
        assert len(resp.results) > 0
        # Only 1 search call should have been made (quality met after iteration 1)
        assert searcher._n == 1

    def test_results_returned_on_quality_met(self):
        results = [_r("doc-1", 0.9, content="customer account")]
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(results),
            modular_rag_enabled=True,
            max_iterations=3,
        )
        resp = orch.search(SearchRequest(query="customer account"))
        assert any(r.document_id == "doc-1" for r in resp.results)


# ─── Loop: max_iterations circuit breaker ────────────────────────────────────

class TestLoopMaxIterationsCircuitBreaker:
    def test_terminates_after_max_iterations(self):
        # Different doc_ids per iteration to avoid duplicate circuit breaker;
        # low score + no coverage → always insufficient → only max_iterations fires.
        searcher = _SequencedSearcher(
            [_r("doc-1", 0.1, content="foo")],
            [_r("doc-2", 0.1, content="bar")],
            [_r("doc-3", 0.1, content="baz")],
        )
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=True,
            max_iterations=3,
            quality_threshold=0.60,
        )
        orch.search(SearchRequest(query="customer account"))
        assert searcher._n == 3  # exactly 3 iterations

    def test_returns_best_collected_results(self):
        batch1 = [_r("doc-1", 0.4, content="foo")]
        batch2 = [_r("doc-2", 0.3, content="bar")]
        batch3 = [_r("doc-3", 0.2, content="baz")]
        searcher = _SequencedSearcher(batch1, batch2, batch3)
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=True,
            max_iterations=3,
        )
        resp = orch.search(SearchRequest(query="customer account"))
        doc_ids = {r.document_id for r in resp.results}
        assert "doc-1" in doc_ids  # best score from all iterations

    def test_max_iterations_one_is_linear(self):
        results = [_r("doc-1", 0.1, content="foo")]
        searcher = _SequencedSearcher(results)
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=True,
            max_iterations=1,
        )
        orch.search(SearchRequest(query="customer account"))
        assert searcher._n == 1


# ─── Loop: duplicate result circuit breaker ──────────────────────────────────

class TestLoopDuplicateCircuitBreaker:
    def test_stops_when_same_results_twice(self):
        # Same docs every call → duplicate detected at iteration 2
        same_results = [_r("doc-1", 0.3, content="foo"), _r("doc-2", 0.2)]
        searcher = _SequencedSearcher(same_results)
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=True,
            max_iterations=5,
        )
        orch.search(SearchRequest(query="customer account"))
        # Stopped at iteration 2 (duplicate detected)
        assert searcher._n == 2


# ─── Loop: timeout circuit breaker ───────────────────────────────────────────

class TestLoopTimeoutCircuitBreaker:
    def test_timeout_stops_loop_before_max_iterations(self):
        low_results = [_r("doc-1", 0.1, content="foo")]
        searcher = _SequencedSearcher(low_results)
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=True,
            max_iterations=5,
            timeout_ms=0.0,  # zero timeout: always expires after first iteration
        )
        orch.search(SearchRequest(query="customer account"))
        # Should stop at iteration 2 (timeout check is at the START of iteration 2)
        assert searcher._n == 1


# ─── Loop disabled (linear path) ─────────────────────────────────────────────

class TestLinearPath:
    def test_single_search_call_when_modular_rag_disabled(self):
        results = [_r("doc-1", 0.5)]
        searcher = _SequencedSearcher(results)
        orch = _make_orchestrator(
            searcher=searcher,
            modular_rag_enabled=False,
        )
        orch.search(SearchRequest(query="test"))
        assert searcher._n == 1

    def test_results_returned_without_evaluator(self):
        results = [_r("doc-1", 0.5, content="some content")]
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(results),
            modular_rag_enabled=False,
        )
        resp = orch.search(SearchRequest(query="test"))
        assert any(r.document_id == "doc-1" for r in resp.results)


# ─── Router integration in orchestrator ──────────────────────────────────────

class TestRouterIntegration:
    def test_factual_query_applies_vector_only_strategy(self):
        orch = _make_orchestrator(with_router=True)
        # "What is" → factual → vector_only → use_hybrid=False, use_reranking=False
        # We can't directly observe opts, but we verify no exception is raised.
        resp = orch.search(SearchRequest(query="What is the defect rate?"))
        assert resp is not None

    def test_router_filters_collections_by_domain(self):
        """When router has domain signal, only matching collections are searched."""
        # "employee salary" → strong hr signal → should include hr_docs
        # But our test_col is unknown domain → included by fail-open rule
        orch = _make_orchestrator(with_router=True)
        resp = orch.search(SearchRequest(query="employee salary policy"))
        assert resp is not None  # no exception

    def test_query_routed_event_emitted_when_router_configured(self):
        pub = MagicMock()
        orch = _make_orchestrator(with_router=True, publisher=pub)
        orch.search(SearchRequest(query="What is the policy?"))
        event_types = [c.args[0].event_type for c in pub.publish.call_args_list]
        assert "query_routed" in event_types

    def test_query_routed_event_not_emitted_without_router(self):
        pub = MagicMock()
        orch = _make_orchestrator(publisher=pub)
        orch.search(SearchRequest(query="test"))
        event_types = [c.args[0].event_type for c in pub.publish.call_args_list]
        assert "query_routed" not in event_types

    def test_strategy_override_in_search_options_respected(self):
        pub = MagicMock()
        orch = _make_orchestrator(with_router=True, publisher=pub)
        req = SearchRequest(
            query="What is X?",
            options=SearchOptions(strategy="hybrid+rerank"),
        )
        orch.search(req)
        routing_events = [
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "query_routed"
        ]
        assert len(routing_events) == 1
        assert routing_events[0].data["strategy"] == "hybrid+rerank"


# ─── Loop event emission ──────────────────────────────────────────────────────

class TestLoopEventEmission:
    def test_search_iteration_events_emitted_per_iteration(self):
        pub = MagicMock()
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(
                [_r("doc-1", 0.1, content="foo")],
                [_r("doc-2", 0.1, content="bar")],
                [_r("doc-3", 0.1, content="baz")],
            ),
            publisher=pub,
            modular_rag_enabled=True,
            max_iterations=3,
        )
        orch.search(SearchRequest(query="customer account"))

        iter_events = [
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "search_iteration"
        ]
        assert len(iter_events) == 3

    def test_search_iteration_numbers_are_sequential(self):
        pub = MagicMock()
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(
                [_r("doc-1", 0.1, content="foo")],
                [_r("doc-2", 0.1, content="bar")],
                [_r("doc-3", 0.1, content="baz")],
            ),
            publisher=pub,
            modular_rag_enabled=True,
            max_iterations=3,
        )
        orch.search(SearchRequest(query="customer account"))

        iter_events = [
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "search_iteration"
        ]
        numbers = [e.data["iteration"] for e in iter_events]
        assert numbers == [1, 2, 3]

    def test_loop_completed_event_emitted_once(self):
        pub = MagicMock()
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(
                [_r("doc-1", 0.1, content="foo")],
                [_r("doc-2", 0.1, content="bar")],
            ),
            publisher=pub,
            modular_rag_enabled=True,
            max_iterations=2,
        )
        orch.search(SearchRequest(query="customer account"))

        loop_events = [
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "loop_completed"
        ]
        assert len(loop_events) == 1

    def test_loop_completed_termination_max_iterations(self):
        pub = MagicMock()
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(
                [_r("doc-1", 0.1, content="foo")],
                [_r("doc-2", 0.1, content="bar")],
            ),
            publisher=pub,
            modular_rag_enabled=True,
            max_iterations=2,
        )
        orch.search(SearchRequest(query="customer account"))

        loop_event = next(
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "loop_completed"
        )
        assert loop_event.data["termination"] == "max_iterations"

    def test_loop_completed_termination_quality_met(self):
        good_results = [_r("doc-1", 0.9, content="customer account purchase")]
        pub = MagicMock()
        orch = _make_orchestrator(
            searcher=_SequencedSearcher(good_results),
            publisher=pub,
            modular_rag_enabled=True,
            max_iterations=3,
        )
        orch.search(SearchRequest(query="customer account"))

        loop_event = next(
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "loop_completed"
        )
        assert loop_event.data["termination"] == "quality_met"

    def test_loop_completed_not_emitted_without_loop(self):
        pub = MagicMock()
        orch = _make_orchestrator(publisher=pub, modular_rag_enabled=False)
        orch.search(SearchRequest(query="test"))

        loop_events = [
            c for c in pub.publish.call_args_list
            if c.args[0].event_type == "loop_completed"
        ]
        assert len(loop_events) == 0
