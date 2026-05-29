"""Unit tests for Priority 1/2 event factory functions and orchestrator emission.

Items covered:
  - Item 3  (P1): event_chunk_retrieved (MR-05-001)
  - Item 5  (P2): event_query_routed    (MR-01, FR-MR-01-004)
  - Item 7  (P2): event_search_iteration, event_loop_completed (MR-03, FR-MR-03-004)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from navigator.events import (
    TraceContext,
    event_chunk_retrieved,
    event_loop_completed,
    event_query_routed,
    event_search_iteration,
)
from navigator.models import SearchResult


# ─── helpers ──────────────────────────────────────────────────────────────────

def _tc(**kwargs) -> TraceContext:
    return TraceContext(
        trace_id="trace-001",
        span_id="span-001",
        tenant_id=kwargs.get("tenant_id", "acme"),
        request_id=kwargs.get("request_id", "req-001"),
    )


# ─── Item 3: event_chunk_retrieved (MR-05-001) ────────────────────────────────

class TestEventChunkRetrieved:
    def test_event_type(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82, 0)
        assert ev.event_type == "chunk_retrieved"

    def test_category_is_lineage(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82, 0)
        assert ev.category == "lineage"

    def test_chunk_id_in_data(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82, 0)
        assert ev.data["chunk_id"] == "doc-1_0002"

    def test_document_id_in_data(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82, 0)
        assert ev.data["document_id"] == "doc-1"

    def test_score_rounded_in_data(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82356789, 0)
        assert isinstance(ev.data["score"], float)
        assert ev.data["score"] == pytest.approx(0.8236, abs=0.001)

    def test_rank_in_data(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82, 3)
        assert ev.data["rank"] == 3

    def test_optional_collection_in_data(self):
        ev = event_chunk_retrieved(_tc(), "doc-1_0002", "doc-1", 0.82, 0, collection="hr_docs")
        assert ev.data["collection"] == "hr_docs"

    def test_status_is_retrieved(self):
        ev = event_chunk_retrieved(_tc(), "x", "y", 0.5, 0)
        assert ev.status == "retrieved"

    def test_action_taken_is_retrieve(self):
        ev = event_chunk_retrieved(_tc(), "x", "y", 0.5, 0)
        assert ev.action_taken == "retrieve"

    def test_trace_id_propagated(self):
        tc = _tc()
        ev = event_chunk_retrieved(tc, "x", "y", 0.5, 0)
        assert ev.trace_id == tc.trace_id

    def test_tenant_id_propagated(self):
        tc = _tc(tenant_id="globex")
        ev = event_chunk_retrieved(tc, "x", "y", 0.5, 0)
        assert ev.tenant_id == "globex"

    def test_module_is_navigator(self):
        ev = event_chunk_retrieved(_tc(), "x", "y", 0.5, 0)
        assert ev.module == "navigator"

    def test_event_id_is_unique(self):
        ev1 = event_chunk_retrieved(_tc(), "x", "y", 0.5, 0)
        ev2 = event_chunk_retrieved(_tc(), "x", "y", 0.5, 0)
        assert ev1.event_id != ev2.event_id


# ─── Item 5: event_query_routed (MR-01, FR-MR-01-004) ────────────────────────

class TestEventQueryRouted:
    def test_event_type(self):
        ev = event_query_routed(_tc(), "factual", "vector_only", ["customer_docs"], [], 0.85, 1.2)
        assert ev.event_type == "query_routed"

    def test_category_is_operational(self):
        ev = event_query_routed(_tc(), "factual", "vector_only", ["customer_docs"], [], 0.85, 1.2)
        assert ev.category == "operational"

    def test_intent_in_data(self):
        ev = event_query_routed(_tc(), "analytical", "hybrid+rerank", [], [], 0.7, 2.0)
        assert ev.data["intent"] == "analytical"

    def test_strategy_in_data(self):
        ev = event_query_routed(_tc(), "analytical", "hybrid+rerank", [], [], 0.7, 2.0)
        assert ev.data["strategy"] == "hybrid+rerank"

    def test_collections_in_data(self):
        ev = event_query_routed(_tc(), "factual", "vector_only", ["hr_docs", "customer_docs"], [], 0.8, 1.5)
        assert ev.data["collections"] == ["hr_docs", "customer_docs"]

    def test_excluded_in_data(self):
        ev = event_query_routed(_tc(), "factual", "vector_only", ["hr_docs"], ["customer_docs"], 0.8, 1.5)
        assert ev.data["excluded"] == ["customer_docs"]

    def test_confidence_rounded(self):
        ev = event_query_routed(_tc(), "factual", "v", [], [], 0.85678, 1.0)
        assert ev.data["confidence"] == pytest.approx(0.857, abs=0.001)

    def test_routing_ms_in_data(self):
        ev = event_query_routed(_tc(), "factual", "v", [], [], 0.8, 3.456)
        assert ev.data["routing_ms"] == pytest.approx(3.46, abs=0.01)

    def test_status_is_routed(self):
        ev = event_query_routed(_tc(), "factual", "v", [], [], 0.8, 1.0)
        assert ev.status == "routed"

    def test_action_taken_is_route(self):
        ev = event_query_routed(_tc(), "factual", "v", [], [], 0.8, 1.0)
        assert ev.action_taken == "route"


# ─── Item 7: event_search_iteration (MR-03, FR-MR-03-004) ────────────────────

class TestEventSearchIteration:
    def test_event_type(self):
        ev = event_search_iteration(_tc(), 1, "sufficient", 0.82, 0.75, None, 42.0)
        assert ev.event_type == "search_iteration"

    def test_category_is_operational(self):
        ev = event_search_iteration(_tc(), 1, "sufficient", 0.82, 0.75, None, 42.0)
        assert ev.category == "operational"

    def test_iteration_number_in_data(self):
        ev = event_search_iteration(_tc(), 2, "insufficient", 0.30, 0.20, "broaden", 55.0)
        assert ev.data["iteration"] == 2

    def test_verdict_in_data(self):
        ev = event_search_iteration(_tc(), 1, "insufficient", 0.30, 0.20, "broaden", 55.0)
        assert ev.data["verdict"] == "insufficient"

    def test_top_score_in_data(self):
        ev = event_search_iteration(_tc(), 1, "sufficient", 0.82, 0.75, None, 42.0)
        assert ev.data["top_score"] == pytest.approx(0.82, abs=0.001)

    def test_coverage_in_data(self):
        ev = event_search_iteration(_tc(), 1, "sufficient", 0.82, 0.75, None, 42.0)
        assert ev.data["coverage"] == pytest.approx(0.75, abs=0.001)

    def test_refinement_none_when_not_applied(self):
        ev = event_search_iteration(_tc(), 1, "sufficient", 0.82, 0.75, None, 42.0)
        assert ev.data["refinement"] is None

    def test_refinement_strategy_recorded(self):
        ev = event_search_iteration(_tc(), 2, "insufficient", 0.30, 0.20, "keyword_expand", 55.0)
        assert ev.data["refinement"] == "keyword_expand"

    def test_status_matches_verdict(self):
        ev = event_search_iteration(_tc(), 1, "insufficient", 0.30, 0.20, None, 55.0)
        assert ev.status == "insufficient"


# ─── Item 7: event_loop_completed (MR-03, FR-MR-03-004) ─────────────────────

class TestEventLoopCompleted:
    def test_event_type(self):
        ev = event_loop_completed(_tc(), 2, "quality_met", 5, 120.0)
        assert ev.event_type == "loop_completed"

    def test_category_is_operational(self):
        ev = event_loop_completed(_tc(), 2, "quality_met", 5, 120.0)
        assert ev.category == "operational"

    def test_total_iterations_in_data(self):
        ev = event_loop_completed(_tc(), 3, "max_iterations", 4, 480.0)
        assert ev.data["total_iterations"] == 3

    def test_termination_reason_in_data(self):
        ev = event_loop_completed(_tc(), 2, "timeout", 3, 510.0)
        assert ev.data["termination"] == "timeout"

    def test_final_result_count_in_data(self):
        ev = event_loop_completed(_tc(), 1, "quality_met", 7, 80.0)
        assert ev.data["final_result_count"] == 7

    def test_total_ms_in_data(self):
        ev = event_loop_completed(_tc(), 2, "quality_met", 5, 123.456)
        assert ev.data["total_ms"] == pytest.approx(123.46, abs=0.01)

    def test_status_is_completed(self):
        ev = event_loop_completed(_tc(), 1, "quality_met", 5, 80.0)
        assert ev.status == "completed"

    def test_action_taken_is_search_loop(self):
        ev = event_loop_completed(_tc(), 1, "quality_met", 5, 80.0)
        assert ev.action_taken == "search_loop"

    @pytest.mark.parametrize("reason", [
        "quality_met", "max_iterations", "timeout", "duplicate",
    ])
    def test_all_termination_reasons_accepted(self, reason):
        ev = event_loop_completed(_tc(), 1, reason, 5, 80.0)
        assert ev.data["termination"] == reason


# ─── Orchestrator: chunk_retrieved emitted per result ─────────────────────────

class TestOrchestratorChunkEventEmission:
    """Verify that the orchestrator emits chunk_retrieved for every result (Item 3)."""

    def _build_orchestrator(self, results: list[SearchResult], publisher: MagicMock):
        from unittest.mock import MagicMock
        from navigator.config import Config, SearchDefaultsConfig, VectorDBConfig, CollectionConfig
        from navigator.orchestrator import Orchestrator
        from navigator.searcher import MockSearcher
        from navigator.vault_client import NoopVaultClient

        cfg = Config()
        cfg.search_defaults = SearchDefaultsConfig(
            top_k=10, over_fetch_multiplier=2,
            use_hybrid=False, use_reranking=False, min_score=0.0,
        )
        cfg.vector_db = VectorDBConfig(collections={"test": CollectionConfig()})

        embedder = MagicMock()
        embedder.embed.return_value = [0.0] * 8
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, r, k: r[:k]

        class _FixedSearcher:
            def vector_search(self, *a, **k): return results
            def sparse_search(self, *a, **k): return []
            def collections(self): return []
            def ensure_collection(self, *a, **k): pass
            def upsert(self, *a, **k): pass

        return Orchestrator(cfg, embedder, _FixedSearcher(), reranker, NoopVaultClient(), publisher=publisher)

    def test_one_event_emitted_per_result(self):
        from navigator.models import SearchRequest
        results = [
            SearchResult(document_id=f"doc-{i}", score=0.8, chunk_id=f"doc-{i}_0000")
            for i in range(3)
        ]
        pub = MagicMock()
        orch = self._build_orchestrator(results, pub)

        orch.search(SearchRequest(query="test query"))

        chunk_events = [
            c for c in pub.publish.call_args_list
            if c.args[0].event_type == "chunk_retrieved"
        ]
        assert len(chunk_events) == 3

    def test_chunk_event_carries_correct_document_id(self):
        from navigator.models import SearchRequest
        results = [SearchResult(document_id="doc-xyz", score=0.8, chunk_id="doc-xyz_0001")]
        pub = MagicMock()
        orch = self._build_orchestrator(results, pub)

        orch.search(SearchRequest(query="test"))

        chunk_event = next(
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "chunk_retrieved"
        )
        assert chunk_event.data["document_id"] == "doc-xyz"

    def test_rank_increases_per_result(self):
        from navigator.models import SearchRequest
        results = [
            SearchResult(document_id=f"d{i}", score=0.9 - i * 0.1, chunk_id=f"d{i}_0000")
            for i in range(3)
        ]
        pub = MagicMock()
        orch = self._build_orchestrator(results, pub)

        orch.search(SearchRequest(query="test"))

        chunk_events = [
            c.args[0] for c in pub.publish.call_args_list
            if c.args[0].event_type == "chunk_retrieved"
        ]
        ranks = [e.data["rank"] for e in chunk_events]
        assert ranks == [0, 1, 2]

    def test_no_chunk_events_when_no_results(self):
        from navigator.models import SearchRequest
        pub = MagicMock()
        orch = self._build_orchestrator([], pub)

        orch.search(SearchRequest(query="test"))

        chunk_events = [
            c for c in pub.publish.call_args_list
            if c.args[0].event_type == "chunk_retrieved"
        ]
        assert len(chunk_events) == 0
