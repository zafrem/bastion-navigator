"""Unit tests for navigator.router (MR-01)."""
from __future__ import annotations

import pytest
from navigator.router import QueryIntent, Router

_COLLECTIONS = ["customer_docs", "manufacturing_docs", "hr_docs"]


@pytest.fixture
def router() -> Router:
    return Router()


class TestIntentClassification:
    def test_factual_query(self, router):
        d = router.route("What is the defect rate for product X?", _COLLECTIONS)
        assert d.intent == QueryIntent.FACTUAL

    def test_analytical_query(self, router):
        d = router.route("Compare Q3 vs Q4 defect trends", _COLLECTIONS)
        assert d.intent == QueryIntent.ANALYTICAL

    def test_procedural_query(self, router):
        d = router.route("How to apply for annual leave?", _COLLECTIONS)
        assert d.intent == QueryIntent.PROCEDURAL

    def test_multi_hop_query(self, router):
        d = router.route("Show employees and also their policy records", _COLLECTIONS)
        assert d.intent == QueryIntent.MULTI_HOP

    def test_ambiguous_query(self, router):
        d = router.route("xyz1234", _COLLECTIONS)
        assert d.intent == QueryIntent.AMBIGUOUS

    def test_korean_factual(self, router):
        d = router.route("고객 계좌 조회해줘", _COLLECTIONS)
        assert d.intent == QueryIntent.FACTUAL

    def test_korean_analytical(self, router):
        d = router.route("생산 불량 비율 분석", _COLLECTIONS)
        assert d.intent == QueryIntent.ANALYTICAL


class TestStrategySelection:
    def test_factual_gives_vector_only(self, router):
        d = router.route("What is the defect rate?", _COLLECTIONS)
        assert d.strategy == "vector_only"

    def test_analytical_gives_hybrid_rerank(self, router):
        d = router.route("Compare Q3 vs Q4 statistics", _COLLECTIONS)
        assert d.strategy == "hybrid+rerank"

    def test_procedural_gives_hybrid(self, router):
        d = router.route("How to apply for leave?", _COLLECTIONS)
        assert d.strategy == "hybrid"

    def test_strategy_override(self, router):
        d = router.route("What is X?", _COLLECTIONS, strategy_override="hybrid")
        assert d.strategy == "hybrid"


class TestCollectionSelection:
    def test_customer_query_targets_customer_docs(self, router):
        d = router.route("고객 계좌 정보 조회", _COLLECTIONS)
        assert "customer_docs" in d.collections

    def test_hr_query_targets_hr_docs(self, router):
        d = router.route("employee salary information", _COLLECTIONS)
        assert "hr_docs" in d.collections

    def test_manufacturing_query_targets_mfg_docs(self, router):
        d = router.route("production defect rate on factory line", _COLLECTIONS)
        assert "manufacturing_docs" in d.collections

    def test_generic_query_returns_all_collections(self, router):
        d = router.route("xyz unknown query", _COLLECTIONS)
        assert set(d.collections) == set(_COLLECTIONS)

    def test_excluded_plus_selected_equals_available(self, router):
        d = router.route("customer purchase history", _COLLECTIONS)
        assert sorted(d.collections + d.excluded) == sorted(_COLLECTIONS)

    def test_empty_collections_returns_empty(self, router):
        d = router.route("something", [])
        assert d.collections == []


class TestRoutingMetadata:
    def test_confidence_is_between_0_and_1(self, router):
        d = router.route("What is X?", _COLLECTIONS)
        assert 0.0 <= d.confidence <= 1.0

    def test_routing_ms_is_positive(self, router):
        d = router.route("What is X?", _COLLECTIONS)
        assert d.routing_ms >= 0.0

    def test_routing_ms_under_5ms(self, router):
        import time
        t0 = time.perf_counter()
        for _ in range(20):
            router.route("What is the defect rate?", _COLLECTIONS)
        avg_ms = (time.perf_counter() - t0) * 1000 / 20
        assert avg_ms < 5.0, f"routing avg {avg_ms:.2f} ms exceeds 5 ms budget"
