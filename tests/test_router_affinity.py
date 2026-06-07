"""Unit tests for embedding-based domain affinity (FR-MR-01-003)."""
from __future__ import annotations

from navigator.router import Router, _cosine
from navigator.searcher import MockSearcher, _fold_centroid

_COLLECTIONS = ["customer_docs", "manufacturing_docs", "hr_docs"]

# Orthogonal unit "topic" vectors, one axis per collection.
_TOPICS = {
    "customer_docs":      [1.0, 0.0, 0.0],
    "manufacturing_docs": [0.0, 1.0, 0.0],
    "hr_docs":            [0.0, 0.0, 1.0],
}


class TestCosine:
    def test_identical_vectors(self):
        assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch(self):
        assert _cosine([1.0, 0.0], [1.0]) == 0.0


class TestFoldCentroid:
    def test_first_batch_is_mean(self):
        c, n = _fold_centroid(None, 0, [[2.0, 0.0], [4.0, 0.0]])
        assert c == [3.0, 0.0] and n == 2

    def test_incremental_running_mean(self):
        c1, n1 = _fold_centroid(None, 0, [[0.0, 0.0]])
        c2, n2 = _fold_centroid(c1, n1, [[2.0, 2.0]])
        assert c2 == [1.0, 1.0] and n2 == 2


class TestEmbeddingSelection:
    def test_query_aligned_with_one_collection(self):
        r = Router()
        # Query points straight at customer_docs; the others are orthogonal (cos=0).
        d = r.route(
            "anything", _COLLECTIONS, routing_threshold=0.25,
            query_vector=[1.0, 0.0, 0.0], topic_vectors=_TOPICS,
        )
        assert d.collections == ["customer_docs"]
        assert sorted(d.excluded) == ["hr_docs", "manufacturing_docs"]

    def test_missing_topic_vector_is_kept(self):
        r = Router()
        partial = {"customer_docs": [1.0, 0.0, 0.0]}  # others have no centroid yet
        d = r.route(
            "anything", _COLLECTIONS, routing_threshold=0.25,
            query_vector=[1.0, 0.0, 0.0], topic_vectors=partial,
        )
        # customer_docs passes on score; the two without centroids fail open.
        assert set(d.collections) == set(_COLLECTIONS)
        assert d.excluded == []

    def test_all_below_threshold_fails_open(self):
        r = Router()
        d = r.route(
            "anything", _COLLECTIONS, routing_threshold=0.99,
            query_vector=[1.0, 1.0, 1.0], topic_vectors=_TOPICS,
        )
        # No collection clears 0.99 → fail open to all, nothing excluded.
        assert set(d.collections) == set(_COLLECTIONS)
        assert d.excluded == []

    def test_falls_back_to_keywords_without_vectors(self):
        r = Router()
        d = r.route("employee salary information", _COLLECTIONS)
        assert "hr_docs" in d.collections


class TestMockSearcherTopics:
    def test_update_then_retrieve(self):
        s = MockSearcher()
        s.update_topic_vector("hr_docs", [[2.0, 0.0], [0.0, 2.0]])
        assert s.get_topic_vector("hr_docs") == [1.0, 1.0]

    def test_running_mean_across_updates(self):
        s = MockSearcher()
        s.update_topic_vector("c", [[0.0, 0.0]])
        s.update_topic_vector("c", [[2.0, 2.0]])
        assert s.get_topic_vector("c") == [1.0, 1.0]

    def test_get_topic_vectors_batch_skips_unknown(self):
        s = MockSearcher()
        s.update_topic_vector("a", [[1.0, 0.0]])
        out = s.get_topic_vectors(["a", "b"])
        assert out == {"a": [1.0, 0.0]}

    def test_empty_vectors_noop(self):
        s = MockSearcher()
        s.update_topic_vector("a", [])
        assert s.get_topic_vector("a") is None


class TestOrchestratorWiring:
    """End-to-end: _do_route honours use_embedding_affinity + per-tenant threshold."""

    def _orch(self, *, use_affinity: bool, tenant_thresholds=None):
        from unittest.mock import MagicMock
        from navigator.config import Config, ModularRAGConfig, RouterConfig
        from navigator.orchestrator import Orchestrator
        from navigator.router import Router
        from navigator.vault_client import NoopVaultClient

        cfg = Config()
        cfg.modular_rag = ModularRAGConfig(
            enabled=True,
            router=RouterConfig(
                routing_threshold=0.25,
                use_embedding_affinity=use_affinity,
                tenant_thresholds=tenant_thresholds or {},
            ),
        )
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]  # points at customer_docs
        searcher = MockSearcher()
        searcher.update_topic_vector("customer_docs", [[1.0, 0.0, 0.0]])
        searcher.update_topic_vector("manufacturing_docs", [[0.0, 1.0, 0.0]])
        searcher.update_topic_vector("hr_docs", [[0.0, 0.0, 1.0]])
        orch = Orchestrator(cfg, embedder, searcher, MagicMock(), NoopVaultClient())
        orch._router = Router()
        return orch

    def _route(self, orch, tenant_id="t1"):
        from navigator.events import TraceContext
        from navigator.models import SearchOptions
        tc = TraceContext(tenant_id=tenant_id)
        cols, _ = orch._do_route("anything", list(_TOPICS.keys()), SearchOptions(), tc)
        return cols

    def test_affinity_enabled_narrows_collections(self):
        cols = self._route(self._orch(use_affinity=True))
        assert cols == ["customer_docs"]

    def test_affinity_disabled_keeps_keyword_behaviour(self):
        # Keyword proxy on "anything" finds no domain signal → all collections.
        cols = self._route(self._orch(use_affinity=False))
        assert set(cols) == set(_TOPICS.keys())

    def test_per_tenant_threshold_override(self):
        # Query leans mostly to customer_docs, partly to manufacturing_docs:
        #   cos(customer)=0.857, cos(manufacturing)=0.514, cos(hr)=0.
        orch = self._orch(use_affinity=True, tenant_thresholds={"vip": 0.7})
        orch._embedder.embed.return_value = [1.0, 0.6, 0.0]
        # Default tenant (threshold 0.25) keeps customer + manufacturing.
        assert set(self._route(orch, tenant_id="t1")) == {"customer_docs", "manufacturing_docs"}
        # vip tenant (threshold 0.7) keeps only customer_docs.
        assert self._route(orch, tenant_id="vip") == ["customer_docs"]
