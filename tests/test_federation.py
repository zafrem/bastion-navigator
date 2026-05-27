"""Unit tests for navigator.federation — RRF merge, FederationRouter, FederatedOrchestrator."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from navigator.federation import (
    FederationConfig,
    FederatedOrchestrator,
    FederationRouter,
    PeerClient,
    PeerConfig,
    build_federated_orchestrator,
    rrf_merge_multi,
)
from navigator.models import SearchMetadata, SearchRequest, SearchResponse, SearchResult


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_result(doc_id: str, score: float = 0.8) -> SearchResult:
    return SearchResult(document_id=doc_id, content="content", score=score)


def make_response(*doc_ids: str, strategy: str = "vector") -> SearchResponse:
    results = [make_result(d) for d in doc_ids]
    return SearchResponse(
        results=results,
        metadata=SearchMetadata(
            total_candidates=len(results),
            final_count=len(results),
            strategy=strategy,
        ),
    )


# ─── rrf_merge_multi ──────────────────────────────────────────────────────────

class TestRRFMerge:
    def test_single_list_preserves_order(self):
        results = [make_result(f"doc{i}", score=1.0 - i * 0.1) for i in range(5)]
        merged = rrf_merge_multi([results], k=60.0)
        ids = [r.document_id for r in merged]
        # RRF with single list re-scores but order should still match rank order
        assert ids == ["doc0", "doc1", "doc2", "doc3", "doc4"]

    def test_two_lists_deduplicates(self):
        list_a = [make_result("shared"), make_result("a_only")]
        list_b = [make_result("shared"), make_result("b_only")]
        merged = rrf_merge_multi([list_a, list_b], k=60.0)
        ids = [r.document_id for r in merged]
        assert len(ids) == len(set(ids)), "duplicates present"
        assert "shared" in ids
        assert "a_only" in ids
        assert "b_only" in ids

    def test_shared_doc_scores_higher_than_unique(self):
        # A doc appearing in both lists should outscore docs in only one list.
        list_a = [make_result("shared"), make_result("a_only")]
        list_b = [make_result("shared"), make_result("b_only")]
        merged = rrf_merge_multi([list_a, list_b], k=60.0)
        scores_by_id = {r.document_id: r.score for r in merged}
        assert scores_by_id["shared"] > scores_by_id["a_only"]
        assert scores_by_id["shared"] > scores_by_id["b_only"]

    def test_empty_input_returns_empty(self):
        assert rrf_merge_multi([]) == []

    def test_all_empty_lists_returns_empty(self):
        assert rrf_merge_multi([[], []]) == []

    def test_rrf_formula(self):
        # For a single list with k=60, score of rank-0 doc = 1/(60+1)
        results = [make_result("only")]
        merged = rrf_merge_multi([results], k=60.0)
        expected = 1.0 / (60.0 + 1)
        assert abs(merged[0].score - expected) < 1e-9

    def test_sorted_descending(self):
        list_a = [make_result("top"), make_result("mid"), make_result("bot")]
        list_b = [make_result("mid")]  # mid gets an extra point
        merged = rrf_merge_multi([list_a, list_b])
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_three_lists(self):
        l1 = [make_result("a"), make_result("b")]
        l2 = [make_result("b"), make_result("c")]
        l3 = [make_result("a"), make_result("c")]
        merged = rrf_merge_multi([l1, l2, l3])
        ids = {r.document_id for r in merged}
        assert ids == {"a", "b", "c"}


# ─── FederationRouter ─────────────────────────────────────────────────────────

class TestFederationRouter:
    def _make_embedder(self, vec: list[float]):
        emb = MagicMock()
        emb.embed.return_value = vec
        return emb

    def _make_peer(self, pid: str, affinity_tags: list[str]) -> PeerConfig:
        return PeerConfig(id=pid, endpoint=f"localhost:90{pid}", topic_affinity=affinity_tags)

    def test_high_confidence_returns_no_peers(self):
        embedder = self._make_embedder([1.0, 0.0])
        peer = self._make_peer("p1", ["finance"])
        cfg = FederationConfig(confidence_threshold=0.7, routing_threshold=0.3)

        router = FederationRouter([peer], embedder)
        peers = router.route([1.0, 0.0], local_confidence=0.9, cfg=cfg)
        assert peers == []

    def test_low_confidence_returns_matching_peers(self):
        embedder = self._make_embedder([1.0, 0.0])
        peer = self._make_peer("p1", ["finance"])
        cfg = FederationConfig(confidence_threshold=0.7, routing_threshold=0.1)

        router = FederationRouter([peer], embedder)
        # The affinity embedding will be [1.0, 0.0] (same as query) → cosine=1.0
        peers = router.route([1.0, 0.0], local_confidence=0.3, cfg=cfg)
        assert len(peers) == 1
        assert peers[0].id == "p1"

    def test_hop_depth_at_max_returns_empty(self):
        embedder = self._make_embedder([1.0, 0.0])
        peer = self._make_peer("p1", ["finance"])
        cfg = FederationConfig(confidence_threshold=0.7, routing_threshold=0.1, max_depth=2)

        router = FederationRouter([peer], embedder)
        peers = router.route([1.0, 0.0], local_confidence=0.1, cfg=cfg, hop_depth=2)
        assert peers == []

    def test_origin_id_excluded(self):
        embedder = self._make_embedder([1.0, 0.0])
        peer = self._make_peer("self-peer", ["finance"])
        cfg = FederationConfig(confidence_threshold=0.7, routing_threshold=0.1)

        router = FederationRouter([peer], embedder)
        peers = router.route([1.0, 0.0], local_confidence=0.1, cfg=cfg, origin_id="self-peer")
        assert peers == []

    def test_below_routing_threshold_excluded(self):
        embedder = self._make_embedder([1.0, 0.0])
        # Perpendicular vector → cosine similarity = 0.0
        peer = self._make_peer("p_irrelevant", ["finance"])
        # Override affinity embedding after construction
        cfg = FederationConfig(confidence_threshold=0.7, routing_threshold=0.9)

        router = FederationRouter([peer], embedder)
        # Force a low-similarity affinity embedding
        router._peers[0].affinity_embedding = np.array([0.0, 1.0])
        peers = router.route([1.0, 0.0], local_confidence=0.1, cfg=cfg)
        assert peers == []

    def test_max_peers_per_query_respected(self):
        embedder = self._make_embedder([1.0, 0.0])
        peers = [self._make_peer(f"p{i}", ["finance"]) for i in range(5)]
        cfg = FederationConfig(confidence_threshold=0.7, routing_threshold=0.1, max_peers_per_query=2)

        router = FederationRouter(peers, embedder)
        # All peers get same affinity embedding [1.0, 0.0]
        selected = router.route([1.0, 0.0], local_confidence=0.1, cfg=cfg)
        assert len(selected) <= 2


# ─── FederatedOrchestrator ────────────────────────────────────────────────────

class TestFederatedOrchestrator:
    def _make_base(self, results=None, embed_vec=None):
        base = MagicMock()
        base.search.return_value = make_response(*(results or ["local1", "local2"]))
        base.embed.return_value = embed_vec or [1.0, 0.0]
        return base

    def _make_router(self, return_peers=None):
        router = MagicMock()
        router.route.return_value = return_peers or []
        return router

    def test_no_peers_selected_returns_local(self):
        base = self._make_base(["local1", "local2"])
        router = self._make_router(return_peers=[])
        orch = FederatedOrchestrator(base, FederationConfig(), "self", router, {})

        req = SearchRequest(query="test")
        resp = orch.search(req)
        assert [r.document_id for r in resp.results] == ["local1", "local2"]

    def test_peer_results_merged_via_rrf(self):
        base = self._make_base(["local1"])
        router = MagicMock()
        peer_cfg = PeerConfig(id="p1", endpoint="localhost:9001")
        router.route.return_value = [peer_cfg]

        peer_client = MagicMock()
        peer_response = make_response("remote1", "local1")  # local1 appears in both
        peer_client.search = AsyncMock(return_value=peer_response)

        orch = FederatedOrchestrator(base, FederationConfig(), "self", router, {"p1": peer_client})
        req = SearchRequest(query="test")
        resp = orch.search(req)

        ids = {r.document_id for r in resp.results}
        assert "local1" in ids
        assert "remote1" in ids

    def test_strategy_tagged_as_federated(self):
        base = self._make_base(["local1"])
        router = MagicMock()
        peer_cfg = PeerConfig(id="p1", endpoint="localhost:9001")
        router.route.return_value = [peer_cfg]

        peer_client = MagicMock()
        peer_client.search = AsyncMock(return_value=make_response("remote1"))

        orch = FederatedOrchestrator(base, FederationConfig(), "self", router, {"p1": peer_client})
        resp = orch.search(SearchRequest(query="test"))
        assert resp.metadata.strategy.startswith("federated+")

    def test_peer_timeout_falls_back_to_local(self):
        base = self._make_base(["local1"])
        router = MagicMock()
        peer_cfg = PeerConfig(id="p1", endpoint="localhost:9001")
        router.route.return_value = [peer_cfg]

        peer_client = MagicMock()
        peer_client.search = AsyncMock(return_value=None)  # timeout → None

        orch = FederatedOrchestrator(base, FederationConfig(), "self", router, {"p1": peer_client})
        resp = orch.search(SearchRequest(query="test"))
        # Only local results since peer returned None
        assert [r.document_id for r in resp.results] == ["local1"]

    def test_embed_delegates_to_base(self):
        base = self._make_base()
        orch = FederatedOrchestrator(base, FederationConfig(), "self", self._make_router(), {})
        result = orch.embed("some text")
        base.embed.assert_called_once_with("some text")
        assert result == [1.0, 0.0]

    def test_collections_delegates_to_base(self):
        base = self._make_base()
        base.collections.return_value = []
        orch = FederatedOrchestrator(base, FederationConfig(), "self", self._make_router(), {})
        orch.collections()
        base.collections.assert_called_once()


# ─── build_federated_orchestrator factory ─────────────────────────────────────

class TestBuildFederatedOrchestrator:
    def test_creates_orchestrator_with_peers(self):
        base = MagicMock()
        base.embed.return_value = [0.1, 0.2]

        cfg = FederationConfig(
            peers=[
                PeerConfig(id="p1", endpoint="localhost:9001", topic_affinity=["finance"]),
                PeerConfig(id="p2", endpoint="localhost:9002", topic_affinity=["hr"]),
            ]
        )
        orch = build_federated_orchestrator(base, cfg, "self-id")
        assert isinstance(orch, FederatedOrchestrator)
        assert "p1" in orch._peer_clients
        assert "p2" in orch._peer_clients

    def test_empty_peers_still_builds(self):
        base = MagicMock()
        cfg = FederationConfig(peers=[])
        orch = build_federated_orchestrator(base, cfg, "self-id")
        assert isinstance(orch, FederatedOrchestrator)
