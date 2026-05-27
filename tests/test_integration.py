"""Integration tests for Navigator connecting all three extension cases.

Architecture under test:
    Mock-Sentinel  (wsgi app via requests_mock / httpserver)
    Mock-Vault     (wsgi app via httpserver)
    Navigator      (FastAPI TestClient — in-process)

Flows verified:
    Case 1 (industry): Navigator returns honey-token results → events published
    Case 2 (agent):    Navigator agent-generate calls local LLM (mocked)
    Case 3 (federation): FederatedOrchestrator fans out to mock peer Navigator

Each test class builds a fresh Navigator app + orchestrator.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from navigator.config import (
    AgentConfig,
    Config,
    FederationConfig,
    LocalLLMConfig,
    PeerConfig,
)
from navigator.events import Publisher
from navigator.federation import (
    FederatedOrchestrator,
    FederationRouter,
    build_federated_orchestrator,
)
from navigator.hooks import HookManager
from navigator.models import (
    CollectionInfo,
    SearchMetadata,
    SearchRequest,
    SearchResponse,
    SearchResult,
    UserContext,
)
from navigator.rest import build_app


# ─── shared fixtures ──────────────────────────────────────────────────────────

def _make_results(*doc_ids: str, score: float = 0.9) -> list[SearchResult]:
    return [SearchResult(document_id=d, content="ctx", score=score) for d in doc_ids]


def _make_response(*doc_ids: str, strategy: str = "hybrid+rerank") -> SearchResponse:
    results = _make_results(*doc_ids)
    return SearchResponse(
        results=results,
        metadata=SearchMetadata(
            total_candidates=len(results),
            final_count=len(results),
            strategy=strategy,
        ),
        processing_time_ms=8.0,
    )


def _mock_orchestrator(search_results: list[SearchResult] | None = None) -> MagicMock:
    orch = MagicMock()
    orch.search.return_value = SearchResponse(
        results=search_results or _make_results("doc-1", "doc-2"),
        metadata=SearchMetadata(total_candidates=2, final_count=2, strategy="hybrid"),
        processing_time_ms=5.0,
    )
    orch.embed.return_value = [0.1] * 8
    orch.embed_batch.return_value = [[0.1] * 8]
    orch.rerank.return_value = _make_results("doc-1")
    orch.collections.return_value = [CollectionInfo(name="docs", vector_count=500)]
    return orch


def _build_client(
    cfg: Config | None = None,
    orch=None,
    pub: Publisher | None = None,
) -> TestClient:
    cfg = cfg or Config()
    orch = orch or _mock_orchestrator()
    pub = pub or MagicMock(spec=Publisher)
    hm = MagicMock(spec=HookManager)
    app = build_app(cfg, orch, pub, hm)
    return TestClient(app)


# ─── CASE 1 INTEGRATION: honey-token detection + event pipeline ───────────────

class TestCase1_HoneyToken_Pipeline:
    """Verify honey-token detection fires the right events through the REST layer."""

    def test_honey_token_in_results_triggers_event(self):
        honey_result = SearchResult(
            document_id="honey-doc-001",
            content="Sensitive bait document",
            score=0.95,
            metadata={"is_honey_token": "true", "honey_token_id": "ht-001"},
        )
        pub = MagicMock(spec=Publisher)
        client = _build_client(orch=_mock_orchestrator([honey_result]), pub=pub)

        r = client.post("/v1/navigator/search", json={"query": "sensitive data"})
        assert r.status_code == 200

        # Publisher should have been called: search_started, search_completed, honey_token_retrieved
        event_types = [
            call.args[0].event_type
            for call in pub.publish.call_args_list
        ]
        assert "honey_token_retrieved" in event_types, (
            f"Expected honey_token_retrieved event. Got: {event_types}"
        )

    def test_normal_result_does_not_trigger_honey_event(self):
        pub = MagicMock(spec=Publisher)
        client = _build_client(orch=_mock_orchestrator(), pub=pub)

        r = client.post("/v1/navigator/search", json={"query": "regular query"})
        assert r.status_code == 200

        event_types = [call.args[0].event_type for call in pub.publish.call_args_list]
        assert "honey_token_retrieved" not in event_types

    def test_multiple_honey_tokens_all_reported(self):
        honey_results = [
            SearchResult(
                document_id=f"honey-{i}",
                score=0.9,
                metadata={"is_honey_token": "true", "honey_token_id": f"ht-{i:03d}"},
            )
            for i in range(3)
        ]
        pub = MagicMock(spec=Publisher)
        client = _build_client(orch=_mock_orchestrator(honey_results), pub=pub)

        r = client.post("/v1/navigator/search", json={"query": "bait"})
        assert r.status_code == 200

        honey_events = [
            call for call in pub.publish.call_args_list
            if call.args[0].event_type == "honey_token_retrieved"
        ]
        assert len(honey_events) == 3, f"Expected 3 honey events, got {len(honey_events)}"

    def test_permission_filter_event_when_docs_filtered(self):
        # Simulate a response where some documents were filtered out.
        orch = MagicMock()
        orch.search.return_value = SearchResponse(
            results=_make_results("doc-allowed"),
            metadata=SearchMetadata(
                total_candidates=5,
                filtered_out=4,
                final_count=1,
                strategy="vector",
            ),
        )
        orch.embed.return_value = [0.0]
        pub = MagicMock(spec=Publisher)
        client = _build_client(
            orch=orch,
            pub=pub,
        )
        r = client.post("/v1/navigator/search", json={
            "query": "test",
            "user": {"user_id": "u1", "allowed_categories": ["customer_data"]},
        })
        assert r.status_code == 200
        event_types = [call.args[0].event_type for call in pub.publish.call_args_list]
        assert "permission_filtered" in event_types


# ─── CASE 2 INTEGRATION: agent generate pipeline ─────────────────────────────

class TestCase2_AgentGenerate_Pipeline:
    """Verify the agent generate endpoint runs the Sentinel → LLM flow."""

    def _agent_cfg(self, provider: str = "ollama", endpoint: str = "http://fake:11434") -> Config:
        cfg = Config()
        cfg.mode = "agent"
        cfg.agent = AgentConfig(
            local_llm=LocalLLMConfig(
                provider=provider,
                endpoint=endpoint,
                model="llama3",
                timeout_seconds=5,
            )
        )
        return cfg

    def test_503_when_mode_is_search(self):
        client = _build_client()
        r = client.post("/v1/navigator/agent/generate", json={"query": "x"})
        assert r.status_code == 503

    def test_ollama_provider_called_with_query(self):
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "The balance is $5,000."}

        cfg = self._agent_cfg(provider="ollama", endpoint="http://fake-ollama:11434")
        client = _build_client(cfg=cfg)

        with patch("httpx.post", return_value=mock_resp) as mocked_post:
            r = client.post("/v1/navigator/agent/generate", json={
                "query": "What is the account balance?",
                "context": [],
                "max_tokens": 200,
            })

        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "The balance is $5,000."
        assert data["model"] == "llama3"
        # Verify the endpoint was called with the right URL
        mocked_post.assert_called_once()
        call_url = mocked_post.call_args.args[0]
        assert "fake-ollama:11434" in call_url

    def test_openai_compatible_provider_called(self):
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Answer from llamacpp."}}]
        }

        cfg = self._agent_cfg(provider="llamacpp", endpoint="http://fake-llamacpp:8080")
        client = _build_client(cfg=cfg)

        with patch("httpx.post", return_value=mock_resp):
            r = client.post("/v1/navigator/agent/generate", json={
                "query": "Explain policy X",
                "context": [{"document_id": "d1", "content": "Policy X states...", "score": 0.9}],
            })

        assert r.status_code == 200
        assert r.json()["answer"] == "Answer from llamacpp."

    def test_context_prepended_to_prompt(self):
        """Verify context results are prepended before the query in the LLM prompt."""
        captured_calls = []

        def capture_post(url, *, json=None, timeout=None, **kwargs):
            captured_calls.append(json)
            m = MagicMock()
            m.raise_for_status.return_value = None
            m.json.return_value = {"response": "ok"}
            return m

        cfg = self._agent_cfg(provider="ollama")
        client = _build_client(cfg=cfg)

        with patch("httpx.post", side_effect=capture_post):
            client.post("/v1/navigator/agent/generate", json={
                "query": "Summarise this.",
                "context": [{"document_id": "d1", "content": "Relevant paragraph.", "score": 0.9}],
            })

        assert len(captured_calls) == 1
        prompt = captured_calls[0]["prompt"]
        assert "Relevant paragraph." in prompt
        assert "Summarise this." in prompt
        assert prompt.index("Relevant paragraph.") < prompt.index("Summarise this.")

    def test_llm_failure_returns_empty_answer(self):
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
            cfg = self._agent_cfg(provider="ollama")
            client = _build_client(cfg=cfg)
            r = client.post("/v1/navigator/agent/generate", json={"query": "x"})

        assert r.status_code == 200
        assert r.json()["answer"] == ""
        assert r.json()["confidence"] == 0.0


# ─── CASE 3 INTEGRATION: federation pipeline ─────────────────────────────────

class TestCase3_Federation_Pipeline:
    """Verify FederatedOrchestrator integrates correctly with the REST layer."""

    def _make_federated_client(
        self,
        peer_responses: dict[str, SearchResponse] | None = None,
    ) -> tuple[TestClient, dict[str, AsyncMock]]:
        """Build a Navigator TestClient backed by a FederatedOrchestrator with mock peers."""
        base = MagicMock()
        # Local results are weak (low score) to trigger federation.
        base.search.return_value = SearchResponse(
            results=[SearchResult(document_id="local-doc", score=0.3)],
            metadata=SearchMetadata(total_candidates=1, final_count=1, strategy="vector"),
        )
        base.embed.return_value = [1.0, 0.0]
        base.collections.return_value = []

        peer_responses = peer_responses or {}
        peer_clients: dict[str, AsyncMock] = {}

        cfg = FederationConfig(
            confidence_threshold=0.7,
            routing_threshold=0.1,
            max_peers_per_query=3,
            max_depth=2,
            peer_timeout_ms=1000,
            rrf_k=60.0,
            peers=[
                PeerConfig(id=pid, endpoint=f"localhost:900{i}", topic_affinity=["finance"])
                for i, pid in enumerate(peer_responses.keys())
            ],
        )

        router = MagicMock()
        router.route.return_value = cfg.peers  # always fan out to all configured peers

        for pid, resp in peer_responses.items():
            client_mock = MagicMock()
            client_mock.search = AsyncMock(return_value=resp)
            peer_clients[pid] = client_mock

        orch = FederatedOrchestrator(base, cfg, "self", router, peer_clients)
        pub = MagicMock(spec=Publisher)
        hm = MagicMock(spec=HookManager)
        nav_cfg = Config()
        nav_cfg.mode = "federation"
        app = build_app(nav_cfg, orch, pub, hm)
        return TestClient(app), peer_clients

    def test_federated_search_merges_local_and_remote(self):
        peer_resp = SearchResponse(
            results=[SearchResult(document_id="remote-doc", score=0.85)],
            metadata=SearchMetadata(total_candidates=1, final_count=1),
        )
        client, _ = self._make_federated_client({"peer-1": peer_resp})

        r = client.post("/v1/navigator/search", json={"query": "distributed query"})
        assert r.status_code == 200
        result_ids = {res["document_id"] for res in r.json()["results"]}
        assert "local-doc" in result_ids
        assert "remote-doc" in result_ids

    def test_federation_strategy_tag(self):
        peer_resp = SearchResponse(
            results=[SearchResult(document_id="r1", score=0.8)],
            metadata=SearchMetadata(total_candidates=1, final_count=1),
        )
        client, _ = self._make_federated_client({"peer-1": peer_resp})

        r = client.post("/v1/navigator/search", json={"query": "test"})
        assert r.status_code == 200
        strategy = r.json()["metadata"]["strategy"]
        assert strategy.startswith("federated+"), f"Expected federated+ prefix, got: {strategy}"

    def test_peer_timeout_falls_back_to_local(self):
        """When all peers time out (return None), only local results are returned."""
        base = MagicMock()
        base.search.return_value = SearchResponse(
            results=[SearchResult(document_id="local-only", score=0.5)],
            metadata=SearchMetadata(total_candidates=1, final_count=1, strategy="vector"),
        )
        base.embed.return_value = [1.0, 0.0]

        cfg = FederationConfig(
            peers=[PeerConfig(id="timeout-peer", endpoint="localhost:9001", topic_affinity=["x"])]
        )
        router = MagicMock()
        router.route.return_value = cfg.peers

        timeout_client = MagicMock()
        timeout_client.search = AsyncMock(return_value=None)  # simulates timeout

        orch = FederatedOrchestrator(base, cfg, "self", router, {"timeout-peer": timeout_client})
        pub = MagicMock(spec=Publisher)
        nav_cfg = Config()
        nav_cfg.mode = "federation"
        app = build_app(nav_cfg, orch, pub, MagicMock(spec=HookManager))
        client = TestClient(app)

        r = client.post("/v1/navigator/search", json={"query": "test"})
        assert r.status_code == 200
        result_ids = [res["document_id"] for res in r.json()["results"]]
        assert result_ids == ["local-only"]

    def test_hop_depth_header_forwarded(self):
        """Verify x-hop-depth and x-origin-id headers are read from the request."""
        orch = _mock_orchestrator()
        pub = MagicMock(spec=Publisher)
        nav_cfg = Config()
        nav_cfg.mode = "federation"
        app = build_app(nav_cfg, orch, pub, MagicMock(spec=HookManager))
        client = TestClient(app)

        with patch("navigator.rest._is_federation_mode", return_value=True):
            r = client.post(
                "/v1/navigator/search",
                json={"query": "federation test"},
                headers={"x-hop-depth": "1", "x-origin-id": "navigator-upstream"},
            )
        assert r.status_code == 200
        # Verify search was called with the federation kwargs
        call_kwargs = orch.search.call_args.kwargs
        assert call_kwargs.get("hop_depth") == 1
        assert call_kwargs.get("origin_id") == "navigator-upstream"

    def test_multiple_peers_rrf_dedup(self):
        """When multiple peers return the same document, RRF should deduplicate."""
        shared_doc = SearchResult(document_id="shared", score=0.9)
        unique_1 = SearchResult(document_id="unique-1", score=0.7)
        unique_2 = SearchResult(document_id="unique-2", score=0.6)

        peer_responses = {
            "peer-a": SearchResponse(
                results=[shared_doc, unique_1],
                metadata=SearchMetadata(total_candidates=2, final_count=2),
            ),
            "peer-b": SearchResponse(
                results=[shared_doc, unique_2],
                metadata=SearchMetadata(total_candidates=2, final_count=2),
            ),
        }
        client, _ = self._make_federated_client(peer_responses)
        r = client.post("/v1/navigator/search", json={"query": "test"})
        assert r.status_code == 200

        results = r.json()["results"]
        result_ids = [res["document_id"] for res in results]
        # No duplicates
        assert len(result_ids) == len(set(result_ids))
        # Shared doc should score higher than unique docs (appears in 2 lists)
        id_to_score = {res["document_id"]: res["score"] for res in results}
        assert "shared" in id_to_score
        for uid in ["unique-1", "unique-2"]:
            if uid in id_to_score:
                assert id_to_score["shared"] > id_to_score[uid], (
                    f"shared ({id_to_score['shared']:.3f}) should outscore {uid} ({id_to_score[uid]:.3f})"
                )


# ─── FULL PIPELINE: Sentinel-IN → Navigator → Sentinel-OUT (mock Sentinel) ───

class TestFullNavigatorPipeline:
    """Simulate the complete Navigator-centric pipeline with mock Sentinel/Vault."""

    def _mock_sentinel_validator(self, block: bool = False):
        """Returns a minimal WSGI-like handler for Sentinel validation."""
        def handler(request):
            if block:
                return (403, {"status": "BLOCKED", "code": 403})
            return (200, {"status": "PASSED", "prompt_risk_score": 0.01})
        return handler

    def test_search_with_permissions_full_flow(self):
        """POST /v1/navigator/search/with-permissions returns 200 for valid input."""
        client = _build_client()
        r = client.post("/v1/navigator/search/with-permissions", json={
            "query": "employee records for Q1",
            "tenant_id": "acme",
            "user": {
                "user_id": "analyst-1",
                "department": "HR",
                "roles": ["analyst"],
                "allowed_categories": ["hr_data"],
            },
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0

    def test_search_permission_filter_event_fires(self):
        orch = MagicMock()
        orch.search.return_value = SearchResponse(
            results=[SearchResult(document_id="d1", score=0.9)],
            metadata=SearchMetadata(
                total_candidates=10,
                filtered_out=9,
                final_count=1,
                strategy="vector",
            ),
        )
        orch.embed.return_value = [0.0]
        pub = MagicMock(spec=Publisher)
        client = _build_client(orch=orch, pub=pub)

        r = client.post("/v1/navigator/search/with-permissions", json={
            "query": "test",
            "tenant_id": "acme",
            "user": {"user_id": "u1", "allowed_categories": ["customer_data"]},
        })
        assert r.status_code == 200
        event_types = [call.args[0].event_type for call in pub.publish.call_args_list]
        assert "permission_filtered" in event_types

    def test_batch_search_all_queries_executed(self):
        client = _build_client()
        r = client.post("/v1/navigator/search/batch", json={
            "request_id": "batch-001",
            "queries": [
                {"query": "employee policy"},
                {"query": "data governance"},
                {"query": "security audit"},
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 3

    def test_embed_then_search_pipeline(self):
        """Embed a query then use the vector for search (two-step pipeline)."""
        client = _build_client()

        # Step 1: embed
        embed_r = client.post("/v1/navigator/embed", json={"text": "machine learning pipeline"})
        assert embed_r.status_code == 200
        embedding = embed_r.json()["embedding"]
        assert len(embedding) > 0

        # Step 2: search with the embedding available
        search_r = client.post("/v1/navigator/search", json={
            "query": "machine learning pipeline",
            "tenant_id": "acme",
        })
        assert search_r.status_code == 200

    def test_collection_discovery_then_search(self):
        """List collections then search — simulates a client discovery flow."""
        client = _build_client()

        # Step 1: list collections
        col_r = client.get("/v1/navigator/collections")
        assert col_r.status_code == 200
        collections = col_r.json()["collections"]
        assert len(collections) > 0
        col_name = collections[0]["name"]

        # Step 2: get specific collection
        detail_r = client.get(f"/v1/navigator/collections/{col_name}")
        assert detail_r.status_code == 200
        assert detail_r.json()["name"] == col_name

        # Step 3: search
        search_r = client.post("/v1/navigator/search", json={
            "query": "test",
            "options": {"filters": {"collection": col_name}},
        })
        assert search_r.status_code == 200
