"""Unit tests for navigator.rest — FastAPI endpoints using TestClient."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from navigator.config import Config
from navigator.events import Publisher
from navigator.hooks import HookManager
from navigator.models import (
    CollectionInfo,
    SearchMetadata,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from navigator.rest import build_app


# ─── fixtures ─────────────────────────────────────────────────────────────────

def _make_response(*doc_ids: str) -> SearchResponse:
    return SearchResponse(
        results=[SearchResult(document_id=d, content="c", score=0.9) for d in doc_ids],
        metadata=SearchMetadata(total_candidates=len(doc_ids), final_count=len(doc_ids)),
        processing_time_ms=5.0,
    )


def _make_client(cfg_overrides: dict | None = None, orch_overrides: dict | None = None) -> TestClient:
    cfg = Config()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    orch = MagicMock()
    orch.search.return_value = _make_response("doc1", "doc2")
    orch.embed.return_value = [0.1] * 10
    orch.embed_batch.return_value = [[0.1] * 10]
    orch.rerank.return_value = [SearchResult(document_id="doc1", score=0.95)]
    orch.collections.return_value = [CollectionInfo(name="customer_docs", vector_count=100)]
    if orch_overrides:
        for k, v in orch_overrides.items():
            setattr(orch, k, v)

    pub = MagicMock(spec=Publisher)
    hm = MagicMock(spec=HookManager)
    app = build_app(cfg, orch, pub, hm)
    return TestClient(app)


# ─── health endpoints ─────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self):
        client = _make_client()
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_live(self):
        client = _make_client()
        r = client.get("/v1/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_ready(self):
        client = _make_client()
        r = client.get("/v1/health/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_metrics_returns_text(self):
        client = _make_client()
        r = client.get("/v1/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]


# ─── search endpoint ──────────────────────────────────────────────────────────

class TestSearch:
    def test_basic_search_returns_results(self):
        client = _make_client()
        r = client.post("/v1/navigator/search", json={"query": "machine learning"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["document_id"] == "doc1"

    def test_search_publishes_started_and_completed_events(self):
        pub = MagicMock(spec=Publisher)
        orch = MagicMock()
        orch.search.return_value = _make_response("d1")
        orch.embed.return_value = [0.0]
        hm = MagicMock(spec=HookManager)
        cfg = Config()
        from navigator.rest import build_app
        app = build_app(cfg, orch, pub, hm)
        client = TestClient(app)
        client.post("/v1/navigator/search", json={"query": "test"})
        # Publisher.publish should be called at least twice (started + completed)
        assert pub.publish.call_count >= 2

    def test_search_with_permissions_missing_categories_returns_400(self):
        client = _make_client()
        r = client.post("/v1/navigator/search/with-permissions", json={
            "query": "test",
            "tenant_id": "t1",
            "user": {"user_id": "u1", "allowed_categories": []},
        })
        assert r.status_code == 400

    def test_search_with_permissions_missing_tenant_returns_400(self):
        client = _make_client()
        r = client.post("/v1/navigator/search/with-permissions", json={
            "query": "test",
            "user": {"user_id": "u1", "allowed_categories": ["cat1"]},
        })
        assert r.status_code == 400

    def test_search_with_permissions_valid_request(self):
        client = _make_client()
        r = client.post("/v1/navigator/search/with-permissions", json={
            "query": "test",
            "tenant_id": "t1",
            "user": {"user_id": "u1", "allowed_categories": ["customer_data"]},
        })
        assert r.status_code == 200

    def test_hybrid_search_sets_use_hybrid_true(self):
        orch = MagicMock()
        captured = []

        def capture_search(req, **kwargs):
            captured.append(req)
            return _make_response("d1")

        orch.search.side_effect = capture_search
        orch.embed.return_value = [0.0]

        pub = MagicMock(spec=Publisher)
        hm = MagicMock(spec=HookManager)
        app = build_app(Config(), orch, pub, hm)
        client = TestClient(app)
        client.post("/v1/navigator/search/hybrid", json={"query": "test"})

        assert len(captured) == 1
        assert captured[0].options.use_hybrid is True

    def test_batch_search(self):
        client = _make_client()
        r = client.post("/v1/navigator/search/batch", json={
            "request_id": "batch-1",
            "queries": [
                {"query": "first"},
                {"query": "second"},
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2


# ─── honey-token detection ────────────────────────────────────────────────────

class TestHoneyToken:
    def test_honey_token_fires_events(self):
        pub = MagicMock(spec=Publisher)
        hm = MagicMock(spec=HookManager)
        orch = MagicMock()
        orch.search.return_value = SearchResponse(
            results=[SearchResult(
                document_id="honey-doc",
                score=0.9,
                metadata={"is_honey_token": "true", "honey_token_id": "ht-001"},
            )],
            metadata=SearchMetadata(),
        )
        orch.embed.return_value = [0.0]

        app = build_app(Config(), orch, pub, hm)
        client = TestClient(app)
        client.post("/v1/navigator/search", json={"query": "test"})

        # Expect at least one publish call for the honey token event
        honey_calls = [
            call for call in pub.publish.call_args_list
            if "honey_token" in str(call)
        ]
        assert len(honey_calls) >= 1


# ─── embed endpoints ──────────────────────────────────────────────────────────

class TestEmbed:
    def test_embed_single(self):
        client = _make_client()
        r = client.post("/v1/navigator/embed", json={"text": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert "embedding" in data
        assert data["dim_count"] == len(data["embedding"])

    def test_embed_batch(self):
        client = _make_client()
        r = client.post("/v1/navigator/embed/batch", json={"texts": ["a", "b"]})
        assert r.status_code == 200
        assert len(r.json()["embeddings"]) == 1  # mock returns one list


# ─── rerank endpoint ──────────────────────────────────────────────────────────

class TestRerank:
    def test_rerank(self):
        client = _make_client()
        r = client.post("/v1/navigator/rerank", json={
            "query": "ml",
            "candidates": [{"document_id": "d1", "content": "content", "score": 0.5}],
            "top_k": 1,
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1


# ─── collections endpoints ────────────────────────────────────────────────────

class TestCollections:
    def test_list_collections(self):
        client = _make_client()
        r = client.get("/v1/navigator/collections")
        assert r.status_code == 200
        assert len(r.json()["collections"]) == 1

    def test_get_collection_found(self):
        client = _make_client()
        r = client.get("/v1/navigator/collections/customer_docs")
        assert r.status_code == 200
        assert r.json()["name"] == "customer_docs"

    def test_get_collection_not_found(self):
        client = _make_client()
        r = client.get("/v1/navigator/collections/nonexistent")
        assert r.status_code == 404


# ─── agent/generate endpoint ──────────────────────────────────────────────────

class TestAgentGenerate:
    def test_returns_503_when_not_agent_mode(self):
        client = _make_client()  # default mode = "search"
        r = client.post("/v1/navigator/agent/generate", json={
            "query": "summarize this",
            "context": [],
        })
        assert r.status_code == 503

    def test_agent_mode_calls_local_llm(self):
        from unittest.mock import patch, MagicMock
        from navigator.config import AgentConfig, LocalLLMConfig

        llm_cfg = LocalLLMConfig(
            provider="ollama",
            endpoint="http://fake-ollama:11434",
            model="llama3",
            timeout_seconds=5,
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "The answer is 42."}

        with patch("httpx.post", return_value=mock_resp):
            client = _make_client(cfg_overrides={
                "mode": "agent",
                "agent": AgentConfig(local_llm=llm_cfg),
            })
            r = client.post("/v1/navigator/agent/generate", json={
                "query": "What is 6*7?",
                "context": [],
            })

        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "The answer is 42."
        assert data["model"] == "llama3"
