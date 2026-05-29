"""Unit tests for source attribution provenance fields — MR-05-002 (Item 8).

Tests:
  - SearchResult and SearchOptions model fields
  - searcher._to_search_result extracts provenance from Qdrant payload
  - orchestrator.index_document stores char_start / char_end / last_indexed
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from navigator.models import SearchResult, SearchOptions
from navigator.searcher import _to_search_result


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mock_hit(score: float, **payload) -> MagicMock:
    hit = MagicMock()
    hit.id = "qdrant-id-abc"
    hit.score = score
    hit.payload = payload
    return hit


# ─── SearchResult model fields (MR-05-002) ────────────────────────────────────

class TestSearchResultProvenanceFields:
    def test_chunk_id_field_exists_with_default(self):
        r = SearchResult(document_id="d1", content="text", score=0.9)
        assert r.chunk_id == ""

    def test_heading_path_field_exists_with_default(self):
        r = SearchResult(document_id="d1", content="text", score=0.9)
        assert r.heading_path == ""

    def test_char_start_field_exists_with_default(self):
        r = SearchResult(document_id="d1", content="text", score=0.9)
        assert r.char_start == 0

    def test_char_end_field_exists_with_default(self):
        r = SearchResult(document_id="d1", content="text", score=0.9)
        assert r.char_end == 0

    def test_last_indexed_field_exists_with_default(self):
        r = SearchResult(document_id="d1", content="text", score=0.9)
        assert r.last_indexed == ""

    def test_provenance_fields_assignable(self):
        r = SearchResult(
            document_id="d1",
            content="text",
            score=0.9,
            chunk_id="d1_0003",
            heading_path="# Policy > ## Leave",
            char_start=120,
            char_end=450,
            last_indexed="2026-05-29T10:00:00+00:00",
        )
        assert r.chunk_id == "d1_0003"
        assert r.heading_path == "# Policy > ## Leave"
        assert r.char_start == 120
        assert r.char_end == 450
        assert r.last_indexed == "2026-05-29T10:00:00+00:00"


class TestSearchOptionsStrategyField:
    def test_strategy_field_exists_with_default(self):
        opts = SearchOptions()
        assert opts.strategy == ""

    def test_strategy_field_assignable(self):
        opts = SearchOptions(strategy="hybrid+rerank")
        assert opts.strategy == "hybrid+rerank"

    def test_strategy_empty_string_is_falsy(self):
        opts = SearchOptions()
        assert not opts.strategy


# ─── _to_search_result: provenance extraction ─────────────────────────────────

class TestToSearchResultProvenance:
    def test_chunk_id_extracted_from_payload(self):
        hit = _mock_hit(0.85, document_id="doc-1", content="text", chunk_id="doc-1_0002")
        r = _to_search_result(hit)
        assert r.chunk_id == "doc-1_0002"

    def test_heading_path_extracted_from_payload(self):
        hit = _mock_hit(0.85, document_id="d1", content="text",
                        heading_path="# Report > ## Line 7")
        r = _to_search_result(hit)
        assert r.heading_path == "# Report > ## Line 7"

    def test_char_start_extracted_from_payload(self):
        hit = _mock_hit(0.85, document_id="d1", content="text", char_start=120)
        r = _to_search_result(hit)
        assert r.char_start == 120

    def test_char_end_extracted_from_payload(self):
        hit = _mock_hit(0.85, document_id="d1", content="text", char_end=450)
        r = _to_search_result(hit)
        assert r.char_end == 450

    def test_last_indexed_extracted_from_payload(self):
        ts = "2026-05-29T10:00:00+00:00"
        hit = _mock_hit(0.85, document_id="d1", content="text", last_indexed=ts)
        r = _to_search_result(hit)
        assert r.last_indexed == ts

    def test_defaults_when_provenance_missing(self):
        hit = _mock_hit(0.85, document_id="d1", content="text")
        r = _to_search_result(hit)
        assert r.chunk_id == ""
        assert r.heading_path == ""
        assert r.char_start == 0
        assert r.char_end == 0
        assert r.last_indexed == ""

    def test_provenance_keys_not_in_metadata(self):
        hit = _mock_hit(
            0.85,
            document_id="d1",
            content="text",
            chunk_id="d1_0001",
            heading_path="# H",
            char_start=0,
            char_end=100,
            last_indexed="2026-05-01T00:00:00Z",
        )
        r = _to_search_result(hit)
        for key in ("chunk_id", "heading_path", "char_start", "char_end", "last_indexed"):
            assert key not in r.metadata, f"'{key}' should not appear in metadata"

    def test_extra_payload_goes_to_metadata(self):
        hit = _mock_hit(0.85, document_id="d1", content="text", tenant_id="acme", category="hr")
        r = _to_search_result(hit)
        assert r.metadata.get("tenant_id") == "acme"
        assert r.metadata.get("category") == "hr"

    def test_char_start_zero_is_not_treated_as_missing(self):
        hit = _mock_hit(0.85, document_id="d1", content="text", char_start=0, char_end=50)
        r = _to_search_result(hit)
        assert r.char_start == 0
        assert r.char_end == 50

    def test_document_id_falls_back_to_hit_id(self):
        hit = _mock_hit(0.70, content="text")
        hit.id = "fallback-uuid"
        hit.payload = {"content": "text"}  # no document_id key
        r = _to_search_result(hit)
        assert r.document_id == "fallback-uuid"

    def test_score_is_assigned(self):
        hit = _mock_hit(0.72, document_id="d1", content="text")
        r = _to_search_result(hit)
        assert r.score == pytest.approx(0.72)


# ─── index_document: payload stores provenance ────────────────────────────────

class TestIndexDocumentProvenancePayload:
    """index_document must write char_start, char_end, last_indexed to Qdrant payload."""

    def _make_orchestrator(self):
        from navigator.config import Config, SearchDefaultsConfig, VectorDBConfig, CollectionConfig
        from navigator.orchestrator import Orchestrator
        from navigator.vault_client import NoopVaultClient

        cfg = Config()
        cfg.search_defaults = SearchDefaultsConfig(top_k=10, over_fetch_multiplier=2)
        cfg.vector_db = VectorDBConfig(collections={"test": CollectionConfig()})

        embedder = MagicMock()
        embedder.embed_batch.return_value = [[0.1] * 8]
        reranker = MagicMock()

        captured = []

        class _CapturingSearcher:
            def vector_search(self, *a, **k): return []
            def sparse_search(self, *a, **k): return []
            def collections(self): return []
            def ensure_collection(self, *a, **k): pass
            def upsert(self, collection, points): captured.extend(points)

        return Orchestrator(cfg, embedder, _CapturingSearcher(), reranker, NoopVaultClient()), captured

    def test_payload_contains_char_start(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        assert len(captured) > 0
        assert "char_start" in captured[0]["payload"]

    def test_payload_contains_char_end(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        assert "char_end" in captured[0]["payload"]

    def test_payload_contains_last_indexed(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        assert "last_indexed" in captured[0]["payload"]

    def test_char_start_is_integer(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        assert isinstance(captured[0]["payload"]["char_start"], int)

    def test_char_end_is_integer(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        assert isinstance(captured[0]["payload"]["char_end"], int)

    def test_last_indexed_is_valid_iso8601(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        ts = captured[0]["payload"]["last_indexed"]
        datetime.fromisoformat(ts)  # raises ValueError if not valid ISO-8601

    def test_last_indexed_is_utc(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nContent.", tenant_id="t"))
        ts = captured[0]["payload"]["last_indexed"]
        # UTC offset must be present
        assert "+00:00" in ts or ts.endswith("Z")

    def test_char_end_greater_than_char_start(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        orch.index_document(IndexRequest(document_id="d1", content="# H\n\nSome long content here.", tenant_id="t"))
        p = captured[0]["payload"]
        assert p["char_end"] > p["char_start"]

    def test_last_indexed_same_for_all_chunks(self):
        from navigator.models import IndexRequest
        orch, captured = self._make_orchestrator()
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[0.1] * 8, [0.1] * 8]
        orch._embedder = embedder
        # Multi-chunk document
        content = "# Section A\n\n" + "Content A. " * 60 + "\n\n# Section B\n\nContent B."
        orch.index_document(IndexRequest(document_id="d1", content=content, tenant_id="t"))
        if len(captured) > 1:
            ts_values = {p["payload"]["last_indexed"] for p in captured}
            assert len(ts_values) == 1, "all chunks should share the same last_indexed timestamp"
