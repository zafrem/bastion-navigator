"""Tests for Priority 4 features: connectors, delta indexing, profiles, decomposer."""
from __future__ import annotations

import csv
import io
import json
import os
import textwrap
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ─── FR-MR-06-001: Source Connector Interface ─────────────────────────────────

class TestSourceConnectorInterface:
    def test_jsonl_connector_lists_documents(self, tmp_path):
        from navigator.connector import JsonlConnector
        f = tmp_path / "docs.jsonl"
        f.write_text(
            '{"id":"d1","content":"hello world","title":"Doc 1"}\n'
            '{"id":"d2","content":"second doc","title":"Doc 2"}\n'
        )
        docs = list(JsonlConnector(str(f)).list_documents())
        assert len(docs) == 2
        assert docs[0].id == "d1"
        assert docs[1].content == "second doc"

    def test_jsonl_connector_get_document(self, tmp_path):
        from navigator.connector import JsonlConnector
        f = tmp_path / "docs.jsonl"
        f.write_text('{"id":"x1","content":"target","title":"T"}\n')
        doc = JsonlConnector(str(f)).get_document("x1")
        assert doc is not None
        assert doc.content == "target"

    def test_jsonl_connector_missing_file_returns_empty(self, tmp_path):
        from navigator.connector import JsonlConnector
        docs = list(JsonlConnector(str(tmp_path / "missing.jsonl")).list_documents())
        assert docs == []

    def test_jsonl_connector_skips_invalid_lines(self, tmp_path):
        from navigator.connector import JsonlConnector
        f = tmp_path / "docs.jsonl"
        f.write_text('{"id":"ok","content":"good"}\nnot json\n{"id":"ok2","content":"also good"}\n')
        docs = list(JsonlConnector(str(f)).list_documents())
        assert len(docs) == 2

    def test_jsonl_since_filter(self, tmp_path):
        from navigator.connector import JsonlConnector
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        f = tmp_path / "docs.jsonl"
        f.write_text(
            '{"id":"old","content":"x","updated_at":"2025-12-01T00:00:00Z"}\n'
            '{"id":"new","content":"y","updated_at":"2026-06-01T00:00:00Z"}\n'
        )
        docs = list(JsonlConnector(str(f)).list_documents(since=cutoff))
        assert len(docs) == 1
        assert docs[0].id == "new"

    def test_directory_connector_lists_md_files(self, tmp_path):
        from navigator.connector import DirectoryConnector
        (tmp_path / "a.md").write_text("# Doc A\nContent A.")
        (tmp_path / "b.txt").write_text("plain text")
        (tmp_path / "skip.exe").write_text("binary")
        docs = list(DirectoryConnector(str(tmp_path)).list_documents())
        ids = {d.id for d in docs}
        assert "a.md" in ids
        assert "b.txt" in ids
        assert not any("exe" in i for i in ids)

    def test_directory_connector_mime_type_from_extension(self, tmp_path):
        from navigator.connector import DirectoryConnector
        (tmp_path / "page.html").write_text("<html><body>hi</body></html>")
        docs = list(DirectoryConnector(str(tmp_path)).list_documents())
        html_docs = [d for d in docs if d.id.endswith(".html")]
        assert html_docs[0].mime_type == "text/html"

    def test_document_content_hash_is_sha256(self):
        from navigator.connector import Document
        import hashlib
        doc = Document(id="x", content="hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert doc.content_hash() == expected

    def test_doc_from_dict_parses_timestamp(self):
        from navigator.connector import _doc_from_dict
        doc = _doc_from_dict({"id": "d1", "content": "x", "updated_at": "2026-05-01T00:00:00Z"})
        assert doc.updated_at is not None
        assert doc.updated_at.year == 2026

    def test_build_connector_jsonl(self, tmp_path):
        from navigator.connector import build_connector
        from navigator.config import ConnectorConfig
        f = tmp_path / "d.jsonl"
        f.write_text("")
        cfg = ConnectorConfig(enabled=True, type="jsonl", path=str(f))
        connector = build_connector(cfg)
        assert connector is not None

    def test_build_connector_disabled_returns_none(self):
        from navigator.connector import build_connector
        from navigator.config import ConnectorConfig
        assert build_connector(ConnectorConfig(enabled=False)) is None

    def test_build_connector_directory(self, tmp_path):
        from navigator.connector import build_connector, DirectoryConnector
        from navigator.config import ConnectorConfig
        cfg = ConnectorConfig(enabled=True, type="directory", path=str(tmp_path))
        assert isinstance(build_connector(cfg), DirectoryConnector)


# ─── FR-MR-06-004: Chunking Profiles ─────────────────────────────────────────

class TestChunkingProfiles:
    def test_profile_for_markdown_mime(self):
        from navigator.chunker import profile_for_mime, PROFILE_MARKDOWN
        assert profile_for_mime("text/markdown") == PROFILE_MARKDOWN

    def test_profile_for_csv_mime(self):
        from navigator.chunker import profile_for_mime, PROFILE_STRUCTURED_CSV
        assert profile_for_mime("text/csv") == PROFILE_STRUCTURED_CSV

    def test_profile_for_json_mime(self):
        from navigator.chunker import profile_for_mime, PROFILE_JSON_RECORD
        assert profile_for_mime("application/json") == PROFILE_JSON_RECORD

    def test_profile_for_html_mime(self):
        from navigator.chunker import profile_for_mime, PROFILE_HTML
        assert profile_for_mime("text/html") == PROFILE_HTML

    def test_profile_fallback_to_extension(self):
        from navigator.chunker import profile_for_mime, PROFILE_PLAIN_TEXT
        assert profile_for_mime("", filename="readme.txt") == PROFILE_PLAIN_TEXT

    def test_profile_default_is_markdown(self):
        from navigator.chunker import profile_for_mime, PROFILE_MARKDOWN
        assert profile_for_mime("") == PROFILE_MARKDOWN

    def test_config_for_plain_text_has_smaller_max_chars(self):
        from navigator.chunker import config_for_profile, PROFILE_PLAIN_TEXT, PROFILE_MARKDOWN
        md_cfg = config_for_profile(PROFILE_MARKDOWN)
        pt_cfg = config_for_profile(PROFILE_PLAIN_TEXT)
        assert pt_cfg.max_chars < md_cfg.max_chars

    def test_chunk_csv_one_chunk_per_row(self):
        from navigator.chunker import chunk_csv
        content = "name,age,city\nAlice,30,Seoul\nBob,25,Busan\nCharlie,35,Incheon"
        chunks = chunk_csv("doc1", content)
        assert len(chunks) == 3  # 3 data rows
        assert "name,age,city" in chunks[0].content  # header prepended
        assert "Alice" in chunks[0].content
        assert "Bob" in chunks[1].content

    def test_chunk_csv_empty_returns_empty(self):
        from navigator.chunker import chunk_csv
        assert chunk_csv("d", "") == []

    def test_chunk_json_array_one_chunk_per_record(self):
        from navigator.chunker import chunk_json
        content = json.dumps([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
        chunks = chunk_json("doc1", content)
        assert len(chunks) == 2
        assert '"id": 1' in chunks[0].content or '"id":1' in chunks[0].content

    def test_chunk_json_single_object(self):
        from navigator.chunker import chunk_json
        content = json.dumps({"id": 1, "data": "test"})
        chunks = chunk_json("doc1", content)
        assert len(chunks) == 1

    def test_chunk_json_invalid_falls_back_to_text(self):
        from navigator.chunker import chunk_json
        chunks = chunk_json("doc1", "not json at all")
        assert len(chunks) >= 1

    def test_chunk_html_strips_tags(self):
        from navigator.chunker import chunk_html
        content = "<html><body><h1>Title</h1><p>Paragraph text here.</p></body></html>"
        chunks = chunk_html("doc1", content)
        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "Title" in combined
        assert "Paragraph text" in combined
        assert "<html>" not in combined
        assert "<p>" not in combined

    def test_chunk_by_profile_dispatches_correctly(self):
        from navigator.chunker import chunk_by_profile, PROFILE_STRUCTURED_CSV
        content = "col1,col2\nval1,val2\nval3,val4"
        chunks = chunk_by_profile("doc1", content, PROFILE_STRUCTURED_CSV)
        assert len(chunks) == 2

    def test_chunk_by_profile_markdown_uses_heading_path(self):
        from navigator.chunker import chunk_by_profile, PROFILE_MARKDOWN
        content = "# Title\n\nContent here."
        chunks = chunk_by_profile("doc1", content, PROFILE_MARKDOWN)
        assert chunks[0].heading_path == ["# Title"]


# ─── FR-MR-02-003: Sub-query Decomposition ───────────────────────────────────

class TestQueryDecomposer:
    def test_non_multi_hop_not_decomposed(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("What is the defect rate?", intent="factual")
        assert result.strategy == "none"
        assert result.sub_queries == ["What is the defect rate?"]

    def test_english_conjunction_split(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("Show sales data and also show defect rates", intent="multi_hop")
        assert result.strategy == "conjunction"
        assert len(result.sub_queries) == 2
        assert any("sales" in q for q in result.sub_queries)

    def test_korean_conjunction_split(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("매출 현황 조회 그리고 불량률 확인", intent="multi_hop")
        assert result.strategy == "conjunction"
        assert len(result.sub_queries) == 2

    def test_max_sub_queries_capped(self):
        from navigator.decomposer import QueryDecomposer, MAX_SUB_QUERIES
        d = QueryDecomposer()
        # Query with many conjunctions
        parts = ["find A", "find B", "find C", "find D", "find E"]
        query = " and also ".join(parts)
        result = d.decompose(query, intent="multi_hop")
        assert len(result.sub_queries) <= MAX_SUB_QUERIES

    def test_no_conjunction_falls_back_to_none(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("What is the annual leave policy?", intent="multi_hop")
        # No conjunctions → strategy=none, single sub-query
        assert result.sub_queries == ["What is the annual leave policy?"]

    def test_empty_query_returns_unchanged(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("", intent="multi_hop")
        assert result.sub_queries == [""]

    def test_duplicate_sub_queries_deduplicated(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("find data and also find data", intent="multi_hop")
        assert len(set(result.sub_queries)) == len(result.sub_queries)

    def test_as_well_as_conjunction(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d.decompose("Show HR data as well as finance records", intent="multi_hop")
        assert result.strategy == "conjunction"

    def test_original_query_preserved(self):
        from navigator.decomposer import QueryDecomposer
        d = QueryDecomposer()
        q = "Find defects and also find employees"
        result = d.decompose(q, intent="multi_hop")
        assert result.original == q


# ─── FR-MR-06-002/003: Delta Indexing ────────────────────────────────────────

class TestDeltaIndexing:
    def _make_orch(self, stored_hash=""):
        """Build a minimal orchestrator with mock searcher."""
        from navigator.orchestrator import Orchestrator
        from navigator.config import Config

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.0] * 1024
        mock_embedder.embed_batch.return_value = [[0.0] * 1024]

        mock_searcher = MagicMock()
        mock_searcher.count_by_document.return_value = 3
        mock_searcher.delete_by_document.return_value = 3
        mock_searcher.ensure_collection.return_value = None
        mock_searcher.upsert.return_value = None
        # Simulate stored hash via vector_search metadata
        mock_result = MagicMock()
        mock_result.metadata = {"content_hash": stored_hash}
        mock_searcher.vector_search.return_value = [mock_result] if stored_hash else []

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = lambda q, c, k: c[:k]

        orch = Orchestrator(
            cfg=Config(),
            embedder=mock_embedder,
            searcher=mock_searcher,
            reranker=mock_reranker,
            vault=MagicMock(),
        )
        return orch, mock_searcher

    def test_same_hash_skips_reindex(self):
        import hashlib
        from navigator.models import DeltaIndexRequest
        content = "hello world"
        h = hashlib.sha256(content.encode()).hexdigest()
        orch, searcher = self._make_orch(stored_hash=h)

        req = DeltaIndexRequest(document_id="d1", content=content)
        resp = orch.delta_index_document(req)

        assert resp.indexed is False
        searcher.delete_by_document.assert_not_called()

    def test_changed_hash_triggers_reindex(self):
        from navigator.models import DeltaIndexRequest
        orch, searcher = self._make_orch(stored_hash="oldhash")

        req = DeltaIndexRequest(document_id="d1", content="new content")
        resp = orch.delta_index_document(req)

        assert resp.indexed is True
        searcher.delete_by_document.assert_called_once()

    def test_force_flag_bypasses_hash_check(self):
        import hashlib
        from navigator.models import DeltaIndexRequest
        content = "same content"
        h = hashlib.sha256(content.encode()).hexdigest()
        orch, searcher = self._make_orch(stored_hash=h)

        req = DeltaIndexRequest(document_id="d1", content=content, force=True)
        resp = orch.delta_index_document(req)

        assert resp.indexed is True
        searcher.delete_by_document.assert_called_once()

    def test_delete_called_before_upsert(self):
        """SC-10: deletion must complete before insertion."""
        from navigator.models import DeltaIndexRequest
        call_order = []

        orch, searcher = self._make_orch(stored_hash="old")
        searcher.delete_by_document.side_effect = lambda *a, **k: call_order.append("delete")
        searcher.upsert.side_effect = lambda *a, **k: call_order.append("upsert")

        req = DeltaIndexRequest(document_id="d1", content="new")
        orch.delta_index_document(req)

        assert call_order.index("delete") < call_order.index("upsert")

    def test_response_includes_content_hash(self):
        import hashlib
        from navigator.models import DeltaIndexRequest
        content = "test content for hash"
        orch, _ = self._make_orch()
        req = DeltaIndexRequest(document_id="d1", content=content)
        resp = orch.delta_index_document(req)
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert resp.content_hash == expected

    def test_index_document_stores_content_hash(self):
        import hashlib
        from navigator.models import IndexRequest
        orch, searcher = self._make_orch()
        req = IndexRequest(document_id="doc1", content="Some content to index", category="default")
        resp = orch.index_document(req)
        assert resp.content_hash == hashlib.sha256(b"Some content to index").hexdigest()
        # Verify the hash appears in the upsert payload
        call_args = searcher.upsert.call_args
        if call_args:
            points = call_args[0][1] if call_args[0] else call_args[1].get("points", [])
            for point in points:
                assert point["payload"]["content_hash"] == resp.content_hash


# ─── FR-MR-04-004: Data Steward (Navigator side) ─────────────────────────────

class TestUpdatePurposes:
    def test_update_purposes_calls_set_payload(self):
        from navigator.orchestrator import Orchestrator
        from navigator.config import Config
        from navigator.models import UpdatePurposesRequest

        mock_searcher = MagicMock()
        mock_searcher.set_payload.return_value = 5

        orch = Orchestrator(
            cfg=Config(),
            embedder=MagicMock(),
            searcher=mock_searcher,
            reranker=MagicMock(),
            vault=MagicMock(),
        )

        req = UpdatePurposesRequest(
            document_id="doc1",
            tenant_id="acme",
            collection="customer_docs",
            permitted_purposes=["customer_support", "audit"],
            steward_user_id="steward.kim",
        )
        resp = orch.update_document_purposes(req)

        mock_searcher.set_payload.assert_called_once_with(
            "customer_docs", "doc1", {"permitted_purposes": "customer_support,audit"}
        )
        assert resp.chunks_updated == 5
        assert resp.permitted_purposes == ["customer_support", "audit"]
