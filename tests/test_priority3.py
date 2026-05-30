"""Tests for Priority 3 features: purpose filter, HyDE, staleness, events.

Items covered:
  FR-MR-04-001/002  Document purpose tagging + request purpose declaration
  FR-MR-04-003      Purpose × RBAC pre-filter in Navigator
  FR-MR-02-002      HyDE transformer
  FR-MR-05-004      Staleness tracking
  Events            event_purpose_filtered, event_chunk_stale, event_query_transformed
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from navigator.models import (
    IndexRequest, SearchOptions, SearchRequest, SearchResult, PERMITTED_PURPOSES,
)
from navigator.events import (
    TraceContext, event_purpose_filtered, event_chunk_stale, event_query_transformed,
)
from navigator.hyde import HyDETransformer
from navigator.orchestrator import _filter_by_purpose, _check_staleness


# ─── helpers ──────────────────────────────────────────────────────────────────

def _tc() -> TraceContext:
    return TraceContext(trace_id="t1", span_id="s1", tenant_id="acme")


def _result(doc_id: str, permitted_purposes: str = "", last_indexed: str = "") -> SearchResult:
    meta: dict[str, str] = {}
    if permitted_purposes:
        meta["permitted_purposes"] = permitted_purposes
    if last_indexed:
        meta["last_indexed"] = last_indexed
    return SearchResult(
        document_id=doc_id,
        content="content",
        score=0.8,
        chunk_id=f"{doc_id}_0000",
        last_indexed=last_indexed,
        metadata=meta,
    )


# ─── MR-04-001: IndexRequest permitted_purposes ───────────────────────────────

class TestIndexRequestPurpose:
    def test_default_permitted_purposes(self):
        req = IndexRequest(document_id="d1", content="text")
        assert req.permitted_purposes == ["customer_support"]

    def test_custom_permitted_purposes(self):
        req = IndexRequest(document_id="d1", content="text",
                           permitted_purposes=["audit", "legal"])
        assert req.permitted_purposes == ["audit", "legal"]

    def test_permitted_purposes_constants_available(self):
        assert "customer_support" in PERMITTED_PURPOSES
        assert "audit" in PERMITTED_PURPOSES
        assert "hr_analytics" in PERMITTED_PURPOSES
        assert "training_data" in PERMITTED_PURPOSES


# ─── MR-04-002: SearchRequest purpose declaration ─────────────────────────────

class TestSearchRequestPurpose:
    def test_default_purpose(self):
        req = SearchRequest(query="test")
        assert req.purpose == "customer_support"

    def test_custom_purpose(self):
        req = SearchRequest(query="test", purpose="audit")
        assert req.purpose == "audit"

    def test_purpose_field_exists(self):
        assert hasattr(SearchRequest(query="x"), "purpose")


# ─── MR-04-002/003: purpose pre-filter ───────────────────────────────────────

class TestFilterByPurpose:
    def test_allows_matching_purpose(self):
        r = _result("d1", permitted_purposes="customer_support,audit")
        allowed, excluded = _filter_by_purpose([r], "customer_support")
        assert "d1" in [x.document_id for x in allowed]
        assert excluded == []

    def test_excludes_non_matching_purpose(self):
        r = _result("d1", permitted_purposes="audit,legal")
        allowed, excluded = _filter_by_purpose([r], "customer_support")
        assert allowed == []
        assert "d1" in [x.document_id for x in excluded]

    def test_allows_when_no_purpose_set_on_document(self):
        r = _result("d1")  # no permitted_purposes in metadata
        allowed, excluded = _filter_by_purpose([r], "customer_support")
        assert "d1" in [x.document_id for x in allowed]
        assert excluded == []

    def test_allows_all_when_purpose_empty(self):
        r = _result("d1", permitted_purposes="audit")
        allowed, excluded = _filter_by_purpose([r], "")
        assert "d1" in [x.document_id for x in allowed]
        assert excluded == []

    def test_splits_comma_separated_purposes_correctly(self):
        r = _result("d1", permitted_purposes="audit,hr_analytics,legal")
        a, _ = _filter_by_purpose([r], "hr_analytics")
        assert len(a) == 1

    def test_multiple_results_mixed_purposes(self):
        results = [
            _result("doc-cs", permitted_purposes="customer_support"),
            _result("doc-audit", permitted_purposes="audit"),
            _result("doc-any"),  # no restrictions
        ]
        allowed, excluded = _filter_by_purpose(results, "customer_support")
        allowed_ids = {r.document_id for r in allowed}
        assert "doc-cs" in allowed_ids
        assert "doc-any" in allowed_ids
        assert "doc-audit" not in allowed_ids
        assert len(excluded) == 1


# ─── event_purpose_filtered ───────────────────────────────────────────────────

class TestEventPurposeFiltered:
    def test_event_type(self):
        ev = event_purpose_filtered(_tc(), "doc-1", "customer_support", ["audit"])
        assert ev.event_type == "purpose_filtered"

    def test_category_is_security(self):
        ev = event_purpose_filtered(_tc(), "doc-1", "customer_support", ["audit"])
        assert ev.category == "security"

    def test_data_fields(self):
        ev = event_purpose_filtered(_tc(), "doc-1", "customer_support", ["audit", "legal"])
        assert ev.data["document_id"] == "doc-1"
        assert ev.data["declared_purpose"] == "customer_support"
        assert ev.data["permitted_purposes"] == ["audit", "legal"]

    def test_status_is_filtered(self):
        ev = event_purpose_filtered(_tc(), "x", "y", [])
        assert ev.status == "filtered"


# ─── MR-02-002: HyDE ─────────────────────────────────────────────────────────

class TestHyDETransformer:
    def test_should_apply_factual_short_query(self):
        h = HyDETransformer()
        assert h.should_apply("What is the defect rate?", "factual", max_words=12) is True

    def test_should_apply_procedural_short_query(self):
        h = HyDETransformer()
        assert h.should_apply("How to apply for leave?", "procedural", max_words=12) is True

    def test_should_not_apply_long_query(self):
        h = HyDETransformer()
        long_q = "word " * 15  # 15 words
        assert h.should_apply(long_q, "factual", max_words=12) is False

    def test_should_not_apply_analytical(self):
        h = HyDETransformer()
        assert h.should_apply("Compare Q3 vs Q4", "analytical", max_words=12) is False

    def test_generate_returns_non_empty_string(self):
        h = HyDETransformer()
        result = h.generate("What is the defect rate?", "factual")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_includes_query_text(self):
        h = HyDETransformer()
        result = h.generate("What is the defect rate?", "factual")
        assert "defect rate" in result.lower() or "What is" in result

    def test_generate_factual_and_procedural_differ(self):
        h = HyDETransformer()
        f = h.generate("What is X?", "factual")
        p = h.generate("How to do X?", "procedural")
        assert f != p

    def test_generate_respects_max_chars(self):
        h = HyDETransformer()
        result = h.generate("What is X?", "factual")
        assert len(result) <= 400

    def test_llm_failure_falls_back_to_template(self):
        h = HyDETransformer(llm_endpoint="http://fake:11434")
        # LLM endpoint unreachable → should fall back to template
        result = h.generate("What is the defect rate?", "factual")
        assert len(result) > 0

    def test_should_apply_ambiguous(self):
        h = HyDETransformer()
        assert h.should_apply("unknown short query", "ambiguous", max_words=12) is True


# ─── event_query_transformed ──────────────────────────────────────────────────

class TestEventQueryTransformed:
    def test_event_type(self):
        ev = event_query_transformed(_tc(), "hyde", 20, 150, 1.5)
        assert ev.event_type == "query_transformed"

    def test_category_is_operational(self):
        ev = event_query_transformed(_tc(), "hyde", 20, 150, 1.5)
        assert ev.category == "operational"

    def test_data_fields(self):
        ev = event_query_transformed(_tc(), "hyde", 20, 150, 2.3)
        assert ev.data["transformation_type"] == "hyde"
        assert ev.data["original_length"] == 20
        assert ev.data["transformed_length"] == 150

    def test_status_is_transformed(self):
        ev = event_query_transformed(_tc(), "hyde", 10, 100, 1.0)
        assert ev.status == "transformed"

    def test_sub_query_count_default_zero(self):
        ev = event_query_transformed(_tc(), "hyde", 10, 100, 1.0)
        assert ev.data["sub_query_count"] == 0


# ─── MR-05-004: staleness tracking ───────────────────────────────────────────

class TestCheckStaleness:
    def _stale_ts(self, days_ago: int) -> str:
        dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return dt.isoformat()

    def test_fresh_chunk_not_flagged(self):
        r = _result("d1", last_indexed=self._stale_ts(3))
        r = r.model_copy(update={"last_indexed": self._stale_ts(3)})
        updated = _check_staleness([r], threshold_days=7, tc=_tc(), publisher=None)
        assert updated[0].metadata.get("stale") != "true"

    def test_stale_chunk_flagged(self):
        r = _result("d1")
        r = r.model_copy(update={"last_indexed": self._stale_ts(10)})
        updated = _check_staleness([r], threshold_days=7, tc=_tc(), publisher=None)
        assert updated[0].metadata.get("stale") == "true"

    def test_days_stale_recorded(self):
        r = _result("d1")
        r = r.model_copy(update={"last_indexed": self._stale_ts(15)})
        updated = _check_staleness([r], threshold_days=7, tc=_tc(), publisher=None)
        days = int(updated[0].metadata.get("days_stale", "0"))
        assert days >= 8  # at least 8 days stale (15 - 7)

    def test_missing_last_indexed_not_flagged(self):
        r = _result("d1", last_indexed="")
        updated = _check_staleness([r], threshold_days=7, tc=_tc(), publisher=None)
        assert updated[0].metadata.get("stale") != "true"

    def test_invalid_timestamp_not_flagged(self):
        r = _result("d1")
        r = r.model_copy(update={"last_indexed": "not-a-date"})
        updated = _check_staleness([r], threshold_days=7, tc=_tc(), publisher=None)
        assert updated[0].metadata.get("stale") != "true"

    def test_stale_event_emitted(self):
        pub = MagicMock()
        r = _result("d1")
        r = r.model_copy(update={"last_indexed": self._stale_ts(10)})
        _check_staleness([r], threshold_days=7, tc=_tc(), publisher=pub)
        events_fired = [c.args[0].event_type for c in pub.publish.call_args_list]
        assert "chunk_stale" in events_fired

    def test_fresh_chunk_no_stale_event(self):
        pub = MagicMock()
        r = _result("d1")
        r = r.model_copy(update={"last_indexed": self._stale_ts(2)})
        _check_staleness([r], threshold_days=7, tc=_tc(), publisher=pub)
        events_fired = [c.args[0].event_type for c in pub.publish.call_args_list]
        assert "chunk_stale" not in events_fired

    def test_result_still_returned_when_stale(self):
        """Fail-open: stale chunks are returned, just flagged."""
        r = _result("d1")
        r = r.model_copy(update={"last_indexed": self._stale_ts(30)})
        updated = _check_staleness([r], threshold_days=7, tc=_tc(), publisher=None)
        assert len(updated) == 1
        assert updated[0].document_id == "d1"


# ─── event_chunk_stale ────────────────────────────────────────────────────────

class TestEventChunkStale:
    def test_event_type(self):
        ev = event_chunk_stale(_tc(), "d1_0001", "d1", "2026-01-01T00:00:00Z", 10)
        assert ev.event_type == "chunk_stale"

    def test_severity_is_warning(self):
        ev = event_chunk_stale(_tc(), "d1_0001", "d1", "2026-01-01T00:00:00Z", 10)
        assert ev.severity == "warning"

    def test_category_is_lineage(self):
        ev = event_chunk_stale(_tc(), "d1_0001", "d1", "2026-01-01T00:00:00Z", 10)
        assert ev.category == "lineage"

    def test_data_fields(self):
        ev = event_chunk_stale(_tc(), "d1_0001", "d1", "2026-01-01T00:00:00Z", 10)
        assert ev.data["chunk_id"] == "d1_0001"
        assert ev.data["document_id"] == "d1"
        assert ev.data["days_stale"] == 10

    def test_status_is_stale(self):
        ev = event_chunk_stale(_tc(), "x", "y", "ts", 5)
        assert ev.status == "stale"


# ─── Orchestrator integration: purpose filter emits events ────────────────────

class TestOrchestratorPurposeIntegration:
    def _make_orch(self, results, publisher=None):
        from navigator.config import Config, SearchDefaultsConfig, VectorDBConfig, CollectionConfig
        from navigator.orchestrator import Orchestrator
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

        class _FS:
            def vector_search(self, *a, **k): return results
            def sparse_search(self, *a, **k): return []
            def collections(self): return []
            def ensure_collection(self, *a, **k): pass
            def upsert(self, *a, **k): pass

        return Orchestrator(cfg, embedder, _FS(), reranker, NoopVaultClient(), publisher=publisher)

    def test_purpose_filtered_event_emitted_for_excluded(self):
        results = [_result("doc-cs", permitted_purposes="customer_support")]
        pub = MagicMock()
        orch = self._make_orch(results, publisher=pub)
        # Declare "audit" purpose — doc-cs only allows customer_support → excluded
        orch.search(SearchRequest(query="test", purpose="audit"))
        fired = [c.args[0].event_type for c in pub.publish.call_args_list]
        assert "purpose_filtered" in fired

    def test_no_purpose_filtered_event_when_purpose_matches(self):
        results = [_result("doc-cs", permitted_purposes="customer_support,audit")]
        pub = MagicMock()
        orch = self._make_orch(results, publisher=pub)
        orch.search(SearchRequest(query="test", purpose="customer_support"))
        fired = [c.args[0].event_type for c in pub.publish.call_args_list]
        assert "purpose_filtered" not in fired

    def test_permitted_purposes_stored_in_index_payload(self):
        from navigator.models import IndexRequest
        captured = []

        class _CS:
            def vector_search(self, *a, **k): return []
            def sparse_search(self, *a, **k): return []
            def collections(self): return []
            def ensure_collection(self, *a, **k): pass
            def upsert(self, c, points): captured.extend(points)

        from navigator.config import Config, SearchDefaultsConfig, VectorDBConfig, CollectionConfig
        from navigator.orchestrator import Orchestrator
        from navigator.vault_client import NoopVaultClient

        cfg = Config()
        cfg.search_defaults = SearchDefaultsConfig(top_k=10, over_fetch_multiplier=2)
        cfg.vector_db = VectorDBConfig(collections={"test": CollectionConfig()})
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[0.0] * 8]
        reranker = MagicMock()
        orch = Orchestrator(cfg, embedder, _CS(), reranker, NoopVaultClient())

        orch.index_document(IndexRequest(
            document_id="d1",
            content="# H\n\nContent.",
            permitted_purposes=["audit", "legal"],
        ))
        assert len(captured) > 0
        assert captured[0]["payload"]["permitted_purposes"] == "audit,legal"
