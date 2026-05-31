"""Qdrant vector + sparse searcher for Navigator."""
from __future__ import annotations

import logging
import time
from typing import Optional

from .models import CollectionInfo, SearchResult
from . import metrics

log = logging.getLogger(__name__)


def _build_filter(filters: dict[str, str]) -> dict:
    """Convert a flat key=value dict into a Qdrant filter."""
    conditions = [
        {"key": k, "match": {"value": v}}
        for k, v in filters.items()
    ]
    return {"must": conditions} if conditions else {}


_PROVENANCE_KEYS = frozenset({
    "content", "document_id", "chunk_id", "heading_path",
    "char_start", "char_end", "last_indexed",
})

def _to_search_result(hit) -> SearchResult:
    payload = hit.payload or {}
    return SearchResult(
        document_id=payload.get("document_id", str(hit.id)),
        content=payload.get("content", ""),
        score=hit.score,
        chunk_id=payload.get("chunk_id", ""),
        heading_path=payload.get("heading_path", ""),
        char_start=int(payload["char_start"]) if payload.get("char_start") is not None else 0,
        char_end=int(payload["char_end"]) if payload.get("char_end") is not None else 0,
        last_indexed=payload.get("last_indexed", ""),
        metadata={k: v for k, v in payload.items() if k not in _PROVENANCE_KEYS},
    )


class QdrantSearcher:
    """Searches Qdrant collections via the Python client."""

    def __init__(self, hosts: list[str]) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        host = hosts[0] if hosts else "localhost"
        self._client = QdrantClient(url=host)

    def vector_search(
        self,
        collection: str,
        vector: list[float],
        filters: dict[str, str],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        start = time.perf_counter()
        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            )
        hits = self._client.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
        )
        metrics.qdrant_call_duration_seconds.labels(operation="vector_search").observe(
            time.perf_counter() - start
        )
        return [_to_search_result(h) for h in hits]

    def sparse_search(
        self,
        collection: str,
        query: str,
        filters: dict[str, str],
        top_k: int,
    ) -> list[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        start = time.perf_counter()
        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            )
        hits = self._client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )[0]
        metrics.qdrant_call_duration_seconds.labels(operation="sparse_search").observe(
            time.perf_counter() - start
        )
        results = [_to_search_result(h) for h in hits]
        q_lower = query.lower()
        for r in results:
            r.score = sum(1 for w in q_lower.split() if w in r.content.lower()) / max(len(q_lower.split()), 1)
        return sorted(results, key=lambda r: r.score, reverse=True)

    def collections(self) -> list[CollectionInfo]:
        cols = self._client.get_collections().collections
        return [CollectionInfo(name=c.name) for c in cols]

    def ensure_collection(self, name: str, vector_size: int = 768) -> None:
        from qdrant_client.models import VectorParams, Distance  # type: ignore
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, points: list[dict]) -> None:
        from qdrant_client.models import PointStruct  # type: ignore
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )

    def delete_by_document(self, collection: str, document_id: str) -> int:
        """Delete all points whose document_id payload field matches document_id.

        Returns the number of points deleted (estimated from Qdrant status).
        Uses FilterSelector so the delete is server-side; no client-side scan.
        """
        from qdrant_client.models import (  # type: ignore
            Filter, FieldCondition, MatchValue, FilterSelector,
        )
        start = time.perf_counter()
        result = self._client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
        )
        metrics.qdrant_call_duration_seconds.labels(operation="delete_by_document").observe(
            time.perf_counter() - start
        )
        # Qdrant returns UpdateResult; deleted count is not always available —
        # return 0 as a safe default when the field is missing.
        return getattr(result, "deleted", 0) or 0

    def set_payload(self, collection: str, document_id: str, payload_patch: dict) -> int:
        """Overwrite selected payload fields on all chunks of a document (MR-04-004).

        Returns estimated count of updated points.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        result = self._client.set_payload(
            collection_name=collection,
            payload=payload_patch,
            points=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        return getattr(result, "updated", 0) or 0

    def count_by_document(self, collection: str, document_id: str) -> int:
        """Count chunks stored for a document (used for delta-index old_chunk_count)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        result = self._client.count(
            collection_name=collection,
            count_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        return result.count if result else 0


class MockSearcher:
    """In-memory mock searcher for tests."""

    def vector_search(self, *args, **kwargs) -> list[SearchResult]:
        return []

    def sparse_search(self, *args, **kwargs) -> list[SearchResult]:
        return []

    def collections(self) -> list[CollectionInfo]:
        return []

    def ensure_collection(self, name: str, vector_size: int = 768) -> None:
        pass

    def upsert(self, collection: str, points: list[dict]) -> None:
        pass

    def delete_by_document(self, collection: str, document_id: str) -> int:
        return 0

    def set_payload(self, collection: str, document_id: str, payload_patch: dict) -> int:
        return 0

    def count_by_document(self, collection: str, document_id: str) -> int:
        return 0
