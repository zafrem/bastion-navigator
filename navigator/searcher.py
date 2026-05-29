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
