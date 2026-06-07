"""Adaptive query routing — MR-01 (FR-MR-01-001/002/003)."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 for zero-magnitude or mismatched-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class QueryIntent(str, Enum):
    FACTUAL    = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    MULTI_HOP  = "multi_hop"
    AMBIGUOUS  = "ambiguous"


_INTENT_STRATEGY: dict[QueryIntent, str] = {
    QueryIntent.FACTUAL:    "vector_only",
    QueryIntent.ANALYTICAL: "hybrid+rerank",
    QueryIntent.PROCEDURAL: "hybrid",
    QueryIntent.MULTI_HOP:  "hybrid+rerank",
    QueryIntent.AMBIGUOUS:  "hybrid",
}

# Default intent-classification patterns. Each is overridable via RouterConfig
# (modular_rag.router.*_pattern in config.yaml). English terms use \b; Korean
# terms have no ASCII word boundary so they are listed in the same alternation.
DEFAULT_ANALYTICAL_PATTERN = (
    r"\b(compare|trend|average|total|distribution|breakdown|percentage|ratio|"
    r"aggregate|summary|statistics)\b"
    r"|분석|비교|통계|현황|추이|평균|합계"
)
DEFAULT_PROCEDURAL_PATTERN = (
    r"\b(how to|procedure|process|policy|guideline|step|apply)\b"
    r"|방법|절차|정책|기준|신청|지침"
)
DEFAULT_MULTI_HOP_PATTERN = (
    r"\b(and also|as well as|both)\b"
    r"|그리고|또한|및"
)
DEFAULT_FACTUAL_PATTERN = (
    r"\b(what is|what are|show me|get|find|list|retrieve)\b"
    r"|조회|확인|찾아|몇|언제|어디"
)

_RE_ANALYTICAL = re.compile(DEFAULT_ANALYTICAL_PATTERN, re.IGNORECASE)
_RE_PROCEDURAL = re.compile(DEFAULT_PROCEDURAL_PATTERN, re.IGNORECASE)
_RE_MULTI_HOP = re.compile(DEFAULT_MULTI_HOP_PATTERN, re.IGNORECASE)
_RE_FACTUAL = re.compile(DEFAULT_FACTUAL_PATTERN, re.IGNORECASE)


def _compile_or(raw: str, default: "re.Pattern") -> "re.Pattern":
    """Compile raw (case-insensitive); fall back to default when empty/invalid."""
    if raw:
        try:
            return re.compile(raw, re.IGNORECASE)
        except re.error:
            return default
    return default

# Lightweight domain keywords per collection (proxy for topic embeddings).
_COLLECTION_DOMAINS: dict[str, list[str]] = {
    "customer_docs":      ["customer", "account", "purchase", "고객", "계좌", "구매", "주문"],
    "manufacturing_docs": ["defect", "production", "factory", "line", "worker", "불량", "생산", "공장", "공정"],
    "hr_docs":            ["employee", "salary", "leave", "hr", "직원", "급여", "연차", "인사", "휴가", "근태"],
}


@dataclass
class RoutingDecision:
    intent: QueryIntent
    strategy: str
    collections: list[str]
    excluded: list[str]
    confidence: float
    routing_ms: float


class Router:
    """Classifies query intent and selects retrieval strategy + target collections."""

    def __init__(self, cfg=None):
        """cfg is an optional RouterConfig (or any object exposing
        *_pattern attributes); empty/missing fields use the module defaults."""
        self._re_analytical = _compile_or(getattr(cfg, "analytical_pattern", ""), _RE_ANALYTICAL)
        self._re_procedural = _compile_or(getattr(cfg, "procedural_pattern", ""), _RE_PROCEDURAL)
        self._re_multi_hop  = _compile_or(getattr(cfg, "multi_hop_pattern", ""), _RE_MULTI_HOP)
        self._re_factual    = _compile_or(getattr(cfg, "factual_pattern", ""), _RE_FACTUAL)

    def route(
        self,
        query: str,
        available_collections: list[str],
        routing_threshold: float = 0.25,
        strategy_override: str = "",
        query_vector: Optional[list[float]] = None,
        topic_vectors: Optional[dict[str, list[float]]] = None,
    ) -> RoutingDecision:
        t0 = time.perf_counter()
        intent, confidence = self._classify(query)
        strategy = strategy_override or _INTENT_STRATEGY[intent]
        # FR-MR-01-003: prefer embedding affinity when a query vector and at least
        # one collection topic vector are available; else fall back to keywords.
        if query_vector and topic_vectors:
            collections, excluded = self._select_by_embedding(
                available_collections, query_vector, topic_vectors, routing_threshold
            )
        else:
            collections, excluded = self._select_collections(query, available_collections)
        routing_ms = (time.perf_counter() - t0) * 1000
        return RoutingDecision(
            intent=intent,
            strategy=strategy,
            collections=collections,
            excluded=excluded,
            confidence=confidence,
            routing_ms=routing_ms,
        )

    def _classify(self, query: str) -> tuple[QueryIntent, float]:
        scores: dict[QueryIntent, float] = {i: 0.0 for i in QueryIntent}

        if self._re_analytical.search(query):
            scores[QueryIntent.ANALYTICAL] += 0.6
        if self._re_procedural.search(query):
            scores[QueryIntent.PROCEDURAL] += 0.6
        if self._re_multi_hop.search(query):
            scores[QueryIntent.MULTI_HOP] += 0.7
        if self._re_factual.search(query):
            scores[QueryIntent.FACTUAL] += 0.5

        fired = sum(1 for v in scores.values() if v > 0)
        if fired == 0:
            return QueryIntent.AMBIGUOUS, 0.3

        best = max(scores, key=scores.__getitem__)
        best_score = scores[best]
        if fired > 1:
            best_score *= 0.7  # conflicting signals lower confidence

        if best_score < 0.4:
            return QueryIntent.AMBIGUOUS, best_score

        return best, min(best_score, 0.95)

    def _select_collections(
        self,
        query: str,
        available: list[str],
    ) -> tuple[list[str], list[str]]:
        """Include collections with at least one keyword hit; fall back to all if none match."""
        q_lower = query.lower()
        hits: dict[str, int] = {}
        for col in available:
            keywords = _COLLECTION_DOMAINS.get(col, [])
            hits[col] = sum(1 for kw in keywords if kw in q_lower)

        if max(hits.values(), default=0) == 0:
            # No domain signal → search all (fail-open, SC-03)
            return list(available), []

        selected = [c for c, h in hits.items() if h > 0]
        excluded = [c for c, h in hits.items() if h == 0]
        return selected, excluded

    def _select_by_embedding(
        self,
        available: list[str],
        query_vector: list[float],
        topic_vectors: dict[str, list[float]],
        routing_threshold: float,
    ) -> tuple[list[str], list[str]]:
        """Score each collection by cosine(query, topic_vector); exclude those below
        routing_threshold. Collections lacking a topic vector are kept (fail-open,
        SC-03) since absence of a centroid is not evidence of irrelevance."""
        selected: list[str] = []
        excluded: list[str] = []
        for col in available:
            topic = topic_vectors.get(col)
            if topic is None:
                selected.append(col)          # no centroid yet → don't exclude
                continue
            if _cosine(query_vector, topic) >= routing_threshold:
                selected.append(col)
            else:
                excluded.append(col)

        # Never return an empty target set; fail open to all available collections.
        if not selected:
            return list(available), []
        return selected, excluded
