"""Adaptive query routing — MR-01 (FR-MR-01-001/002/003)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum


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

# English terms use \b; Korean terms have no ASCII word boundary so listed separately.
_RE_ANALYTICAL = re.compile(
    r"\b(compare|trend|average|total|distribution|breakdown|percentage|ratio|"
    r"aggregate|summary|statistics)\b"
    r"|분석|비교|통계|현황|추이|평균|합계",
    re.IGNORECASE,
)
_RE_PROCEDURAL = re.compile(
    r"\b(how to|procedure|process|policy|guideline|step|apply)\b"
    r"|방법|절차|정책|기준|신청|지침",
    re.IGNORECASE,
)
_RE_MULTI_HOP = re.compile(
    r"\b(and also|as well as|both)\b"
    r"|그리고|또한|및",
    re.IGNORECASE,
)
_RE_FACTUAL = re.compile(
    r"\b(what is|what are|show me|get|find|list|retrieve)\b"
    r"|조회|확인|찾아|몇|언제|어디",
    re.IGNORECASE,
)

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

    def route(
        self,
        query: str,
        available_collections: list[str],
        routing_threshold: float = 0.25,
        strategy_override: str = "",
    ) -> RoutingDecision:
        t0 = time.perf_counter()
        intent, confidence = self._classify(query)
        strategy = strategy_override or _INTENT_STRATEGY[intent]
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

        if _RE_ANALYTICAL.search(query):
            scores[QueryIntent.ANALYTICAL] += 0.6
        if _RE_PROCEDURAL.search(query):
            scores[QueryIntent.PROCEDURAL] += 0.6
        if _RE_MULTI_HOP.search(query):
            scores[QueryIntent.MULTI_HOP] += 0.7
        if _RE_FACTUAL.search(query):
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
