"""Sub-query decomposition — MR-02-003 (FR-MR-02-003).

Splits a multi-hop query into N ≤ 4 independent sub-queries using rule-based
conjunction detection. No LLM call required.

Security: SC-04 — each sub-query is an independent retrieval surface. The
caller (orchestrator) executes sub-queries independently and merges results
with RRF. Per-sub-query Sentinel-IN / Vault Phase-1 re-entry is deferred
pending cross-service client infrastructure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_SUB_QUERIES = 4

# ─── splitting patterns ───────────────────────────────────────────────────────
# English conjunctions — use \b to avoid matching mid-word.
_RE_EN = re.compile(
    r"\band\s+also\b|\bas\s+well\s+as\b|\bboth\b"
    r"|\bmoreover\b|\badditionally\b|\bfurthermore\b"
    r"|\balso\b(?=\s+(?:find|show|get|list|what|who|when|where))",
    re.IGNORECASE,
)
# Korean conjunctions — no word boundary needed.
_RE_KR = re.compile(r"그리고|또한|및|더불어|아울러")

# Temporal / conditional splits.
_RE_TEMPORAL = re.compile(
    r"\b(?:before|after|since|until|when|while)\b"
    r"|이전에|이후에|동안에|전에|후에",
    re.IGNORECASE,
)

# Sentence separators — used to find natural boundaries when conjunctions are absent.
_RE_SENTENCE_SEP = re.compile(r"[.!?。！？]+\s+")


@dataclass
class DecomposedQuery:
    sub_queries: list[str]
    strategy: str   # "conjunction" | "temporal" | "sentence" | "none"
    original: str


class QueryDecomposer:
    """Rule-based query decomposer for multi-hop intent."""

    def decompose(self, query: str, intent: str = "multi_hop") -> DecomposedQuery:
        """Return a DecomposedQuery. strategy='none' when decomposition is not helpful."""
        if intent != "multi_hop" or not query.strip():
            return DecomposedQuery(sub_queries=[query], strategy="none", original=query)

        # Try conjunction split first.
        parts = self._split_conjunctions(query)
        if len(parts) > 1:
            return DecomposedQuery(
                sub_queries=_cap_and_clean(parts),
                strategy="conjunction",
                original=query,
            )

        # Try temporal/conditional clause split.
        parts = self._split_temporal(query)
        if len(parts) > 1:
            return DecomposedQuery(
                sub_queries=_cap_and_clean(parts),
                strategy="temporal",
                original=query,
            )

        # Last resort: split at sentence boundaries for very long multi-hop queries.
        parts = self._split_sentences(query)
        if len(parts) > 1:
            return DecomposedQuery(
                sub_queries=_cap_and_clean(parts),
                strategy="sentence",
                original=query,
            )

        return DecomposedQuery(sub_queries=[query], strategy="none", original=query)

    # ── private ───────────────────────────────────────────────────────────────

    def _split_conjunctions(self, query: str) -> list[str]:
        """Split at English + Korean conjunction patterns."""
        combined = _RE_EN.pattern + "|" + _RE_KR.pattern
        parts = re.split(combined, query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    def _split_temporal(self, query: str) -> list[str]:
        """Split at temporal/conditional clause boundaries."""
        parts = _RE_TEMPORAL.split(query)
        return [p.strip() for p in parts if p.strip()]

    def _split_sentences(self, query: str) -> list[str]:
        """Split at sentence boundaries (period / question mark / Korean full stop)."""
        parts = _RE_SENTENCE_SEP.split(query)
        return [p.strip() for p in parts if p.strip()]


def _cap_and_clean(parts: list[str]) -> list[str]:
    """Deduplicate, strip empty, enforce MAX_SUB_QUERIES cap."""
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) == MAX_SUB_QUERIES:
            break
    return out
