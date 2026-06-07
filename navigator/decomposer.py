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
# Defaults; each is overridable via DecomposerConfig (decomposer.*_pattern).
# English conjunctions — use \b to avoid matching mid-word.
DEFAULT_CONJUNCTION_EN_PATTERN = (
    r"\band\s+also\b|\bas\s+well\s+as\b|\bboth\b"
    r"|\bmoreover\b|\badditionally\b|\bfurthermore\b"
    r"|\balso\b(?=\s+(?:find|show|get|list|what|who|when|where))"
)
# Korean conjunctions — no word boundary needed.
DEFAULT_CONJUNCTION_KR_PATTERN = r"그리고|또한|및|더불어|아울러"
# Temporal / conditional splits.
DEFAULT_TEMPORAL_PATTERN = (
    r"\b(?:before|after|since|until|when|while)\b"
    r"|이전에|이후에|동안에|전에|후에"
)
# Sentence separators — natural boundaries when conjunctions are absent.
DEFAULT_SENTENCE_SEP_PATTERN = r"[.!?。！？]+\s+"

_RE_EN = re.compile(DEFAULT_CONJUNCTION_EN_PATTERN, re.IGNORECASE)
_RE_KR = re.compile(DEFAULT_CONJUNCTION_KR_PATTERN)
_RE_TEMPORAL = re.compile(DEFAULT_TEMPORAL_PATTERN, re.IGNORECASE)
_RE_SENTENCE_SEP = re.compile(DEFAULT_SENTENCE_SEP_PATTERN)


def _compile_or(raw: str, default: "re.Pattern", flags: int = 0) -> "re.Pattern":
    """Compile raw with flags; fall back to default when empty/invalid."""
    if raw:
        try:
            return re.compile(raw, flags)
        except re.error:
            return default
    return default


@dataclass
class DecomposedQuery:
    sub_queries: list[str]
    strategy: str   # "conjunction" | "temporal" | "sentence" | "none"
    original: str


class QueryDecomposer:
    """Rule-based query decomposer for multi-hop intent."""

    def __init__(self, cfg=None):
        """cfg is an optional DecomposerConfig (or any object exposing
        *_pattern attributes); empty/missing fields use the module defaults."""
        self._re_en       = _compile_or(getattr(cfg, "conjunction_en_pattern", ""), _RE_EN, re.IGNORECASE)
        self._re_kr       = _compile_or(getattr(cfg, "conjunction_kr_pattern", ""), _RE_KR)
        self._re_temporal = _compile_or(getattr(cfg, "temporal_pattern", ""), _RE_TEMPORAL, re.IGNORECASE)
        self._re_sentence = _compile_or(getattr(cfg, "sentence_separator_pattern", ""), _RE_SENTENCE_SEP)

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
        combined = self._re_en.pattern + "|" + self._re_kr.pattern
        parts = re.split(combined, query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    def _split_temporal(self, query: str) -> list[str]:
        """Split at temporal/conditional clause boundaries."""
        parts = self._re_temporal.split(query)
        return [p.strip() for p in parts if p.strip()]

    def _split_sentences(self, query: str) -> list[str]:
        """Split at sentence boundaries (period / question mark / Korean full stop)."""
        parts = self._re_sentence.split(query)
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
