"""Retrieval quality evaluator — MR-03-001/002."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .models import SearchResult

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "what", "who", "how", "me", "my", "i",
    "의", "이", "가", "을", "를", "은", "는", "에", "에서", "로", "와", "과",
})


class QualityVerdict(str, Enum):
    SUFFICIENT   = "sufficient"
    INSUFFICIENT = "insufficient"
    UNCERTAIN    = "uncertain"


@dataclass
class QualitySignals:
    top_score:  float = 0.0
    score_gap:  float = 0.0
    coverage:   float = 0.0
    diversity:  float = 0.0


@dataclass
class EvaluatorConfig:
    quality_threshold:  float = 0.60
    coverage_threshold: float = 0.40
    uncertain_low:      float = 0.45


class Evaluator:
    """Scores retrieval quality and generates rule-based query refinements."""

    def evaluate(
        self,
        query: str,
        results: list[SearchResult],
        cfg: EvaluatorConfig,
    ) -> tuple[QualityVerdict, QualitySignals]:
        if not results:
            return QualityVerdict.INSUFFICIENT, QualitySignals()

        top_score = results[0].score
        score_gap = (
            results[0].score - results[1].score if len(results) > 1 else top_score
        )

        keywords = _keywords(query)
        combined = " ".join(r.content.lower() for r in results[:5])
        coverage = (
            sum(1 for kw in keywords if kw in combined) / len(keywords)
            if keywords else 1.0
        )

        paths = {r.heading_path for r in results if r.heading_path}
        diversity = len(paths) / len(results) if results else 0.0

        signals = QualitySignals(
            top_score=top_score,
            score_gap=score_gap,
            coverage=coverage,
            diversity=diversity,
        )

        if top_score >= cfg.quality_threshold and coverage >= cfg.coverage_threshold:
            return QualityVerdict.SUFFICIENT, signals
        if top_score >= cfg.uncertain_low:
            return QualityVerdict.UNCERTAIN, signals
        return QualityVerdict.INSUFFICIENT, signals

    def refine_query(
        self, query: str, signals: QualitySignals
    ) -> tuple[str, str]:
        """Return (refined_query, strategy). Rule-based, no LLM, < 10 ms."""
        if signals.top_score < 0.30:
            # Low score: broaden by removing the most specific (longest) tokens.
            tokens = re.findall(r"\S+", query)
            if len(tokens) > 3:
                tokens = sorted(tokens, key=len)[: max(3, len(tokens) - 1)]
            return " ".join(tokens), "broaden"

        if signals.coverage < 0.40:
            # Low coverage: strip opaque Vault tokens that block keyword matching.
            refined = re.sub(r"\b[A-Z_]+_[0-9a-f]{6}\b", "", query).strip()
            return refined or query, "keyword_expand"

        if signals.diversity < 0.30:
            # Low diversity: ask for varied aspects.
            return query + " 개요 요약", "diversify"

        return query, "none"


def _keywords(query: str) -> list[str]:
    tokens = re.findall(r"\w+", query.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
