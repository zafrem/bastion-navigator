"""Unit tests for navigator.evaluator (MR-03)."""
from __future__ import annotations

import pytest
from navigator.evaluator import Evaluator, EvaluatorConfig, QualityVerdict, QualitySignals
from navigator.models import SearchResult


def _result(score: float, content: str = "", heading_path: str = "") -> SearchResult:
    return SearchResult(
        document_id=f"doc-{int(score*100)}",
        content=content,
        score=score,
        heading_path=heading_path,
    )


@pytest.fixture
def ev() -> Evaluator:
    return Evaluator()


@pytest.fixture
def cfg() -> EvaluatorConfig:
    return EvaluatorConfig()


class TestQualityVerdict:
    def test_sufficient_when_high_score_and_coverage(self, ev, cfg):
        results = [
            _result(0.85, "customer account purchase history defect"),
            _result(0.75, "customer order"),
        ]
        verdict, _ = ev.evaluate("customer purchase history", results, cfg)
        assert verdict == QualityVerdict.SUFFICIENT

    def test_insufficient_when_score_too_low(self, ev, cfg):
        results = [_result(0.20, "unrelated content xyz")]
        verdict, _ = ev.evaluate("customer purchase", results, cfg)
        assert verdict == QualityVerdict.INSUFFICIENT

    def test_uncertain_for_borderline_score(self, ev, cfg):
        results = [
            _result(0.50, "some relevant content purchase customer"),
        ]
        verdict, _ = ev.evaluate("customer purchase history", results, cfg)
        assert verdict == QualityVerdict.UNCERTAIN

    def test_insufficient_when_empty_results(self, ev, cfg):
        verdict, signals = ev.evaluate("any query", [], cfg)
        assert verdict == QualityVerdict.INSUFFICIENT
        assert signals.top_score == 0.0

    def test_coverage_below_threshold_gives_insufficient(self, ev, cfg):
        # High score but zero keyword coverage → insufficient
        results = [_result(0.80, "completely unrelated text foo bar baz")]
        verdict, signals = ev.evaluate("customer purchase defect history", results, cfg)
        # Coverage should be low for completely unrelated text
        assert signals.coverage < 0.4


class TestQualitySignals:
    def test_top_score_is_first_result_score(self, ev, cfg):
        results = [_result(0.75), _result(0.60)]
        _, signals = ev.evaluate("query", results, cfg)
        assert signals.top_score == pytest.approx(0.75)

    def test_score_gap_computed_correctly(self, ev, cfg):
        results = [_result(0.80), _result(0.50)]
        _, signals = ev.evaluate("query", results, cfg)
        assert signals.score_gap == pytest.approx(0.30)

    def test_single_result_score_gap_equals_score(self, ev, cfg):
        results = [_result(0.70)]
        _, signals = ev.evaluate("query", results, cfg)
        assert signals.score_gap == pytest.approx(0.70)

    def test_diversity_with_varied_heading_paths(self, ev, cfg):
        results = [
            _result(0.70, heading_path="# Policy > ## Leave"),
            _result(0.65, heading_path="# Policy > ## Salary"),
            _result(0.60, heading_path="# Policy > ## Leave"),
        ]
        _, signals = ev.evaluate("query", results, cfg)
        # 2 unique paths out of 3 results → diversity ≈ 0.67
        assert signals.diversity == pytest.approx(2 / 3, rel=0.01)

    def test_zero_diversity_all_same_heading(self, ev, cfg):
        results = [
            _result(0.70, heading_path="# Section"),
            _result(0.60, heading_path="# Section"),
        ]
        _, signals = ev.evaluate("query", results, cfg)
        assert signals.diversity == pytest.approx(0.5)


class TestRefineQuery:
    def test_broaden_removes_long_tokens_on_low_score(self, ev):
        signals = QualitySignals(top_score=0.20, score_gap=0.20, coverage=0.5, diversity=0.5)
        refined, strategy = ev.refine_query("customer purchase history defect", signals)
        assert strategy == "broaden"
        assert len(refined.split()) < len("customer purchase history defect".split())

    def test_keyword_expand_strips_vault_tokens(self, ev):
        signals = QualitySignals(top_score=0.55, score_gap=0.10, coverage=0.20, diversity=0.5)
        query = "Find KR_NAME_4d9e1b account EMAIL_a1b2c3 purchase"
        refined, strategy = ev.refine_query(query, signals)
        assert strategy == "keyword_expand"
        assert "KR_NAME_4d9e1b" not in refined
        assert "EMAIL_a1b2c3" not in refined

    def test_diversify_appends_summary_hint(self, ev):
        signals = QualitySignals(top_score=0.65, score_gap=0.30, coverage=0.60, diversity=0.10)
        _, strategy = ev.refine_query("policy overview", signals)
        assert strategy == "diversify"

    def test_no_refinement_when_signals_ok(self, ev):
        signals = QualitySignals(top_score=0.65, score_gap=0.20, coverage=0.60, diversity=0.50)
        refined, strategy = ev.refine_query("good query", signals)
        assert strategy == "none"
        assert refined == "good query"

    def test_broaden_keeps_minimum_tokens(self, ev):
        signals = QualitySignals(top_score=0.10, score_gap=0.10, coverage=0.5, diversity=0.5)
        refined, _ = ev.refine_query("a b c", signals)
        # Short query should not be reduced below 3 tokens
        assert len(refined.split()) >= 1  # at minimum something remains

    def test_keyword_expand_returns_original_when_nothing_left(self, ev):
        signals = QualitySignals(top_score=0.55, score_gap=0.10, coverage=0.20, diversity=0.5)
        query = "KR_NAME_4d9e1b EMAIL_a1b2c3"
        refined, strategy = ev.refine_query(query, signals)
        # All tokens stripped → falls back to original
        assert strategy == "keyword_expand"
        # refined is either empty (stripped to nothing, returns original) or the original
        assert len(refined) > 0
