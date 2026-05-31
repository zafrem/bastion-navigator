"""Tests for the PII split guard in navigator.chunker (FR-MR-12)."""
from __future__ import annotations

import os
import re
import tempfile
import textwrap

import pytest

from navigator.chunker import (
    ChunkerConfig,
    _default_pii_patterns,
    _pii_spans,
    _safe_cut,
    _split_oversized,
    chunk_document,
    load_pii_patterns,
)


# ─── load_pii_patterns ────────────────────────────────────────────────────────

class TestLoadPiiPatterns:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert load_pii_patterns(str(tmp_path)) == []

    def test_nonexistent_dir_returns_empty(self):
        assert load_pii_patterns("/does/not/exist/ever") == []

    def test_empty_string_returns_empty(self):
        assert load_pii_patterns("") == []

    def test_loads_python_pattern_from_yaml(self, tmp_path):
        (tmp_path / "test.yml").write_text(textwrap.dedent("""
            namespace: test
            patterns:
              - id: test_01
                pattern: '\\d{3}-\\d{4}'
                langs:
                  python: '\\d{3}-\\d{4}'
        """))
        patterns = load_pii_patterns(str(tmp_path))
        assert len(patterns) == 1
        assert patterns[0].search("123-4567")

    def test_falls_back_to_pattern_field_when_no_langs(self, tmp_path):
        (tmp_path / "test.yml").write_text(textwrap.dedent("""
            namespace: test
            patterns:
              - id: test_01
                pattern: '[A-Z]{3}\\d{4}'
        """))
        patterns = load_pii_patterns(str(tmp_path))
        assert len(patterns) == 1
        assert patterns[0].search("ABC1234")

    def test_skips_invalid_regex_gracefully(self, tmp_path):
        (tmp_path / "bad.yml").write_text(textwrap.dedent("""
            namespace: test
            patterns:
              - id: bad_01
                pattern: '[unclosed'
                langs:
                  python: '[unclosed'
              - id: good_01
                pattern: '\\d+'
                langs:
                  python: '\\d+'
        """))
        patterns = load_pii_patterns(str(tmp_path))
        assert len(patterns) == 1
        assert patterns[0].search("42")

    def test_ignorecase_flag_applied(self, tmp_path):
        (tmp_path / "test.yml").write_text(textwrap.dedent("""
            namespace: test
            patterns:
              - id: ci_01
                pattern: 'hello'
                flags: [IGNORECASE]
                langs:
                  python: 'hello'
        """))
        patterns = load_pii_patterns(str(tmp_path))
        assert patterns[0].search("HELLO")
        assert patterns[0].search("hello")

    def test_walks_subdirectories(self, tmp_path):
        subdir = tmp_path / "kr"
        subdir.mkdir()
        (subdir / "phone.yml").write_text(textwrap.dedent("""
            namespace: kr
            patterns:
              - id: mobile_01
                pattern: '01[01679][\\d]{7,8}'
                langs:
                  python: '01[01679][\\d]{7,8}'
        """))
        patterns = load_pii_patterns(str(tmp_path))
        assert len(patterns) == 1

    def test_non_yml_files_ignored(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not yaml")
        (tmp_path / "config.json").write_text("{}")
        assert load_pii_patterns(str(tmp_path)) == []

    def test_malformed_yaml_skipped_gracefully(self, tmp_path):
        (tmp_path / "bad.yml").write_text(": this is: not: valid: yaml:::")
        (tmp_path / "good.yml").write_text(textwrap.dedent("""
            namespace: test
            patterns:
              - id: ok_01
                pattern: '\\d+'
                langs:
                  python: '\\d+'
        """))
        patterns = load_pii_patterns(str(tmp_path))
        assert len(patterns) == 1


# ─── _pii_spans ───────────────────────────────────────────────────────────────

class TestPiiSpans:
    RRN = re.compile(r'\d{6}-\d{7}')
    EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    def test_no_pii_returns_empty(self):
        assert _pii_spans("plain text with no PII", [self.RRN]) == []

    def test_single_match_correct_span(self):
        text = "ID: 900101-1234567 found"
        spans = _pii_spans(text, [self.RRN])
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == "900101-1234567"

    def test_multiple_matches_sorted(self):
        text = "a@b.com and 900101-1234567 and c@d.com"
        spans = _pii_spans(text, [self.RRN, self.EMAIL])
        assert len(spans) == 3
        starts = [s for s, _ in spans]
        assert starts == sorted(starts)

    def test_empty_patterns_returns_empty(self):
        assert _pii_spans("900101-1234567", []) == []


# ─── _safe_cut ────────────────────────────────────────────────────────────────

class TestSafeCut:
    def test_no_spans_returns_pos_unchanged(self):
        assert _safe_cut(50, []) == 50

    def test_pos_before_span_returns_pos(self):
        assert _safe_cut(5, [(10, 20)]) == 5

    def test_pos_after_span_returns_pos(self):
        assert _safe_cut(25, [(10, 20)]) == 25

    def test_pos_exactly_on_span_start_not_bisected(self):
        # pos == start is not "inside" (start < pos < end is False when pos==start)
        assert _safe_cut(10, [(10, 20)]) == 10

    def test_pos_inside_span_advances_to_end(self):
        assert _safe_cut(15, [(10, 20)]) == 20

    def test_pos_exactly_on_span_end_not_advanced(self):
        # pos == end: start < end < end is False
        assert _safe_cut(20, [(10, 20)]) == 20

    def test_two_consecutive_spans_advances_past_both(self):
        # pos=12 is inside (10,18); advancing to 18 then hits (16,25) → advance to 25
        assert _safe_cut(12, [(10, 18), (16, 25)]) == 25

    def test_nested_spans_resolved(self):
        assert _safe_cut(5, [(3, 15), (8, 20)]) == 20

    def test_non_overlapping_only_first_matters(self):
        assert _safe_cut(12, [(10, 20), (30, 40)]) == 20


# ─── _split_oversized with PII guard ─────────────────────────────────────────

_RRN_PAT = re.compile(r'\d{6}-\d{7}')
_EMAIL_PAT = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')


class TestSplitOversizedPiiGuard:
    """Tests for the character-fallback path with PII protection."""

    def _make_text(self, filler: str, pii: str, filler_len: int) -> str:
        """Build: <filler × n><pii><filler × n>"""
        pad = "x" * filler_len
        return pad + pii + pad

    def test_rrn_not_split_across_parts(self):
        # Place RRN exactly at the max_chars boundary so a naive cut bisects it.
        rrn = "900101-1234567"
        # pad = max_chars - len(rrn)//2 positions RRN straddling the cut
        prefix_len = 80 - 7  # cut at 80; RRN starts 7 chars before it
        text = "a" * prefix_len + rrn + "b" * 60
        parts = _split_oversized(text, max_chars=80, overlap_chars=0, pii_patterns=[_RRN_PAT])
        for part in parts:
            # Either the full RRN is present in the part, or it's absent —
            # never a partial fragment
            if rrn[:6] in part:
                assert rrn in part, f"RRN was bisected in part: {part!r}"

    def test_email_not_split_across_parts(self):
        email = "hong.gildong@example.com"
        prefix_len = 80 - 10  # cut bisects the email
        text = "a" * prefix_len + email + "b" * 60
        parts = _split_oversized(text, max_chars=80, overlap_chars=0, pii_patterns=[_EMAIL_PAT])
        for part in parts:
            if "@" in part and "hong" in part:
                assert email in part, f"Email was bisected in part: {part!r}"

    def test_no_patterns_behavior_unchanged(self):
        # Without patterns, the cut happens exactly at max_chars (modulo strip).
        text = "x" * 200
        parts_with = _split_oversized(text, 100, 0, pii_patterns=[_RRN_PAT])
        parts_without = _split_oversized(text, 100, 0, pii_patterns=None)
        # No PII in text → both should produce the same result
        assert parts_with == parts_without

    def test_pii_spanning_entire_remainder_kept_whole(self):
        # PII span covers the entire remaining text → kept as one part
        rrn = "900101-1234567"
        text = "a" * 80 + rrn  # total 94 chars, max_chars=80
        parts = _split_oversized(text, max_chars=80, overlap_chars=0, pii_patterns=[_RRN_PAT])
        combined = " ".join(parts)
        assert rrn in combined

    def test_multiple_pii_in_text_all_preserved(self):
        rrn1 = "900101-1234567"
        rrn2 = "851230-2345678"
        text = "a" * 40 + rrn1 + "b" * 40 + rrn2 + "c" * 20
        parts = _split_oversized(text, max_chars=60, overlap_chars=0, pii_patterns=[_RRN_PAT])
        for part in parts:
            # Each RRN that appears at all must appear whole
            for rrn in (rrn1, rrn2):
                if rrn[:6] in part:
                    assert rrn in part


# ─── chunk_document with PII guard ───────────────────────────────────────────

class TestChunkDocumentPiiGuard:
    """Integration tests: PII guard applied during full chunking."""

    RRN = re.compile(r'\d{6}-\d{7}')
    MOBILE = re.compile(r'01[01679][ -]?\d{3,4}[ -]?\d{4}')

    def _cfg(self, max_chars=200, overlap_chars=40) -> ChunkerConfig:
        return ChunkerConfig(
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            pii_patterns=[self.RRN, self.MOBILE],
        )

    def test_rrn_stays_in_single_chunk(self):
        rrn = "900101-1234567"
        # Build a document where the RRN straddles a natural chunk boundary.
        filler = "가나다라마바사아자차카타파하. " * 10  # ~200 chars
        doc = filler + rrn + " " + filler
        chunks = chunk_document("doc", doc, self._cfg(max_chars=200, overlap_chars=0))
        owner = [c for c in chunks if rrn in c.content]
        assert len(owner) >= 1, "RRN not found in any chunk"
        for c in chunks:
            if rrn[:6] in c.content:
                assert rrn in c.content, f"RRN bisected in chunk: {c.content!r}"

    def test_mobile_number_stays_in_single_chunk(self):
        mobile = "010-1234-5678"
        filler = "This is filler text. " * 10
        doc = filler + mobile + " " + filler
        chunks = chunk_document("doc", doc, self._cfg(max_chars=200, overlap_chars=0))
        for c in chunks:
            if "010" in c.content:
                assert mobile in c.content, f"Mobile number bisected in chunk: {c.content!r}"

    def test_overlap_does_not_start_mid_rrn(self):
        rrn = "900101-1234567"
        # Position RRN near the end of a chunk so overlap might clip it.
        filler = "a" * 150
        doc = filler + rrn + "b" * 10  # total ~173 chars; will chunk around 200
        chunks = chunk_document("doc", doc, self._cfg(max_chars=160, overlap_chars=20))
        if len(chunks) >= 2:
            second = chunks[1].content
            if rrn[:6] in second:
                assert rrn in second, (
                    f"Overlap started mid-RRN: {second[:30]!r}"
                )

    def test_no_pii_patterns_behaves_as_before(self):
        doc = "This is a long document. " * 60
        chunks_with = chunk_document("d", doc, ChunkerConfig(pii_patterns=[self.RRN]))
        chunks_without = chunk_document("d", doc, ChunkerConfig(pii_patterns=[]))
        # No PII in document → same chunk count and content
        assert len(chunks_with) == len(chunks_without)
        for a, b in zip(chunks_with, chunks_without):
            assert a.content == b.content

    def test_explicit_patterns_override_env_default(self, monkeypatch):
        monkeypatch.setenv("BASTION_PATTERN_DIR", "/nonexistent")
        custom_pat = re.compile(r"CUSTOM_\d{4}")
        cfg = ChunkerConfig(pii_patterns=[custom_pat])
        # Verify the custom pattern is what gets used (env dir is invalid → no patterns there)
        assert cfg.pii_patterns == [custom_pat]


# ─── default pattern cache ────────────────────────────────────────────────────

class TestDefaultPiiPatterns:
    def test_returns_list(self):
        result = _default_pii_patterns()
        assert isinstance(result, list)

    def test_env_var_empty_returns_empty(self, monkeypatch):
        # Patch the cache to force re-evaluation
        import navigator.chunker as chunker_mod
        original = chunker_mod._DEFAULT_PII_PATTERNS
        chunker_mod._DEFAULT_PII_PATTERNS = None
        monkeypatch.setenv("BASTION_PATTERN_DIR", "")
        try:
            result = _default_pii_patterns()
            assert result == []
        finally:
            chunker_mod._DEFAULT_PII_PATTERNS = original

    def test_env_var_with_valid_dir_loads_patterns(self, monkeypatch, tmp_path):
        (tmp_path / "test.yml").write_text(textwrap.dedent("""
            namespace: test
            patterns:
              - id: t_01
                pattern: 'TEST\\d+'
                langs:
                  python: 'TEST\\d+'
        """))
        import navigator.chunker as chunker_mod
        original = chunker_mod._DEFAULT_PII_PATTERNS
        chunker_mod._DEFAULT_PII_PATTERNS = None
        monkeypatch.setenv("BASTION_PATTERN_DIR", str(tmp_path))
        try:
            result = _default_pii_patterns()
            assert len(result) == 1
            assert result[0].search("TEST123")
        finally:
            chunker_mod._DEFAULT_PII_PATTERNS = original
