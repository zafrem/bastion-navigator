"""Unit tests for navigator.token_rewriter (MR-02-001)."""
from __future__ import annotations

import pytest
from pathlib import Path
from navigator.token_rewriter import TokenRewriter, _RE_TOKEN

_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "sentinel" / "external" / "pii-pattern-engine" / "datas"
)


@pytest.fixture(scope="module")
def rw() -> TokenRewriter:
    return TokenRewriter(data_dir=_DATA_DIR)


class TestKoreanName:
    def test_kr_name_produces_english_name(self, rw: TokenRewriter):
        result = rw.rewrite_text("KR_NAME_4d9e1b")
        # Must contain the hex suffix so Vault Phase 2 can reverse it
        assert "_4d9e1b" in result
        # Must not be the original token
        assert result != "KR_NAME_4d9e1b"

    def test_kr_name_no_hangul_in_output(self, rw: TokenRewriter):
        result = rw._replace("KR_NAME", "4d9e1b")
        # Cross-language mapping: output must be ASCII (English name)
        assert result.isascii() or "_4d9e1b" in result
        for char in result.replace(f"_{{}}", "").replace("_4d9e1b", ""):
            assert ord(char) < 128 or char == " ", f"non-ASCII char in KR_NAME output: {char!r}"

    def test_kr_name_deterministic(self, rw: TokenRewriter):
        a = rw._replace("KR_NAME", "4d9e1b")
        b = rw._replace("KR_NAME", "4d9e1b")
        assert a == b

    def test_different_hex_produce_different_names(self, rw: TokenRewriter):
        a = rw._replace("KR_NAME", "000000")
        b = rw._replace("KR_NAME", "ffffff")
        assert a != b


class TestEmail:
    def test_email_produces_example_com(self, rw: TokenRewriter):
        result = rw._replace("EMAIL", "a1b2c3")
        assert "@example.com_a1b2c3" in result

    def test_email_has_hex_suffix(self, rw: TokenRewriter):
        result = rw._replace("EMAIL", "a1b2c3")
        assert result.endswith("_a1b2c3")

    def test_email_deterministic(self, rw: TokenRewriter):
        assert rw._replace("EMAIL", "a1b2c3") == rw._replace("EMAIL", "a1b2c3")


class TestMobile:
    def test_mobile_uses_unissued_prefix(self, rw: TokenRewriter):
        result = rw._replace("MOBILE", "000001")
        assert result.startswith("010-0000-")

    def test_mobile_has_hex_suffix(self, rw: TokenRewriter):
        result = rw._replace("MOBILE", "000001")
        assert result.endswith("_000001")

    def test_mobile_4digit_body(self, rw: TokenRewriter):
        result = rw._replace("MOBILE", "000001")
        # Format: 010-0000-DDDD_hex
        parts = result.split("_")
        body = parts[0]  # 010-0000-DDDD
        last_group = body.split("-")[-1]
        assert len(last_group) == 4 and last_group.isdigit()

    def test_mobile_deterministic(self, rw: TokenRewriter):
        assert rw._replace("MOBILE", "abc123") == rw._replace("MOBILE", "abc123")


class TestGenericLabels:
    def test_email_honey_returns_class_label(self, rw: TokenRewriter):
        assert rw._replace("EMAIL_honey", "4d9e1b") == "[EMAIL]"

    def test_rrn_token_returns_id_number(self, rw: TokenRewriter):
        assert rw._replace("RRN_TOKEN", "4d9e1b") == "[ID_NUMBER]"

    def test_emp_returns_employee_id(self, rw: TokenRewriter):
        assert rw._replace("EMP", "4d9e1b") == "[EMPLOYEE_ID]"

    def test_wrk_returns_worker_id(self, rw: TokenRewriter):
        assert rw._replace("WRK", "4d9e1b") == "[WORKER_ID]"


class TestRewriteText:
    def test_token_in_prose_is_replaced(self, rw: TokenRewriter):
        text = "Customer KR_NAME_4d9e1b called about their account."
        result = rw.rewrite_text(text)
        assert "KR_NAME_4d9e1b" not in result
        assert "_4d9e1b" in result

    def test_multiple_tokens_in_text(self, rw: TokenRewriter):
        text = "Name: KR_NAME_4d9e1b, Email: EMAIL_a1b2c3, Mobile: MOBILE_ff0011"
        result = rw.rewrite_text(text)
        assert "KR_NAME_4d9e1b" not in result
        assert "EMAIL_a1b2c3" not in result
        assert "MOBILE_ff0011" not in result

    def test_honey_email_in_text_becomes_label(self, rw: TokenRewriter):
        text = "Contact EMAIL_honey_4d9e1b for details."
        result = rw.rewrite_text(text)
        assert "[EMAIL]" in result
        assert "EMAIL_honey_4d9e1b" not in result

    def test_regular_email_not_confused_with_honey(self, rw: TokenRewriter):
        text = "Email: EMAIL_4d9e1b"
        result = rw.rewrite_text(text)
        assert "@example.com" in result

    def test_non_token_text_unchanged(self, rw: TokenRewriter):
        text = "No tokens here. Just normal text."
        assert rw.rewrite_text(text) == text

    def test_rrn_in_text_becomes_id_number(self, rw: TokenRewriter):
        text = "RRN: RRN_TOKEN_deadbe"
        result = rw.rewrite_text(text)
        assert "[ID_NUMBER]" in result

    def test_rewrite_is_deterministic_across_calls(self, rw: TokenRewriter):
        text = "KR_NAME_4d9e1b EMAIL_a1b2c3"
        assert rw.rewrite_text(text) == rw.rewrite_text(text)


class TestMissingDataDir:
    def test_graceful_fallback_when_csv_missing(self, tmp_path):
        # Empty data dir: pick() returns hex_sfx instead of a name
        rw = TokenRewriter(data_dir=tmp_path)
        result = rw._replace("KR_NAME", "4d9e1b")
        # Should still include hex suffix and not raise
        assert "4d9e1b" in result

    def test_mobile_still_works_without_csv(self, tmp_path):
        rw = TokenRewriter(data_dir=tmp_path)
        result = rw._replace("MOBILE", "000001")
        assert result.startswith("010-0000-")
