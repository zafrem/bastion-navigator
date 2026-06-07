"""Faker token rewriting for MR-02-001.

Vault Phase 1 tokenizes PII into opaque tokens (KR_NAME_4d9e1b, EMAIL_4d9e1b …).
This module rewrites those tokens into semantically meaningful pseudonyms so
BGE-M3 embeddings carry real contextual signal while Vault Phase 2 can still
reverse them by extracting the hex suffix.

Cross-language mapping (Korean names → English pseudonyms) avoids collision
with real Korean users who share the same name as the generated faker value.
"""
from __future__ import annotations

import csv
import re
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "sentinel" / "external" / "pii-pattern-engine" / "datas"
)

# EMAIL_honey must precede EMAIL in the alternation so it wins over the shorter match.
# Overridable via TokenRewriterConfig.token_pattern (token_rewriter.token_pattern
# in config.yaml). The pattern must expose two capture groups: token kind and hex suffix.
DEFAULT_TOKEN_PATTERN = r'\b(KR_NAME|EMAIL_honey|EMAIL|MOBILE|RRN_TOKEN|EMP|WRK)_([0-9a-f]{6})\b'

_RE_TOKEN = re.compile(DEFAULT_TOKEN_PATTERN)


def _load_csv(path: Path, col: str) -> list[str]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [row[col] for row in csv.DictReader(f) if row.get(col)]
    except FileNotFoundError:
        log.warning("[token_rewriter] data file not found: %s", path)
        return []


def _pick(names: list[str], hex_sfx: str, offset: int = 0) -> str:
    if not names:
        return hex_sfx
    return names[(int(hex_sfx, 16) + offset) % len(names)]


class TokenRewriter:
    """Replaces Vault tokens embedded in retrieved text with deterministic faker pseudonyms.

    Determinism guarantee: the same token always maps to the same pseudonym,
    derived from the hex suffix alone, so repeated searches produce stable context.
    """

    def __init__(self, data_dir: Optional[Path] = None, token_pattern: str = "") -> None:
        d = Path(data_dir) if data_dir else _DATA_DIR
        self._en_given = _load_csv(d / "en_given_names.csv", "name")
        self._en_sur   = _load_csv(d / "en_surnames.csv", "surname")
        self._kr_given = _load_csv(d / "kr_given_names.csv", "given_name")
        self._kr_sur   = _load_csv(d / "kr_surnames.csv", "surname")
        # Empty/invalid override falls back to the built-in default pattern.
        self._re_token = _RE_TOKEN
        if token_pattern:
            try:
                self._re_token = re.compile(token_pattern)
            except re.error:
                log.warning("[token_rewriter] invalid token_pattern override; using default")

    def rewrite_text(self, text: str) -> str:
        """Replace all Vault tokens in *text* with faker pseudonyms in-place."""
        return self._re_token.sub(lambda m: self._replace(m.group(1), m.group(2)), text)

    def _replace(self, kind: str, hex_sfx: str) -> str:
        if kind == "KR_NAME":
            first = _pick(self._en_given, hex_sfx)
            last  = _pick(self._en_sur, hex_sfx, offset=len(self._en_given))
            return f"{first} {last}_{hex_sfx}"

        if kind == "EMAIL":
            given = _pick(self._kr_given, hex_sfx)
            sur   = _pick(self._kr_sur, hex_sfx, offset=len(self._kr_given))
            return f"{given}.{sur}@example.com_{hex_sfx}"

        if kind == "MOBILE":
            digits = int(hex_sfx, 16) % 10000
            return f"010-0000-{digits:04d}_{hex_sfx}"

        if kind == "EMAIL_honey":
            return "[EMAIL]"

        if kind == "RRN_TOKEN":
            return "[ID_NUMBER]"

        if kind == "EMP":
            return "[EMPLOYEE_ID]"

        if kind == "WRK":
            return "[WORKER_ID]"

        return f"{kind}_{hex_sfx}"
