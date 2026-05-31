"""Semantic chunker: splits Markdown documents at structural boundaries.

Preserves heading hierarchy (H1→H2→H3 breadcrumb), keeps tables atomic,
keeps code fences atomic, and carries overlap between paragraph chunks so
context is not lost at boundaries.

PII split guard (FR-MR-12): loads regex patterns from the pii-pattern-engine
YAML directory (env var BASTION_PATTERN_DIR) and ensures character-level cuts
never bisect a PII span such as an RRN, mobile number, or email address.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ─── public types ─────────────────────────────────────────────────────────────

@dataclass
class ChunkerConfig:
    max_chars: int = 1200       # ~300 tokens at 4 chars/token
    overlap_chars: int = 120    # tail of previous chunk prepended to next
    min_chars: int = 80         # chunks smaller than this are merged into previous
    pii_patterns: list = field(default_factory=list)  # compiled re.Pattern objects for PII guard


@dataclass
class Chunk:
    chunk_id: str                        # "{parent_id}_{index:04d}"
    parent_document_id: str
    chunk_index: int
    content: str                         # raw text (tables intact, links intact)
    heading_path: list[str] = field(default_factory=list)
    contains_table: bool = False
    contains_link: bool = False
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """Returns text sent to the embedder: heading breadcrumb + content."""
        if self.heading_path:
            breadcrumb = " > ".join(self.heading_path)
            return f"{breadcrumb}\n\n{self.content}"
        return self.content

    def stable_uuid(self) -> str:
        """Deterministic UUID for Qdrant point ID, derived from chunk_id."""
        return str(uuid.uuid5(_UUID_NS, self.chunk_id))


_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


# ─── internal block type ──────────────────────────────────────────────────────

@dataclass
class _Block:
    text: str
    is_table: bool = False
    is_heading: bool = False
    heading_level: int = 0
    char_start: int = 0


# ─── regex ────────────────────────────────────────────────────────────────────

_RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)")
_RE_TABLE_ROW = re.compile(r"^\|")
_RE_FENCE = re.compile(r"^```")
_RE_LINK = re.compile(r"\[[^\]]*\]\([^)]+\)")


# ─── PII pattern loader ───────────────────────────────────────────────────────

def load_pii_patterns(pattern_dir: str) -> list:
    """Load compiled Python regex patterns from a pii-pattern-engine YAML directory.

    Walks pattern_dir recursively. Each YAML file may contain a ``patterns``
    list; each entry is compiled using ``langs.python`` (preferred) or the
    top-level ``pattern`` field. Entries with uncompilable patterns are silently
    skipped so a single bad YAML file does not abort indexing.

    Returns an empty list when pattern_dir is empty or does not exist.
    """
    patterns: list = []
    if not pattern_dir or not os.path.isdir(pattern_dir):
        return patterns
    try:
        import yaml  # type: ignore
    except ImportError:
        return patterns
    for root, _, files in os.walk(pattern_dir):
        for fname in sorted(files):
            if not fname.endswith(".yml"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                for entry in data.get("patterns", []):
                    raw = (
                        entry.get("langs", {}).get("python")
                        or entry.get("pattern", "")
                    )
                    if not raw:
                        continue
                    flags = 0
                    for flag_name in entry.get("flags", []):
                        if flag_name == "IGNORECASE":
                            flags |= re.IGNORECASE
                    try:
                        patterns.append(re.compile(raw, flags))
                    except re.error:
                        pass
            except Exception:
                pass
    return patterns


# Module-level cache: populated on first call; avoids re-walking the directory
# on every chunk_document() call while keeping the loader side-effect-free at
# import time.
_DEFAULT_PII_PATTERNS: Optional[list] = None


def _default_pii_patterns() -> list:
    """Return patterns loaded from BASTION_PATTERN_DIR (cached after first call)."""
    global _DEFAULT_PII_PATTERNS
    if _DEFAULT_PII_PATTERNS is None:
        pdir = os.environ.get("BASTION_PATTERN_DIR", "")
        _DEFAULT_PII_PATTERNS = load_pii_patterns(pdir)
    return _DEFAULT_PII_PATTERNS


# ─── PII guard helpers ────────────────────────────────────────────────────────

def _pii_spans(text: str, patterns: list) -> list:
    """Return all PII match spans in text as a sorted list of (start, end) tuples."""
    spans = []
    for pat in patterns:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return sorted(spans)


def _safe_cut(pos: int, spans: list) -> int:
    """Return the earliest position >= pos that does not bisect a PII span.

    If ``pos`` falls strictly inside a span ``(start, end)`` — i.e.
    ``start < pos < end`` — the cut is advanced to ``end``. Multiple nested or
    adjacent spans are handled by iterating in sorted order.
    """
    adjusted = pos
    for start, end in spans:
        if start >= adjusted:
            break  # spans are sorted; nothing further can overlap adjusted
        if adjusted < end:
            adjusted = end  # advance past this span
    return adjusted


# ─── parser ───────────────────────────────────────────────────────────────────

def _parse_blocks(text: str) -> list[_Block]:
    """Split document text into typed blocks (headings, tables, code, paragraphs)."""
    lines = text.splitlines()
    blocks: list[_Block] = []
    i = 0
    char_pos = 0

    while i < len(lines):
        line = lines[i]
        line_start = char_pos
        char_pos += len(line) + 1

        # Heading
        m = _RE_HEADING.match(line)
        if m:
            blocks.append(_Block(
                text=line,
                is_heading=True,
                heading_level=len(m.group(1)),
                char_start=line_start,
            ))
            i += 1
            continue

        # Code fence — collect until closing ```
        if _RE_FENCE.match(line):
            fence_start = line_start
            collected = [line]
            i += 1
            while i < len(lines) and not _RE_FENCE.match(lines[i]):
                char_pos += len(lines[i]) + 1
                collected.append(lines[i])
                i += 1
            if i < len(lines):
                char_pos += len(lines[i]) + 1
                collected.append(lines[i])
                i += 1
            blocks.append(_Block(text="\n".join(collected), char_start=fence_start))
            continue

        # Table — collect consecutive | rows
        if _RE_TABLE_ROW.match(line):
            table_start = line_start
            collected = [line]
            i += 1
            while i < len(lines) and _RE_TABLE_ROW.match(lines[i]):
                char_pos += len(lines[i]) + 1
                collected.append(lines[i])
                i += 1
            blocks.append(_Block(
                text="\n".join(collected),
                is_table=True,
                char_start=table_start,
            ))
            continue

        # Blank line — skip
        if not line.strip():
            i += 1
            continue

        # Paragraph — accumulate until blank / heading / table / fence
        para_start = line_start
        collected = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (
                not nxt.strip()
                or _RE_HEADING.match(nxt)
                or _RE_TABLE_ROW.match(nxt)
                or _RE_FENCE.match(nxt)
            ):
                break
            char_pos += len(nxt) + 1
            collected.append(nxt)
            i += 1
        blocks.append(_Block(text="\n".join(collected), char_start=para_start))

    return blocks


def _split_oversized(
    text: str,
    max_chars: int,
    overlap_chars: int,
    pii_patterns: Optional[list] = None,
) -> list[str]:
    """Split a block exceeding max_chars at sentence boundaries using NLTK Punkt.

    Falls back to hard-cut with overlap if the punkt data is not available.
    Punkt handles abbreviations (Dr., U.S.A., Fig.) and Korean sentence endings
    that the previous regex pattern could not.

    When pii_patterns is provided the character-level fallback advances each
    cut point past any PII span it would bisect, keeping values like RRNs and
    mobile numbers intact within a single part.
    """
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = None

    if sentences:
        return _pack_sentences(sentences, max_chars, overlap_chars, text)

    # Fallback: hard cut with PII guard
    active = pii_patterns or []
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = max_chars
        if active:
            spans = _pii_spans(remaining, active)
            cut = _safe_cut(max_chars, spans)
            if cut >= len(remaining):
                # PII span covers the rest of the text; keep it whole.
                break
        parts.append(remaining[:cut].strip())
        remaining = remaining[max(0, cut - overlap_chars):].lstrip()
    if remaining.strip():
        parts.append(remaining.strip())
    return parts or [text]


def _pack_sentences(
    sentences: list[str],
    max_chars: int,
    overlap_chars: int,
    original: str,
) -> list[str]:
    """Pack sentences into parts up to max_chars, carrying overlap between parts."""
    parts: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = (buf + " " + sent).lstrip() if buf else sent
        if buf and len(candidate) > max_chars:
            parts.append(buf.strip())
            overlap = buf[-overlap_chars:].lstrip() if overlap_chars else ""
            buf = (overlap + " " + sent).lstrip() if overlap else sent
        else:
            buf = candidate
    if buf.strip():
        parts.append(buf.strip())
    return parts or [original]


# ─── public API ───────────────────────────────────────────────────────────────

def chunk_document(
    document_id: str,
    content: str,
    cfg: Optional[ChunkerConfig] = None,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """Chunk a document, returning a list of Chunk objects in order.

    Each chunk carries:
    - heading_path: the ancestor heading hierarchy at the chunk's position
    - contains_table / contains_link: structural flags for downstream routing
    - embed_text(): breadcrumb-prefixed text ready for the embedding model
    """
    if not content or not content.strip():
        return []

    cfg = cfg or ChunkerConfig()
    metadata = metadata or {}
    blocks = _parse_blocks(content)

    # Resolve active PII patterns: caller-supplied takes priority over the
    # module-level cache from BASTION_PATTERN_DIR.
    active_patterns: list = cfg.pii_patterns if cfg.pii_patterns else _default_pii_patterns()

    heading_path: list[str] = []
    chunks: list[Chunk] = []
    buf_pieces: list[str] = []
    buf_start = 0
    buf_is_table = False
    overlap = ""

    def _flush(end: int) -> None:
        nonlocal buf_pieces, buf_start, buf_is_table, overlap
        text = "\n\n".join(buf_pieces).strip()
        if not text:
            return
        idx = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{document_id}_{idx:04d}",
            parent_document_id=document_id,
            chunk_index=idx,
            content=text,
            heading_path=list(heading_path),
            contains_table=buf_is_table,
            contains_link=bool(_RE_LINK.search(text)),
            char_start=buf_start,
            char_end=end,
            metadata=dict(metadata),
        ))
        # PII guard on overlap: don't let the overlap window start mid-PII span.
        raw_overlap = text[-cfg.overlap_chars:] if cfg.overlap_chars else ""
        if raw_overlap and active_patterns:
            overlap_start = len(text) - len(raw_overlap)
            safe_start = _safe_cut(overlap_start, _pii_spans(text, active_patterns))
            overlap = text[safe_start:] if safe_start < len(text) else ""
        else:
            overlap = raw_overlap
        buf_pieces = []
        buf_is_table = False

    def _append_text_block(block: _Block) -> None:
        nonlocal buf_pieces, buf_start, overlap

        # A single block may itself exceed max_chars — split it first.
        if len(block.text) > cfg.max_chars:
            sub_texts = _split_oversized(
                block.text, cfg.max_chars, cfg.overlap_chars, active_patterns
            )
        else:
            sub_texts = [block.text]

        for sub in sub_texts:
            projected = sum(len(p) + 2 for p in buf_pieces) + len(sub)
            if buf_pieces and projected > cfg.max_chars:
                _flush(block.char_start)
                buf_start = block.char_start
                buf_pieces = [sub]
                if overlap:
                    buf_pieces = [overlap, sub]
                    overlap = ""
            else:
                if not buf_pieces:
                    buf_start = block.char_start
                    if overlap:
                        buf_pieces = [overlap, sub]
                        overlap = ""
                    else:
                        buf_pieces = [sub]
                else:
                    buf_pieces.append(sub)

    for block in blocks:
        block_end = block.char_start + len(block.text)

        if block.is_heading:
            _flush(block.char_start)
            level = block.heading_level
            heading_path = heading_path[:level - 1]
            heading_path.append(block.text)
            overlap = ""
            buf_start = block_end
            continue

        if block.is_table:
            _flush(block.char_start)
            text = block.text.strip()
            if text:
                idx = len(chunks)
                chunks.append(Chunk(
                    chunk_id=f"{document_id}_{idx:04d}",
                    parent_document_id=document_id,
                    chunk_index=idx,
                    content=text,
                    heading_path=list(heading_path),
                    contains_table=True,
                    contains_link=bool(_RE_LINK.search(text)),
                    char_start=block.char_start,
                    char_end=block_end,
                    metadata=dict(metadata),
                ))
            overlap = ""
            buf_start = block_end
            continue

        _append_text_block(block)

    _flush(len(content))

    # Merge orphan tail chunk into its predecessor only when they are in the
    # same heading section — never merge across section boundaries.
    if (
        len(chunks) >= 2
        and len(chunks[-1].content) < cfg.min_chars
        and chunks[-1].heading_path == chunks[-2].heading_path
    ):
        tail = chunks.pop()
        prev = chunks[-1]
        merged = prev.content + "\n\n" + tail.content
        chunks[-1] = Chunk(
            chunk_id=prev.chunk_id,
            parent_document_id=prev.parent_document_id,
            chunk_index=prev.chunk_index,
            content=merged,
            heading_path=prev.heading_path,
            contains_table=prev.contains_table or tail.contains_table,
            contains_link=prev.contains_link or tail.contains_link,
            char_start=prev.char_start,
            char_end=tail.char_end,
            metadata=prev.metadata,
        )

    return chunks


# ─── Schema-aware chunking profiles (FR-MR-06-004) ───────────────────────────

PROFILE_MARKDOWN      = "markdown"
PROFILE_PLAIN_TEXT    = "plain_text"
PROFILE_STRUCTURED_CSV = "structured_csv"
PROFILE_JSON_RECORD   = "json_record"
PROFILE_HTML          = "html"

# Profile → (max_chars, overlap_chars). CSV and JSON use dedicated chunkers below.
_PROFILE_CONFIG: dict[str, tuple[int, int]] = {
    PROFILE_MARKDOWN:   (1200, 120),
    PROFILE_PLAIN_TEXT: (800,  80),
    PROFILE_HTML:       (1200, 120),
}

# MIME type → profile name.
_MIME_PROFILE: dict[str, str] = {
    "text/markdown":      PROFILE_MARKDOWN,
    "text/plain":         PROFILE_PLAIN_TEXT,
    "text/csv":           PROFILE_STRUCTURED_CSV,
    "application/json":   PROFILE_JSON_RECORD,
    "text/html":          PROFILE_HTML,
    "text/htm":           PROFILE_HTML,
}

# File extension → profile name (fallback when MIME type is absent).
_EXT_PROFILE: dict[str, str] = {
    ".md":   PROFILE_MARKDOWN,
    ".txt":  PROFILE_PLAIN_TEXT,
    ".csv":  PROFILE_STRUCTURED_CSV,
    ".json": PROFILE_JSON_RECORD,
    ".html": PROFILE_HTML,
    ".htm":  PROFILE_HTML,
}


def profile_for_mime(mime_type: str, filename: str = "") -> str:
    """Return the profile name for a MIME type, falling back to file extension."""
    if mime_type and mime_type in _MIME_PROFILE:
        return _MIME_PROFILE[mime_type]
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in _EXT_PROFILE:
            return _EXT_PROFILE[ext]
    return PROFILE_MARKDOWN


def config_for_profile(profile: str) -> ChunkerConfig:
    """Return a ChunkerConfig pre-set for the given profile.

    CSV and JSON profiles return a sentinel config (max_chars=0) that signals
    the caller should use ``chunk_csv`` / ``chunk_json`` instead.
    """
    if profile in (PROFILE_STRUCTURED_CSV, PROFILE_JSON_RECORD):
        return ChunkerConfig(max_chars=0, overlap_chars=0)
    max_c, overlap_c = _PROFILE_CONFIG.get(profile, (1200, 120))
    return ChunkerConfig(max_chars=max_c, overlap_chars=overlap_c)


def chunk_by_profile(
    document_id: str,
    content: str,
    profile: str,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """Dispatch to the correct chunker for the given profile."""
    metadata = metadata or {}
    if profile == PROFILE_STRUCTURED_CSV:
        return chunk_csv(document_id, content, metadata)
    if profile == PROFILE_JSON_RECORD:
        return chunk_json(document_id, content, metadata)
    if profile == PROFILE_HTML:
        return chunk_html(document_id, content, metadata)
    cfg = config_for_profile(profile)
    return chunk_document(document_id, content, cfg, metadata)


def chunk_csv(document_id: str, content: str, metadata: Optional[dict] = None) -> list[Chunk]:
    """One Chunk per CSV row. The header row is prepended to every data row."""
    import csv, io
    metadata = metadata or {}
    chunks: list[Chunk] = []
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return chunks
    header = rows[0]
    header_text = ",".join(header)
    char_pos = len(content.splitlines()[0]) + 1 if content else 0
    for idx, row in enumerate(rows[1:]):
        row_text = ",".join(row)
        combined = f"{header_text}\n{row_text}"
        end = char_pos + len(row_text)
        chunks.append(Chunk(
            chunk_id=f"{document_id}_{idx:04d}",
            parent_document_id=document_id,
            chunk_index=idx,
            content=combined,
            heading_path=[],
            char_start=char_pos,
            char_end=end,
            metadata=dict(metadata),
        ))
        char_pos = end + 1
    return chunks


def chunk_json(document_id: str, content: str, metadata: Optional[dict] = None) -> list[Chunk]:
    """One Chunk per top-level JSON object (array) or one Chunk for a single object."""
    import json as _json
    metadata = metadata or {}
    chunks: list[Chunk] = []
    try:
        data = _json.loads(content)
    except _json.JSONDecodeError:
        # Fall back to plain-text chunking on invalid JSON.
        return chunk_document(document_id, content, ChunkerConfig(max_chars=800), metadata)

    records = data if isinstance(data, list) else [data]
    for idx, record in enumerate(records):
        text = _json.dumps(record, ensure_ascii=False)
        chunks.append(Chunk(
            chunk_id=f"{document_id}_{idx:04d}",
            parent_document_id=document_id,
            chunk_index=idx,
            content=text,
            heading_path=[],
            metadata=dict(metadata),
        ))
    return chunks


def chunk_html(
    document_id: str,
    content: str,
    metadata: Optional[dict] = None,
    cfg: Optional[ChunkerConfig] = None,
) -> list[Chunk]:
    """Strip HTML tags, then chunk the resulting plain text as markdown."""
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
        def handle_data(self, data: str) -> None:
            self._parts.append(data)
        def get_text(self) -> str:
            return " ".join(self._parts)

    stripper = _Stripper()
    stripper.feed(content)
    plain = stripper.get_text().strip()
    cfg = cfg or ChunkerConfig(max_chars=1200, overlap_chars=120)
    return chunk_document(document_id, plain, cfg, metadata or {})
