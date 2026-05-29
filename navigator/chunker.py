"""Semantic chunker: splits Markdown documents at structural boundaries.

Preserves heading hierarchy (H1→H2→H3 breadcrumb), keeps tables atomic,
keeps code fences atomic, and carries overlap between paragraph chunks so
context is not lost at boundaries.
"""
from __future__ import annotations

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


def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a block exceeding max_chars at sentence boundaries using NLTK Punkt.

    Falls back to hard-cut with overlap if the punkt data is not available.
    Punkt handles abbreviations (Dr., U.S.A., Fig.) and Korean sentence endings
    that the previous regex pattern could not.
    """
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = None

    if sentences:
        return _pack_sentences(sentences, max_chars, overlap_chars, text)

    # Fallback: hard cut with overlap (punkt data not installed)
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        parts.append(remaining[:max_chars].strip())
        remaining = remaining[max(0, max_chars - overlap_chars):].lstrip()
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
        overlap = text[-cfg.overlap_chars:] if cfg.overlap_chars else ""
        buf_pieces = []
        buf_is_table = False

    def _append_text_block(block: _Block) -> None:
        nonlocal buf_pieces, buf_start, overlap

        # A single block may itself exceed max_chars — split it first.
        if len(block.text) > cfg.max_chars:
            sub_texts = _split_oversized(block.text, cfg.max_chars, cfg.overlap_chars)
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
