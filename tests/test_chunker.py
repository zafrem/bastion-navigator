"""Unit tests for navigator.chunker."""
from __future__ import annotations

import pytest
from navigator.chunker import Chunk, ChunkerConfig, chunk_document


# ─── helpers ──────────────────────────────────────────────────────────────────

def _chunk(content: str, **cfg_kwargs) -> list[Chunk]:
    cfg = ChunkerConfig(**cfg_kwargs) if cfg_kwargs else None
    return chunk_document("test-doc", content, cfg)


# ─── basic structure ──────────────────────────────────────────────────────────

class TestEmptyDocument:
    def test_empty_string(self):
        assert chunk_document("d", "") == []

    def test_whitespace_only(self):
        assert chunk_document("d", "   \n\n   ") == []


class TestSingleHeadingSection:
    DOC = "# Warranty Terms\n\nThis product is warranted for 24 months."

    def test_produces_one_chunk(self):
        chunks = _chunk(self.DOC)
        assert len(chunks) == 1

    def test_heading_path_set(self):
        chunks = _chunk(self.DOC)
        assert chunks[0].heading_path == ["# Warranty Terms"]

    def test_content_does_not_include_heading_line(self):
        chunks = _chunk(self.DOC)
        assert "# Warranty Terms" not in chunks[0].content
        assert "warranted for 24 months" in chunks[0].content

    def test_embed_text_has_breadcrumb(self):
        chunks = _chunk(self.DOC)
        et = chunks[0].embed_text()
        assert et.startswith("# Warranty Terms\n\n")
        assert "warranted for 24 months" in et


class TestHeadingHierarchy:
    DOC = (
        "# Policy\n\n"
        "Intro paragraph.\n\n"
        "## Section A\n\n"
        "Content A.\n\n"
        "### Subsection A1\n\n"
        "Detail A1."
    )

    def test_produces_three_chunks(self):
        chunks = _chunk(self.DOC)
        assert len(chunks) == 3

    def test_first_chunk_has_h1_only(self):
        chunks = _chunk(self.DOC)
        assert chunks[0].heading_path == ["# Policy"]

    def test_second_chunk_has_h1_and_h2(self):
        chunks = _chunk(self.DOC)
        assert chunks[1].heading_path == ["# Policy", "## Section A"]

    def test_third_chunk_has_full_path(self):
        chunks = _chunk(self.DOC)
        assert chunks[2].heading_path == ["# Policy", "## Section A", "### Subsection A1"]

    def test_sibling_heading_resets_path_correctly(self):
        doc = "# Root\n\n## A\n\nText A.\n\n## B\n\nText B."
        chunks = _chunk(doc)
        b_chunk = next(c for c in chunks if "Text B" in c.content)
        assert b_chunk.heading_path == ["# Root", "## B"]


class TestSentenceSplitting:
    """Verify NLTK Punkt is used instead of the old regex splitter."""

    def test_abbreviation_not_split(self):
        # Old regex split on "Dr." — NLTK should not
        doc = ("Dr. Smith reviewed the defect report. "
               "The 0.42% rate was acceptable. " * 30)
        chunks = _chunk(doc, max_chars=300)
        for c in chunks:
            assert not c.content.startswith("Smith"), (
                "'Dr.' was incorrectly treated as a sentence boundary"
            )

    def test_korean_sentence_split(self):
        # NLTK splits Korean sentences correctly; old regex missed them entirely
        sentence = "박민준 고객의 계좌를 조회해줘. 이름을 확인하세요. "
        doc = sentence * 30
        chunks = _chunk(doc, max_chars=300)
        assert len(chunks) >= 2

    def test_fallback_on_missing_punkt(self):
        # If punkt data is unavailable, _split_oversized falls back to hard cut.
        # Simulate by patching sent_tokenize to raise LookupError.
        from unittest.mock import patch
        from navigator.chunker import _split_oversized
        with patch("nltk.tokenize.sent_tokenize", side_effect=LookupError):
            sentence = "This is a sentence. " * 40
            parts = _split_oversized(sentence, max_chars=200, overlap_chars=40)
            assert len(parts) >= 2
            assert all(len(p) <= 200 for p in parts)


class TestTableIsolation:
    DOC = (
        "# Coverage\n\n"
        "Intro text.\n\n"
        "| Item | Covered |\n"
        "|---|---|\n"
        "| Defects | Yes |\n\n"
        "Footer text."
    )

    def test_table_is_its_own_chunk(self):
        chunks = _chunk(self.DOC)
        table_chunks = [c for c in chunks if c.contains_table]
        assert len(table_chunks) == 1

    def test_table_content_is_complete(self):
        chunks = _chunk(self.DOC)
        tbl = next(c for c in chunks if c.contains_table)
        assert "| Item | Covered |" in tbl.content
        assert "| Defects | Yes |" in tbl.content

    def test_table_inherits_heading_path(self):
        chunks = _chunk(self.DOC)
        tbl = next(c for c in chunks if c.contains_table)
        assert tbl.heading_path == ["# Coverage"]

    def test_table_rows_not_split_across_chunks(self):
        # Every line of the table must appear in the same chunk
        chunks = _chunk(self.DOC)
        for chunk in chunks:
            if "| Item" in chunk.content:
                assert "| Defects" in chunk.content
                break
        else:
            pytest.fail("Table header row not found in any chunk")


class TestCodeFenceIsolation:
    DOC = (
        "# Example\n\n"
        "Explanation.\n\n"
        "```python\n"
        "x = 1\n"
        "y = 2\n"
        "```\n\n"
        "More text."
    )

    def test_code_fence_is_not_split(self):
        chunks = _chunk(self.DOC)
        for chunk in chunks:
            if "```python" in chunk.content:
                assert "x = 1" in chunk.content
                assert "y = 2" in chunk.content
                return
        pytest.fail("Code fence content not found in any chunk")


# ─── size and overlap ─────────────────────────────────────────────────────────

class TestOversizedParagraphSplit:
    def test_long_paragraph_produces_multiple_chunks(self):
        sentence = "This is a sentence about nothing in particular. "
        doc = sentence * 30  # ~1440 chars, exceeds default 1200
        chunks = _chunk(doc)
        assert len(chunks) >= 2

    def test_overlap_content_appears_in_next_chunk(self):
        sentence = "Each sentence is unique number {i}. "
        doc = "".join(sentence.format(i=i) for i in range(40))
        chunks = _chunk(doc, max_chars=300, overlap_chars=60)
        if len(chunks) >= 2:
            # The tail of chunk[0] should partially appear at the start of chunk[1]
            tail = chunks[0].content[-60:]
            # At least some characters should overlap (check a 20-char window)
            assert any(
                tail[j : j + 20] in chunks[1].content
                for j in range(0, len(tail) - 20, 5)
            ), "No overlap found between consecutive chunks"

    def test_chunk_ids_are_sequential(self):
        sentence = "Sentence number {i} in the document. "
        doc = "".join(sentence.format(i=i) for i in range(50))
        chunks = _chunk(doc, max_chars=200)
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"test-doc_{i:04d}"
            assert c.chunk_index == i


# ─── links ────────────────────────────────────────────────────────────────────

class TestLinkDetection:
    def test_markdown_link_flagged(self):
        doc = "See [our policy](https://example.com/policy) for details."
        chunks = _chunk(doc)
        assert chunks[0].contains_link is True

    def test_plain_text_no_link(self):
        doc = "Contact support at support@example.com."
        chunks = _chunk(doc)
        assert chunks[0].contains_link is False

    def test_link_text_preserved_in_content(self):
        doc = "Read [the guide](https://example.com) here."
        chunks = _chunk(doc)
        assert "[the guide](https://example.com)" in chunks[0].content


# ─── Korean text ─────────────────────────────────────────────────────────────

class TestKoreanText:
    def test_korean_paragraph_produces_chunk(self):
        doc = "# 보증 정책\n\n이 제품은 구매일로부터 24개월 동안 보증됩니다."
        chunks = _chunk(doc)
        assert len(chunks) == 1
        assert "보증됩니다" in chunks[0].content

    def test_korean_heading_in_path(self):
        doc = "# 보증 정책\n\n## 적용 범위\n\n부품 결함에 한함."
        chunks = _chunk(doc)
        assert chunks[0].heading_path == ["# 보증 정책", "## 적용 범위"]

    def test_mixed_korean_english(self):
        doc = "# HR Policy\n\n홍길동 employee ID E003, annual leave 15 days.\n\n## Details\n\n See table below."
        chunks = _chunk(doc)
        assert any("홍길동" in c.content for c in chunks)

    def test_korean_oversized_splits(self):
        sentence = "이것은 아무 의미 없는 한국어 문장입니다. "
        doc = sentence * 40
        chunks = _chunk(doc, max_chars=300)
        assert len(chunks) >= 2


# ─── chunk metadata ───────────────────────────────────────────────────────────

class TestChunkMetadata:
    def test_parent_document_id(self):
        chunks = chunk_document("my-doc-id", "Some content here.")
        assert all(c.parent_document_id == "my-doc-id" for c in chunks)

    def test_metadata_passed_through(self):
        chunks = chunk_document("d", "Content.", metadata={"tenant_id": "acme", "lang": "en"})
        assert chunks[0].metadata["tenant_id"] == "acme"
        assert chunks[0].metadata["lang"] == "en"

    def test_stable_uuid_is_deterministic(self):
        doc = "# H\n\nContent."
        chunks1 = chunk_document("doc-abc", doc)
        chunks2 = chunk_document("doc-abc", doc)
        assert chunks1[0].stable_uuid() == chunks2[0].stable_uuid()

    def test_stable_uuid_differs_between_chunks(self):
        doc = "# A\n\nParagraph one.\n\n# B\n\nParagraph two."
        chunks = chunk_document("doc-x", doc)
        uuids = [c.stable_uuid() for c in chunks]
        assert len(set(uuids)) == len(uuids)

    def test_embed_text_no_heading_returns_content(self):
        doc = "Just plain text, no headings."
        chunks = _chunk(doc)
        assert chunks[0].embed_text() == chunks[0].content


# ─── edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_heading_only_document(self):
        # Headings with no body produce no text chunks
        chunks = _chunk("# H1\n\n## H2\n\n### H3")
        assert chunks == []

    def test_multiple_tables_each_isolated(self):
        doc = (
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            "Middle text.\n\n"
            "| C | D |\n|---|---|\n| 3 | 4 |"
        )
        chunks = _chunk(doc)
        table_chunks = [c for c in chunks if c.contains_table]
        assert len(table_chunks) == 2

    def test_min_chars_merges_tail(self):
        # A document ending with a tiny chunk should merge it into the previous one
        doc = (
            "A longer paragraph with enough content to be its own chunk. " * 5 + "\n\n"
            "Tiny."
        )
        chunks = _chunk(doc, max_chars=200, min_chars=80)
        # "Tiny." (5 chars) should have been merged into the last real chunk
        assert not any(c.content.strip() == "Tiny." for c in chunks)
