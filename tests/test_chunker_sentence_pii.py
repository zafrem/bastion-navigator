"""Sentence-aware splitting must stay active when PII patterns are supplied, without bisecting PII."""
import re

from navigator.chunker import ChunkerConfig, chunk_document

RRN = "900101-1234567"


def _text() -> str:
    sentences = [f"Sentence number {i} talks about quarterly results and planning." for i in range(12)]
    sentences.insert(6, f"The applicant registration number is {RRN} on file.")
    return " ".join(sentences)


def _cfg(patterns):
    return ChunkerConfig(max_chars=200, overlap_chars=0, min_chars=20, pii_patterns=patterns)


def test_sentence_boundaries_used_with_pii_patterns():
    chunks = chunk_document("d1", _text(), _cfg([re.compile(r"\d{6}-\d{7}")]))
    assert len(chunks) > 1
    for c in chunks:
        assert c.content.rstrip().endswith((".", "file.")), f"chunk cut mid-sentence: {c.content[-30:]!r}"


def test_pii_span_never_split_across_chunks():
    chunks = chunk_document("d1", _text(), _cfg([re.compile(r"\d{6}-\d{7}")]))
    assert any(RRN in c.content for c in chunks)
    assert not any(("900101-" in c.content) != (RRN in c.content) for c in chunks)
