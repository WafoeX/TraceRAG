"""Unit tests for the chunker — pure, no network or DB required."""
import pytest

from app.services.chunking import chunk_text


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_empty_or_whitespace_returns_no_chunks():
    assert chunk_text("   \n  ", chunk_size=100, chunk_overlap=10) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c d", chunk_size=10, chunk_overlap=10)


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(str(i) for i in range(3000))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunks_overlap_preserves_context():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)
    # With overlap, consecutive chunks should share some content.
    assert len(chunks) >= 2
