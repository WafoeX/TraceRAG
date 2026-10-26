"""Token-aware text chunking with overlap.

Splitting on token boundaries keeps every chunk safely within the embedding
model's context window, while the overlap preserves meaning that would
otherwise be lost across a hard cut.
"""
from __future__ import annotations

import os

# Serverless filesystems are read-only except /tmp — point tiktoken's cache there
# so the encoder can download/cache on a cold start (e.g. Vercel).
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/tiktoken-cache")

import tiktoken  # noqa: E402

_encoder = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    tokens = _encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        piece = _encoder.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(tokens):
            break
    return chunks
