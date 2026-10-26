"""Retrieval orchestration: hybrid retrieval → re-rank → top_k.

This is the single entry point the API and scripts use. The pipeline is
configurable via settings (use_hybrid, use_rerank) so you can A/B the stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.reranking import rerank
from app.services.retrieval import (
    RetrievedChunk,
    keyword_search,
    reciprocal_rank_fusion,
    vector_search,
)
from app.services.tracing import stage


@dataclass
class SearchResult:
    """Final chunks plus a retrieval-confidence signal.

    `confidence` is the top vector cosine similarity (0..1) — an interpretable,
    pre-fusion measure of how well the best chunk matches the question. The API
    uses it to decline off-topic questions instead of letting the LLM guess.
    """

    chunks: list[RetrievedChunk]
    confidence: float


def search(db: Session, question: str, top_k: int) -> SearchResult:
    settings = get_settings()
    n = max(settings.retrieval_candidates, top_k)

    # Vector search runs in every mode, and its top cosine similarity is the
    # cleanest relevance signal — capture it before fusion mangles the scale.
    with stage("dense_retrieval"):
        vector = vector_search(db, question, n)
    confidence = vector[0].score if vector else 0.0

    if settings.use_hybrid:
        with stage("keyword_retrieval"):
            keyword = keyword_search(db, question, n)
        with stage("fusion"):
            candidates = reciprocal_rank_fusion([vector, keyword])[:n]
    else:
        candidates = vector

    if settings.use_rerank and candidates:
        with stage("reranking"):
            chunks = rerank(question, candidates, top_k)
    else:
        chunks = candidates[:top_k]

    return SearchResult(chunks=chunks, confidence=confidence)
