"""RAG query endpoints (JSON answer + streaming)."""

from __future__ import annotations

import json
import re
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Chunk, ChunkLocation, DocumentVersion
from app.schemas import (
    Citation,
    ContextCitation,
    QueryEvidenceResponse,
    QueryRequest,
    QueryResponse,
)
from app.services import tracing
from app.services.generation import generate_answer, stream_answer
from app.services.resilience import (
    ProviderUnavailable,
    answer_content,
    failure_code,
    warn,
    warnings,
)
from app.services.retrieval import RetrievedChunk
from app.services.search import SearchResult
from app.services.search import search as run_search

GENERATION_FAILED = "Answer generation is unavailable. Retrieved evidence is available below."


def search(db: Session, question: str, top_k: int) -> SearchResult:
    """Prevent SQL/provider exception parameters from reaching response/server logs."""
    try:
        return run_search(db, question, top_k)
    except ProviderUnavailable:
        raise
    except Exception:
        raise ProviderUnavailable("retrieval_unavailable") from None


def safe_answer(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[str, Literal["completed", "failed"]]:
    try:
        with tracing.stage("generation"):
            return answer_content(generate_answer(question, chunks)), "completed"
    except Exception as exc:
        code = exc.code if isinstance(exc, ProviderUnavailable) else failure_code(exc)
        warn("generation_" + code)
        return GENERATION_FAILED, "failed"


router = APIRouter(prefix="/query", tags=["query"])

# Returned (instead of an LLM-generated guess) when retrieval confidence is low.
NO_ANSWER = (
    "I couldn't find anything about that in the current knowledge base. "
    "Try rephrasing, or add a document that covers it."
)

# The streaming response is "answer text" + this byte + "citations JSON". 0x1F
# (ASCII unit separator) never appears in natural-language output, so the client
# can split on it without escaping.
CITATION_SEPARATOR = "\x1f"


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            score=c.score,
            snippet=c.content[:240].strip(),
        )
        for c in chunks
    ]


def evidence_metadata(db: Session, chunks: list[RetrievedChunk]) -> dict[int, dict]:
    if not chunks:
        return {}
    rows = db.execute(
        select(
            Chunk.id,
            Chunk.document_id,
            DocumentVersion.logical_id,
            DocumentVersion.number,
            ChunkLocation.page_number,
            DocumentVersion.raw_content.is_not(None).label("has_original"),
        )
        .select_from(Chunk)
        .outerjoin(DocumentVersion, DocumentVersion.document_id == Chunk.document_id)
        .outerjoin(ChunkLocation, ChunkLocation.chunk_id == Chunk.id)
        .where(Chunk.id.in_([c.chunk_id for c in chunks]))
    ).all()
    return {
        row.id: {
            "logical_id": row.logical_id,
            "version_number": row.number,
            "page_number": row.page_number,
            "original_url": (
                f"/documents/{row.document_id}/original"
                + (f"#page={row.page_number}" if row.page_number is not None else "")
            )
            if row.has_original
            else None,
        }
        for row in rows
    }


def _context_citations(
    chunks: list[RetrievedChunk],
    metadata: dict[int, dict] | None = None,
) -> list[ContextCitation]:
    return [
        ContextCitation(
            **citation.model_dump(),
            context_number=i,
            content=c.content,
            evidence_url=f"/documents/{c.document_id}/chunks/{c.chunk_index}",
            **(metadata or {}).get(c.chunk_id, {}),
        )
        for i, (citation, c) in enumerate(zip(_citations(chunks), chunks, strict=True), start=1)
    ]


def invalid_citations(answer: str, count: int) -> list[int]:
    """Syntax/range check only; NOT a semantic support or sufficiency judgment."""
    numbers = {int(n) for n in re.findall(r"\[([0-9]+)\]", answer) if len(n) <= 100}
    # Huge numeric tokens are also invalid; bound conversion to avoid Python's
    # integer parsing limit. -1 is a reserved malformed-number warning sentinel.
    if any(len(n) > 100 for n in re.findall(r"\[([0-9]+)\]", answer)):
        numbers.add(-1)
    return sorted(n for n in numbers if n < 1 or n > count)


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_db)):
    """Retrieve relevant context and answer with inline citations."""
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    tracing.begin_query(payload.question, payload.retain_trace_content, top_k)
    with tracing.stage("retrieval"):
        result = search(db, payload.question, top_k)
    accepted = result.confidence >= settings.min_relevance
    tracing.context(result.chunks if accepted else [], generation_attempted=accepted)
    if not accepted:
        tracing.answer(NO_ANSWER, "skipped")
        return QueryResponse(answer=NO_ANSWER, citations=[])
    answer, status = safe_answer(payload.question, result.chunks)
    tracing.answer(answer, status)
    return QueryResponse(answer=answer, citations=_citations(result.chunks))


@router.post("/evidence", response_model=QueryEvidenceResponse)
def query_evidence(payload: QueryRequest, db: Session = Depends(get_db)):
    """Full actual context snapshots with range-only citation audit, not a judge."""
    settings = get_settings()
    tracing.begin_query(
        payload.question, payload.retain_trace_content, payload.top_k or settings.top_k
    )
    with tracing.stage("retrieval"):
        result = search(db, payload.question, payload.top_k or settings.top_k)
    accepted = result.confidence >= settings.min_relevance
    tracing.context(result.chunks if accepted else [], generation_attempted=accepted)
    if not accepted:
        tracing.answer(NO_ANSWER, "skipped")
        return QueryEvidenceResponse(
            answer=NO_ANSWER, citations=[], generation_status="skipped", warnings=warnings()
        )
    with tracing.stage("evidence_metadata"):
        try:
            metadata = evidence_metadata(db, result.chunks)
        except Exception:
            raise ProviderUnavailable("evidence_unavailable") from None
    answer, status = safe_answer(payload.question, result.chunks)
    tracing.answer(answer, status)
    return QueryEvidenceResponse(
        answer=answer,
        citations=_context_citations(result.chunks, metadata),
        invalid_citation_numbers=invalid_citations(answer, len(result.chunks)),
        warnings=warnings(),
        generation_status=status,
    )


@router.post("/stream")
def query_stream(payload: QueryRequest, db: Session = Depends(get_db)):
    """Yield answer text, then one trailing JSON citations frame.

    Wire format: answer text, CITATION_SEPARATOR, then `{"citations": [...]}`.
    Provider adapters may buffer a complete response for usage accounting; this
    route does not guarantee token-by-token delivery or first-token latency.
    """
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    tracing.begin_query(payload.question, payload.retain_trace_content, top_k)
    with tracing.stage("retrieval"):
        result = search(db, payload.question, top_k)
    accepted = result.confidence >= settings.min_relevance
    tracing.context(result.chunks if accepted else [], generation_attempted=accepted)

    def frame(citations: list[Citation]) -> str:
        data = {"citations": [c.model_dump() for c in citations]}
        return CITATION_SEPARATOR + json.dumps(data)

    if result.confidence < settings.min_relevance:

        def gated():
            tracing.answer(NO_ANSWER, "skipped")
            yield NO_ANSWER
            yield frame([])

        return StreamingResponse(gated(), media_type="text/plain; charset=utf-8")

    citations = _citations(result.chunks)

    def generate():
        parts = []
        size = 0
        status = "completed"
        try:
            with tracing.stage("generation"):
                for part in stream_answer(payload.question, result.chunks):
                    if payload.retain_trace_content and size <= tracing.MAX_SNAPSHOT_BYTES:
                        parts.append(part)
                        size += len(part.encode())
                    yield part
        except Exception:
            # Never replay a partially emitted stream; always finish the evidence frame.
            status = "failed"
            parts.append(GENERATION_FAILED)
            yield GENERATION_FAILED
        tracing.answer("".join(parts), status)
        yield frame(citations)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
