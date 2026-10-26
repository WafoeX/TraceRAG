"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512, examples=["Company Handbook"])
    content: str = Field(..., min_length=1, max_length=10 * 1024 * 1024)
    source: str | None = Field(None, max_length=512, examples=["handbook.pdf"])


class IngestResponse(BaseModel):
    document_id: int
    title: str
    chunks_created: int


class DocumentSummary(BaseModel):
    id: int
    title: str
    source: str | None
    chunks: int


class DocumentCatalogSummary(DocumentSummary):
    created_at: datetime
    logical_id: int | None = None
    version_number: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=4000, examples=["What is the refund policy?"]
    )
    top_k: int | None = Field(None, ge=1, le=20)
    retain_trace_content: bool = Field(
        False,
        strict=True,
        description=(
            "Explicit consent for this request only; "
            "question, answer and contexts retained up to 7 days"
        ),
    )


class Citation(BaseModel):
    document_id: int
    document_title: str
    chunk_index: int
    score: float
    snippet: str


class ContextCitation(Citation):
    context_number: int
    content: str
    evidence_url: str
    logical_id: int | None = None
    version_number: int | None = None
    page_number: int | None = None
    original_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class QueryEvidenceResponse(BaseModel):
    answer: str
    citations: list[ContextCitation]
    invalid_citation_numbers: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generation_status: Literal["completed", "failed", "skipped"] = "completed"


class JobSummary(BaseModel):
    id: int
    title: str
    state: str
    attempts: int
    error_code: str | None
    document_id: int | None
    created_at: datetime
    target_logical_id: int | None = None
    base_document_id: int | None = None


class ChunkEvidence(BaseModel):
    document_id: int
    document_title: str
    chunk_index: int
    content: str
    logical_id: int | None = None
    version_number: int | None = None
    page_number: int | None = None
    original_url: str | None = None


class VersionSummary(BaseModel):
    document_id: int
    title: str
    logical_id: int | None
    version_number: int
    active: bool
    created_at: datetime
    file_hash: str | None
    original_available: bool
