"""Synchronous ingestion and a durable PostgreSQL queue with fenced publication.

Model calls run outside transactions and may be repeated after lease expiry.
Only the current unexpired worker may atomically publish a document.
"""

from __future__ import annotations

import hashlib
import io
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from pypdf import PdfReader
from sqlalchemy import and_, func, or_, select, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    Chunk,
    ChunkLocation,
    Document,
    DocumentVersion,
    IngestionJob,
    LogicalDocument,
)
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.resilience import ProviderUnavailable, application_active
from app.services.retrieval import library_id

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_ATTEMPTS = 3
LEASE_SECONDS = 900
ALLOWED_SUFFIXES = (".txt", ".md", ".pdf")


def ingest_document(db: Session, title: str, content: str, source: str | None) -> Document:
    """Historical synchronous API, preserved for scripts and reproducibility."""
    settings = get_settings()
    if application_active() and len(content.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("Text exceeds 10 MiB")
    pieces = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    if not pieces:
        raise ValueError("Document produced no chunks (empty content?)")
    if application_active() and len(pieces) > 512:
        raise ValueError("Document exceeds 512 chunks")
    embeddings = embed_texts(pieces)
    document = Document(title=title, source=source)
    document.chunks = [
        Chunk(chunk_index=i, content=piece, embedding=embedding)
        for i, (piece, embedding) in enumerate(zip(pieces, embeddings, strict=True))
    ]
    db.add(document)
    db.flush()
    logical = LogicalDocument(knowledge_base_id=library_id(db))
    db.add(logical)
    db.flush()
    db.add(
        DocumentVersion(
            document_id=document.id,
            logical_id=logical.id,
            number=1,
            active=True,
        )
    )
    db.commit()
    db.refresh(document)
    return document


class VersionConflict(ValueError):
    """An update is based on an archived version; never silently overwrite."""


def adopt_version(db: Session, document_id: int) -> DocumentVersion:
    """Lazily adopt post-migration legacy synchronous documents; no invented raw file."""
    version = db.get(DocumentVersion, document_id)
    if version is not None:
        return version
    document = db.scalar(select(Document).where(Document.id == document_id).with_for_update())
    if document is None:
        raise LookupError("Document not found")
    version = db.get(DocumentVersion, document_id)
    if version is None:
        logical = LogicalDocument(knowledge_base_id=library_id(db))
        db.add(logical)
        db.flush()
        version = DocumentVersion(
            document_id=document_id, logical_id=logical.id, number=1, active=True
        )
        db.add(version)
        db.flush()
    return version


def enqueue(
    db: Session,
    title: str,
    filename: str,
    data: bytes,
    base_document_id: int | None = None,
) -> IngestionJob:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must contain 1 byte to 10 MiB")
    if not filename.lower().endswith(ALLOWED_SUFFIXES):
        raise ValueError("Only .txt, .md and .pdf are supported")
    if not title.strip() or len(title) > 512 or len(filename) > 512:
        raise ValueError("Title and filename must contain at most 512 characters")
    digest = hashlib.sha256(data).hexdigest()
    logical_id = None
    kb_id = library_id(db)
    prefix = f"kb:{kb_id}:" if kb_id is not None else ""
    dedup_key = prefix + "new:" + digest
    if base_document_id is not None:
        version = adopt_version(db, base_document_id)
        logical_id = version.logical_id
        root = db.scalar(
            select(LogicalDocument).where(LogicalDocument.id == logical_id).with_for_update()
        )
        if root is None or (kb_id is not None and root.knowledge_base_id != kb_id):
            raise LookupError("Document not found")
        active = db.scalar(
            select(DocumentVersion).where(
                DocumentVersion.logical_id == logical_id, DocumentVersion.active.is_(True)
            )
        )
        if active is None or active.document_id != base_document_id:
            raise VersionConflict("Version changed; refresh before updating")
        if active.file_hash == digest and active.raw_content is not None:
            original = db.scalar(
                select(IngestionJob)
                .where(
                    IngestionJob.document_id == base_document_id,
                    IngestionJob.state == "completed",
                )
                .order_by(IngestionJob.id.desc())
                .limit(1)
            )
            if original is not None:
                return original
        dedup_key = prefix + f"update:{logical_id}:{base_document_id}:{digest}"
    existing = db.scalar(select(IngestionJob).where(IngestionJob.dedup_key == dedup_key))
    if existing is not None:
        return existing
    job = IngestionJob(
        file_hash=digest,
        dedup_key=dedup_key,
        knowledge_base_id=kb_id,
        title=title,
        filename=filename,
        payload=data,
        target_logical_id=logical_id,
        base_document_id=base_document_id,
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(select(IngestionJob).where(IngestionJob.dedup_key == dedup_key))
        if existing is None:
            raise
        return existing
    db.refresh(job)
    return job


@dataclass(frozen=True)
class ClaimedJob:
    id: int
    token: str
    title: str
    filename: str
    payload: bytes
    target_logical_id: int | None = None
    base_document_id: int | None = None
    knowledge_base_id: int | None = None


def claim(
    db: Session, lease_seconds: int = LEASE_SECONDS, *, require_owned: bool = False
) -> ClaimedJob | None:
    job = db.scalar(
        select(IngestionJob)
        .where(
            or_(
                IngestionJob.state == "queued",
                and_(
                    IngestionJob.state.in_(["parsing", "embedding"]),
                    IngestionJob.lease_until <= func.now(),
                ),
            )
        )
        .where(IngestionJob.knowledge_base_id.is_not(None) if require_owned else true())
        .order_by(IngestionJob.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if job is None:
        db.rollback()
        return None
    if job.attempts >= MAX_ATTEMPTS:
        job.state, job.error_code = "failed", "lease_attempts_exhausted"
        job.lease_token = None
        job.lease_until = None
        db.commit()
        return None
    token = uuid.uuid4().hex
    job.state, job.error_code = "parsing", None
    job.attempts += 1
    job.lease_token = token
    now = db.execute(select(func.now())).scalar_one()
    job.lease_until = now + timedelta(seconds=lease_seconds)
    result = ClaimedJob(
        job.id,
        token,
        job.title,
        job.filename,
        job.payload or b"",
        job.target_logical_id,
        job.base_document_id,
        job.knowledge_base_id,
    )
    db.commit()
    return result


def _owned(db: Session, item: ClaimedJob) -> IngestionJob | None:
    return db.scalar(
        select(IngestionJob)
        .where(
            IngestionJob.id == item.id,
            IngestionJob.lease_token == item.token,
            IngestionJob.knowledge_base_id == item.knowledge_base_id,
            IngestionJob.state.in_(["parsing", "embedding"]),
            IngestionJob.lease_until > func.now(),
        )
        .with_for_update()
    )


def extract_sections(item: ClaimedJob) -> list[tuple[int | None, str]]:
    if item.filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(item.payload))
        if reader.is_encrypted or len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError("Encrypted or oversized PDF")
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    return [(None, item.payload.decode("utf-8"))]


def extract_payload(item: ClaimedJob) -> str:
    return "\n".join(content for _, content in extract_sections(item))


def process(
    session_factory: Callable[[], Session],
    item: ClaimedJob,
    embed: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> bool:
    """Return True only after publication; stale workers are harmless no-ops."""
    error_code = "invalid_document"
    try:
        settings = get_settings()
        located = [
            (page, piece)
            for page, content in extract_sections(item)
            for piece in chunk_text(content, settings.chunk_size, settings.chunk_overlap)
        ]
        pieces = [piece for _, piece in located]
        if not pieces:
            raise ValueError("No extractable text")
        if (application_active() or item.knowledge_base_id is not None) and len(pieces) > 512:
            raise ValueError("Document exceeds 512 chunks")
        with session_factory() as db:
            job = _owned(db, item)
            if job is None:
                return False
            if item.target_logical_id is not None:
                active = db.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.logical_id == item.target_logical_id,
                        DocumentVersion.active.is_(True),
                    )
                )
                if active is None or active.document_id != item.base_document_id:
                    raise VersionConflict("Obsolete update")
            job.state = "embedding"
            db.commit()
        error_code = "embedding_failed"
        vectors = embed(pieces)
        if len(vectors) != len(pieces) or any(
            len(v) != settings.embedding_dim or not all(math.isfinite(x) for x in v)
            for v in vectors
        ):
            raise ValueError("Invalid embedding shape or values")
        error_code = "publication_failed"
        with session_factory() as db:
            # Publication/deletion/enqueue share root -> task lock order.
            logical = None
            if item.target_logical_id is not None:
                logical = db.scalar(
                    select(LogicalDocument)
                    .where(LogicalDocument.id == item.target_logical_id)
                    .with_for_update()
                )
                if logical is None:
                    return False
                if logical.knowledge_base_id != item.knowledge_base_id:
                    raise VersionConflict("Task scope mismatch")
            job = _owned(db, item)
            if job is None:
                return False
            number = 1
            if logical is not None:
                active = db.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.logical_id == logical.id,
                        DocumentVersion.active.is_(True),
                    )
                )
                if active is None or active.document_id != item.base_document_id:
                    raise VersionConflict("Obsolete update")
                number = active.number + 1
                active.active = False
                db.flush()
            else:
                logical = LogicalDocument(knowledge_base_id=item.knowledge_base_id)
                db.add(logical)
                db.flush()
            document = Document(title=item.title, source=item.filename, created_at=job.created_at)
            document.chunks = [
                Chunk(chunk_index=i, content=piece, embedding=vector)
                for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
            ]
            db.add(document)
            db.flush()
            db.add(
                DocumentVersion(
                    document_id=document.id,
                    logical_id=logical.id,
                    number=number,
                    active=True,
                    filename=item.filename,
                    raw_content=item.payload,
                    file_hash=hashlib.sha256(item.payload).hexdigest(),
                    created_at=job.created_at,
                )
            )
            for chunk, (page, _) in zip(document.chunks, located, strict=True):
                if page is not None:
                    db.add(ChunkLocation(chunk_id=chunk.id, page_number=page))
            job.document_id = document.id
            job.state = "completed"
            job.payload = None
            job.lease_token = None
            job.lease_until = None
            db.commit()
        return True
    except Exception as exc:  # No provider exception text/private content in API errors.
        if isinstance(exc, VersionConflict):
            error_code = "version_conflict"
        elif isinstance(exc, ProviderUnavailable) and exc.code.startswith("budget_"):
            error_code = exc.code
        with session_factory() as db:
            job = _owned(db, item)
            if job is not None:
                job.state, job.error_code = "failed", error_code
                job.lease_token = None
                job.lease_until = None
                db.commit()
        return False
