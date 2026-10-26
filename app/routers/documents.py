"""Document ingestion endpoints."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import func, or_, select, true
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.database import get_db
from app.models import (
    Chunk,
    ChunkLocation,
    Document,
    DocumentVersion,
    IngestionJob,
    KnowledgeBase,
    LogicalDocument,
    User,
)
from app.schemas import (
    ChunkEvidence,
    DocumentCatalogSummary,
    DocumentSummary,
    IngestRequest,
    IngestResponse,
    JobSummary,
    VersionSummary,
)
from app.services.extract import extract_text
from app.services.ingestion import (
    MAX_UPLOAD_BYTES,
    VersionConflict,
    enqueue,
    ingest_document,
)
from app.services.retrieval import library_id, visible_documents

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)):
    """List ingested documents and their chunk counts (powers the sidebar)."""
    rows = db.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .where(visible_documents(db))
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [DocumentSummary(id=d.id, title=d.title, source=d.source, chunks=n) for d, n in rows]


@router.get("/catalog", response_model=list[DocumentCatalogSummary])
def document_catalog(db: Session = Depends(get_db)):
    """Active versions only; stable logical IDs and immutable document IDs."""
    rows = db.execute(
        select(Document, func.count(Chunk.id), DocumentVersion.logical_id, DocumentVersion.number)
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .outerjoin(DocumentVersion, DocumentVersion.document_id == Document.id)
        .where(visible_documents(db))
        .group_by(Document.id, DocumentVersion.document_id)
        .order_by(Document.created_at.desc())
    ).all()
    return [
        DocumentCatalogSummary(
            id=d.id,
            title=d.title,
            source=d.source,
            chunks=n,
            created_at=d.created_at,
            logical_id=logical_id,
            version_number=number,
        )
        for d, n, logical_id, number in rows
    ]


def _job_summary(job: IngestionJob) -> JobSummary:
    return JobSummary(
        id=job.id,
        title=job.title,
        state=job.state,
        attempts=job.attempts,
        error_code=job.error_code,
        document_id=job.document_id,
        created_at=job.created_at,
        target_logical_id=job.target_logical_id,
        base_document_id=job.base_document_id,
    )


@router.post("/jobs", response_model=JobSummary, status_code=202)
def queue_text(payload: IngestRequest, db: Session = Depends(get_db)):
    try:
        job = enqueue(db, payload.title, "pasted.txt", payload.content.encode("utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _job_summary(job)


async def _upload_task(file: UploadFile, db: Session, base_document_id: int | None = None):
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds 10 MiB")
        name = file.filename or "upload.txt"
        job = await run_in_threadpool(enqueue, db, name, name, data, base_document_id)
        return _job_summary(job)
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


@router.post("/jobs/upload", response_model=JobSummary, status_code=202)
async def queue_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await _upload_task(file, db)


@router.post("/{document_id}/updates/upload", response_model=JobSummary, status_code=202)
async def update_upload(
    document_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await _upload_task(file, db, document_id)


@router.post("/{document_id}/updates", response_model=JobSummary, status_code=202)
def update_text(document_id: int, payload: IngestRequest, db: Session = Depends(get_db)):
    try:
        return _job_summary(
            enqueue(db, payload.title, "pasted.txt", payload.content.encode("utf-8"), document_id)
        )
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/jobs", response_model=list[JobSummary])
def list_jobs(db: Session = Depends(get_db)):
    scope = library_id(db)
    jobs = db.scalars(
        select(IngestionJob)
        .where(IngestionJob.knowledge_base_id == scope if scope is not None else true())
        .order_by(IngestionJob.id.desc())
        .limit(100)
    )
    return [_job_summary(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _job_summary(job)


@router.post("/jobs/{job_id}/retry", response_model=JobSummary, status_code=202)
def retry_job(job_id: int, db: Session = Depends(get_db)):
    job = db.scalar(select(IngestionJob).where(IngestionJob.id == job_id).with_for_update())
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if job.state != "failed" or job.payload is None:
        raise HTTPException(status_code=409, detail="Only failed tasks can be retried")
    if job.target_logical_id is not None:
        current = db.scalar(
            select(DocumentVersion.document_id).where(
                DocumentVersion.logical_id == job.target_logical_id,
                DocumentVersion.active.is_(True),
            )
        )
        if current != job.base_document_id:
            raise HTTPException(status_code=409, detail="Version changed; submit a new update")
    job.state, job.error_code, job.attempts = "queued", None, 0
    job.lease_token = None
    job.lease_until = None
    db.commit()
    return _job_summary(job)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_failed_job(job_id: int, db: Session = Depends(get_db)):
    job = db.scalar(select(IngestionJob).where(IngestionJob.id == job_id).with_for_update())
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if job.state != "failed":
        raise HTTPException(status_code=409, detail="Only failed tasks can be discarded")
    db.delete(job)
    db.commit()
    return Response(status_code=204)


@router.get("/{document_id}/chunks/{chunk_index}", response_model=ChunkEvidence)
def get_evidence(document_id: int, chunk_index: int, db: Session = Depends(get_db)):
    row = db.execute(
        select(
            Document.title,
            Chunk.content,
            DocumentVersion.logical_id,
            DocumentVersion.number,
            ChunkLocation.page_number,
            DocumentVersion.raw_content.is_not(None).label("has_original"),
        )
        .select_from(Document)
        .join(Chunk, Chunk.document_id == Document.id)
        .outerjoin(DocumentVersion, DocumentVersion.document_id == Document.id)
        .outerjoin(ChunkLocation, ChunkLocation.chunk_id == Chunk.id)
        .where(Document.id == document_id, Chunk.chunk_index == chunk_index)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Document or chunk not found")
    return ChunkEvidence(
        document_id=document_id,
        document_title=row.title,
        chunk_index=chunk_index,
        content=row.content,
        logical_id=row.logical_id,
        version_number=row.number,
        page_number=row.page_number,
        original_url=(
            f"/documents/{document_id}/original"
            + (f"#page={row.page_number}" if row.page_number is not None else "")
        )
        if row.has_original
        else None,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    from app.services.tracing import clear_snapshots

    owner_id = db.scalar(select(KnowledgeBase.owner_id).where(KnowledgeBase.id == library_id(db)))
    if owner_id is not None:
        db.scalar(select(User.id).where(User.id == owner_id).with_for_update())
        clear_snapshots(db, owner_id)
    logical_id = db.scalar(
        select(DocumentVersion.logical_id).where(DocumentVersion.document_id == document_id)
    )
    logical = None
    if logical_id is not None:
        logical = db.scalar(
            select(LogicalDocument).where(LogicalDocument.id == logical_id).with_for_update()
        )
        ids = list(
            db.scalars(
                select(DocumentVersion.document_id).where(DocumentVersion.logical_id == logical_id)
            )
        )
    else:
        ids = [document_id]
    documents = db.scalars(select(Document).where(Document.id.in_(ids)).with_for_update()).all()
    if not documents:
        raise HTTPException(status_code=404, detail="Document not found")
    # Root -> tasks is the same lock order used by update publication.
    condition = IngestionJob.document_id.in_(ids)
    if logical_id is not None:
        condition = or_(condition, IngestionJob.target_logical_id == logical_id)
    jobs = db.scalars(
        select(IngestionJob).where(condition).order_by(IngestionJob.id).with_for_update()
    ).all()
    for job in jobs:
        db.delete(job)
    db.flush()
    for document in documents:
        db.delete(document)
    db.flush()  # DB cascades versions, raw originals and page locations.
    if logical is not None:
        db.delete(logical)
    db.commit()
    return Response(status_code=204)


@router.get("/{document_id}/versions", response_model=list[VersionSummary])
def document_versions(document_id: int, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    logical_id = db.scalar(
        select(DocumentVersion.logical_id).where(DocumentVersion.document_id == document_id)
    )
    if logical_id is None:
        return [
            VersionSummary(
                document_id=document.id,
                title=document.title,
                logical_id=None,
                version_number=1,
                active=True,
                created_at=document.created_at,
                file_hash=None,
                original_available=False,
            )
        ]
    rows = db.execute(
        select(
            Document.id,
            Document.title,
            DocumentVersion.logical_id,
            DocumentVersion.number,
            DocumentVersion.active,
            DocumentVersion.created_at,
            DocumentVersion.file_hash,
            DocumentVersion.raw_content.is_not(None).label("has_original"),
        )
        .join(DocumentVersion, DocumentVersion.document_id == Document.id)
        .where(DocumentVersion.logical_id == logical_id)
        .order_by(DocumentVersion.number.desc())
    ).all()
    return [
        VersionSummary(
            document_id=row.id,
            title=row.title,
            logical_id=row.logical_id,
            version_number=row.number,
            active=row.active,
            created_at=row.created_at,
            file_hash=row.file_hash,
            original_available=row.has_original,
        )
        for row in rows
    ]


@router.get("/{document_id}/original")
def original_file(document_id: int, db: Session = Depends(get_db)):
    version = db.get(DocumentVersion, document_id)
    if version is None or version.raw_content is None:
        raise HTTPException(status_code=404, detail="Original file unavailable (legacy or deleted)")
    filename = version.filename or "document.txt"
    return Response(
        content=version.raw_content,
        media_type="application/pdf" if filename.lower().endswith(".pdf") else "text/plain",
        headers={
            "Content-Disposition": "inline; filename*=UTF-8''" + quote(filename, safe=""),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "sandbox",
        },
    )


def _to_response(document) -> IngestResponse:
    return IngestResponse(
        document_id=document.id,
        title=document.title,
        chunks_created=len(document.chunks),
    )


@router.post("", response_model=IngestResponse, status_code=201)
def create_document(payload: IngestRequest, db: Session = Depends(get_db)):
    """Ingest a raw-text document."""
    try:
        document = ingest_document(db, payload.title, payload.content, payload.source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(document)


@router.post("/upload", response_model=IngestResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Ingest an uploaded file (.txt, .md, .pdf)."""
    content = await extract_text(file)
    try:
        document = await run_in_threadpool(
            ingest_document, db, file.filename or "upload", content, file.filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(document)
