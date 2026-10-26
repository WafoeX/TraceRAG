"""Fixed-label owner feedback and explicit manual-review downloads.

No model calls, content retention, automatic anonymization or trace TTL extension.
"""

from __future__ import annotations

from contextlib import contextmanager
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth import current_user
from app.database import get_db
from app.models import RequestFeedback, RequestTrace, User
from app.services.feedback_cases import (
    CATEGORIES,
    ExportInput,
    FeedbackInput,
    RegressionFile,
    metadata_export,
    parse_metadata,
)

router = APIRouter(prefix="/traces", tags=["feedback"])


@contextmanager
def boundary():
    try:
        yield
    except SQLAlchemyError:
        raise HTTPException(503, "Feedback unavailable") from None


def owned_trace(db: Session, owner: int, identifier: str, *, lock: bool = False) -> RequestTrace:
    try:
        UUID(identifier)
    except ValueError:
        raise HTTPException(404, "Trace not found") from None
    if lock:
        # Owner -> trace is consistent with trace/doc erasure and cross-process writes.
        db.scalar(select(User.id).where(User.id == owner).with_for_update())
    statement = select(RequestTrace).where(
        RequestTrace.id == identifier,
        RequestTrace.user_id == owner,
        RequestTrace.expires_at > func.clock_timestamp(),
    )
    if lock:
        statement = statement.with_for_update()
    row = db.scalar(statement)
    if row is None:
        raise HTTPException(404, "Trace not found")
    return row


def labels(row: RequestFeedback) -> FeedbackInput:
    return FeedbackInput(
        categories=[key for key in CATEGORIES if getattr(row, key)],
        citation_number=row.citation_number,
    )


def summary(row: RequestFeedback | None):
    if row is None:
        return None
    return {**labels(row).model_dump(), "updated_at": row.updated_at}


def trace_metadata(row: RequestTrace) -> dict:
    try:
        return parse_metadata(row.metadata_json)
    except ValueError:
        raise HTTPException(503, "Trace unavailable") from None


@router.get("/{trace_id}/feedback")
def get_feedback(trace_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    with boundary():
        owned_trace(db, user.id, trace_id)
        return {"feedback": summary(db.get(RequestFeedback, trace_id))}


@router.put("/{trace_id}/feedback")
def put_feedback(
    trace_id: str,
    payload: FeedbackInput,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    with boundary():
        trace = owned_trace(db, user.id, trace_id, lock=True)
        contexts = trace_metadata(trace).get("contexts")
        numbers = (
            {
                c.get("number")
                for c in contexts
                if isinstance(c, dict) and type(c.get("number")) is int
            }
            if isinstance(contexts, list)
            else set()
        )
        if payload.citation_number is not None and payload.citation_number not in numbers:
            raise HTTPException(422, "Citation number is not in this trace")
        row = db.get(RequestFeedback, trace_id)
        fields: dict[str, bool | int | None] = {
            key: key in payload.categories for key in CATEGORIES
        }
        fields["citation_number"] = payload.citation_number
        if row is None:
            row = RequestFeedback(trace_id=trace_id, **fields)
            db.add(row)
        elif any(getattr(row, key) != value for key, value in fields.items()):
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = func.clock_timestamp()
        db.commit()
        db.refresh(row)
        return {"feedback": summary(row), "expires_at": trace.expires_at}


@router.delete("/{trace_id}/feedback", status_code=204)
def delete_feedback(
    trace_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)
):
    with boundary():
        owned_trace(db, user.id, trace_id, lock=True)
        db.execute(delete(RequestFeedback).where(RequestFeedback.trace_id == trace_id))
        db.commit()
        return Response(status_code=204)


@router.post("/{trace_id}/export")
def export_feedback(
    trace_id: str,
    payload: ExportInput,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    with boundary():
        trace = owned_trace(db, user.id, trace_id, lock=True)
        row = db.get(RequestFeedback, trace_id)
        if row is None:
            raise HTTPException(409, "Submit feedback before exporting")
        if payload.mode == "metadata":
            data = metadata_export(trace_metadata(trace), labels(row))
            filename = "tracerag-feedback-metadata.json"
        else:
            # Already validated; no snapshot fallback or merge of any private trace fields.
            if payload.reviewed_case is None:
                raise HTTPException(422, "Reviewed case required")
            data = RegressionFile(
                schema_version="tracerag-regression-v1",
                feedback=labels(row),
                case=payload.reviewed_case,
            ).model_dump()
            filename = "tracerag-regression.json"
        return JSONResponse(
            data,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Content-Type-Options": "nosniff",
            },
        )
