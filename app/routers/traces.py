"""Owner-only traces: expiry enforced in SQL, mutations use normal CSRF checks."""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.database import get_db
from app.models import RequestTrace, User

router = APIRouter(prefix="/traces", tags=["traces"])


def visible(user_id: int):
    return (RequestTrace.user_id == user_id, RequestTrace.expires_at > func.now())


@router.get("")
def list_traces(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(
        select(RequestTrace)
        .where(*visible(user.id))
        .order_by(RequestTrace.created_at.desc())
        .limit(20)
    )
    return [
        {
            "trace_id": row.id,
            "created_at": row.created_at,
            "expires_at": row.expires_at,
            "knowledge_base_id": row.knowledge_base_id,
            "has_content": row.snapshot_json is not None,
        }
        for row in rows
    ]


def owned(db: Session, user_id: int, trace_id: str) -> RequestTrace:
    try:
        UUID(trace_id)
    except ValueError:
        raise HTTPException(404, "Trace not found") from None
    row = db.scalar(select(RequestTrace).where(RequestTrace.id == trace_id, *visible(user_id)))
    if row is None:
        raise HTTPException(404, "Trace not found")
    return row


@router.get("/{trace_id}")
def get_trace(trace_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = owned(db, user.id, trace_id)
    try:
        metadata = json.loads(row.metadata_json)
        snapshot = json.loads(row.snapshot_json) if row.snapshot_json is not None else None
    except (ValueError, TypeError):
        raise HTTPException(503, "Trace unavailable") from None
    return {
        "trace_id": row.id,
        "created_at": row.created_at,
        "expires_at": row.expires_at,
        "knowledge_base_id": row.knowledge_base_id,
        "content_consent": row.content_consent,
        "metadata": metadata,
        "snapshot": snapshot,
        "cost_notice": "CNY estimates, not provider charges; null means unknown, not zero",
    }


@router.delete("/{trace_id}", status_code=204)
def delete_trace(trace_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    # Match persistence/deletion lock order; prevents races with document erasure.
    db.scalar(select(User.id).where(User.id == user.id).with_for_update())
    row = owned(db, user.id, trace_id)
    db.execute(
        delete(RequestTrace).where(RequestTrace.id == row.id, RequestTrace.user_id == user.id)
    )
    db.commit()
    return Response(status_code=204)
