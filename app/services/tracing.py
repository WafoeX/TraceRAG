"""Owner-only query traces. No content unless this request explicitly consents.

Trace failure never replays a model call or changes the independent budget ledger.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

RETENTION_DAYS = 7
MAX_SNAPSHOT_BYTES = 1024 * 1024


@dataclass
class TraceState:
    id: str = field(default_factory=lambda: str(uuid4()))
    started: float = field(default_factory=time.perf_counter)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_id: int | None = None
    library_id: int | None = None
    consent: bool = False
    snapshot: dict | None = None
    data: dict = field(
        default_factory=lambda: {"stages": [], "calls": [], "contexts": [], "warnings": []}
    )


_state: ContextVar[TraceState | None] = ContextVar("query_trace", default=None)


def bind_owner(user_id: int, library_id: int) -> None:
    state = _state.get()
    if state is not None:
        state.user_id, state.library_id = user_id, library_id


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    outcome = "completed"
    try:
        yield
    except BaseException:
        outcome = "failed"
        raise
    finally:
        state = _state.get()
        if state is not None and len(state.data["stages"]) < 100:
            state.data["stages"].append(
                {
                    "name": name,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                    "status": outcome,
                }
            )


def warning(code: str) -> None:
    state = _state.get()
    if state is not None and code not in state.data["warnings"]:
        state.data["warnings"].append(code)


def begin_query(question: str, consent: bool, top_k: int) -> None:
    from app.config import get_settings

    state = _state.get()
    if state is None:
        return
    settings = get_settings()
    state.consent = consent
    state.data["config"] = {
        key: getattr(settings, key)
        for key in (
            "use_hybrid",
            "use_rerank",
            "retrieval_candidates",
            "min_relevance",
            "model_provider",
            "openai_embedding_model",
            "openai_chat_model",
            "llm_rerank_model",
            "rerank_passage_mode",
            "rerank_prompt_version",
            "llm_rerank_enable_thinking",
            "embedding_dim",
            "application_generation_max_tokens",
            "application_rerank_max_tokens",
            "provider_max_attempts",
            "provider_timeout_seconds",
        )
    }
    state.data["config"]["top_k"] = top_k
    if consent:
        state.snapshot = {"question": question}


def context(chunks, *, generation_attempted: bool) -> None:
    state = _state.get()
    if state is None:
        return
    state.data["generation_attempted"] = generation_attempted
    state.data["contexts"] = [
        {
            "number": i,
            "document_id": c.document_id,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "score": c.score,
        }
        for i, c in enumerate(chunks, 1)
    ]
    if state.snapshot is not None:
        state.snapshot["contexts"] = [
            {
                "number": i,
                "document_id": c.document_id,
                "title": c.document_title,
                "content": c.content,
            }
            for i, c in enumerate(chunks, 1)
        ]


def answer(text: str, status: str) -> None:
    state = _state.get()
    if state is not None:
        state.data["generation_status"] = status
        if state.snapshot is not None:
            state.snapshot["answer"] = text


def generation_payload(payload: dict) -> None:
    """Exact selected generation messages, never credentials or arbitrary payload fields."""
    state = _state.get()
    messages = payload.get("messages")
    if (
        state is not None
        and state.snapshot is not None
        and isinstance(messages, list)
        and all(
            isinstance(m, dict)
            and m.get("role") in {"system", "user", "assistant"}
            and isinstance(m.get("content"), str)
            for m in messages
        )
    ):
        state.snapshot["generation_messages"] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]


def call_event(event: dict) -> None:
    state = _state.get()
    if state is not None and len(state.data["calls"]) < 100:
        state.data["calls"].append(event)


def clear_snapshots(db: Session, user_id: int) -> None:
    """Caller holds owner lock; conservatively erase ALL owner's content copies."""
    from app.models import RequestTrace

    db.execute(
        update(RequestTrace).where(RequestTrace.user_id == user_id).values(snapshot_json=None)
    )


def persist(state: TraceState) -> None:
    from app.database import SessionLocal
    from app.models import DocumentVersion, LogicalDocument, RequestTrace, User

    if state.user_id is None or state.library_id is None:
        return
    # Content is bounded, never silently truncated into a purported exact snapshot.
    snapshot = (
        json.dumps(state.snapshot, ensure_ascii=False) if state.snapshot is not None else None
    )
    if snapshot is not None and len(snapshot.encode()) > MAX_SNAPSHOT_BYTES:
        snapshot = None
        state.data["warnings"].append("trace_snapshot_too_large")
    with SessionLocal() as db:
        # Serialize snapshot persistence with document deletion, then recheck all IDs
        # using a fresh READ COMMITTED transaction, not retrieval's old snapshot.
        if db.scalar(select(User.id).where(User.id == state.user_id).with_for_update()) is None:
            return
        ids = {c["document_id"] for c in state.data["contexts"]}
        present = set(
            db.scalars(
                select(DocumentVersion.document_id)
                .join(LogicalDocument)
                .where(
                    DocumentVersion.document_id.in_(ids),
                    LogicalDocument.knowledge_base_id == state.library_id,
                )
            )
        )
        if ids - present:
            snapshot = None
            state.data["warnings"].append("trace_source_deleted")
        db.add(
            RequestTrace(
                id=state.id,
                user_id=state.user_id,
                knowledge_base_id=state.library_id,
                created_at=state.created_at,
                expires_at=state.created_at + timedelta(days=RETENTION_DAYS),
                metadata_json=json.dumps(state.data),
                snapshot_json=snapshot,
                content_consent=state.consent,
            )
        )
        db.commit()


class TraceBoundary:
    """ASGI lifetime covers sync threads and buffered streaming; no body capture."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in {"/query", "/query/evidence", "/query/stream"}
        ):
            return await self.app(scope, receive, send)
        state = TraceState()
        token = _state.set(state)
        status = 500
        attempted = False

        async def save():
            nonlocal attempted
            if attempted:
                return
            attempted = True
            state.data["http_status"] = status
            state.data["duration_ms"] = round((time.perf_counter() - state.started) * 1000, 3)
            try:
                await run_in_threadpool(persist, state)
            except Exception:
                # No exception text/URL/content, and no replay of provider calls.
                import logging

                logging.getLogger(__name__).warning("trace_persistence_unavailable")

        async def traced_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-trace-id", state.id.encode())
                ]
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await save()
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except BaseException:
            state.data["interrupted"] = True
            raise
        finally:
            try:
                await save()
            finally:
                _state.reset(token)
