"""Shared PostgreSQL admission, plus bounded ASGI bodies before multipart parsing.

Uses the existing v3 login-window table with a separate key namespace (no DDL).
Transaction advisory locks release on close/crash, not lease expiry. No model
calls occur on the dedicated admission connection; data transactions stay separate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

from anyio import CancelScope
from fastapi import HTTPException, Request
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app import database
from app.config import get_settings
from app.services.resilience import ProviderUnavailable, application_calls

MAX_BODY_BYTES = 10 * 1024 * 1024 + 64 * 1024  # file plus bounded multipart/JSON overhead
QUERY_BODY_BYTES = 32 * 1024
FEEDBACK_BODY_BYTES = 1024 * 1024


def _slot(db: Session, namespace: str, limit: int) -> bool:
    for slot in range(limit):
        key = int.from_bytes(
            hashlib.sha256(f"tracerag:p2:{namespace}:{slot}".encode()).digest()[:8],
            "big",
            signed=True,
        )
        if (
            db.connection()
            .exec_driver_sql("SELECT pg_try_advisory_xact_lock(%(key)s)", {"key": key})
            .scalar_one()
        ):
            return True
    return False


def admit_account(user_id: int) -> Session:
    settings = get_settings()
    db = database.SessionLocal()
    try:
        # DB clock and capped counter: shared across all libraries, IPs and processes.
        count = (
            db.connection()
            .exec_driver_sql(
                "INSERT INTO auth_login_windows(key, attempts) "
                "VALUES ('rag:' || %(user)s || ':' || "
                "floor(extract(epoch from clock_timestamp()) / 60)::bigint::text, 1) "
                "ON CONFLICT (key) DO UPDATE SET attempts = auth_login_windows.attempts + 1 "
                "WHERE auth_login_windows.attempts < %(cap)s RETURNING attempts",
                {"user": str(user_id), "cap": settings.rate_limit_per_minute},
            )
            .scalar_one_or_none()
        )
        db.commit()
        if count is None or not _slot(db, f"account:{user_id}", settings.account_concurrency):
            raise HTTPException(429, "Account call limit exceeded", headers={"Retry-After": "60"})
        return db
    except BaseException:
        db.close()
        raise


async def require_capacity(request: Request):
    """Authorize first; ASGI boundary (not yield dependency) owns slot lifetime."""
    if request.method != "POST" or not get_settings().rate_limit_enabled:
        return
    try:
        db = await run_in_threadpool(admit_account, request.state.user_id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "Call limiter unavailable") from None
    request.state._admission_db = db


@contextmanager
def provider_capacity() -> Iterator[None]:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        yield
        return
    try:
        db = database.SessionLocal()
    except Exception:
        raise ProviderUnavailable("provider_limiter_unavailable") from None
    try:
        try:
            available = _slot(db, "provider", settings.application_concurrency)
        except Exception:
            raise ProviderUnavailable("provider_limiter_unavailable") from None
        if not available:
            raise ProviderUnavailable("provider_busy")
        yield
    finally:
        db.close()


class ApplicationBoundary:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        try:
            await self._bounded(scope, receive, send)
        finally:
            # FastAPI 0.115 closes yield dependencies BEFORE streaming iteration.
            # Own this connection until response completion/disconnect instead.
            db = scope.get("state", {}).pop("_admission_db", None)
            if db is not None:
                with CancelScope(shield=True):
                    await run_in_threadpool(db.close)

    async def _bounded(self, scope: Scope, receive: Receive, send: Send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith(("/query", "/documents", "/traces/")):
            await self.app(scope, receive, send)
            return
        with application_calls():
            if scope.get("method") not in {"POST", "PUT"}:
                await self.app(scope, receive, send)
                return
            limit = (
                FEEDBACK_BODY_BYTES
                if path.startswith("/traces/")
                else QUERY_BODY_BYTES
                if path.startswith("/query")
                else MAX_BODY_BYTES
            )
            data = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body = message.get("body", b"")
                if len(data) + len(body) > limit:
                    response = JSONResponse(
                        {"detail": "Request body exceeds application limit"},
                        status_code=413,
                        headers={"Cache-Control": "no-store", "Vary": "Cookie"},
                    )
                    await response(scope, receive, send)
                    return
                data.extend(body)
                if not message.get("more_body", False):
                    break
            consumed = False

            async def bounded_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": bytes(data), "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, send)
