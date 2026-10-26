"""Session authentication and owner-only knowledge-base boundary for every RAG route."""

from __future__ import annotations

import hmac
import secrets
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import (
    AuthSession,
    DocumentVersion,
    IngestionJob,
    KnowledgeBase,
    LogicalDocument,
    User,
)
from app.security import hash_password, normalize_username, token_hash, verify_password
from app.services.application_schema import DatabaseNotReady, check_guard
from app.services.resilience import set_budget_owner

router = APIRouter(tags=["identity"])
COOKIE = "tracerag_session"
_DUMMY_HASH = hash_password("dummy-password-not-an-account")


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=1, max_length=128, repr=False)


class LibraryInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)


def same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin:
        expected = urlsplit(str(request.base_url))
        try:
            actual = urlsplit(origin)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Invalid origin") from exc
        if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
            raise HTTPException(status_code=403, detail="Cross-origin mutation rejected")


def active_database(db: Session) -> None:
    try:
        check_guard(db)
    except DatabaseNotReady:
        raise HTTPException(
            503, "Application recovery review or database migration required"
        ) from None


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    raw = request.cookies.get(COOKIE, "")
    if not raw or len(raw) > 128:
        raise HTTPException(status_code=401, detail="Login required")
    active_database(db)
    row = db.execute(
        select(User, AuthSession)
        .join(AuthSession)
        .where(
            AuthSession.token_hash == token_hash(raw),
            AuthSession.expires_at > func.now(),
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=401, detail="Session expired or revoked")
    user, session = row
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        same_origin(request)
        supplied = request.headers.get("x-csrf-token", "")
        if not hmac.compare_digest(supplied.encode(), session.csrf_token.encode()):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
    request.state.auth_session = session
    request.state.user_id = user.id
    return user


def require_library(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    knowledge_base_id: int | None = Query(
        None,
        ge=1,
        le=2147483647,
        description="Select a library for lists/uploads/queries; owned resource URLs can infer it",
    ),
) -> int:
    """No anonymous fallback; infer owned scope for immutable evidence URLs."""
    selected_id = knowledge_base_id
    resource_key = "job_id" if "job_id" in request.path_params else "document_id"
    resource_id = None
    if resource_key in request.path_params:
        try:
            resource_id = int(request.path_params[resource_key])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid resource ID") from exc
        if not 1 <= resource_id <= 2147483647:
            raise HTTPException(status_code=404, detail="Resource not found")
    statement = select(KnowledgeBase.id).where(KnowledgeBase.owner_id == user.id)
    if "job_id" in request.path_params:
        statement = statement.join(
            IngestionJob, IngestionJob.knowledge_base_id == KnowledgeBase.id
        ).where(IngestionJob.id == resource_id)
    elif "document_id" in request.path_params:
        statement = (
            statement.join(LogicalDocument, LogicalDocument.knowledge_base_id == KnowledgeBase.id)
            .join(DocumentVersion, DocumentVersion.logical_id == LogicalDocument.id)
            .where(DocumentVersion.document_id == resource_id)
        )
    elif selected_id is None:
        raise HTTPException(status_code=400, detail="Select a knowledge base")
    if selected_id is not None:
        statement = statement.where(KnowledgeBase.id == selected_id)
    identifier = db.scalar(statement)
    if identifier is None:
        raise HTTPException(status_code=404, detail="Resource not found")
    db.info["knowledge_base_id"] = identifier
    set_budget_owner(user.id)
    from app.services.tracing import bind_owner

    bind_owner(user.id, identifier)
    return identifier


@router.post("/auth/login")
def login(
    payload: Credentials, request: Request, response: Response, db: Session = Depends(get_db)
):
    same_origin(request)
    active_database(db)
    # Use the ASGI client address, never parse forwarded headers here. Configure
    # trusted proxies at the server boundary (the local demo disables them).
    ip = request.client.host if request.client else "unknown"
    bucket = token_hash(ip) + ":" + str(int(time.time()) // 300)
    count = (
        db.connection()
        .exec_driver_sql(
            "INSERT INTO auth_login_windows(key, attempts) VALUES (%(key)s, 1) "
            "ON CONFLICT (key) DO UPDATE SET attempts = auth_login_windows.attempts + 1 "
            "RETURNING attempts",
            {"key": bucket},
        )
        .scalar_one()
    )
    db.commit()
    if count > 20:
        raise HTTPException(status_code=429, detail="Too many login attempts; wait five minutes")
    try:
        username = normalize_username(payload.username)
    except ValueError:
        username = ""
    user = db.scalar(select(User).where(User.username == username))
    valid = verify_password(payload.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not valid:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    raw, csrf = secrets.token_urlsafe(32), secrets.token_hex(32)
    ttl = get_settings().session_ttl_seconds
    db.add(
        AuthSession(
            token_hash=token_hash(raw),
            user_id=user.id,
            csrf_token=csrf,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl),
        )
    )
    db.commit()
    response.set_cookie(
        COOKIE,
        raw,
        max_age=ttl,
        httponly=True,
        secure=get_settings().session_cookie_secure,
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {"id": user.id, "username": user.username, "csrf_token": csrf}


@router.get("/auth/me")
def me(request: Request, response: Response, user: User = Depends(current_user)):
    response.headers["Cache-Control"] = "no-store"
    return {
        "id": user.id,
        "username": user.username,
        "csrf_token": request.state.auth_session.csrf_token,
    }


@router.post("/auth/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    db.execute(
        delete(AuthSession).where(AuthSession.token_hash == request.state.auth_session.token_hash)
    )
    db.commit()
    response.delete_cookie(
        COOKIE,
        path="/",
        secure=get_settings().session_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"


@router.get("/knowledge-bases")
def libraries(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(
        select(KnowledgeBase).where(KnowledgeBase.owner_id == user.id).order_by(KnowledgeBase.id)
    ).all()
    return [{"id": row.id, "name": row.name} for row in rows]


@router.post("/knowledge-bases", status_code=201)
def create_library(
    payload: LibraryInput, db: Session = Depends(get_db), user: User = Depends(current_user)
):
    if not payload.name.strip():
        raise HTTPException(status_code=422, detail="Knowledge-base name is empty")
    kb = KnowledgeBase(owner_id=user.id, name=payload.name.strip())
    db.add(kb)
    db.commit()
    return {"id": kb.id, "name": kb.name}
