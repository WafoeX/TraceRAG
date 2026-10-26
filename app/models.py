"""SQLAlchemy ORM models: documents and their embedded chunks."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.database import Base

settings = get_settings()


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RecoveryState(Base):
    __tablename__ = "application_recovery_state"
    __table_args__ = (CheckConstraint("id = 1"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    quarantined: Mapped[bool] = mapped_column(Boolean, server_default="false")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))

    document: Mapped[Document] = relationship(back_populates="chunks")


class User(Base):
    __tablename__ = "app_users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_users.id", ondelete="CASCADE"))
    csrf_token: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LoginWindow(Base):
    __tablename__ = "auth_login_windows"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    attempts: Mapped[int] = mapped_column(Integer)


class BudgetAccount(Base):
    __tablename__ = "application_budget_accounts"
    __table_args__ = (
        CheckConstraint("limit_cny >= 0 AND spent_cny >= 0 AND held_cny >= 0"),
        CheckConstraint("spent_cny + held_cny <= limit_cny"),
        CheckConstraint(
            "(key = 'global' AND user_id IS NULL) OR "
            "(user_id IS NOT NULL AND key = 'user:' || user_id::text)"
        ),
    )
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("app_users.id"), unique=True)
    limit_cny: Mapped[Decimal] = mapped_column(Numeric(30, 12))
    spent_cny: Mapped[Decimal] = mapped_column(Numeric(30, 12), server_default="0")
    held_cny: Mapped[Decimal] = mapped_column(Numeric(30, 12), server_default="0")
    blocked: Mapped[bool] = mapped_column(Boolean, server_default="false")


class BudgetReservation(Base):
    __tablename__ = "application_budget_reservations"
    __table_args__ = (
        CheckConstraint("reserved_cny >= 0 AND (estimated_cny IS NULL OR estimated_cny >= 0)"),
        CheckConstraint("status IN ('held','unknown','settled','released')"),
        CheckConstraint(
            "(status IN ('held','unknown') AND estimated_cny IS NULL) OR "
            "(status IN ('settled','released') AND estimated_cny IS NOT NULL)"
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(256))
    profile_json: Mapped[str] = mapped_column(Text)
    reserved_cny: Mapped[Decimal] = mapped_column(Numeric(30, 12))
    estimated_cny: Mapped[Decimal | None] = mapped_column(Numeric(30, 12))
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BudgetAdminEvent(Base):
    __tablename__ = "application_budget_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action: Mapped[str] = mapped_column(String(32))
    scope: Mapped[str] = mapped_column(String(64))
    reservation_id: Mapped[str | None] = mapped_column(
        ForeignKey("application_budget_reservations.id")
    )
    amount_cny: Mapped[Decimal | None] = mapped_column(Numeric(30, 12))
    reason: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("app_users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RequestTrace(Base):
    __tablename__ = "application_request_traces"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_users.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    content_consent: Mapped[bool] = mapped_column(Boolean)
    metadata_json: Mapped[str] = mapped_column(Text)
    snapshot_json: Mapped[str | None] = mapped_column(Text)


class RequestFeedback(Base):
    __tablename__ = "application_request_feedback"
    __table_args__ = (
        CheckConstraint("unhelpful OR unsupported_citation OR outdated"),
        CheckConstraint("citation_number IS NULL OR citation_number BETWEEN 1 AND 20"),
    )
    trace_id: Mapped[str] = mapped_column(
        ForeignKey("application_request_traces.id", ondelete="CASCADE"), primary_key=True
    )
    unhelpful: Mapped[bool] = mapped_column(Boolean)
    unsupported_citation: Mapped[bool] = mapped_column(Boolean)
    outdated: Mapped[bool] = mapped_column(Boolean)
    citation_number: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LogicalDocument(Base):
    __tablename__ = "logical_documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    knowledge_base_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_bases.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DocumentVersion(Base):
    """Each existing document ID remains an immutable version/evidence identity."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("logical_id", "number", name="uq_document_version_number"),
        Index(
            "uq_document_version_active", "logical_id", unique=True, postgresql_where=text("active")
        ),
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    logical_id: Mapped[int] = mapped_column(
        ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ChunkLocation(Base):
    __tablename__ = "chunk_locations"
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(Integer)


class IngestionJob(Base):
    """Durable single-library upload queue; workers publish documents atomically."""

    __tablename__ = "ingestion_jobs"

    knowledge_base_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_bases.id"), nullable=True, index=True
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dedup_key: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    target_logical_id: Mapped[int | None] = mapped_column(
        ForeignKey("logical_documents.id", ondelete="CASCADE"), nullable=True, index=True
    )
    base_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(512))
    payload: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
