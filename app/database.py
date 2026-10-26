"""Database engine, session, and schema/extension bootstrap."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

# prepare_threshold=None disables psycopg3 prepared statements so the app works
# over pgbouncer/pooled endpoints (Neon, Supabase) as well as direct connections.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args={
        "prepare_threshold": None,
        "connect_timeout": settings.database_connect_timeout_seconds,
    },
    pool_timeout=settings.database_connect_timeout_seconds,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    """Enable pgvector, create tables, and add an ANN index.

    Idempotent — safe to call on every startup.
    """
    # Import models so they register on Base.metadata before create_all.
    from app import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    inspector = inspect(engine)
    if inspector.has_table("ingestion_jobs") and "dedup_key" not in {
        col["name"] for col in inspector.get_columns("ingestion_jobs")
    }:
        raise RuntimeError(
            "Application schema v1: stop services and run scripts.migrate_application"
        )
    if inspector.has_table("logical_documents") and "knowledge_base_id" not in {
        col["name"] for col in inspector.get_columns("logical_documents")
    }:
        raise RuntimeError(
            "Application schema v2: stop services and run scripts.migrate_application_auth"
        )
    if inspector.has_table("app_users") and not all(
        inspector.has_table(name)
        for name in (
            "application_budget_accounts",
            "application_budget_reservations",
            "application_budget_events",
        )
    ):
        raise RuntimeError(
            "Application schema v3: stop services and run scripts.migrate_application_budget"
        )
    if inspector.has_table("application_budget_accounts") and not inspector.has_table(
        "application_request_traces"
    ):
        raise RuntimeError(
            "Application schema v4: stop services and run scripts.migrate_application_traces"
        )
    if inspector.has_table("application_request_traces") and not inspector.has_table(
        "application_request_feedback"
    ):
        raise RuntimeError(
            "Application schema v5: stop services and run scripts.migrate_application_feedback"
        )
    if inspector.has_table("application_request_feedback") and not inspector.has_table(
        "application_recovery_state"
    ):
        raise RuntimeError(
            "Application schema v6: stop services and run scripts.migrate_application_recovery"
        )
    Base.metadata.create_all(bind=engine)

    # HNSW index for fast approximate nearest-neighbour search at scale.
    # pgvector's HNSW index supports at most 2000 dimensions; above that we keep
    # the column (exact search) but skip the index.
    if settings.embedding_dim <= 2000:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )

    # GIN index powering keyword (full-text) search for hybrid retrieval.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_content_fts "
                "ON chunks USING gin (to_tsvector('english', content))"
            )
        )


def get_db(request: Request) -> Iterator[Session]:
    """Query retrieval stages share one version snapshot, even during activation."""
    db = SessionLocal()
    try:
        if request.url.path == "/query" or request.url.path.startswith("/query/"):
            db.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        yield db
    finally:
        db.close()
