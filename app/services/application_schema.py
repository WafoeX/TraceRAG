"""Database-only readiness/restore guard. No settings, providers, or .env imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pgvector.sqlalchemy import Vector  # registers vector reflection with PostgreSQL
from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 7
MIGRATIONS = (
    ("001_document_versions", "migrate_application"),
    ("002_owner_isolation", "migrate_application_auth"),
    ("003_application_budget", "migrate_application_budget"),
    ("004_application_traces", "migrate_application_traces"),
    ("005_application_feedback", "migrate_application_feedback"),
    ("006_recovery_guard", "migrate_application_recovery"),
)
# Vector import intentionally registers reflection, without importing runtime models.
_VECTOR_TYPE = Vector


class DatabaseNotReady(RuntimeError):
    pass


def check_guard(db: Session) -> None:
    """Used on login/auth/worker entry too, not just a cosmetic health indicator."""
    try:
        blocked = (
            db.connection()
            .exec_driver_sql("SELECT quarantined FROM application_recovery_state WHERE id=1")
            .scalar_one_or_none()
        )
    except Exception:
        raise DatabaseNotReady("Application database unavailable") from None
    if blocked:
        raise DatabaseNotReady("Application recovery review required")


def verify_schema(conn: Connection, embedding_dim: int, *, allow_quarantine: bool = False) -> None:
    inspector = inspect(conn)
    try:
        manifest = json.loads((ROOT / "deploy/schema-manifest.json").read_text())
    except Exception:
        raise DatabaseNotReady("Schema manifest unavailable") from None
    if not conn.exec_driver_sql(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')"
    ).scalar_one():
        raise DatabaseNotReady("Vector extension missing")
    tables = set(inspector.get_table_names())
    if not set(manifest) <= tables:
        raise DatabaseNotReady("Explicit migration required")
    for name, spec in manifest.items():
        columns = {c["name"]: c for c in inspector.get_columns(name)}
        for field, expected in spec["columns"].items():
            column = columns.get(field)
            kind = expected["type"].replace("__EMBEDDING_DIM__", str(embedding_dim))
            if (
                column is None
                or str(column["type"].compile(dialect=conn.dialect)) != kind
                or column["nullable"] != expected["nullable"]
            ):
                raise DatabaseNotReady("Schema column mismatch")
        if inspector.get_pk_constraint(name)["constrained_columns"] != spec["pk"]:
            raise DatabaseNotReady("Schema primary key mismatch")
        found = inspector.get_foreign_keys(name)
        for fk in spec["fks"]:
            if not any(
                f["constrained_columns"] == fk["columns"]
                and f["referred_table"] == fk["table"]
                and f["referred_columns"] == fk["remote"]
                and f.get("options", {}).get("ondelete") == fk["ondelete"]
                for f in found
            ):
                raise DatabaseNotReady("Schema foreign key mismatch")
    indexes = {i["name"]: i for i in inspector.get_indexes("chunks")}
    if "chunks_content_fts" not in indexes or (
        embedding_dim <= 2000 and "chunks_embedding_hnsw" not in indexes
    ):
        raise DatabaseNotReady("Required search index missing")
    active = {i["name"]: i for i in inspector.get_indexes("document_versions")}.get(
        "uq_document_version_active"
    )
    if (
        not active
        or not active["unique"]
        or active["column_names"] != ["logical_id"]
        or str(active.get("dialect_options", {}).get("postgresql_where", ""))
        .replace("(", "")
        .replace(")", "")
        .strip()
        != "active"
    ):
        raise DatabaseNotReady("Version activation index missing")
    for table, columns in (
        ("app_users", ["username"]),
        ("ingestion_jobs", ["dedup_key"]),
        ("application_budget_accounts", ["user_id"]),
        ("document_versions", ["logical_id", "number"]),
    ):
        unique = [u["column_names"] for u in inspector.get_unique_constraints(table)]
        unique += [i["column_names"] for i in inspector.get_indexes(table) if i["unique"]]
        if columns not in unique:
            raise DatabaseNotReady("Required unique constraint missing")
    verify_ledger(conn)
    blocked = conn.exec_driver_sql(
        "SELECT quarantined FROM application_recovery_state WHERE id=1"
    ).scalar_one_or_none()
    if blocked and not allow_quarantine:
        raise DatabaseNotReady("Application recovery review required")


def verify_ledger(conn: Connection) -> None:
    if inspect(conn).has_table("application_migrations"):
        known = {key: ROOT / "scripts" / (module + ".py") for key, module in MIGRATIONS}
        for version, digest in conn.exec_driver_sql(
            "SELECT version,sha256 FROM application_migrations"
        ):
            if (
                version not in known
                or hashlib.sha256(known[version].read_bytes()).hexdigest() != digest
            ):
                raise DatabaseNotReady("Migration checksum mismatch")


def ready(engine: Engine, embedding_dim: int) -> bool:
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SET LOCAL statement_timeout='1500ms'")
            verify_schema(conn, embedding_dim)
        return True
    except Exception:
        return False
