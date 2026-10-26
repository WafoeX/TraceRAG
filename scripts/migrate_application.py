"""Explicit, transactional v1 -> document-version migration; no model imports.

Default is a plan only. --apply requires an explicit DB URL, never .env.
Stop API and workers and take a backup before applying to a real app database.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

MIGRATION = "001_document_versions"


def upgrade(engine: Engine) -> bool:
    """Return True when applied, False on verified replay. Roll back all on error."""
    # This reviewed Python migration is immutable after deployment. Never execute
    # SQL read from a runtime file or interpolated from a URL/schema argument.
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "SELECT pg_advisory_xact_lock(hashtext(current_database() || current_schema() "
            "|| 'tracerag-application-migration'))"
        )
        if not all(inspect(conn).has_table(t) for t in ("documents", "chunks", "ingestion_jobs")):
            raise ValueError("Requires the P1.1 application schema; no tables changed")
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS application_migrations "
            "(version VARCHAR(80) PRIMARY KEY, sha256 VARCHAR(64) NOT NULL, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        old = conn.scalar(
            text("SELECT sha256 FROM application_migrations WHERE version=:v"), {"v": MIGRATION}
        )
        if old is not None:
            if old != digest:
                raise ValueError(
                    "Applied migration checksum differs; restore the original migration"
                )
            return False
        if "dedup_key" in {col["name"] for col in inspect(conn).get_columns("ingestion_jobs")}:
            raise ValueError(
                "Schema already versioned without a migration ledger; refusing backfill"
            )
        conn.exec_driver_sql(
            "LOCK TABLE documents, chunks, ingestion_jobs IN SHARE ROW EXCLUSIVE MODE"
        )
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS logical_documents (
                id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_versions (
                document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                logical_id INTEGER NOT NULL REFERENCES logical_documents(id) ON DELETE CASCADE,
                number INTEGER NOT NULL, active BOOLEAN NOT NULL,
                filename VARCHAR(512), raw_content BYTEA, file_hash VARCHAR(64),
                created_at TIMESTAMPTZ NOT NULL,
                CONSTRAINT uq_document_version_number UNIQUE(logical_id, number)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_document_version_active
                ON document_versions(logical_id) WHERE active;
            CREATE INDEX IF NOT EXISTS ix_document_versions_logical_id
                ON document_versions(logical_id);
            CREATE TABLE IF NOT EXISTS chunk_locations (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL
            );
            ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS dedup_key VARCHAR(160);
            ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS target_logical_id INTEGER
                REFERENCES logical_documents(id) ON DELETE CASCADE;
            ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS base_document_id INTEGER
                REFERENCES documents(id) ON DELETE CASCADE;
            UPDATE ingestion_jobs SET dedup_key = 'new:' || file_hash WHERE dedup_key IS NULL;
            ALTER TABLE ingestion_jobs ALTER COLUMN dedup_key SET NOT NULL;
            ALTER TABLE ingestion_jobs DROP CONSTRAINT IF EXISTS ingestion_jobs_file_hash_key;
            CREATE UNIQUE INDEX IF NOT EXISTS ingestion_jobs_dedup_key_key
                ON ingestion_jobs(dedup_key);
            CREATE INDEX IF NOT EXISTS ix_ingestion_jobs_target_logical_id
                ON ingestion_jobs(target_logical_id);
            INSERT INTO logical_documents(id, created_at)
                SELECT d.id, d.created_at FROM documents d WHERE NOT EXISTS
                (SELECT 1 FROM document_versions v WHERE v.document_id = d.id)
                ON CONFLICT (id) DO NOTHING;
            INSERT INTO document_versions(document_id, logical_id, number, active, created_at)
                SELECT d.id, d.id, 1, true, d.created_at FROM documents d WHERE NOT EXISTS
                (SELECT 1 FROM document_versions v WHERE v.document_id = d.id);
            SELECT setval(pg_get_serial_sequence('logical_documents', 'id'),
                COALESCE((SELECT max(id) FROM logical_documents), 1),
                EXISTS(SELECT 1 FROM logical_documents));
        """)
        conn.execute(
            text("INSERT INTO application_migrations(version, sha256) VALUES (:v, :h)"),
            {"v": MIGRATION, "h": digest},
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url", help="Explicit application DB, NOT an experiment DB")
    args = parser.parse_args()
    if not args.apply:
        print(
            f"PLAN ONLY: {MIGRATION}. Stop API/workers, back up, then use --apply --database-url."
        )
        print("Preserves document/chunk IDs and embeddings; old raw files/pages remain unknown.")
        return
    if not args.database_url:
        parser.error("--apply requires --database-url; .env is deliberately not used")
    url = args.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine: Engine | None = None
    try:
        engine = create_engine(url)
        print("Migration applied" if upgrade(engine) else "Migration already applied (checksum OK)")
    except Exception:
        # Connection exceptions may contain credentials. Do not echo URLs/errors.
        raise SystemExit(
            "Migration failed or outcome unknown; verify the ledger safely before retrying"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
