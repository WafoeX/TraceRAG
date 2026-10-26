"""Explicit migration 005: traces v5 -> fixed-label feedback v6.

Plan is offline and does not load .env. Stop services/back up before apply.
No feedback backfill, provider calls, credits or changes to migrations 001-004.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

MIGRATION = "005_application_feedback"
TABLE = "application_request_feedback"


def upgrade(engine: Engine) -> bool:
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "SELECT pg_advisory_xact_lock(hashtext(current_database() || current_schema() "
            "|| 'tracerag-application-migration'))"
        )
        if not all(
            inspect(conn).has_table(t)
            for t in (
                "app_users",
                "knowledge_bases",
                "application_request_traces",
                "application_budget_accounts",
                "application_budget_reservations",
                "application_budget_events",
            )
        ):
            raise ValueError("Requires traces v5 application schema")
        if not {"id", "user_id", "expires_at", "snapshot_json", "metadata_json"} <= {
            c["name"] for c in inspect(conn).get_columns("application_request_traces")
        }:
            raise ValueError("Requires complete v5 traces")
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS application_migrations "
            "(version VARCHAR(80) PRIMARY KEY, sha256 VARCHAR(64) NOT NULL, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        old = conn.exec_driver_sql(
            "SELECT sha256 FROM application_migrations WHERE version=%(v)s", {"v": MIGRATION}
        ).scalar_one_or_none()
        present = inspect(conn).has_table(TABLE)
        if old is not None:
            if old != digest or not present:
                raise ValueError("Migration checksum/schema differs")
            return False
        if present:
            raise ValueError("Native v6 schema; no blind adoption")
        conn.exec_driver_sql("LOCK TABLE app_users IN SHARE ROW EXCLUSIVE MODE")
        conn.exec_driver_sql("""
            CREATE TABLE application_request_feedback (
                trace_id VARCHAR(36) PRIMARY KEY
                    REFERENCES application_request_traces(id) ON DELETE CASCADE,
                unhelpful BOOLEAN NOT NULL,
                unsupported_citation BOOLEAN NOT NULL,
                outdated BOOLEAN NOT NULL,
                citation_number INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (unhelpful OR unsupported_citation OR outdated),
                CHECK (citation_number IS NULL OR citation_number BETWEEN 1 AND 20)
            )
        """)
        conn.exec_driver_sql(
            "INSERT INTO application_migrations(version,sha256) VALUES (%(v)s,%(h)s)",
            {"v": MIGRATION, "h": digest},
        )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url")
    args = parser.parse_args()
    if not args.apply:
        print(
            "PLAN ONLY: 005 creates fixed-label feedback, expiry follows trace; "
            "no content backfill or provider calls. Stop services/back up before apply."
        )
        return
    if not args.database_url:
        parser.error("--apply requires --database-url")
    engine = None
    try:
        engine = create_engine(
            args.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
        print("Migration applied" if upgrade(engine) else "Already applied (checksum OK)")
    except Exception:
        raise SystemExit(
            "Migration failed or outcome unknown; verify ledger safely before retrying"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
