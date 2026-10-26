"""Explicit migration 004: budget v4 -> owner-only request traces v5.

Default plan is offline, does not load .env. Stop services and back up first.
Frozen migrations 001/002/003 are not changed. No provider calls or budget edits.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

MIGRATION = "004_application_traces"
TABLE = "application_request_traces"


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
                "application_budget_accounts",
                "application_budget_reservations",
                "application_budget_events",
            )
        ):
            raise ValueError("Requires budget v4 application schema")
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
            raise ValueError("Native v5 schema; no blind adoption")
        conn.exec_driver_sql("LOCK TABLE app_users IN SHARE ROW EXCLUSIVE MODE")
        conn.exec_driver_sql("""
            CREATE TABLE application_request_traces (
                id VARCHAR(36) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                knowledge_base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                content_consent BOOLEAN NOT NULL,
                metadata_json TEXT NOT NULL,
                snapshot_json TEXT
            );
            CREATE INDEX ix_application_request_traces_user_id
                ON application_request_traces(user_id);
            CREATE INDEX ix_application_request_traces_expires_at
                ON application_request_traces(expires_at);
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
            "PLAN ONLY: 004 creates request traces; default metadata only, "
            "optional per-request content, 7 day retention. Stop services/back up before apply."
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
