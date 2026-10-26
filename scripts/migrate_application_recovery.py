"""Explicit 006: v6 -> v7 restore quarantine guard; plan never reads .env."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

MIGRATION = "006_recovery_guard"


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
                "application_request_traces",
                "application_request_feedback",
                "app_users",
                "application_budget_accounts",
                "ingestion_jobs",
            )
        ):
            raise ValueError("Requires v6 application schema")
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS application_migrations "
            "(version VARCHAR(80) PRIMARY KEY, sha256 VARCHAR(64) NOT NULL, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        old = conn.exec_driver_sql(
            "SELECT sha256 FROM application_migrations WHERE version=%(v)s", {"v": MIGRATION}
        ).scalar_one_or_none()
        present = inspect(conn).has_table("application_recovery_state")
        if old is not None:
            if old != digest or not present:
                raise ValueError("Migration checksum/schema differs")
            return False
        if present:
            raise ValueError("Native v7 schema; no blind adoption")
        conn.exec_driver_sql(
            "CREATE TABLE application_recovery_state ("
            "id INTEGER PRIMARY KEY CHECK(id=1), "
            "quarantined BOOLEAN NOT NULL DEFAULT false, reviewed_at TIMESTAMPTZ)"
        )
        conn.exec_driver_sql(
            "INSERT INTO application_recovery_state(id,quarantined) VALUES (1,false)"
        )
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
        print("PLAN ONLY: 006 adds recovery guard; stop services/back up before apply")
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
        raise SystemExit("Migration failed or outcome unknown; verify ledger safely") from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
