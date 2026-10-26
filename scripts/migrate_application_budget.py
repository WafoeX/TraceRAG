"""Explicit migration 003: owner-isolated v3 -> cumulative budget ledger v4.

Plan is offline and does not load .env. Stop API/workers and back up before apply.
No accounts are credited by migration; frozen migrations 001/002 remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

MIGRATION = "003_application_budget"


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
                "auth_sessions",
                "auth_login_windows",
                "ingestion_jobs",
            )
        ) or "knowledge_base_id" not in {
            c["name"] for c in inspect(conn).get_columns("logical_documents")
        }:
            raise ValueError("Requires owner-isolated v3 application schema")
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS application_migrations "
            "(version VARCHAR(80) PRIMARY KEY, sha256 VARCHAR(64) NOT NULL, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        old = conn.exec_driver_sql(
            "SELECT sha256 FROM application_migrations WHERE version=%(v)s", {"v": MIGRATION}
        ).scalar_one_or_none()
        tables = (
            "application_budget_accounts",
            "application_budget_reservations",
            "application_budget_events",
        )
        present = [inspect(conn).has_table(t) for t in tables]
        if old is not None:
            if old != digest or not all(present):
                raise ValueError("Migration checksum/schema differs")
            return False
        if any(present):
            raise ValueError("Native v4 or partial budget schema; no blind adoption")
        conn.exec_driver_sql("LOCK TABLE app_users IN SHARE ROW EXCLUSIVE MODE")
        conn.exec_driver_sql("""
            CREATE TABLE application_budget_accounts (
                key VARCHAR(64) PRIMARY KEY,
                user_id INTEGER UNIQUE REFERENCES app_users(id),
                limit_cny NUMERIC(30,12) NOT NULL,
                spent_cny NUMERIC(30,12) NOT NULL DEFAULT 0,
                held_cny NUMERIC(30,12) NOT NULL DEFAULT 0,
                blocked BOOLEAN NOT NULL DEFAULT false,
                CHECK (limit_cny >= 0 AND spent_cny >= 0 AND held_cny >= 0),
                CHECK (spent_cny + held_cny <= limit_cny),
                CHECK ((key='global' AND user_id IS NULL) OR
                       (user_id IS NOT NULL AND key='user:' || user_id::text))
            );
            CREATE TABLE application_budget_reservations (
                id VARCHAR(36) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES app_users(id),
                role VARCHAR(32) NOT NULL, model VARCHAR(256) NOT NULL,
                profile_json TEXT NOT NULL,
                reserved_cny NUMERIC(30,12) NOT NULL,
                estimated_cny NUMERIC(30,12),
                status VARCHAR(16) NOT NULL, reason VARCHAR(64),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (reserved_cny >= 0 AND (estimated_cny IS NULL OR estimated_cny >= 0)),
                CHECK (status IN ('held','unknown','settled','released')),
                CHECK ((status IN ('held','unknown') AND estimated_cny IS NULL) OR
                       (status IN ('settled','released') AND estimated_cny IS NOT NULL))
            );
            CREATE INDEX ix_application_budget_reservations_user_id
                ON application_budget_reservations(user_id);
            CREATE TABLE application_budget_events (
                id VARCHAR(36) PRIMARY KEY,
                action VARCHAR(32) NOT NULL, scope VARCHAR(64) NOT NULL,
                reservation_id VARCHAR(36) REFERENCES application_budget_reservations(id),
                amount_cny NUMERIC(30,12), reason VARCHAR(512) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
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
            "PLAN ONLY: 003 creates budget ledger; no credit, no provider calls. "
            "Stop services/back up before apply."
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
