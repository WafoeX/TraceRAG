"""Explicit v2 -> owner isolation migration 002. Never edits frozen migration 001.

Default is plan-only. Apply after stopping API/workers and backing up an explicit
application database. Legacy ownership must be supplied; no first-user takeover.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.security import hash_password, normalize_username, verify_password

MIGRATION = "002_owner_isolation"


def upgrade(engine: Engine, owner: str, password: str, library_name: str) -> bool:
    username = normalize_username(owner)
    if not library_name.strip() or len(library_name) > 200:
        raise ValueError("Invalid library name")
    encoded = hash_password(password)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "SELECT pg_advisory_xact_lock(hashtext(current_database() || current_schema() "
            "|| 'tracerag-application-migration'))"
        )
        if not all(
            inspect(conn).has_table(t)
            for t in ("documents", "logical_documents", "document_versions", "ingestion_jobs")
        ):
            raise ValueError("Requires the v2 application schema")
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
                raise ValueError("Applied migration checksum differs")
            return False
        if "knowledge_base_id" in {
            c["name"] for c in inspect(conn).get_columns("logical_documents")
        }:
            raise ValueError("Native v3 schema does not require legacy assignment")
        conn.exec_driver_sql(
            "LOCK TABLE documents, chunks, logical_documents, document_versions, "
            "ingestion_jobs IN SHARE ROW EXCLUSIVE MODE"
        )
        if conn.scalar(
            text(
                "SELECT EXISTS(SELECT 1 FROM documents d WHERE NOT EXISTS "
                "(SELECT 1 FROM document_versions v WHERE v.document_id=d.id))"
            )
        ):
            raise ValueError("Unversioned documents require explicit v1 adoption first")
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS app_users (
                id SERIAL PRIMARY KEY, username VARCHAR(64) NOT NULL UNIQUE,
                password_hash VARCHAR(256) NOT NULL, created_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id SERIAL PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES app_users(id),
                name VARCHAR(200) NOT NULL, created_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_knowledge_bases_owner_id ON knowledge_bases(owner_id);
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash VARCHAR(64) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                csrf_token VARCHAR(64) NOT NULL, expires_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_auth_sessions_expires_at ON auth_sessions(expires_at);
            CREATE TABLE IF NOT EXISTS auth_login_windows (
                key VARCHAR(100) PRIMARY KEY, attempts INTEGER NOT NULL
            );
            ALTER TABLE logical_documents ADD COLUMN knowledge_base_id INTEGER
                REFERENCES knowledge_bases(id);
            ALTER TABLE ingestion_jobs ADD COLUMN knowledge_base_id INTEGER
                REFERENCES knowledge_bases(id);
            CREATE INDEX ix_logical_documents_knowledge_base_id
                ON logical_documents(knowledge_base_id);
            CREATE INDEX ix_ingestion_jobs_knowledge_base_id ON ingestion_jobs(knowledge_base_id);
        """)
        row = conn.exec_driver_sql(
            "SELECT id,password_hash FROM app_users WHERE username=%(u)s", {"u": username}
        ).first()
        if row is None:
            user_id = conn.exec_driver_sql(
                "INSERT INTO app_users(username,password_hash,created_at) "
                "VALUES (%(u)s,%(p)s,now()) RETURNING id",
                {"u": username, "p": encoded},
            ).scalar_one()
        else:
            if not verify_password(password, row.password_hash):
                raise ValueError("Existing owner credentials do not match")
            user_id = row.id
        kb_id = conn.exec_driver_sql(
            "INSERT INTO knowledge_bases(owner_id,name,created_at) "
            "VALUES (%(u)s,%(n)s,now()) RETURNING id",
            {"u": user_id, "n": library_name.strip()},
        ).scalar_one()
        conn.exec_driver_sql(
            "UPDATE logical_documents SET knowledge_base_id=%(kb)s WHERE knowledge_base_id IS NULL",
            {"kb": kb_id},
        )
        conn.exec_driver_sql(
            "UPDATE ingestion_jobs SET knowledge_base_id=%(kb)s, "
            "dedup_key='kb:' || CAST(%(kb)s AS TEXT) || ':' || dedup_key "
            "WHERE knowledge_base_id IS NULL",
            {"kb": kb_id},
        )
        conn.exec_driver_sql(
            "INSERT INTO application_migrations(version,sha256) VALUES (%(v)s,%(h)s)",
            {"v": MIGRATION, "h": digest},
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--owner")
    parser.add_argument("--library-name")
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            "PLAN ONLY: 002 owner isolation; stop services, back up, "
            "explicitly choose owner/library."
        )
        return
    if not all((args.database_url, args.owner, args.library_name)):
        parser.error("--apply requires --database-url, --owner and --library-name")
    engine = None
    try:
        password = (
            sys.stdin.readline().rstrip("\r\n")
            if args.password_stdin
            else getpass.getpass("Owner password: ")
        )
        engine = create_engine(
            args.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
        print(
            "Migration applied"
            if upgrade(engine, args.owner, password, args.library_name)
            else "Already applied (checksum OK)"
        )
    except Exception:
        raise SystemExit(
            "Migration failed or outcome unknown; safely verify ledger before retrying"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
