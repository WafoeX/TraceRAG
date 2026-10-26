"""Explicit offline-plan/init/migration-chain/check/recovery review operations.

Never reads .env or imports application settings/models/providers. Mutations need
an explicit URL, database-name confirmation, --apply and services-stopped consent.
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import sys

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Connection, Engine

from app.services.application_schema import MIGRATIONS, ROOT, verify_ledger, verify_schema


def confirm(conn: Connection, expected: str) -> None:
    if conn.exec_driver_sql("SELECT current_database()").scalar_one() != expected:
        raise ValueError("Database name confirmation mismatch")
    if conn.exec_driver_sql("SELECT current_schema()").scalar_one() != "public":
        raise ValueError("Deployment tools require public schema")


def indexes(conn: Connection, dim: int) -> None:
    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS chunks_content_fts ON chunks "
        "USING gin (to_tsvector('english', content))"
    )
    if dim <= 2000:
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )


def initialize(engine: Engine, expected: str, dim: int) -> None:
    if type(dim) is not int or not 1 <= dim <= 65535:
        raise ValueError("Invalid embedding dimension")
    with engine.begin() as conn:
        confirm(conn, expected)
        conn.exec_driver_sql(
            "SELECT pg_advisory_xact_lock(hashtext(current_database() || current_schema() "
            "|| 'tracerag-application-migration'))"
        )
        inspector = inspect(conn)
        occupied = conn.exec_driver_sql(
            "SELECT EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n "
            "ON n.oid=c.relnamespace WHERE n.nspname='public' "
            "AND c.relkind IN ('r','p','v','m','S','f'))"
        ).scalar_one()
        if occupied or any(
            s not in {"public", "information_schema"} and not s.startswith("pg_")
            for s in inspector.get_schema_names()
        ):
            raise ValueError("Bootstrap requires a new empty database")
        # Only the packaged reviewed SQL, never a caller-supplied file/SQL/schema.
        sql = (
            (ROOT / "deploy/application-schema.sql")
            .read_text()
            .replace("__EMBEDDING_DIM__", str(dim))
        )
        conn.exec_driver_sql(sql)
        indexes(conn, dim)
        verify_schema(conn, dim)


def version(conn: Connection) -> int:
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if not {"documents", "chunks", "ingestion_jobs"} <= tables:
        raise ValueError("Not a supported application database")
    if "dedup_key" not in {c["name"] for c in inspector.get_columns("ingestion_jobs")}:
        return 1
    if not {"logical_documents", "document_versions", "chunk_locations"} <= tables:
        raise ValueError("Partial version schema")
    if "knowledge_base_id" not in {c["name"] for c in inspector.get_columns("logical_documents")}:
        return 2
    for target, required in (
        (
            3,
            {
                "application_budget_accounts",
                "application_budget_reservations",
                "application_budget_events",
            },
        ),
        (4, {"application_request_traces"}),
        (5, {"application_request_feedback"}),
        (6, {"application_recovery_state"}),
    ):
        if not tables.intersection(required):
            return target
        if not required <= tables:
            raise ValueError("Partial application schema")
    return 7


def migrate(
    engine: Engine,
    expected: str,
    source: int,
    dim: int,
    owner: str | None,
    password: str | None,
    library: str | None,
) -> None:
    with engine.connect() as conn:
        confirm(conn, expected)
        if version(conn) != source:
            raise ValueError("Explicit source version differs; inspect before retry")
        # Check every existing ledger entry before committing any next migration.
        verify_ledger(conn)
    for number in range(source - 1, len(MIGRATIONS)):
        _, module_name = MIGRATIONS[number]
        module = importlib.import_module("scripts." + module_name)
        if number == 1:
            if not owner or password is None or not library:
                raise ValueError("Explicit legacy owner required")
            module.upgrade(engine, owner, password, library)
        else:
            module.upgrade(engine)
    with engine.begin() as conn:
        indexes(conn, dim)
        verify_schema(conn, dim)


def release(engine: Engine, expected: str, dim: int) -> None:
    with engine.begin() as conn:
        confirm(conn, expected)
        verify_schema(conn, dim, allow_quarantine=True)
        conn.exec_driver_sql(
            "DELETE FROM application_request_traces WHERE expires_at <= clock_timestamp()"
        )
        conn.exec_driver_sql(
            "UPDATE application_recovery_state SET quarantined=false, "
            "reviewed_at=clock_timestamp() WHERE id=1 AND quarantined"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        nargs="?",
        default="plan",
        choices=["plan", "init", "migrate", "check", "release"],
    )
    parser.add_argument("--database-url")
    parser.add_argument("--confirm-database")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--from-version", type=int, choices=range(1, 8))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-services-stopped", action="store_true")
    parser.add_argument("--confirm-privacy-reviewed", action="store_true")
    parser.add_argument("--confirm-billing-reviewed", action="store_true")
    parser.add_argument("--owner")
    parser.add_argument("--library-name")
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()
    if args.operation == "plan" or (args.operation != "check" and not args.apply):
        print(
            "PLAN ONLY: native v7 bootstrap or explicit chain "
            "001 -> 002 -> 003 -> 004 -> 005 -> 006; stop services/back up. "
            "Restore remains quarantined until manual privacy/billing review."
        )
        return
    if not args.database_url or not 1 <= args.embedding_dim <= 65535:
        parser.error("Explicit --database-url and valid dimension required")
    if args.operation != "check" and not (args.confirm_database and args.confirm_services_stopped):
        parser.error("Mutations require --confirm-database and --confirm-services-stopped")
    if args.operation == "release" and not (
        args.confirm_privacy_reviewed and args.confirm_billing_reviewed
    ):
        parser.error("Release requires privacy and external-billing review confirmations")
    if args.operation == "migrate" and args.from_version is None:
        parser.error("Migration requires explicit --from-version")
    engine = None
    try:
        engine = create_engine(
            args.database_url.replace("postgresql://", "postgresql+psycopg://", 1),
            connect_args={"connect_timeout": 3},
            pool_timeout=3,
        )
        if args.operation == "init":
            initialize(engine, args.confirm_database, args.embedding_dim)
        elif args.operation == "migrate":
            password = None
            if args.from_version <= 2:
                if not args.owner or not args.library_name:
                    raise ValueError("Explicit legacy owner/library required")
                password = (
                    sys.stdin.readline().rstrip("\r\n")
                    if args.password_stdin
                    else getpass.getpass("Legacy owner password: ")
                )
            migrate(
                engine,
                args.confirm_database,
                args.from_version,
                args.embedding_dim,
                args.owner,
                password,
                args.library_name,
            )
        elif args.operation == "release":
            release(engine, args.confirm_database, args.embedding_dim)
        else:
            with engine.connect() as conn:
                conn.exec_driver_sql("SET LOCAL statement_timeout='1500ms'")
                verify_schema(conn, args.embedding_dim)
        print(
            "Database operation complete; recovery release never unblocks budgets or retries tasks"
        )
    except Exception:
        raise SystemExit(
            "Database operation failed or outcome unknown; inspect safely before retrying"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
