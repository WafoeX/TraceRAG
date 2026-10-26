"""Explicit expiry cleanup; no .env loading, provider calls, or budget changes.

Expired rows are hidden immediately by the API; physical deletion needs this CLI.
Database backups and provider-side retention are separate operator obligations.
"""

from __future__ import annotations

import argparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def purge_expired(engine: Engine) -> int:
    with engine.begin() as conn:
        result = conn.exec_driver_sql(
            "DELETE FROM application_request_traces WHERE expires_at <= now()"
        )
        return result.rowcount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--database-url")
    args = parser.parse_args()
    if not args.apply:
        print("PLAN ONLY: delete expired request traces only; budget ledger unchanged")
        return
    if not args.database_url:
        parser.error("--apply requires --database-url")
    engine = None
    try:
        engine = create_engine(
            args.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
        print(f"Expired traces deleted: {purge_expired(engine)}")
    except Exception:
        raise SystemExit("Trace cleanup failed or outcome unknown; verify safely") from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
