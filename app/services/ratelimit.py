"""Per-IP fixed-window rate limiting, backed by the existing database.

In-memory limiting is unreliable on serverless — each instance has its own
memory, so counts reset on every cold start and aren't shared across concurrent
instances. Postgres *is* shared, so we count requests per IP per minute there:
cheap, no extra infra, and effective for protecting a public demo.
"""
from __future__ import annotations

from sqlalchemy import text

from app.config import get_settings
from app.database import engine

_table_ready = False


def _ensure_table() -> None:
    """Create the counter table once per process (works even when RUN_DB_INIT=false)."""
    global _table_ready
    if _table_ready:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    ip           text        NOT NULL,
                    window_start timestamptz NOT NULL,
                    count        integer     NOT NULL DEFAULT 0,
                    PRIMARY KEY (ip, window_start)
                )
                """
            )
        )
    _table_ready = True


def check_rate_limit(ip: str) -> bool:
    """Atomically count this request in the current minute; True if under the cap."""
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return True
    _ensure_table()
    with engine.begin() as conn:
        count = conn.execute(
            text(
                """
                INSERT INTO rate_limits (ip, window_start, count)
                VALUES (:ip, date_trunc('minute', now()), 1)
                ON CONFLICT (ip, window_start)
                DO UPDATE SET count = rate_limits.count + 1
                RETURNING count
                """
            ),
            {"ip": ip},
        ).scalar()
    return count <= settings.rate_limit_per_minute
