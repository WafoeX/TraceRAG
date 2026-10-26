"""Synthetic recovery safety checks; executed only by opt-in drill."""

import json

from app.database import engine

queries = {
    "guard": "SELECT quarantined FROM application_recovery_state WHERE id=1",
    "sessions": "SELECT count(*) FROM auth_sessions",
    "blocked": "SELECT bool_and(blocked) FROM application_budget_accounts",
    "pending": (
        "SELECT count(*) FROM ingestion_jobs WHERE state IN ('queued','parsing','embedding')"
    ),
    "expired": "SELECT count(*) FROM application_request_traces WHERE expires_at <= now()",
    "expired_feedback": (
        "SELECT count(*) FROM application_request_feedback "
        "WHERE trace_id='22222222-2222-4222-8222-222222222222'"
    ),
}
with engine.connect() as conn:
    print(
        json.dumps({name: conn.exec_driver_sql(sql).scalar_one() for name, sql in queries.items()})
    )
