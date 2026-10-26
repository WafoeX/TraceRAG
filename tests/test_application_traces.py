"""Real auth/SQL trace privacy in a disposable PG schema, fake models only."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import inspect

from app.models import RequestTrace
from app.routers import query as query_module
from app.services import tracing
from app.services.resilience import ProviderUnavailable
from app.services.retrieval import RetrievedChunk
from app.services.search import SearchResult
from scripts.manage_application_traces import purge_expired
from scripts.migrate_application_traces import upgrade

from .test_application_budget import bank as _bank_fixture
from .test_application_budget import config as _config_fixture
from .test_application_budget import raw_rows
from .test_application_budget import sdk as _sdk_fixture
from .test_application_lifecycle import database as _database_fixture
from .test_application_resilience import completion
from .test_auth_isolation import clients as _clients_fixture
from .test_auth_isolation import finish, submit

database = _database_fixture
clients = _clients_fixture
config = _config_fixture
bank = _bank_fixture
sdk = _sdk_fixture
pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in disposable DB"),
]


@pytest.fixture
def prepared(clients, database, monkeypatch):
    alice, bob, a, b = clients
    doc = finish(database, submit(alice, a, "SECRET DOCUMENT"))
    chunk = RetrievedChunk(1, doc, "SECRET TITLE", 0, "SECRET DOCUMENT", 0.95)
    monkeypatch.setattr(query_module, "search", lambda *args: SearchResult([chunk], 0.95))
    monkeypatch.setattr(query_module, "evidence_metadata", lambda *args: {})
    monkeypatch.setattr(query_module, "generate_answer", lambda *args: "SECRET ANSWER [1]")
    monkeypatch.setattr(query_module, "stream_answer", lambda *args: iter(["SECRET ANSWER [1]"]))
    return alice, bob, a, b, doc


def ask(alice, kb, consent=False, path="/query/evidence"):
    result = alice.post(
        f"{path}?knowledge_base_id={kb}",
        json={"question": "SECRET QUESTION", "retain_trace_content": consent},
        headers={"X-Trace-ID": "forged"},
    )
    assert result.status_code == 200, result.text
    identifier = result.headers["x-trace-id"]
    assert identifier != "forged"
    detail = alice.get("/traces/" + identifier)
    assert detail.status_code == 200, detail.text
    assert detail.headers["cache-control"] == "no-store"
    return identifier, detail.json()


@pytest.mark.parametrize("path", ["/query", "/query/evidence", "/query/stream"])
@pytest.mark.parametrize("consent", [False, True])
def test_query_trace_compatibility_and_content(prepared, database, path, consent):
    alice, bob, a, _, _ = prepared
    identifier, detail = ask(alice, a, consent, path)
    assert detail["content_consent"] == consent
    assert (detail["snapshot"] is not None) == consent
    assert "SECRET" not in json.dumps(detail["metadata"])
    assert detail["metadata"]["generation_status"] == "completed"
    assert detail["metadata"]["contexts"][0]["number"] == 1
    assert len(detail["metadata"]["stages"]) >= 2
    assert detail["metadata"]["calls"] == []  # fake generator, not fabricated zero usage
    if consent:
        assert detail["snapshot"]["contexts"][0]["content"] == "SECRET DOCUMENT"
        assert detail["snapshot"]["question"] == "SECRET QUESTION"
    assert bob.get("/traces/" + identifier).status_code == 404
    assert bob.delete("/traces/" + identifier).status_code == 404
    assert bob.get("/traces").json() == []
    assert len(alice.get("/traces").json()) == 1
    assert alice.get("/traces/invalid").status_code == 404
    token = alice.headers.pop("X-CSRF-Token")
    assert alice.delete("/traces/" + identifier).status_code == 403
    alice.headers["X-CSRF-Token"] = token
    assert alice.delete("/traces/" + identifier).status_code == 204
    assert alice.get("/traces/" + identifier).status_code == 404


def test_expiry_hidden_and_purged_without_budget_changes(prepared, database):
    alice, _, a, _, _ = prepared
    identifier, _ = ask(alice, a, True)
    with database() as db:
        row = db.get(RequestTrace, identifier)
        assert row is not None
        assert timedelta(days=7) == row.expires_at - row.created_at
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    assert alice.get("/traces/" + identifier).status_code == 404
    assert alice.get("/traces").json() == []
    assert purge_expired(database.kw["bind"]) == 1
    assert purge_expired(database.kw["bind"]) == 0


def test_document_deletion_erases_snapshot_and_late_persistence(prepared, database):
    alice, _, a, _, doc = prepared
    identifier, _ = ask(alice, a, True)
    with database() as db:
        row = db.get(RequestTrace, identifier)
        assert row is not None
        owner_id = row.user_id
    assert alice.delete(f"/documents/{doc}?knowledge_base_id={a}").status_code == 204
    assert alice.get("/traces/" + identifier).json()["snapshot"] is None
    late = tracing.TraceState(
        user_id=owner_id, library_id=a, consent=True, snapshot={"question": "SECRET"}
    )
    late.data["contexts"] = [{"document_id": doc}]
    tracing.persist(late)
    with database() as db:
        saved = db.get(RequestTrace, late.id)
        assert saved is not None and saved.snapshot_json is None


def test_generation_failure_and_gate_are_traced(prepared, monkeypatch):
    alice, _, a, _, _ = prepared

    def fail(*args):
        raise ProviderUnavailable("provider_timeout")

    monkeypatch.setattr(query_module, "generate_answer", fail)
    _, detail = ask(alice, a)
    assert detail["metadata"]["generation_status"] == "failed"
    assert "generation_provider_timeout" in detail["metadata"]["warnings"]
    monkeypatch.setattr(query_module, "search", lambda *args: SearchResult([], 0))
    _, detail = ask(alice, a)
    assert detail["metadata"]["generation_status"] == "skipped"
    assert detail["metadata"]["contexts"] == []


def test_snapshot_limit_not_silent_truncation(prepared, monkeypatch):
    alice, _, a, _, _ = prepared
    monkeypatch.setattr(tracing, "MAX_SNAPSHOT_BYTES", 8)
    _, detail = ask(alice, a, True)
    assert detail["snapshot"] is None
    assert "trace_snapshot_too_large" in detail["metadata"]["warnings"]


def make_v4(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE application_request_feedback")
        conn.exec_driver_sql("DROP TABLE application_request_traces")


def test_migration_concurrent_idempotent_checksum(database):
    engine = database.kw["bind"]
    make_v4(engine)
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: upgrade(engine), range(2))) == [False, True]
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE application_migrations SET sha256='tampered' "
            "WHERE version='004_application_traces'"
        )
    with pytest.raises(ValueError, match="checksum"):
        upgrade(engine)


def test_native_schema_not_adopted(database):
    with pytest.raises(ValueError, match="Native"):
        upgrade(database.kw["bind"])


def test_init_v4_requires_explicit_migration(database, monkeypatch):
    from app import database as module

    engine = database.kw["bind"]
    make_v4(engine)
    monkeypatch.setattr(module, "engine", engine)
    with pytest.raises(RuntimeError, match="v4"):
        module.init_db()
    assert not inspect(engine).has_table("application_request_traces")


def test_migration_ddl_failure_rolls_back(database):
    from sqlalchemy import event

    engine = database.kw["bind"]
    make_v4(engine)

    def fail(conn, cursor, statement, *args):
        if "CREATE TABLE application_request_traces" in statement:
            raise RuntimeError("injected")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            upgrade(engine)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    assert not inspect(engine).has_table("application_request_traces")
    assert not inspect(engine).has_table("application_migrations")


@pytest.mark.parametrize("consent", [False, True])
def test_real_sdk_retry_trace_matches_durable_ledger(bank, sdk, consent):
    from app.services import generation
    from app.services.resilience import application_calls

    engine, owner, _ = bank
    calls, outcomes = sdk
    outcomes.extend(["timeout", completion()])
    state = tracing.TraceState(snapshot={"question": "SECRET question"} if consent else None)
    token = tracing._state.set(state)
    try:
        with application_calls(user_id=owner):
            assert generation.generate_answer("SECRET question", [])
    finally:
        tracing._state.reset(token)
    events = state.data["calls"]
    rows = raw_rows(engine)
    assert len(calls) == len(events) == len(rows) == 2
    assert [e["status"] for e in events] == ["unknown", "settled"]
    assert events[0]["usage"] is None and events[0]["estimated_cny"] is None
    assert events[1]["usage"]["total_tokens"] > 0
    assert all(e["reservation_id"] == row["id"] for e, row in zip(events, rows, strict=True))
    assert events[1]["estimated_cny"] == str(rows[1]["estimated_cny"])
    assert "SECRET" not in json.dumps(state.data)
    assert "budget_charge_unknown" in state.data["warnings"]
    if consent:
        assert state.snapshot is not None
        assert "SECRET question" in state.snapshot["generation_messages"][1]["content"]
        assert state.snapshot["generation_messages"][0]["role"] == "system"
    else:
        assert state.snapshot is None


def test_retrieval_failure_traced_without_raw_exception(prepared, monkeypatch):
    alice, _, kb, _, _ = prepared

    def fail(*args):
        raise ProviderUnavailable("provider_timeout")

    monkeypatch.setattr(query_module, "search", fail)
    result = alice.post(
        f"/query/evidence?knowledge_base_id={kb}", json={"question": "SECRET QUESTION"}
    )
    assert result.status_code == 503
    detail = alice.get("/traces/" + result.headers["x-trace-id"]).json()
    assert detail["metadata"]["http_status"] == 503
    assert detail["metadata"]["stages"][0]["status"] == "failed"
    assert "SECRET" not in json.dumps(detail)


def test_concurrent_snapshot_persistence_and_document_erasure(prepared, database):
    alice, _, kb, _, doc = prepared
    identifier, _ = ask(alice, kb, True)
    with database() as db:
        saved = db.get(RequestTrace, identifier)
        assert saved is not None
        owner = saved.user_id
    states = [
        tracing.TraceState(
            user_id=owner, library_id=kb, consent=True, snapshot={"question": "SECRET"}
        )
        for _ in range(6)
    ]
    for state in states:
        state.data["contexts"] = [{"document_id": doc}]
    with ThreadPoolExecutor(7) as pool:
        pending = [pool.submit(tracing.persist, state) for state in states]
        deletion = pool.submit(alice.delete, f"/documents/{doc}?knowledge_base_id={kb}")
        for future in pending:
            future.result()
        assert deletion.result().status_code == 204
    with database() as db:
        assert all(db.get(RequestTrace, state.id).snapshot_json is None for state in states)
