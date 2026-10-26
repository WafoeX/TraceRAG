"""Opt-in disposable PostgreSQL admission checks, no provider requests or DDL upgrade."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app import database as database_module
from app.config import get_settings
from app.models import LoginWindow
from app.services import application_limits as limits
from app.services.resilience import ProviderUnavailable

from .test_application_lifecycle import TEST_URL
from .test_application_lifecycle import database as _database_fixture
from .test_auth_isolation import clients as _clients_fixture

# Both fixtures use the fixed disposable URL; never developer DATABASE_URL.
database = _database_fixture
clients = _clients_fixture
pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in isolated DB"),
]


@pytest.fixture
def admission(database, monkeypatch):
    monkeypatch.setattr(database_module, "SessionLocal", database)
    monkeypatch.setattr(get_settings(), "rate_limit_enabled", True)
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 100)
    monkeypatch.setattr(get_settings(), "account_concurrency", 1)
    monkeypatch.setattr(get_settings(), "application_concurrency", 1)
    return database


def test_account_concurrency_spans_connections_and_releases_on_close(admission):
    held = limits.admit_account(1)
    try:
        with pytest.raises(HTTPException) as exc:
            limits.admit_account(1)
        assert exc.value.status_code == 429
        other = limits.admit_account(2)
        other.close()
    finally:
        held.close()
    limits.admit_account(1).close()


def test_atomic_rate_counter_has_no_parallel_overshoot(admission, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 3)
    monkeypatch.setattr(get_settings(), "account_concurrency", 16)

    def attempt(_):
        try:
            limits.admit_account(1).close()
            return True
        except HTTPException as exc:
            assert exc.status_code == 429
            return False

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sum(workers.map(attempt, range(12))) == 3
    with admission() as db:
        row = db.scalar(select(LoginWindow).where(LoginWindow.key.like("rag:1:%")))
        assert row is not None and row.attempts == 3
    limits.admit_account(2).close()  # different account, not a shared quota


def test_provider_global_slot_releases_after_exception(admission):
    with pytest.raises(ValueError), limits.provider_capacity():
        with pytest.raises(ProviderUnavailable, match="provider_busy"), limits.provider_capacity():
            pytest.fail("global slot bypassed")
        raise ValueError("synthetic")
    with limits.provider_capacity():
        pass


def _child_try_slot():
    # Separate interpreter, no inherited connections or app/.env import.
    script = """
import hashlib, sys
from sqlalchemy import create_engine
key = int.from_bytes(hashlib.sha256(b'tracerag:p2:provider:0').digest()[:8], 'big', signed=True)
engine = create_engine(sys.argv[1])
with engine.begin() as conn:
    result = conn.exec_driver_sql('SELECT pg_try_advisory_xact_lock(%(key)s)', {'key': key})
    print(result.scalar_one())
engine.dispose()
"""
    return subprocess.run(
        [sys.executable, "-c", script, TEST_URL],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def test_slot_cannot_be_bypassed_by_another_process(admission):
    with limits.provider_capacity():
        assert _child_try_slot() == "False"
    assert _child_try_slot() == "True"


def test_authenticated_rate_limit_ignores_forwarded_headers_and_library_switch(
    clients, admission, monkeypatch
):
    alice, bob, a, b = clients
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 1)
    first = alice.post(
        f"/documents/jobs?knowledge_base_id={a}", json={"title": "one", "content": "first"}
    )
    assert first.status_code == 202
    second_library = alice.post("/knowledge-bases", json={"name": "other"}).json()["id"]
    second = alice.post(
        f"/documents/jobs?knowledge_base_id={second_library}",
        json={"title": "two", "content": "second"},
        headers={"x-forwarded-for": "198.51.100.29"},
    )
    assert second.status_code == 429 and second.headers["retry-after"] == "60"
    assert second.headers["cache-control"] == "no-store"
    assert (
        bob.post(
            f"/documents/jobs?knowledge_base_id={b}", json={"title": "three", "content": "third"}
        ).status_code
        == 202
    )
    assert alice.get(f"/documents/jobs?knowledge_base_id={a}").status_code == 200


def test_admission_error_closes_connection_and_fails_closed(clients, admission, monkeypatch):
    alice, _, a, _ = clients
    monkeypatch.setattr(
        limits, "_slot", lambda *args: (_ for _ in ()).throw(RuntimeError("private detail"))
    )
    response = alice.post(f"/query/evidence?knowledge_base_id={a}", json={"question": "q"})
    assert response.status_code == 503
    assert "private detail" not in response.text


def test_provider_limiter_outage_does_not_invoke_model(admission, monkeypatch):
    from unittest.mock import Mock

    from app.services.resilience import application_calls, call_provider

    invoke = Mock()
    monkeypatch.setattr(
        limits, "_slot", lambda *args: (_ for _ in ()).throw(RuntimeError("private"))
    )
    with (
        application_calls(),
        pytest.raises(ProviderUnavailable, match="provider_limiter_unavailable"),
    ):
        call_provider(invoke)
    invoke.assert_not_called()


def test_stream_holds_account_slot_and_releases_after_final_frame(clients, admission, monkeypatch):
    import threading

    from app.routers import query
    from app.services.retrieval import RetrievedChunk
    from app.services.search import SearchResult

    alice, _, a, _ = clients
    entered, release = threading.Event(), threading.Event()
    chunk = RetrievedChunk(1, 1, "fake", 0, "fake evidence", 0.8)
    monkeypatch.setattr(query, "search", lambda *args: SearchResult([chunk], 0.8))

    def streamed(*args):
        entered.set()
        assert release.wait(10)
        yield "answer [1]"

    monkeypatch.setattr(query, "stream_answer", streamed)
    path = f"/query/stream?knowledge_base_id={a}"
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(alice.post, path, json={"question": "q"})
        try:
            assert entered.wait(10)
            blocked = alice.post(path, json={"question": "q"})
            assert blocked.status_code == 429
        finally:
            release.set()
        response = pending.result(timeout=10)
        assert response.status_code == 200 and response.text.count("\x1f") == 1
    # Validation error must also release its dependency slot.
    assert alice.post(path, json={"question": ""}).status_code == 422
    assert alice.post(path, json={"question": "q"}).status_code == 200
