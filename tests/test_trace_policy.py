"""Offline trace privacy and ASGI lifetime checks. No database or provider IO."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest

from app.schemas import QueryRequest
from app.services import tracing
from app.services.retrieval import RetrievedChunk


@pytest.mark.parametrize("consent", [False, True])
def test_content_requires_per_request_consent(consent):
    state = tracing.TraceState()
    token = tracing._state.set(state)
    try:
        tracing.begin_query("SECRET QUESTION", consent, 3)
        tracing.context(
            [RetrievedChunk(2, 1, "SECRET TITLE", 0, "SECRET TEXT", 0.9)], generation_attempted=True
        )
        tracing.answer("SECRET ANSWER", "completed")
        with tracing.stage("generation"):
            pass
        serialized = json.dumps(state.data)
        assert "SECRET" not in serialized
        assert (state.snapshot is not None) == consent
        assert state.data["contexts"][0]["document_id"] == 1
        assert state.data["stages"][0]["duration_ms"] >= 0
        assert "password" not in serialized and "base_url" not in serialized
    finally:
        tracing._state.reset(token)
    assert tracing._state.get() is None


def test_stage_failure_never_records_exception_content():
    state = tracing.TraceState()
    token = tracing._state.set(state)
    try:
        with pytest.raises(RuntimeError), tracing.stage("retrieval"):
            raise RuntimeError("SECRET API KEY")
        assert state.data["stages"][0]["status"] == "failed"
        assert "SECRET" not in json.dumps(state.data)
    finally:
        tracing._state.reset(token)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_consent_is_strict_boolean(value):
    with pytest.raises(ValueError):
        QueryRequest(question="hi", top_k=None, retain_trace_content=value)


@pytest.mark.parametrize("stream,fail_persist", [(False, False), (True, False), (False, True)])
def test_asgi_trace_lifetime(monkeypatch, stream, fail_persist):
    records, sent = [], []

    def persist(state):
        records.append(state)
        if fail_persist:
            raise RuntimeError("SECRET DATABASE URL")

    monkeypatch.setattr(tracing, "persist", persist)

    async def endpoint(scope, receive, send):
        tracing.bind_owner(1, 2)
        tracing.begin_query("private", False, 3)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        if stream:
            await send({"type": "http.response.body", "body": b"part", "more_body": True})
            assert not records
        tracing.answer("answer", "completed")
        await send({"type": "http.response.body", "body": b"done"})

    async def send(message):
        if message["type"] == "http.response.body" and not message.get("more_body"):
            assert len(records) == 1  # detail immediately available after response
        sent.append(message)

    asyncio.run(
        tracing.TraceBoundary(endpoint)(
            {
                "type": "http",
                "method": "POST",
                "path": "/query/stream" if stream else "/query/evidence",
            },
            None,
            send,
        )
    )
    assert len(records) == 1
    assert records[0].data["generation_status"] == "completed"
    assert records[0].snapshot is None
    UUID(dict(sent[0]["headers"])[b"x-trace-id"].decode())
    assert tracing._state.get() is None


def test_context_isolated_across_threads():
    def work(i):
        state = tracing.TraceState()
        token = tracing._state.set(state)
        try:
            tracing.bind_owner(i, i)
            tracing.begin_query(str(i), True, 1)
            return state
        finally:
            tracing._state.reset(token)

    with ThreadPoolExecutor(4) as pool:
        records = list(pool.map(work, range(12)))
    assert len({r.id for r in records}) == 12
    assert all(r.snapshot == {"question": str(r.user_id)} for r in records)


@pytest.mark.parametrize(
    "module", ["scripts.migrate_application_traces", "scripts.manage_application_traces"]
)
def test_cli_plan_and_redacted_apply(module):
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", module], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0 and "PLAN ONLY" in result.stdout
    result = subprocess.run(
        [sys.executable, "-m", module, "--apply", "--database-url", "bad://SECRET"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "SECRET" not in result.stdout + result.stderr


@pytest.mark.parametrize("phase", ["retrieval", "metadata"])
def test_query_errors_do_not_propagate_sql_parameters(monkeypatch, phase):
    from unittest.mock import MagicMock

    from app.routers import query
    from app.services.resilience import ProviderUnavailable
    from app.services.search import SearchResult

    def fail(*args):
        raise RuntimeError("SECRET SQL parameters")

    monkeypatch.setattr(query, "run_search", fail)
    code = "retrieval_unavailable"
    if phase == "metadata":
        code = "evidence_unavailable"
        monkeypatch.setattr(query, "search", lambda *args: SearchResult([], 1.0))
        monkeypatch.setattr(query, "evidence_metadata", fail)
    with pytest.raises(ProviderUnavailable, match=code) as caught:
        query.query_evidence(
            QueryRequest(question="SECRET QUESTION", top_k=None, retain_trace_content=False),
            MagicMock(),
        )
    assert "SECRET" not in str(caught.value)
    assert caught.value.__suppress_context__
