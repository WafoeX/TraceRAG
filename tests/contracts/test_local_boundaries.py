"""Offline route/counter boundaries only; no real model, DB writes or clock claims."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import HTTPException, Request

from app.auth import require_library
from app.main import app
from app.routers import query
from app.services import application_limits, ratelimit
from app.services.search import SearchResult


@pytest.fixture
def capacity_boundary(client, monkeypatch):
    monkeypatch.setattr(application_limits.get_settings(), "rate_limit_enabled", True)

    def owned(request: Request):
        request.state.user_id = 7
        return 1

    app.dependency_overrides[require_library] = owned
    return client


@pytest.mark.parametrize("endpoint", ["/query", "/query/stream"])
@pytest.mark.parametrize("confidence,generate", [(0.0, False), (0.119999, False), (0.12, True)])
def test_refusal_threshold_strictly_below_and_no_generation(
    client, chunks, monkeypatch, endpoint, confidence, generate
):
    monkeypatch.setattr(query, "get_settings", lambda: SimpleNamespace(top_k=1, min_relevance=0.12))
    search = Mock(return_value=SearchResult(chunks[:1], confidence))
    answer = "Synthetic refund answer within 30 days [1]."
    generated = Mock(return_value=answer)
    streamed = Mock(return_value=iter([answer]))
    monkeypatch.setattr(query, "search", search)
    monkeypatch.setattr(query, "generate_answer", generated)
    monkeypatch.setattr(query, "stream_answer", streamed)
    response = client.post(endpoint, json={"question": "refund?", "top_k": 1})
    assert response.status_code == 200
    if endpoint.endswith("/stream"):
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text.count(query.CITATION_SEPARATOR) == 1
        text, frame = response.text.split(query.CITATION_SEPARATOR)
        citations = json.loads(frame)["citations"]
    else:
        text, citations = response.json()["answer"], response.json()["citations"]
    if generate:
        assert text == answer and len(citations) == 1
        (streamed if endpoint.endswith("/stream") else generated).assert_called_once()
    else:
        assert text == query.NO_ANSWER and citations == []
        generated.assert_not_called()
        streamed.assert_not_called()
    search.assert_called_once()


@pytest.mark.parametrize("endpoint", ["/query", "/query/stream"])
def test_429_is_returned_before_retrieval_and_models(capacity_boundary, monkeypatch, endpoint):
    client = capacity_boundary
    limiter = Mock(side_effect=HTTPException(429, "Account call limit exceeded"))
    retrieve = Mock(side_effect=AssertionError("429 must not retrieve"))
    generation = Mock(side_effect=AssertionError("429 must not generate"))
    monkeypatch.setattr(application_limits, "admit_account", limiter)
    monkeypatch.setattr(query, "search", retrieve)
    monkeypatch.setattr(query, "generate_answer", generation)
    monkeypatch.setattr(query, "stream_answer", generation)
    response = client.post(
        endpoint,
        json={"question": "refund?"},
        headers={"x-forwarded-for": "192.0.2.1, 198.51.100.1"},
    )
    assert response.status_code == 429
    assert "Account call limit exceeded" in response.json()["detail"]
    limiter.assert_called_once_with(7)  # forwarded headers are irrelevant
    retrieve.assert_not_called()
    generation.assert_not_called()


@pytest.mark.parametrize("endpoint", ["/query", "/query/stream"])
def test_limiter_failure_is_fail_closed_without_retrieval(capacity_boundary, monkeypatch, endpoint):
    client = capacity_boundary
    limiter = Mock(side_effect=RuntimeError("synthetic limiter failure"))
    retrieve = Mock(return_value=SearchResult([], 0.0))
    generation = Mock(side_effect=AssertionError("No context must not generate"))
    monkeypatch.setattr(application_limits, "admit_account", limiter)
    monkeypatch.setattr(query, "search", retrieve)
    monkeypatch.setattr(query, "generate_answer", generation)
    monkeypatch.setattr(query, "stream_answer", generation)
    response = client.post(endpoint, json={"question": "refund?"})
    assert response.status_code == 503
    assert "synthetic limiter failure" not in response.text
    retrieve.assert_not_called()
    generation.assert_not_called()


@pytest.mark.parametrize("count,allowed", [(1, True), (2, True), (3, False)])
def test_counter_boundary_with_fake_engine_only(monkeypatch, count, allowed):
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.scalar.return_value = count
    monkeypatch.setattr(ratelimit, "engine", engine)
    monkeypatch.setattr(ratelimit, "_table_ready", True)
    monkeypatch.setattr(
        ratelimit,
        "get_settings",
        lambda: SimpleNamespace(rate_limit_enabled=True, rate_limit_per_minute=2),
    )
    assert ratelimit.check_rate_limit("192.0.2.1") is allowed
    assert connection.execute.call_args.args[1] == {"ip": "192.0.2.1"}
    assert "ON CONFLICT" in str(connection.execute.call_args.args[0])


def test_disabled_counter_never_opens_database(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(ratelimit, "engine", engine)
    monkeypatch.setattr(ratelimit, "_table_ready", False)
    monkeypatch.setattr(
        ratelimit, "get_settings", lambda: SimpleNamespace(rate_limit_enabled=False)
    )
    assert ratelimit.check_rate_limit("192.0.2.1") is True
    engine.begin.assert_not_called()


def test_counter_table_created_once_with_fake_engine_not_real_ddl(monkeypatch):
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.scalar.return_value = 1
    monkeypatch.setattr(ratelimit, "engine", engine)
    monkeypatch.setattr(ratelimit, "_table_ready", False)
    monkeypatch.setattr(
        ratelimit,
        "get_settings",
        lambda: SimpleNamespace(rate_limit_enabled=True, rate_limit_per_minute=2),
    )
    assert ratelimit.check_rate_limit("192.0.2.1") is True
    assert ratelimit.check_rate_limit("192.0.2.1") is True
    statements = [str(call.args[0]) for call in connection.execute.call_args_list]
    assert sum("CREATE TABLE" in statement for statement in statements) == 1
    assert sum("INSERT INTO" in statement for statement in statements) == 2
