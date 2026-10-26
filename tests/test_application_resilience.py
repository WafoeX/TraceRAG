"""No paid models: real SDK with MockTransport + ASGI application boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from app.auth import require_library
from app.config import get_settings
from app.database import get_db
from app.main import app
from app.routers import documents, query
from app.services import embeddings, generation, reranking, resilience
from app.services.application_limits import MAX_BODY_BYTES, QUERY_BODY_BYTES
from app.services.retrieval import RetrievedChunk
from app.services.search import SearchResult


@pytest.fixture(autouse=True)
def isolate_budget_boundary(monkeypatch):
    """This suite isolates SDK retry policy; real ledger acceptance is separate."""
    from app.services import application_budget

    def passthrough(role, model, payload, invoke):
        return invoke()

    for module in (application_budget, embeddings, generation, reranking):
        monkeypatch.setattr(module, "priced_call", passthrough)


@pytest.fixture
def chunk():
    return RetrievedChunk(1, 11, "Private paper", 0, "The study measured liver cells.", 0.8)


@pytest.fixture
def sdk(monkeypatch):
    calls = []
    outcomes = []

    def handle(request):
        calls.append(request)
        outcome = outcomes.pop(0)
        if outcome == "timeout":
            raise httpx.ReadTimeout("secret provider detail", request=request)
        if outcome == "connection":
            raise httpx.ConnectError("secret provider detail", request=request)
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"error": {"message": "secret"}}, request=request)
        return httpx.Response(200, json=outcome, request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as http,
        OpenAI(api_key="fake-only", http_client=http, max_retries=5) as client,
    ):
        for module in (generation, reranking, embeddings):
            monkeypatch.setattr(module, "get_client", lambda: client)
        monkeypatch.setattr(get_settings(), "provider_retry_backoff_seconds", 0)
        yield calls, outcomes


def completion(text="Grounded answer [1].", *, choices=True, finish="stop"):
    return {
        "id": "fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ]
        if choices
        else [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize("failure", ["timeout", "connection", 429, 500, 503])
def test_retryable_failure_has_exactly_two_visible_attempts(sdk, chunk, failure):
    calls, outcomes = sdk
    outcomes.extend([failure, completion()])
    with resilience.application_calls():
        assert generation.generate_answer("question", [chunk]) == "Grounded answer [1]."
    assert len(calls) == 2
    assert calls[0].extensions["timeout"]["read"] == 20
    assert json.loads(calls[0].content)["max_tokens"] == 2048


@pytest.mark.parametrize("empty", [completion(""), completion("  "), completion(choices=False)])
def test_empty_answer_retries_without_reasoning_or_fabricated_answer(sdk, chunk, empty):
    calls, outcomes = sdk
    outcomes.extend([empty, completion()])
    with resilience.application_calls():
        assert generation.generate_answer("q", [chunk])
    assert len(calls) == 2


@pytest.mark.parametrize(
    "failure,code", [("timeout", "provider_timeout"), (429, "provider_rate_limited")]
)
def test_exhaustion_is_sanitized_and_bounded(sdk, chunk, failure, code):
    calls, outcomes = sdk
    outcomes.extend([failure, failure, completion("must never call")])
    with resilience.application_calls(), pytest.raises(resilience.ProviderUnavailable) as exc:
        generation.generate_answer("q", [chunk])
    assert str(exc.value) == code
    assert len(calls) == 2 and len(outcomes) == 1


@pytest.mark.parametrize("failure", [400, 401, 403, completion(finish="length")])
def test_rejected_or_truncated_response_is_not_retried(sdk, chunk, failure):
    calls, outcomes = sdk
    outcomes.extend([failure, completion()])
    with resilience.application_calls(), pytest.raises(resilience.ProviderUnavailable):
        generation.generate_answer("q", [chunk])
    assert len(calls) == 1


def test_offline_experiment_remains_single_call_and_has_no_app_token_cap(sdk, chunk):
    calls, outcomes = sdk
    outcomes.append(completion())
    assert generation.generate_answer("q", [chunk])
    assert "max_tokens" not in json.loads(calls[0].content)
    assert not resilience.application_active()


def test_embeddings_use_same_timeout_and_visible_retry(sdk, monkeypatch):
    calls, outcomes = sdk
    monkeypatch.setattr(get_settings(), "embedding_dim", 3)
    outcomes.extend(
        [
            "timeout",
            {
                "object": "list",
                "model": "fake",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        ]
    )
    with resilience.application_calls():
        assert embeddings.embed_texts(["study"]) == [[1.0, 0.0, 0.0]]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "failure,expected_calls", [("timeout", 2), (401, 1), (completion("bad JSON"), 1)]
)
def test_rerank_degrades_with_explicit_reason_and_original_scores(
    sdk, chunk, failure, expected_calls
):
    calls, outcomes = sdk
    outcomes.extend([failure, failure])
    with resilience.application_calls() as state:
        result = reranking.rerank("q", [chunk], 1)
    assert result == [chunk] and result[0].score == 0.8
    assert len(calls) == expected_calls
    assert "rerank_retrieval_order" in state.warnings
    assert any(code.startswith("rerank_provider_") for code in state.warnings)


@pytest.mark.parametrize("mode,text", [("baseline", "[8]"), ("keyed_v1", '{"scores":{"0":8}}')])
def test_valid_rerank_does_not_falsely_degrade(sdk, chunk, monkeypatch, mode, text):
    _, outcomes = sdk
    monkeypatch.setattr(get_settings(), "rerank_prompt_version", mode)
    outcomes.append(completion(text))
    with resilience.application_calls() as state:
        assert reranking.rerank("q", [chunk], 1)[0].score == 8
    assert state.warnings == []


@pytest.fixture
def api(monkeypatch, chunk):
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[require_library] = lambda: 1
    app.dependency_overrides[get_db] = lambda: None
    monkeypatch.setattr(query, "search", lambda *args: SearchResult([chunk], 0.8))
    monkeypatch.setattr(query, "evidence_metadata", lambda *args: {})
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


@pytest.mark.parametrize("path", ["/query", "/query/evidence", "/query/stream"])
def test_generation_exhaustion_keeps_evidence_and_finishes_protocol(api, sdk, path):
    calls, outcomes = sdk
    outcomes.extend(["timeout", "timeout"])
    response = api.post(path, json={"question": "study?"})
    assert response.status_code == 200
    assert "secret" not in response.text
    if path.endswith("stream"):
        assert response.text.count("\x1f") == 1
        answer, frame = response.text.split("\x1f")
        assert answer == query.GENERATION_FAILED
        assert json.loads(frame)["citations"][0]["document_id"] == 11
    else:
        data = response.json()
        assert data["answer"] == query.GENERATION_FAILED
        assert data["citations"][0]["document_id"] == 11
        if path.endswith("evidence"):
            assert data["generation_status"] == "failed"
            assert data["warnings"] == ["generation_provider_timeout"]
            assert data["citations"][0]["content"] == "The study measured liver cells."
    assert len(calls) == 2
    assert response.headers["cache-control"] == "no-store"


def test_retrieval_failure_returns_sanitized_503_not_false_refusal(api, monkeypatch):
    monkeypatch.setattr(
        query, "search", Mock(side_effect=resilience.ProviderUnavailable("provider_timeout"))
    )
    response = api.post("/query/evidence", json={"question": "q"})
    assert response.status_code == 503
    assert response.json() == {"detail": "provider_timeout"}


def test_rerank_warning_reaches_evidence_response(api, sdk, chunk, monkeypatch):
    _, outcomes = sdk
    outcomes.extend([401, completion()])

    def search(*args):
        return SearchResult(reranking.rerank("q", [chunk], 1), 0.8)

    monkeypatch.setattr(query, "search", search)
    response = api.post("/query/evidence", json={"question": "q"})
    assert response.json()["generation_status"] == "completed"
    assert "rerank_provider_rejected" in response.json()["warnings"]
    assert response.json()["citations"][0]["score"] == 0.8


@pytest.mark.parametrize(
    "path,limit",
    [("/query/evidence", QUERY_BODY_BYTES), ("/documents/jobs/upload", MAX_BODY_BYTES)],
)
def test_body_limit_without_content_length_rejects_before_routes(api, monkeypatch, path, limit):
    model = Mock()
    monkeypatch.setattr(query, "search", model)
    response = api.post(path, content=iter([b"x" * limit, b"x"]))
    assert response.status_code == 413
    model.assert_not_called()


def test_question_and_title_limits_reject_before_models(api, monkeypatch):
    model = Mock()
    monkeypatch.setattr(query, "search", model)
    assert api.post("/query/evidence", json={"question": "x" * 4001}).status_code == 422
    assert (
        api.post("/documents/jobs", json={"title": "x" * 513, "content": "ok"}).status_code == 422
    )
    model.assert_not_called()


@pytest.mark.parametrize("path", ["/documents/upload", "/documents/jobs/upload"])
def test_legacy_and_queued_upload_have_same_byte_limit(api, path):
    response = api.post(path, files={"file": ("paper.txt", b"x" * (10 * 1024 * 1024 + 1))})
    assert response.status_code == 413


def test_legacy_pdf_page_limit_precedes_embedding(api, monkeypatch):
    from app.services import extract

    monkeypatch.setattr(
        extract, "PdfReader", lambda _: SimpleNamespace(is_encrypted=False, pages=[None] * 201)
    )
    model = Mock()
    monkeypatch.setattr(documents, "ingest_document", model)
    assert api.post("/documents/upload", files={"file": ("p.pdf", b"fake")}).status_code == 422
    model.assert_not_called()


def test_context_does_not_leak_between_requests(api, sdk, monkeypatch):
    _, outcomes = sdk
    outcomes.extend([401, completion()])
    with resilience.application_calls():
        resilience.warn("outside")
        response = api.post("/query/evidence", json={"question": "q"})
    assert response.json()["warnings"] == ["generation_provider_rejected"]
    response = api.post("/query/evidence", json={"question": "q"})
    assert response.json()["warnings"] == []
    assert not resilience.application_active()


@pytest.mark.parametrize("failure", ["timeout", 429, completion("")])
def test_siliconflow_retry_records_every_attempt_including_unknown_cost(
    sdk, chunk, monkeypatch, tmp_path, failure
):
    from app.services import siliconflow
    from app.services.provider_costs import CostTracker
    from tests.test_siliconflow import price_config

    config = price_config()
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(config))
    tracker = CostTracker(path)
    monkeypatch.setattr(siliconflow, "get_cost_tracker", lambda: tracker)
    monkeypatch.setattr(get_settings(), "model_provider", "siliconflow")
    monkeypatch.setattr(get_settings(), "openai_chat_model", config["roles"]["generation"]["model"])
    calls, outcomes = sdk
    outcomes.extend([failure, completion()])
    with resilience.application_calls():
        assert generation.generate_answer("q", [chunk]) == "Grounded answer [1]."
    assert len(calls) == 2
    assert len(tracker.records) == 2
    if failure in ("timeout", 429):
        assert tracker.records[0]["cost_status"] == "unknown"
        assert tracker.records[0]["estimated_cost_cny"] is None
    assert tracker.records[1]["cost_status"] == "estimated"


def test_empty_embedding_retries_and_malformed_vector_fails_closed(sdk, monkeypatch):
    calls, outcomes = sdk
    monkeypatch.setattr(get_settings(), "embedding_dim", 3)
    outcomes.extend(
        [
            {
                "object": "list",
                "model": "fake",
                "data": [],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            {
                "object": "list",
                "model": "fake",
                "data": [{"index": 0, "embedding": [1.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        ]
    )
    with (
        resilience.application_calls(),
        pytest.raises(resilience.ProviderUnavailable, match="provider_invalid_response"),
    ):
        embeddings.embed_query("q")
    assert len(calls) == 2


def test_response_cancellation_closes_admission_connection():
    import asyncio

    from app.services.application_limits import ApplicationBoundary

    closed = Mock()
    scope = {"type": "http", "path": "/query/stream", "method": "POST", "state": {}}

    async def endpoint(scope, receive, send):
        scope["state"]["_admission_db"] = SimpleNamespace(close=closed)
        raise asyncio.CancelledError()

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ApplicationBoundary(endpoint)(scope, receive, send))
    closed.assert_called_once()
    assert not resilience.application_active()


def test_synchronous_chunk_limit_precedes_embedding(monkeypatch):
    from app.services import ingestion

    monkeypatch.setattr(ingestion, "chunk_text", lambda *args: ["chunk"] * 513)
    embed = Mock()
    monkeypatch.setattr(ingestion, "embed_texts", embed)
    with resilience.application_calls(), pytest.raises(ValueError, match="512 chunks"):
        ingestion.ingest_document(Mock(), "title", "content", None)
    embed.assert_not_called()
