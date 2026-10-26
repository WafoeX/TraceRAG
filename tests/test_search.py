"""The relevance gate: answer when retrieval is confident, decline otherwise.

These run fully offline — the DB session, rate limiter, settings, retrieval,
and generation are all stubbed, so we exercise only the gate's decision logic.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app
from app.routers import query as query_module
from app.services.retrieval import RetrievedChunk
from app.services.search import SearchResult


def _sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        document_id=1,
        document_title="Handbook",
        chunk_index=0,
        content="Customers may request a full refund within 30 days.",
        score=0.9,
    )


@pytest.fixture
def client(monkeypatch):
    # No real DB or rate-limit table, and a fixed threshold independent of any
    # local .env so the test is deterministic everywhere (including CI).
    monkeypatch.setattr(
        query_module,
        "get_settings",
        lambda: SimpleNamespace(top_k=4, min_relevance=0.12),
    )

    def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_answers_when_retrieval_is_confident(client, monkeypatch):
    generated = {"called": False}

    def fake_search(db, question, top_k):
        return SearchResult(chunks=[_sample_chunk()], confidence=0.82)

    def fake_generate(question, chunks):
        generated["called"] = True
        return "You have 30 days for a full refund."

    monkeypatch.setattr(query_module, "search", fake_search)
    monkeypatch.setattr(query_module, "generate_answer", fake_generate)

    res = client.post("/query", json={"question": "What is the refund window?"})
    body = res.json()

    assert res.status_code == 200
    assert generated["called"] is True
    assert body["answer"] == "You have 30 days for a full refund."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["document_title"] == "Handbook"


def test_declines_when_below_threshold_without_calling_the_llm(client, monkeypatch):
    generated = {"called": False}

    def fake_search(db, question, top_k):
        # Top match is essentially unrelated to the question.
        return SearchResult(chunks=[_sample_chunk()], confidence=0.03)

    def fake_generate(question, chunks):
        generated["called"] = True
        return "this should never be returned"

    monkeypatch.setattr(query_module, "search", fake_search)
    monkeypatch.setattr(query_module, "generate_answer", fake_generate)

    res = client.post("/query", json={"question": "What's the weather today?"})
    body = res.json()

    assert res.status_code == 200
    assert generated["called"] is False  # the gate saved an LLM call
    assert body["citations"] == []
    assert "couldn't find" in body["answer"].lower()


def test_stream_endpoint_also_declines_when_not_confident(client, monkeypatch):
    monkeypatch.setattr(
        query_module,
        "search",
        lambda db, question, top_k: SearchResult(chunks=[], confidence=0.0),
    )

    res = client.post("/query/stream", json={"question": "totally off topic"})

    assert res.status_code == 200
    assert "couldn't find" in res.text.lower()


def test_stream_emits_answer_text_then_a_citations_frame(client, monkeypatch):
    monkeypatch.setattr(
        query_module,
        "search",
        lambda db, question, top_k: SearchResult(chunks=[_sample_chunk()], confidence=0.8),
    )
    monkeypatch.setattr(
        query_module,
        "stream_answer",
        lambda question, chunks: iter(["You have ", "30 days."]),
    )

    res = client.post("/query/stream", json={"question": "refund window?"})
    answer, sep, frame = res.text.partition("\x1f")

    assert res.status_code == 200
    assert sep == "\x1f"  # the separator is present
    assert answer == "You have 30 days."  # streamed tokens, concatenated
    citations = json.loads(frame)["citations"]  # trailing JSON parses
    assert citations[0]["document_title"] == "Handbook"
