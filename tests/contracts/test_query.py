"""Freeze answer/citation ordering, validation, and the non-SSE streaming protocol."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from app.routers import query
from app.services.generation import _build_context
from app.services.search import SearchResult


@pytest.mark.parametrize("endpoint", ["/query", "/query/stream"])
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {},
        {"question": "refund?", "top_k": 0},
        {"question": "refund?", "top_k": 21},
    ],
)
def test_query_validation_does_not_retrieve(client, monkeypatch, endpoint, payload) -> None:
    retrieve = Mock()
    monkeypatch.setattr(query, "search", retrieve)
    assert client.post(endpoint, json=payload).status_code == 422
    retrieve.assert_not_called()


def test_json_answer_preserves_citation_order_and_fields(client, db, chunks, monkeypatch) -> None:
    # Deliberately reverse retrieval order: citations must not be sorted by ID or score.
    ordered = list(reversed(chunks))
    retrieve = Mock(return_value=SearchResult(chunks=ordered, confidence=0.9))
    generate = Mock(return_value="Processing takes 5-7 days [1]; refunds within 30 days [2].")
    monkeypatch.setattr(query, "search", retrieve)
    monkeypatch.setattr(query, "generate_answer", generate)
    response = client.post("/query", json={"question": "refund?", "top_k": 2})
    assert response.status_code == 200
    assert response.json() == {
        "answer": generate.return_value,
        "citations": [
            {
                "document_id": 23,
                "document_title": "Processing Times",
                "chunk_index": 2,
                "score": 0.8,
                "snippet": chunks[1].content,
            },
            {
                "document_id": 17,
                "document_title": "Refund Policy",
                "chunk_index": 0,
                "score": 0.9,
                "snippet": chunks[0].content,
            },
        ],
    }
    retrieve.assert_called_once_with(db, "refund?", 2)
    generate.assert_called_once_with("refund?", ordered)
    context = _build_context(ordered)
    first = context.index('[1] (from "Processing Times")')
    second = context.index('[2] (from "Refund Policy")')
    assert first < second


def test_citation_snippet_remains_truncated_to_240_characters(chunks) -> None:
    chunks[0].content = "x" * 300
    citation = query._citations(chunks[:1])[0]
    assert citation.snippet == "x" * 240
    assert isinstance(citation.document_id, int)


def test_stream_uses_text_and_one_final_citation_frame(client, chunks, monkeypatch) -> None:
    monkeypatch.setattr(query, "search", Mock(return_value=SearchResult(chunks, 0.9)))
    stream = Mock(return_value=iter(["Refunds ", "退款 ", "within 30 days [1]."]))
    monkeypatch.setattr(query, "stream_answer", stream)
    response = client.post("/query/stream", json={"question": "refund?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "text/event-stream" not in response.headers["content-type"]
    assert response.text.count("\x1f") == 1
    answer, frame = response.text.split("\x1f")
    assert answer == "Refunds 退款 within 30 days [1]."
    assert json.loads(frame) == {"citations": [c.model_dump() for c in query._citations(chunks)]}
    stream.assert_called_once_with("refund?", chunks)


@pytest.mark.parametrize("endpoint", ["/query", "/query/stream"])
def test_empty_knowledge_base_returns_refusal_without_generation(
    client, monkeypatch, endpoint
) -> None:
    monkeypatch.setattr(query, "search", Mock(return_value=SearchResult([], 0.0)))
    generate = Mock(side_effect=AssertionError("Empty retrieval must not generate"))
    monkeypatch.setattr(query, "generate_answer", generate)
    monkeypatch.setattr(query, "stream_answer", generate)
    response = client.post(endpoint, json={"question": "refund?"})
    assert response.status_code == 200
    if endpoint.endswith("/stream"):
        answer, frame = response.text.split("\x1f")
        assert answer == query.NO_ANSWER
        assert json.loads(frame) == {"citations": []}
    else:
        assert response.json() == {"answer": query.NO_ANSWER, "citations": []}
    generate.assert_not_called()
