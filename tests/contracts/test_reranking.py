"""Freeze the original LLM reranker and its failure fallback without API calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services import reranking


def _install_response(monkeypatch, content: str | None = None, error=None) -> Mock:
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        ),
        side_effect=error,
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(reranking, "get_client", lambda: client)
    return create


def test_empty_candidates_do_not_call_model(monkeypatch) -> None:
    client = Mock(side_effect=AssertionError("No client required for empty candidates"))
    monkeypatch.setattr(reranking, "get_client", client)
    assert reranking.rerank("refund?", [], 2) == []
    client.assert_not_called()


@pytest.mark.parametrize("content", ["not JSON", "[]", "[0, 0]", None])
def test_invalid_or_all_zero_response_preserves_original_order_and_scores(
    chunks, monkeypatch, content
) -> None:
    create = _install_response(monkeypatch, content)
    original = [(c.chunk_id, c.score) for c in chunks]
    result = reranking.rerank("refund?", chunks, 1)
    assert result == chunks[:1]
    assert result[0] is chunks[0]
    assert [(c.chunk_id, c.score) for c in chunks] == original
    create.assert_called_once()


def test_provider_exception_falls_back_without_destroying_scores(chunks, monkeypatch) -> None:
    create = _install_response(monkeypatch, error=RuntimeError("provider unavailable"))
    assert reranking.rerank("refund?", chunks, 2) == chunks
    create.assert_called_once()


def test_valid_scores_reorder_copies_without_mutating_candidates(chunks, monkeypatch) -> None:
    create = _install_response(monkeypatch, "Scores: [2, 9]")
    result = reranking.rerank("refund?", chunks, 2)
    assert [c.chunk_id for c in result] == [102, 101]
    assert [c.score for c in result] == [9.0, 2.0]
    assert [c.score for c in chunks] == [0.9, 0.8]
    assert result[0].document_id == chunks[1].document_id
    assert result[0].content == chunks[1].content
    assert result[0] is not chunks[1]
    kwargs = create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "gpt-4o-mini"
    assert "Question: refund?" in kwargs["messages"][1]["content"]


def test_prompt_keeps_baseline_600_character_passage_limit(chunks, monkeypatch) -> None:
    chunks[0].content = "A" * 600 + "NEVER_IN_PROMPT"
    create = _install_response(monkeypatch, "[8, 2]")
    reranking.rerank("refund?", chunks, 2)
    prompt = create.call_args.kwargs["messages"][1]["content"]
    assert "A" * 600 in prompt
    assert "NEVER_IN_PROMPT" not in prompt


@pytest.mark.parametrize(
    ("content", "n", "expected"),
    [
        ('[{"score": 8}, {"relevance": 2}]', 2, [8.0, 2.0]),
        ("[8]", 2, [8.0, 0.0]),
        ("[8, 2, 1]", 2, [8.0, 2.0]),
        ('["bad", 4]', 2, [0.0, 4.0]),
    ],
)
def test_baseline_lenient_parser_is_explicitly_frozen(content, n, expected) -> None:
    # Stricter validation is a future, separately measured change, not this baseline.
    assert reranking._parse_scores(content, n) == expected
