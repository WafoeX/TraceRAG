"""Fixture guard and plan tests only: no PostgreSQL or model network calls."""
from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import scripts.siliconflow_e2e as e2e
from app.config import Settings


@pytest.fixture
def guard(tmp_path):
    state = {"calls": [], "known_estimated_cost_cny": "0", "estimated_cost_cny": "0"}
    return e2e.FixtureCalls(state, tmp_path / "report.json", Decimal("20"), "SECRET-KEY")


def embedding_reply():
    return SimpleNamespace(
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 10}),
        data=[SimpleNamespace(index=0, embedding=[0.1] * 1024)],
    )


def embed_payload(text):
    return {"model": "BAAI/bge-m3", "input": [text], "encoding_format": "float"}


def test_strict_embedding_sequence_and_extra_call_denied(guard):
    api = Mock(return_value=embedding_reply())
    guard("embedding", "BAAI/bge-m3", api, embed_payload(e2e.DOCUMENT))
    guard("embedding", "BAAI/bge-m3", api, embed_payload(e2e.QUESTION))
    with pytest.raises(ValueError):
        guard("embedding", "BAAI/bge-m3", api, embed_payload(e2e.QUESTION))
    assert api.call_count == 2
    assert guard.spent == 0
    assert guard.state["calls"][0]["dimensions"] == [1024]


@pytest.mark.parametrize("content", ["private text", "", e2e.DOCUMENT * 100])
def test_unapproved_documents_or_oversized_requests_not_sent(guard, content):
    api = Mock()
    with pytest.raises(ValueError):
        guard("embedding", "BAAI/bge-m3", api, embed_payload(content))
    api.assert_not_called()


def test_unknown_usage_stops_even_if_caller_swallowed_exception(guard):
    api = Mock(return_value=SimpleNamespace(usage=None))
    with pytest.raises(ValueError):
        guard("embedding", "BAAI/bge-m3", api, embed_payload(e2e.DOCUMENT))
    with pytest.raises(ValueError):
        guard("embedding", "BAAI/bge-m3", api, embed_payload(e2e.QUESTION))
    assert guard.blocked
    assert guard.state["estimated_cost_cny"] is None
    api.assert_called_once()


def test_chat_reservation_prevents_api_call(guard):
    guard.state["calls"] = [{"role": "embedding"}, {"role": "embedding"}]
    guard.limit = Decimal("0.01")
    api = Mock()
    with pytest.raises(ValueError):
        guard("llm_rerank", "deepseek-ai/DeepSeek-V4.1-Flash", api,
              {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "messages": [], "max_tokens": 64})
    api.assert_not_called()


def test_key_redacted_and_serialization_failure_is_controlled(guard):
    assert guard.safe({"content": "SECRET-KEY"}) == {"content": "[REDACTED]"}
    with pytest.raises(ValueError):
        guard.safe({"bad": object()})


def test_plan_does_not_connect_database_or_model(tmp_path, monkeypatch):
    settings = Settings.model_construct(
        openai_api_key="offline", model_provider="siliconflow",
        openai_base_url="https://api.siliconflow.cn/v1", embedding_dim=1024,
        openai_embedding_model="BAAI/bge-m3",
        openai_chat_model="deepseek-ai/DeepSeek-V4.1-Flash",
        siliconflow_max_cost_cny=Decimal("20"),
    )
    monkeypatch.setattr(e2e, "get_settings", lambda: settings)
    database = Mock(side_effect=AssertionError("No DB in plan"))
    model = Mock(side_effect=AssertionError("No model in plan"))
    monkeypatch.setattr(e2e, "database_snapshot", database)
    monkeypatch.setattr(e2e, "prepare_schema", database)
    monkeypatch.setattr(e2e, "OpenAI", model)
    output = tmp_path / "plan"
    state = e2e.run(output, prior_cost=Decimal("0.002292"))
    assert state["status"] == "planned"
    assert state["calls"] == []
    assert state["available_budget_cny"] == "19.997708"
    assert json.loads((output / "report.json").read_text())["schema"] is None
    database.assert_not_called()
    model.assert_not_called()
    with pytest.raises(FileExistsError):
        e2e.run(output)
