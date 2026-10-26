"""One-shot probe safety: fake SDK responses only; sockets remain blocked."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import Mock

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from app.config import Settings
from scripts import smoke_siliconflow as smoke


@pytest.fixture
def client(monkeypatch):
    settings = Settings.model_construct(
        openai_api_key="offline-secret-never-log",
        model_provider="siliconflow",
        openai_base_url="https://api.siliconflow.cn/v1",
        openai_embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        openai_chat_model="deepseek-ai/DeepSeek-V4.1-Flash",
        llm_rerank_model="deepseek-ai/DeepSeek-V4.1-Flash",
        siliconflow_max_cost_cny=Decimal("20"),
    )
    monkeypatch.setattr(smoke, "get_settings", lambda: settings)
    api = Mock()
    api.with_options.return_value = api
    api.embeddings.create.return_value = CreateEmbeddingResponse.model_validate(
        {
            "object": "list",
            "model": "BAAI/bge-m3",
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.1] * 1024} for i in (1, 0)
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
    )

    def reply(text):
        return ChatCompletion.model_validate(
            {
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    api.chat.completions.create.side_effect = [reply("[8, 2]"), reply("OK")]
    return settings, api


def test_plan_never_constructs_client(tmp_path, client):
    factory = Mock(side_effect=AssertionError("no client"))
    result = smoke.run(tmp_path / "plan", client_factory=factory)
    assert result["model_attempts"] == 0
    assert result["status"] == "planned"
    factory.assert_not_called()


def test_probe_three_calls_bounded_accounted_no_secret(tmp_path, client):
    _, api = client
    output = tmp_path / "probe"
    state = smoke.run(output, True, lambda: api)
    assert state["status"] == "complete"
    assert state["model_attempts"] == 3
    assert Decimal(state["estimated_cost_cny"]) == Decimal("0.00012")
    assert state["production_profiles_modified"] is False
    assert state["database_operations"] == 0
    assert state["calls"][0]["dimensions"] == [1024, 1024]
    assert "dimensions" not in api.embeddings.create.call_args.kwargs
    api.with_options.assert_called_once_with(max_retries=0, timeout=60)
    assert all(c.kwargs["max_tokens"] == 64 for c in api.chat.completions.create.call_args_list)
    assert "offline-secret-never-log" not in (output / "report.json").read_text()
    api.close.assert_called_once()


def test_insufficient_budget_prevents_client_and_output(tmp_path, client):
    client[0].siliconflow_max_cost_cny = Decimal("0.01")
    factory = Mock()
    with pytest.raises(ValueError):
        smoke.run(tmp_path / "none", True, factory)
    factory.assert_not_called()
    assert not (tmp_path / "none").exists()


def test_unknown_usage_stops_without_retry_or_chat(tmp_path, client):
    api = client[1]
    api.embeddings.create.return_value.usage = None
    state = smoke.run(tmp_path / "unknown", True, lambda: api)
    assert state["status"] == "partial"
    assert state["model_attempts"] == 1
    assert state["estimated_cost_cny"] is None
    api.embeddings.create.assert_called_once()
    api.chat.completions.create.assert_not_called()


def test_provider_error_body_not_logged(tmp_path, client):
    api = client[1]
    api.embeddings.create.side_effect = RuntimeError("SECRET_PROVIDER_BODY")
    output = tmp_path / "error"
    state = smoke.run(output, True, lambda: api)
    assert state["status"] == "partial"
    assert state["estimated_cost_cny"] is None
    assert "SECRET_PROVIDER_BODY" not in json.dumps(state)
    api.chat.completions.create.assert_not_called()


def test_existing_output_never_overwritten(tmp_path, client):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "report.json"
    marker.write_text("original")
    factory = Mock()
    with pytest.raises(FileExistsError):
        smoke.run(output, True, factory)
    assert marker.read_text() == "original"
    factory.assert_not_called()


def test_anomalous_chat_usage_saved_before_validation_without_retry(tmp_path, client):
    api = client[1]
    reply = next(api.chat.completions.create.side_effect)
    assert reply.usage is not None
    reply.usage.completion_tokens = 65
    reply.usage.total_tokens = reply.usage.prompt_tokens + 65
    api.chat.completions.create.side_effect = [reply]
    output = tmp_path / "anomaly"
    state = smoke.run(output, True, lambda: api)
    assert state["status"] == "partial"
    assert state["model_attempts"] == 2
    assert state["estimated_cost_cny"] is None
    assert state["calls"][-1]["stage"] == "usage_validation"
    assert state["calls"][-1]["raw_usage"]["completion_tokens"] == 65
    persisted = json.loads((output / "report.json").read_text())
    assert persisted["calls"][-1]["raw_usage"]["completion_tokens"] == 65
    api.chat.completions.create.assert_called_once()


def test_single_chat_diagnostic_deducts_prior_cost_without_other_calls(tmp_path, client):
    api = client[1]
    state = smoke.run(
        tmp_path / "single",
        True,
        lambda: api,
        only_role="llm_rerank",
        prior_cost_cny=Decimal("0.001"),
    )
    assert state["status"] == "complete"
    assert state["selected_roles"] == ["llm_rerank"]
    assert state["available_budget_cny"] == "19.999"
    assert state["model_attempts"] == 1
    assert state["calls"][0]["response"]["choices"][0]["message"]["content"] == "[8, 2]"
    api.embeddings.create.assert_not_called()
    api.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("prior", ["-1", "NaN", "20"])
def test_invalid_or_exhausted_prior_cost_prevents_calls(tmp_path, client, prior):
    factory = Mock()
    with pytest.raises(ValueError):
        smoke.run(
            tmp_path / "none", True, factory, only_role="llm_rerank", prior_cost_cny=Decimal(prior)
        )
    factory.assert_not_called()


def test_response_body_redacts_reflected_key_even_on_protocol_failure(tmp_path, client):
    api = client[1]
    reply = next(api.chat.completions.create.side_effect)
    reply.choices[0].message.content = "offline-secret-never-log"
    api.chat.completions.create.side_effect = [reply]
    output = tmp_path / "redacted"
    state = smoke.run(output, True, lambda: api, only_role="llm_rerank")
    assert state["status"] == "partial"
    assert "offline-secret-never-log" not in (output / "report.json").read_text()
    assert "[REDACTED]" in (output / "report.json").read_text()
    api.chat.completions.create.assert_called_once()


def test_observed_deepseek_usage_bills_reasoning_once_not_as_visible_limit():
    observed = {
        "prompt_tokens": 79,
        "completion_tokens": 118,
        "total_tokens": 197,
        "completion_tokens_details": {"reasoning_tokens": 113},
    }
    usage, cost = smoke.settle(observed, "llm_rerank")
    assert usage["reasoning_tokens"] == 113
    assert usage["visible_completion_tokens"] == 5
    assert usage["completion_tokens"] == 118
    assert cost == Decimal("0.001102")
    smoke.validate_llm_scores("[10,0]", 2)


@pytest.mark.parametrize("reasoning", [-1, 119, True, "113"])
def test_invalid_reasoning_usage_is_not_silently_subtracted(reasoning):
    with pytest.raises(ValueError):
        smoke.settle(
            {
                "prompt_tokens": 79,
                "completion_tokens": 118,
                "total_tokens": 197,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
            "llm_rerank",
        )
