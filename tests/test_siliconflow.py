"""All provider contracts use fake SDK/HTTP responses; sockets remain disabled."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from app.config import Settings
from app.services import embeddings, generation, openai_client, reranking, siliconflow
from app.services.provider_budget import ProviderBudget, ProviderBudgetError
from app.services.provider_costs import CostTracker, ProviderCostError
from app.services.retrieval import RetrievedChunk


def sf_settings(**overrides):
    values: dict[str, Any] = {
        "_env_file": None,
        "openai_api_key": "offline-test-only",
        # _env_file=None does not exclude inherited process environment overrides.
        "deepseek_api_key": "",
        "deepseek_base_url": None,
        "model_provider": "siliconflow",
        "openai_base_url": "https://api.siliconflow.cn/v1",
        "openai_embedding_model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "openai_chat_model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "llm_rerank_model": "fixture-separate-chat",
        "run_db_init": False,
    }
    values.update(overrides)
    return Settings(**values)


def test_sf_fixture_ignores_chat_override_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-env-key")
    settings = sf_settings()
    assert settings.deepseek_base_url is None
    assert settings.deepseek_api_key == ""


def price_config():
    config = json.loads(Path("configs/providers/siliconflow.pricing.json").read_text())
    for role, profile in config["roles"].items():
        profile.update(
            verified=True,
            input_rate="1",
            output_rate="0" if role in {"embedding", "native_rerank"} else "2",
            input_token_upper_bound=10000,
            output_token_upper_bound=4096 if role in {"llm_rerank", "generation"} else 0,
            source="fixture-not-real-pricing",
            checked_at="fixture",
            limits_source="fixture",
            usage_source="fixture",
            request_byte_limit=100000,
        )
    config["roles"]["llm_rerank"]["model"] = "fixture-separate-chat"
    return config


@pytest.fixture
def provider(tmp_path, monkeypatch):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(price_config()))
    settings = sf_settings()
    for module in (embeddings, generation, reranking, siliconflow, openai_client):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    tracker = CostTracker(path)
    monkeypatch.setattr(siliconflow, "get_cost_tracker", lambda: tracker)
    return settings, tracker


@pytest.fixture
def legacy_provider(tmp_path):
    """Historical standalone budget class remains covered, not wired into runtime."""
    path = tmp_path / "legacy-pricing.json"
    path.write_text(json.dumps(price_config()))
    return sf_settings(), ProviderBudget(str(path), Decimal("1"))


def completion(content="[2, 9]", usage=None, finish="stop"):
    return ChatCompletion.model_validate(
        {
            "id": "fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "not the answer",
                    },
                }
            ],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def chunk(i):
    return RetrievedChunk(i, 1, "fixture", i, "passage", 0.8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_dim", 1536),
        ("openai_embedding_model", "text-embedding-3-small"),
        ("openai_base_url", "https://other.example/v1"),
        ("openai_chat_model", "BAAI/bge-reranker-v2-m3"),
        ("llm_rerank_model", "BAAI/bge-reranker-v2-m3"),
        ("siliconflow_max_cost_cny", "-0.01"),
        ("siliconflow_max_cost_cny", "NaN"),
    ],
)
def test_settings_refuse_role_dimension_url_and_budget_mismatch(field, value):
    with pytest.raises(ValidationError):
        sf_settings(**{field: value})


def test_env_loads_blank_key_position_without_leaking_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=fixture-local-secret\n")
    options: dict[str, Any] = {"_env_file": env}
    settings = Settings(**options)
    assert settings.openai_api_key == "fixture-local-secret"
    assert "fixture-local-secret" not in repr(settings)
    env.write_text("OPENAI_API_KEY=\n")
    with pytest.raises(ValidationError):
        Settings(**options)


def test_embedding_sdk_wire_omits_dimensions_reorders_and_accounts(provider, monkeypatch):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "BAAI/bge-m3",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(i)] * 1024}
                    for i in (1, 0)
                ],
                "usage": {"prompt_tokens": 12, "total_tokens": 12},
            },
        )

    with OpenAI(
        api_key="fixture",
        base_url="https://api.siliconflow.cn/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(embeddings, "get_client", lambda: client)
        vectors = embeddings.embed_texts(["first", "second"])
    assert [v[0] for v in vectors] == [0, 1]
    assert payloads == [
        {"model": "BAAI/bge-m3", "input": ["first", "second"], "encoding_format": "float"}
    ]
    assert provider[1].spent == Decimal("0.000012")
    assert provider[1].records[0]["estimated_cost_cny"] == "0.000012"
    assert provider[1].records[0]["currency"] == "CNY"
    assert "estimated_cost_usd" not in provider[1].records[0]
    assert provider[1].records[0]["usage"]["total_tokens"] == 12


@pytest.mark.parametrize(
    "data,count",
    [
        ([], 1),
        ([(-1, 1024)], 1),
        ([(0, 1024), (0, 1024)], 2),
        ([(1, 1024)], 1),
        ([(0, 768)], 1),
    ],
)
def test_embedding_shape_indices_rejected(data, count):
    rows = [SimpleNamespace(index=i, embedding=[0.1] * dim) for i, dim in data]
    with pytest.raises(siliconflow.ProviderResponseError):
        siliconflow.ordered_embeddings(rows, count)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "bad"])
def test_embedding_nonfinite_values_rejected(value):
    with pytest.raises(siliconflow.ProviderResponseError):
        siliconflow.ordered_embeddings([SimpleNamespace(index=0, embedding=[value] * 1024)], 1)


def test_empty_embedding_does_not_construct_client(monkeypatch):
    client = Mock(side_effect=AssertionError("no API"))
    monkeypatch.setattr(embeddings, "get_client", client)
    assert embeddings.embed_texts([]) == []
    client.assert_not_called()


def test_llm_baseline_retained_with_independent_chat_role_and_usage(provider, monkeypatch):
    create = Mock(return_value=completion("```json\n[2, 9]\n```"))
    monkeypatch.setattr(
        reranking,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    result = reranking.rerank("q", [chunk(1), chunk(2)], 2)
    assert [c.chunk_id for c in result] == [2, 1]
    assert [c.score for c in result] == [9, 2]
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "fixture-separate-chat"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0
    assert "Return ONLY a JSON array" in kwargs["messages"][1]["content"]
    assert provider[1].records[0]["role"] == "llm_rerank"
    assert provider[1].spent == Decimal("0.00002")


@pytest.mark.parametrize(
    "content,finish",
    [
        ("[2, 9]", "length"),
        (None, "stop"),
        ("not JSON", "stop"),
        ("[NaN, 9]", "stop"),
        ("[true, 9]", "stop"),
        ("[11, 9]", "stop"),
        ("[8]", "stop"),
        ("[0, 0]", "stop"),
    ],
)
def test_llm_bad_content_keeps_original_candidates(provider, monkeypatch, content, finish):
    create = Mock(return_value=completion(content, finish=finish))
    monkeypatch.setattr(
        reranking,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    candidates = [chunk(1), chunk(2)]
    assert reranking.rerank("q", candidates, 2) == candidates
    assert create.call_count == 1


def test_usage_missing_records_unknown_without_blocking_later_calls(provider, monkeypatch):
    reply = completion()
    reply.usage = None
    create = Mock(return_value=reply)
    monkeypatch.setattr(
        reranking,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    candidates = [chunk(1), chunk(2)]
    assert [c.chunk_id for c in reranking.rerank("q", candidates, 2)] == [2, 1]
    assert [c.chunk_id for c in reranking.rerank("q", candidates, 2)] == [2, 1]
    assert create.call_count == 2
    assert provider[1].summary()["estimated_cost_cny"] is None


def test_generation_and_stream_buffer_use_generation_role(provider, monkeypatch):
    create = Mock(return_value=completion("Answer [1]"))
    monkeypatch.setattr(
        generation,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    assert list(generation.stream_answer("q", [chunk(1)])) == ["Answer [1]"]
    assert create.call_args.kwargs["model"] == provider[0].openai_chat_model
    assert create.call_args.kwargs["max_tokens"] == 2048
    assert "stream" not in create.call_args.kwargs
    assert provider[1].records[0]["role"] == "generation"


def test_native_rerank_index_mapping_and_usage_no_chat(provider, monkeypatch):
    response = Mock()
    response.json.return_value = {
        "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}],
        "usage": {"total_tokens": 10},
    }
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.post.return_value = response
    factory = Mock(return_value=client)
    monkeypatch.setattr(siliconflow.httpx, "Client", factory)
    assert siliconflow.native_rerank("q", ["first", "second"], 2) == [(1, 0.9), (0, 0.2)]
    assert client.post.call_args.args == ("https://api.siliconflow.cn/v1/rerank",)
    assert client.post.call_args.kwargs["json"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert factory.call_args.kwargs["follow_redirects"] is False
    assert provider[1].records[0]["role"] == "native_rerank"


@pytest.mark.parametrize(
    "usage", [None, {}, {"total_tokens": True}, {"total_tokens": -1}, {"total_tokens": 10001}]
)
def test_historical_invalid_usage_stops_future_requests(legacy_provider, usage):
    budget = legacy_provider[1]
    call = Mock(return_value={"usage": usage})
    with pytest.raises(ProviderBudgetError):
        budget.call("embedding", "BAAI/bge-m3", call)
    with pytest.raises(ProviderBudgetError):
        budget.call("embedding", "BAAI/bge-m3", call)
    assert call.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "unverified",
        "wrong_model",
        "missing_rate",
        "currency",
        "unit",
        "negative",
        "nan",
    ],
)
def test_bad_pricing_prevents_invoke(provider, change):
    budget = provider[1]
    p = budget.config["roles"]["embedding"]
    if change == "unverified":
        p["price_verified_from_screenshot"] = False
    elif change == "wrong_model":
        p["model"] = "Pro/BAAI/bge-m3"
    elif change == "missing_rate":
        p["input_rate"] = None
    elif change == "currency":
        p["currency"] = "USD"
    elif change == "unit":
        p["unit"] = "per_request"
    elif change == "negative":
        p["input_rate"] = "-1"
    elif change == "nan":
        p["input_rate"] = "NaN"
    elif change == "bound":
        p["input_token_upper_bound"] = True
    elif change == "byte_limit":
        p["request_byte_limit"] = None
    invoke = Mock()
    with pytest.raises((ProviderCostError, ValueError)):
        siliconflow.guarded_call("embedding", "BAAI/bge-m3", invoke, {"input": ["test"]})
    invoke.assert_not_called()


def test_runtime_ignores_historical_budget_bounds(provider):
    provider[1].config["roles"]["embedding"].update(
        verified=False,
        input_token_upper_bound=None,
        request_byte_limit=None,
    )
    invoke = Mock(return_value={"usage": {"total_tokens": 100001}})
    siliconflow.guarded_call("embedding", "BAAI/bge-m3", invoke, {"input": ["test"]})
    invoke.assert_called_once()
    assert provider[1].summary()["budget_enforced"] is False


def test_unverified_repository_profile_and_zero_budget_prevent_calls():
    path = "configs/providers/siliconflow.pricing.json"
    with pytest.raises(ProviderBudgetError):
        ProviderBudget(path, Decimal("0"))
    with pytest.raises(ProviderBudgetError):
        ProviderBudget(path, Decimal("1")).profile("embedding", "BAAI/bge-m3")


def test_native_not_substituted_in_search():
    from app.services import search

    assert search.rerank is reranking.rerank


def test_sdk_response_contract_is_not_dict():
    response = CreateEmbeddingResponse.model_validate(
        {
            "data": [],
            "model": "fixture",
            "object": "list",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    )
    assert response.data == []


def test_screenshot_prices_normalized_from_thousand_to_million():
    config = json.loads(Path("configs/providers/siliconflow.pricing.json").read_text())
    assert config["ledger_currency"] == "CNY"
    assert "fx" not in config
    cards = config["price_cards"]
    for model in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
        card = cards[model]
        assert Decimal(card["raw_input_rate"]) * 1000 == Decimal(card["input_per_million"]) == 0
    chat = cards["deepseek-ai/DeepSeek-V4.1-Flash"]
    assert chat["time_zone"] is None
    for tier in chat["tiers"]:
        for key in ("cached_input_rate", "input_rate", "output_rate"):
            assert Decimal(tier["raw_" + key]) * 1000 == Decimal(tier[key])
    for role in ("llm_rerank", "generation"):
        profile = config["roles"][role]
        assert profile["price_verified_from_screenshot"] is True
        assert profile["verified"] is False
        assert Decimal(profile["input_rate"]) == max(
            Decimal(t["input_rate"]) for t in chat["tiers"]
        )
        assert Decimal(profile["output_rate"]) == max(
            Decimal(t["output_rate"]) for t in chat["tiers"]
        )


@pytest.mark.parametrize("value", ["2", "10"])
def test_explicit_cny_budget_is_not_the_old_one_usd_cap(value):
    settings = sf_settings(siliconflow_max_cost_cny=value)
    assert settings.siliconflow_max_cost_cny == Decimal(value)


def test_old_usd_env_setting_is_not_reinterpreted_as_cny(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_MAX_COST_CNY", raising=False)
    monkeypatch.setenv("SILICONFLOW_MAX_COST_USD", "1")
    assert sf_settings().siliconflow_max_cost_cny == 0


@pytest.mark.parametrize("version,currency", [(1, "CNY"), (2, "USD")])
def test_old_or_foreign_currency_ledger_schema_rejected(tmp_path, version, currency):
    config = price_config()
    config.update(schema_version=version, ledger_currency=currency)
    path = tmp_path / "incompatible.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ProviderBudgetError):
        ProviderBudget(str(path), Decimal("1"))


def test_cny_budget_deducts_cny_and_reserves_before_request(legacy_provider):
    provider = legacy_provider
    budget = provider[1]
    budget.config["roles"]["generation"].update(
        input_rate="2",
        output_rate="8",
        input_token_upper_bound=100,
        output_token_upper_bound=10,
    )
    model = provider[0].openai_chat_model
    reply = completion(usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
    request = Mock(return_value=reply)
    budget.limit = Decimal("0.00028")
    assert budget.call("generation", model, request, max_output=10) is reply
    assert budget.spent == Decimal("0.00028")
    with pytest.raises(ProviderBudgetError):
        budget.call("generation", model, request, max_output=10)
    request.assert_called_once()


def test_free_encoder_still_requires_complete_usage_and_positive_budget(legacy_provider):
    budget = legacy_provider[1]
    budget.config["roles"]["embedding"]["input_rate"] = "0"
    budget.call("embedding", "BAAI/bge-m3", lambda: {"usage": {"total_tokens": 10}})
    assert budget.spent == 0
    assert budget.records[-1]["estimated_cost_cny"] == "0"
    with pytest.raises(ProviderBudgetError):
        budget.call("embedding", "BAAI/bge-m3", lambda: {"usage": None})
    assert budget.blocked
