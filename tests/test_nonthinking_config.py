"""Offline wire/config contracts and frozen development plan checks."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError

from app.config import Settings
from app.services import generation, reranking
from scripts import validate_scifact_nonthinking as full
from tests.test_siliconflow import chunk, completion, provider, sf_settings  # noqa: F401


@pytest.mark.parametrize("control", [None, False, True])
def test_runtime_control_wire_and_cost_payload(provider, monkeypatch, control):  # noqa: F811
    settings, costs = provider
    costs.capture_payloads = True
    settings.llm_rerank_enable_thinking = control
    create = Mock(return_value=completion())
    monkeypatch.setattr(
        reranking,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    assert [c.chunk_id for c in reranking.rerank("q", [chunk(1), chunk(2)], 2)] == [2, 1]
    wire = create.call_args.kwargs["extra_body"]
    if control is None:
        assert wire is None
        assert "enable_thinking" not in costs.records[0]["request"]
    else:
        assert wire == {"enable_thinking": control}
        assert costs.records[0]["request"]["enable_thinking"] is control


@pytest.mark.parametrize("control", [None, False, True])
def test_sdk_http_field_and_generation_isolation(provider, monkeypatch, control):  # noqa: F811
    settings, _ = provider
    settings.llm_rerank_enable_thinking = control
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion().model_dump())

    with OpenAI(
        api_key="fixture",
        base_url="https://api.siliconflow.cn/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as c:
        monkeypatch.setattr(reranking, "get_client", lambda: c)
        monkeypatch.setattr(generation, "get_client", lambda: c)
        reranking.rerank("q", [chunk(1), chunk(2)], 2)
        generation.generate_answer("q", [chunk(1)])
    if control is None:
        assert "enable_thinking" not in sent[0]
    else:
        assert sent[0]["enable_thinking"] is control
    assert "enable_thinking" not in sent[1]
    assert "extra_body" not in sent[0]


def test_nonthinking_all_zero_retains_retrieval(provider, monkeypatch):  # noqa: F811
    provider[0].llm_rerank_enable_thinking = False
    create = Mock(return_value=completion("[0,0]"))
    monkeypatch.setattr(
        reranking,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    original = [chunk(1), chunk(2)]
    assert reranking.rerank("q", original, 2) == original
    assert provider[1].records[0]["estimated_cost_cny"] is not None


def test_settings_opt_in_is_provider_specific(monkeypatch):
    monkeypatch.delenv("LLM_RERANK_ENABLE_THINKING", raising=False)
    assert sf_settings().llm_rerank_enable_thinking is None
    with pytest.raises(ValidationError, match="only by SiliconFlow"):
        options = {
            "_env_file": None,
            "openai_api_key": "fixture",
            "model_provider": "openai",
            "llm_rerank_enable_thinking": False,
        }
        Settings(**options)


@pytest.fixture
def frozen_inputs(monkeypatch):
    # Synthetic data only: tests must also pass without ignored benchmark artifacts.
    data = {
        "queries": {str(i): f"claim {i}" for i in range(101)},
        "corpus": {"d": {"title": "title"}},
    }
    chunks = {str(i): {"document_id": "d", "text": "passage"} for i in range(100)}
    fused = [{"chunk_id": str(i), "score": 0.1} for i in range(100)]
    first = [
        {
            "query_id": q,
            "fused": fused,
            "payload": {
                "model": "fixture",
                "messages": reranking.build_messages(question, ['(from "title") passage'] * 10),
            },
        }
        for q, question in list(data["queries"].items())[:100]
    ]
    old = [
        {
            "query_id": c["query_id"],
            "mode": "llm_no_thinking",
            "call": {"request": {**c["payload"], "enable_thinking": False}},
        }
        for c in first
    ]
    retrieval = [{"query_id": q, "chunks": fused} for q in data["queries"]]
    monkeypatch.setattr(full.experiment, "inputs", lambda *args: (data, chunks, first))
    monkeypatch.setattr(full, "read_rows", lambda p: retrieval if p.name.startswith("fts") else old)
    return retrieval, old


def test_full_plan_is_offline_and_reuses_frozen_first100(frozen_inputs):
    data, chunks, cases, rows = full.full_inputs(
        full.experiment.SOURCE, full.experiment.CHUNKS, full.RETRIEVAL, full.REUSE
    )
    assert len(cases) == len(data["queries"]) == 101
    assert len(rows) == 100
    assert all(len(c["fused"]) >= 100 for c in cases)
    assert all(len(c["documents"]) == 10 for c in cases)
    assert chunks


@pytest.mark.parametrize("tamper", ["cohort", "reuse", "payload"])
def test_full_plan_refuses_changed_evidence(frozen_inputs, tamper):
    retrieval, old = frozen_inputs
    if tamper == "cohort":
        retrieval.pop()
    elif tamper == "reuse":
        old.append(old[0])
    else:
        old[0]["call"]["request"]["enable_thinking"] = True
    with pytest.raises(ValueError):
        full.full_inputs(full.experiment.SOURCE, full.experiment.CHUNKS, full.RETRIEVAL, full.REUSE)


def test_unknown_reasoning_not_counted_as_zero(tmp_path):
    row = {
        "query_id": "q",
        "chunks": [{"chunk_id": "c", "score": 1}],
        "rerank_ms": 1,
        "fallback": "error",
        "call": {
            "usage": None,
            "estimated_cost_cny": None,
            "role": "llm_rerank",
            "model": "fixture",
            "cost_status": "unknown",
        },
    }
    score = full.summary(
        [row],
        [{"query_id": "q"}],
        {"qrels": {"q": {"d": 1}}},
        {"c": {"document_id": "d"}},
        "configs/providers/siliconflow.pricing.json",
    )
    assert score["usage"]["reasoning_zero_calls"] == 0
    assert score["fallback_rate"] == 1
