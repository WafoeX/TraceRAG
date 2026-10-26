"""Offline prompt ablation contracts; no real corpus, DB or API required."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI

from app.services import reranking
from app.services.rerank_prompts import VERSIONS, ScoreContractError, build_messages, parse_scores
from scripts import reduce_scifact_fallbacks as experiment
from tests.test_siliconflow import chunk, completion, provider  # noqa: F401


def keyed(values):
    return json.dumps({"scores": {str(i): v for i, v in enumerate(values)}})


def test_id_mapping_not_json_member_order():
    assert parse_scores('{"scores":{"1":8,"0":2}}', 2) == [2.0, 8.0]
    assert parse_scores(keyed([0, 0]), 2) == [0, 0]


@pytest.mark.parametrize(
    "content",
    [
        "[2,9]",
        '{"scores":[2,9]}',
        '{"scores":{"0":2}}',
        '{"scores":{"0":2,"1":9,"2":1}}',
        '{"scores":{"0":2,"0":3,"1":9}}',
        '{"scores":{"0":2,"1":9},"scores":{"0":1,"1":2}}',
        keyed([True, 2]),
        keyed(["2", 9]),
        keyed([float("nan"), 2]),
        keyed([float("inf"), 2]),
        keyed([-1, 2]),
        keyed([11, 2]),
        '{"scores":{"00":2,"1":9}}',
        "```json\n" + keyed([2, 9]) + "\n```",
    ],
)
def test_contract_never_silently_repairs(content):
    with pytest.raises(ScoreContractError):
        parse_scores(content, 2)


@pytest.mark.parametrize("version", VERSIONS)
def test_prompt_no_labels_and_complete_contract(version):
    text = build_messages("claim", ["docA", "docB"], version)[1].get("content")
    assert isinstance(text, str)
    assert "[0] docA\n\n[1] docB" in text
    assert '"0", "1"' in text
    assert "No arrays" in text
    if version == "relevance_v1":
        assert "supporting OR contradicting" in text
        assert "All scores may be zero" in text


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("values", [[2, 9], [0, 0]])
def test_application_contract_keeps_original_zero_fallback(provider, monkeypatch, version, values):  # noqa: F811
    settings, tracker = provider
    settings.rerank_prompt_version = version
    settings.llm_rerank_enable_thinking = False
    tracker.capture_payloads = True
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion(keyed(values)).model_dump())

    with OpenAI(
        api_key="fixture",
        base_url="https://api.siliconflow.cn/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(reranking, "get_client", lambda: client)
        original = [chunk(1), chunk(2)]
        result = reranking.rerank("claim", original, 2)
    assert [c.chunk_id for c in result] == ([2, 1] if any(values) else [1, 2])
    assert sent[0]["enable_thinking"] is False
    assert sent[0]["messages"] == build_messages("claim", ['(from "fixture") passage'] * 2, version)
    assert "response_format" not in sent[0]
    assert tracker.records[0]["request"] == sent[0]


def test_screen_rule_does_not_trade_quality_for_zero_fallbacks():
    baseline = {"metrics": {"nDCG@10": 0.75}, "fallbacks": 25}
    scores = {
        "keyed_v1": {"metrics": {"nDCG@10": 0.75}, "fallbacks": 5},
        "relevance_v1": {"metrics": {"nDCG@10": 0.70}, "fallbacks": 0},
    }
    assert experiment.choose(scores, baseline) == "keyed_v1"
    scores["keyed_v1"]["fallbacks"] = 25
    assert experiment.choose(scores, baseline) is None


def test_measure_and_replay_keep_tail_and_real_zero_reason(tmp_path):
    from app.services.provider_costs import CostTracker

    tracker = CostTracker(
        "configs/providers/siliconflow.pricing.json",
        capture_payloads=True,
        context=lambda: (experiment.full.experiment.CONTEXT.get() or {}).copy(),
    )
    base = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "temperature": 0, "max_tokens": 1024}
    case = {
        "query_id": "q",
        "question": "claim",
        "documents": ["passage"] * 10,
        "payload": base,
        "fused": [{"chunk_id": str(i), "score": 0.1} for i in range(12)],
    }
    create = Mock(return_value=completion(keyed([0] * 10)))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    row = experiment.measure(case, "relevance_v1", client, tracker)
    assert row["fallback"] == "all_zero_scores"
    assert row["chunks"] == case["fused"]
    assert experiment.decode(row["call"], 10) == ([0.0] * 10, "all_zero_scores")
    assert row["call"]["estimated_cost_cny"] is not None
    assert "messages" not in base


def test_request_failure_remains_unknown(tmp_path):
    from app.services.provider_costs import CostTracker

    tracker = CostTracker(
        "configs/providers/siliconflow.pricing.json",
        capture_payloads=True,
        context=lambda: (experiment.full.experiment.CONTEXT.get() or {}).copy(),
    )
    case = {
        "query_id": "q",
        "question": "claim",
        "documents": ["passage"] * 10,
        "payload": {"model": "deepseek-ai/DeepSeek-V4.1-Flash"},
        "fused": [{"chunk_id": str(i), "score": 0.1} for i in range(12)],
    }
    create = Mock(side_effect=TimeoutError("fixture"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    row = experiment.measure(case, "keyed_v1", client, tracker)
    assert row["fallback"] == "request_error"
    assert row["chunks"] == case["fused"]
    assert row["call"]["estimated_cost_cny"] is None
