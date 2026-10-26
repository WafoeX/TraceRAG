"""Offline checks of fixed-candidate experiments; sockets disabled by pytest."""

from types import SimpleNamespace

import pytest

from app.services.provider_costs import CostTracker
from scripts import optimize_scifact_rerank as experiment
from tests.test_siliconflow import sf_settings


def case():
    return {
        "query_id": "q",
        "question": "claim",
        "documents": ["passage"] * 10,
        "fused": [{"chunk_id": str(i), "score": 0.1} for i in range(12)],
        "payload": {
            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": "original prompt"}],
            "temperature": 0,
            "max_tokens": 1024,
        },
    }


def tracker(tmp_path):
    return CostTracker(
        "configs/providers/siliconflow.pricing.json",
        log_path=tmp_path / "cost.jsonl",
        capture_payloads=True,
        context=lambda: experiment.CONTEXT.get() or {},
    )


@pytest.mark.parametrize("mode", ["llm_low", "llm_no_thinking"])
def test_chat_controls_preserve_prompt_and_tail(tmp_path, mode):
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 30,
                "total_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="[0,1,2,3,4,5,6,7,8,9]")
                )
            ],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    original = case()
    result = experiment.measure(original, mode, client, tracker(tmp_path), sf_settings())
    assert result["fallback"] is None
    assert seen == [{**original["payload"], "extra_body": experiment.CONTROLS[mode]}]
    assert result["call"]["request"] == {**original["payload"], **experiment.CONTROLS[mode]}
    assert result["chunks"][:10] == [
        {"chunk_id": str(i), "score": float(i)} for i in range(9, -1, -1)
    ]
    assert result["chunks"][10:] == original["fused"][10:]
    assert "extra_body" not in original["payload"]


def test_request_failure_keeps_denominator_and_unknown_cost(tmp_path):
    def fail(**kwargs):
        raise TimeoutError("not logged")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    original = case()
    result = experiment.measure(original, "llm_low", client, tracker(tmp_path), sf_settings())
    assert result["chunks"] == original["fused"]
    assert result["fallback"] == "ProviderRequestError"
    assert result["call"]["estimated_cost_cny"] is None


def test_invalid_scores_fallback_not_free(tmp_path):
    def create(**kwargs):
        return SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="[1]"))],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = experiment.measure(case(), "llm_no_thinking", client, tracker(tmp_path), sf_settings())
    assert result["fallback"] == "ProviderResponseError"
    assert result["call"]["estimated_cost_cny"] is not None


def test_stable_ties_and_tail():
    original = case()["fused"]
    assert experiment.reorder(original, [1] * 10)[:10] == [
        {"chunk_id": str(i), "score": 1.0} for i in range(10)
    ]
    assert experiment.reorder(original, [1] * 10)[10:] == original[10:]


def test_all_zero_scores_preserve_fused_order(tmp_path):
    def create(**kwargs):
        return SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 30},
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content="[0,0,0,0,0,0,0,0,0,0]"))],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    original = case()
    result = experiment.measure(
        original, "llm_no_thinking", client, tracker(tmp_path), sf_settings()
    )
    assert result["fallback"] is not None
    assert result["chunks"] == original["fused"]


def test_native_maps_indices_without_rescaling_or_losing_tail(tmp_path, monkeypatch):
    from app.services import siliconflow

    costs = tracker(tmp_path)
    settings = sf_settings()

    def native(question, documents, top_n):
        assert (question, documents, top_n) == ("claim", ["passage"] * 10, 10)
        costs.call("native_rerank", settings.native_rerank_model, lambda: {"results": []})
        return [(i, i / 10) for i in range(9, -1, -1)]

    monkeypatch.setattr(siliconflow, "native_rerank", native)
    result = experiment.measure(case(), "native", None, costs, settings)
    assert result["fallback"] is None
    assert result["chunks"][:10] == [
        {"chunk_id": str(i), "score": i / 10} for i in range(9, -1, -1)]
    assert result["chunks"][10:] == case()["fused"][10:]
    assert result["call"]["estimated_cost_cny"] is None
