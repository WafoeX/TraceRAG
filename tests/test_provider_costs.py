"""Cost-only accounting: missing cost is never zero or a budget stop."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.services.provider_costs import CostTracker, ProviderCostError, ProviderRequestError


@pytest.fixture
def tracker(tmp_path):
    return CostTracker(
        "configs/providers/siliconflow.pricing.json",
        log_path=tmp_path / "costs.jsonl",
        secret="secret-key",
        capture_payloads=True,
    )


def test_reasoning_included_not_double_counted(tracker):
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 110,
        "total_tokens": 210,
        "completion_tokens_details": {"reasoning_tokens": 100},
    }
    tracker.call("generation", "deepseek-ai/DeepSeek-V4.1-Flash", lambda: {"usage": usage})
    row = tracker.records[0]
    assert row["usage"] == usage
    assert Decimal(row["estimated_cost_cny"]) == Decimal("0.00108")
    assert row["actual_cost_cny"] is None
    assert tracker.summary()["by_model"][row["model"]]["calls"] == 1


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": True}, {"total_tokens": -1}])
def test_unknown_does_not_block_or_become_free(tracker, usage):
    for _ in range(2):
        tracker.call("embedding", "BAAI/bge-m3", lambda: {"usage": usage})
    summary = tracker.summary()
    assert summary["estimated_cost_cny"] is None
    assert summary["known_estimated_cost_cny"] == "0"
    assert summary["unpriced_calls"] == 2
    assert summary["by_role"]["embedding"]["estimated_cost_cny"] is None
    assert not summary["budget_enforced"]


def test_provider_error_logged_and_sanitized_then_can_continue(tracker):
    def fail():
        raise RuntimeError("secret-key")

    with pytest.raises(ProviderRequestError, match="Provider request failed"):
        tracker.call("embedding", "BAAI/bge-m3", fail, request={"key": "secret-key"})
    tracker.call("embedding", "BAAI/bge-m3", lambda: {"usage": {"total_tokens": 1}})
    text = Path(tracker.log_path).read_text()
    assert "secret-key" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert [r["event"] for r in rows] == ["started", "finished", "started", "finished"]
    assert tracker.summary()["estimated_cost_cny"] is None
    assert tracker.failed_requests == 1


def test_unknown_reasoning_invalid_not_priced(tracker):
    tracker.call(
        "generation",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        lambda: {
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        },
    )
    assert tracker.records[0]["estimated_cost_cny"] is None


def test_model_mismatch_never_guesses_price(tracker):
    with pytest.raises(ProviderCostError):
        tracker.call("generation", "wrong-model", lambda: pytest.fail("must not invoke"))
