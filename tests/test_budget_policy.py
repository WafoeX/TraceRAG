"""Pure pricing/usage/CLI bootstrap tests; no DB, credentials or provider calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from app.services.budget_policy import BudgetError, cost, load_profile, money, usage_cost

FIXTURE = Path(__file__).parent / "fixtures/application-budget-pricing.json"
PAYLOAD = {"model": "fixture-chat", "messages": [], "max_tokens": 2048}


def profile(path=FIXTURE):
    return load_profile(
        str(path), "https://offline.invalid/v1", "generation", "fixture-chat", PAYLOAD
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("verified", False),
        ("input_rate", None),
        ("input_rate", "NaN"),
        ("input_rate", "0.0000000000001"),
        ("input_rate", "-1"),
        ("input_rate", True),
        ("output_rate", "Infinity"),
        ("input_token_upper_bound", None),
        ("input_token_upper_bound", True),
        ("request_byte_limit", 0),
        ("output_token_upper_bound", 100),
        ("limits_source", None),
        ("price_source", ""),
        ("usage_source", ""),
        ("checked_at", ""),
    ],
)
def test_unverified_price_or_bound_rejects_before_call(tmp_path, field, value):
    data = json.loads(FIXTURE.read_text())
    data["profiles"][1][field] = value
    path = tmp_path / "price.json"
    path.write_text(json.dumps(data))
    with pytest.raises(BudgetError):
        profile(path)


@pytest.mark.parametrize("field,value", [("currency", "USD"), ("unit", "per_thousand_tokens")])
def test_exact_endpoint_and_model_required_and_no_fx(tmp_path, field, value):
    with pytest.raises(BudgetError):
        load_profile(
            str(FIXTURE), "https://api.deepseek.com/v1", "generation", "fixture-chat", PAYLOAD
        )
    data = json.loads(FIXTURE.read_text())
    data[field] = value
    path = tmp_path / "usd.json"
    path.write_text(json.dumps(data))
    with pytest.raises(BudgetError):
        profile(path)


def test_duplicate_profile_and_credential_url_are_rejected(tmp_path):
    data = json.loads(FIXTURE.read_text())
    data["profiles"].append(data["profiles"][1])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(data))
    with pytest.raises(BudgetError):
        profile(path)
    with pytest.raises(BudgetError):
        load_profile(
            str(FIXTURE),
            "https://secret:password@offline.invalid/v1",
            "generation",
            "fixture-chat",
            PAYLOAD,
        )


def test_payload_byte_bound_and_default_example_fail_closed():
    with pytest.raises(BudgetError, match="budget_request_bound_exceeded"):
        load_profile(
            str(FIXTURE),
            "https://offline.invalid/v1",
            "generation",
            "fixture-chat",
            {**PAYLOAD, "messages": ["x" * 100000]},
        )
    with pytest.raises(BudgetError):
        load_profile(
            "configs/providers/application-budget.example.json",
            "https://api.openai.com/v1",
            "generation",
            "gpt-4o-mini",
            PAYLOAD,
        )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 13},
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    ],
)
def test_unknown_usage_is_not_zero(usage):
    with pytest.raises(BudgetError, match="budget_usage_unknown"):
        usage_cost(profile(), "generation", {"usage": usage})


def test_known_usage_and_reasoning_not_double_counted():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 4},
    }
    assert usage_cost(profile(), "generation", {"usage": usage}) == Decimal("0.00002")
    with pytest.raises(BudgetError, match="budget_bound_violation"):
        usage_cost(
            profile(),
            "generation",
            {"usage": {"prompt_tokens": 10001, "completion_tokens": 1, "total_tokens": 10002}},
        )


@pytest.mark.parametrize("value", ["NaN", "-1", "Infinity", True, "0.0000000000001", "1000000001"])
def test_money_rejects_unsafe_amounts(value):
    with pytest.raises(BudgetError):
        money(value)


def test_estimates_round_up_not_down():
    assert cost({"input_rate": "0.00000001", "output_rate": "0"}, 1, 0) == Decimal("0.000000000001")


@pytest.mark.parametrize(
    "module,args",
    [
        ("scripts.migrate_application_budget", []),
        ("scripts.manage_application_budget", ["--help"]),
    ],
)
def test_cli_plan_and_help_do_not_load_dotenv_or_require_model_key(tmp_path, module, args):
    root = str(Path(__file__).resolve().parents[1])
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=must-not-load\nDATABASE_URL=invalid-private-url\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": root,
        "OPENAI_API_KEY": "",
        "DATABASE_URL": "private-invalid-sentinel",
    }
    result = subprocess.run(
        [sys.executable, "-m", module, *args], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "private-invalid" not in result.stdout + result.stderr
    if not args:
        assert "PLAN ONLY" in result.stdout


def test_cli_failure_does_not_echo_database_credentials():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.manage_application_budget",
            "--database-url",
            "unknown-driver://private-user:private-password@localhost/private-db",
            "show",
            "--global",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "private-password" not in result.stdout + result.stderr
    assert "private-db" not in result.stdout + result.stderr
