"""Pure pricing/usage validation. No environment, DB, provider or implicit FX access."""

from __future__ import annotations

import json
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

QUANTUM = Decimal("0.000000000001")
MAX_MONEY = Decimal("1000000000")
ROLES = {"embedding", "llm_rerank", "generation"}


class BudgetError(RuntimeError):
    """Fixed safe public code; never include config, prompts or database errors."""


def money(value) -> Decimal:
    try:
        if isinstance(value, bool):
            raise ValueError
        result = Decimal(str(value))
        if not result.is_finite() or not 0 <= result <= MAX_MONEY:
            raise ValueError
        if result != result.quantize(QUANTUM):
            raise ValueError
        return result.quantize(QUANTUM)
    except Exception:
        raise BudgetError("budget_invalid_amount") from None


def rate(value) -> Decimal:
    try:
        if isinstance(value, bool):
            raise ValueError
        result = Decimal(str(value))
        if not result.is_finite() or not 0 <= result <= MAX_MONEY:
            raise ValueError
        if result != result.quantize(QUANTUM):
            raise ValueError
        return result.quantize(QUANTUM)
    except Exception:
        raise BudgetError("budget_pricing_unverified") from None


def endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        return value.rstrip("/")
    except Exception:
        raise BudgetError("budget_pricing_unverified") from None


def cost(profile: dict, inputs: int, outputs: int) -> Decimal:
    # Rates have at most 12 fractional places, token bounds at most 100M.
    # Use enough precision to avoid losing a tiny positive remainder before ceil.
    with localcontext() as context:
        context.prec = 60
        value = (
            inputs * rate(profile["input_rate"]) + outputs * rate(profile["output_rate"])
        ) / 1_000_000
        return money(value.quantize(QUANTUM, rounding=ROUND_CEILING))


def load_profile(path: str, base_url: str, role: str, model: str, payload: dict) -> dict:
    try:
        config = json.loads(Path(path).read_text())
        if (
            config["schema_version"] != 1
            or config["currency"] != "CNY"
            or config["unit"] != "per_million_tokens"
            or role not in ROLES
        ):
            raise ValueError
        profiles = [
            p
            for p in config["profiles"]
            if p["role"] == role
            and p["model"] == model
            and endpoint(p["base_url"]) == endpoint(base_url)
        ]
        if len(profiles) != 1:
            raise ValueError
        profile = profiles[0]
        if (
            profile["verified"] is not True
            or profile.get("currency", "CNY") != "CNY"
            or profile.get("unit", "per_million_tokens") != "per_million_tokens"
        ):
            raise ValueError
        for key in ("price_source", "limits_source", "usage_source", "checked_at"):
            if not isinstance(profile[key], str) or not profile[key].strip():
                raise ValueError
        for key in ("input_token_upper_bound", "output_token_upper_bound", "request_byte_limit"):
            if type(profile[key]) is not int or not 0 <= profile[key] <= 100_000_000:
                raise ValueError
        if not profile["input_token_upper_bound"] or not profile["request_byte_limit"]:
            raise ValueError
        rate(profile["input_rate"])
        output_rate = rate(profile["output_rate"])
        if role == "embedding":
            if output_rate or profile["output_token_upper_bound"]:
                raise ValueError
        else:
            maximum = payload.get("max_tokens")
            if type(maximum) is not int or not 0 < maximum <= profile["output_token_upper_bound"]:
                raise ValueError
        encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
        if len(encoded) > profile["request_byte_limit"]:
            raise BudgetError("budget_request_bound_exceeded")
        # An operator attests this WHOLE-request bound for every payload admitted
        # by request_byte_limit; not a guessed tokenizer count or per-item bound.
        cost(profile, profile["input_token_upper_bound"], profile["output_token_upper_bound"])
        return {
            **profile,
            "currency": "CNY",
            "unit": "per_million_tokens",
            "input_rate": str(rate(profile["input_rate"])),
            "output_rate": str(rate(profile["output_rate"])),
        }
    except BudgetError:
        raise
    except Exception:
        raise BudgetError("budget_pricing_unverified") from None


def usage_cost(profile: dict, role: str, response) -> Decimal:
    raw: Any = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        raise BudgetError("budget_usage_unknown")
    keys = ("prompt_tokens", "completion_tokens") if role != "embedding" else ("total_tokens",)
    if any(type(raw.get(k)) is not int or raw[k] < 0 for k in keys):
        raise BudgetError("budget_usage_unknown")
    inputs = raw[keys[0]]
    outputs = raw[keys[1]] if len(keys) == 2 else 0
    if len(keys) == 2 and (
        type(raw.get("total_tokens")) is not int or raw["total_tokens"] != inputs + outputs
    ):
        raise BudgetError("budget_usage_unknown")
    details = raw.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        reasoning = details["reasoning_tokens"]
        if type(reasoning) is not int or not 0 <= reasoning <= outputs:
            raise BudgetError("budget_usage_unknown")
    if inputs > profile["input_token_upper_bound"] or outputs > profile["output_token_upper_bound"]:
        raise BudgetError("budget_bound_violation")
    return cost(profile, inputs, outputs)
