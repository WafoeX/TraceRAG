"""Fail-closed, process-local SiliconFlow estimate budget (not a billing guarantee)."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar

T = TypeVar("T")


class ProviderBudgetError(RuntimeError):
    """Pricing, usage or reservation is unsafe; do not issue another request."""


def number(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception:
        raise ProviderBudgetError("Missing/invalid CNY price") from None
    if not result.is_finite() or result < 0:
        raise ProviderBudgetError("Rates must be finite and nonnegative")
    return result


class ProviderBudget:
    """Serialize reserve/request/settle with lock; unknown usage poisons this budget.

    All rates, reservations, limits and records are CNY. No FX conversion.
    High uncached rates are conservative estimates, not exact provider invoices.
    """

    def __init__(self, path: str, limit: Decimal):
        if not limit.is_finite() or limit <= 0:
            raise ProviderBudgetError("Set an explicit positive budget in CNY")
        try:
            self.config = json.loads(Path(path).read_text())
        except Exception:
            raise ProviderBudgetError("Cannot load pricing configuration") from None
        if (
            not isinstance(self.config, dict)
            or self.config.get("schema_version") != 2
            or self.config.get("ledger_currency") != "CNY"
        ):
            raise ProviderBudgetError("CNY pricing schema v2 required; no implicit USD migration")
        self.limit = limit
        self.spent = Decimal(0)
        self.blocked = False
        self.lock = RLock()
        self.records: list[dict[str, Any]] = []

    def profile(self, role: str, model: str) -> dict[str, Any]:
        try:
            p = self.config["roles"][role]
            if p["model"] != model or p["verified"] is not True:
                raise ProviderBudgetError("Model-specific verified pricing required")
            for key in ("source", "checked_at", "limits_source", "usage_source"):
                if not isinstance(p[key], str) or not p[key].strip():
                    raise ProviderBudgetError("Price/limits/usage provenance required")
            if p["unit"] != "per_million_tokens" or p["currency"] != "CNY":
                raise ProviderBudgetError("Unsupported billing currency/unit")
            for key in ("input_rate", "output_rate"):
                number(p[key])
            for key in ("input_token_upper_bound", "output_token_upper_bound"):
                if type(p[key]) is not int or p[key] < (1 if key.startswith("input") else 0):
                    raise ProviderBudgetError("Verified request token bounds required")
            if role in {"llm_rerank", "generation"} and not p["output_token_upper_bound"]:
                raise ProviderBudgetError("Chat requires a positive output bound")
            if role in {"embedding", "native_rerank"} and number(p["output_rate"]) != 0:
                raise ProviderBudgetError("Encoder pricing must not invent output tokens")
            return p
        except (KeyError, TypeError):
            raise ProviderBudgetError("Incomplete pricing configuration") from None

    def cost(self, p: dict[str, Any], inputs: int, outputs: int) -> Decimal:
        return (inputs * number(p["input_rate"]) + outputs * number(p["output_rate"])) / 1_000_000

    def call(self, role: str, model: str, invoke: Callable[[], T], max_output: int = 0) -> T:
        with self.lock:
            if self.blocked:
                raise ProviderBudgetError("Unknown billing/usage; restart only after audit")
            p = self.profile(role, model)
            # Reserve the declared maximum for the WHOLE request, not per batch item.
            reserve = self.cost(p, p["input_token_upper_bound"], p["output_token_upper_bound"])
            if max_output > p["output_token_upper_bound"]:
                raise ProviderBudgetError("Configured max_tokens exceeds verified output bound")
            if self.spent + reserve > self.limit:
                raise ProviderBudgetError("Request exceeds remaining estimate budget")
            try:
                response = invoke()
                raw: Any = response
                usage: Any = raw.get("usage") if isinstance(raw, dict) else raw.usage
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                if not isinstance(usage, dict):
                    raise ProviderBudgetError("Missing usage; further calls disabled")
                keys = (
                    ("prompt_tokens", "completion_tokens")
                    if role in {"llm_rerank", "generation"}
                    else ("total_tokens",)
                )
                if any(type(usage.get(k)) is not int or usage[k] < 0 for k in keys):
                    raise ProviderBudgetError("Invalid token usage; further calls disabled")
                inputs = usage[keys[0]]
                outputs = usage[keys[1]] if len(keys) == 2 else 0
                if (
                    "total_tokens" in usage
                    and len(keys) == 2
                    and (
                        type(usage["total_tokens"]) is not int
                        or usage["total_tokens"] != inputs + outputs
                    )
                ):
                    raise ProviderBudgetError("Inconsistent chat token usage")
                if inputs > p["input_token_upper_bound"] or outputs > p["output_token_upper_bound"]:
                    raise ProviderBudgetError("Usage exceeds declared reservation; audit billing")
                cost = self.cost(p, inputs, outputs)
                self.spent += cost
                self.records.append(
                    {
                        "role": role,
                        "model": model,
                        "usage": usage,
                        "source_currency": p["currency"],
                        "currency": "CNY",
                        "estimated_cost_cny": str(cost),
                        "pricing_policy": "conservative_full_input_no_cache_discount",
                    }
                )
                return response
            except Exception:
                self.blocked = True
                raise ProviderBudgetError(
                    "Request failed or usage invalid; audit before retry"
                ) from None
