"""Per-wire budget gate: application contexts require durable preauthorization."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.engine import Engine

from app.config import get_settings
from app.services import budget_ledger as ledger
from app.services import tracing
from app.services.budget_policy import BudgetError, cost, load_profile, usage_cost
from app.services.resilience import (
    ProviderUnavailable,
    application_active,
    budget_owner,
    warn,
)

T = TypeVar("T")


def budget_engine() -> Engine:
    from app.database import SessionLocal

    with SessionLocal() as db:
        bind = db.get_bind()
    if not isinstance(bind, Engine):
        raise BudgetError("budget_unavailable")
    return bind


def priced_call(role: str, model: str, payload: dict, invoke: Callable[[], T]) -> T:
    if not application_active():
        return invoke()
    event = {
        "role": role,
        "model": model,
        "status": "not_sent",
        "estimated_cny": None,
        "usage": None,
    }
    event["config"] = {
        key: payload[key]
        for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed")
        if type(payload.get(key)) in (int, float, bool)
    }
    start = time.perf_counter()
    try:
        return _priced_call(role, model, payload, invoke, event)
    except Exception as exc:
        from app.services.resilience import failure_code

        event["error"] = failure_code(exc)
        raise
    finally:
        event["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
        tracing.call_event(event)


def _priced_call(role: str, model: str, payload: dict, invoke: Callable[[], T], event: dict) -> T:
    try:
        settings = get_settings()
        base_url = (settings.deepseek_base_url if role != "embedding" else None) or (
            settings.openai_base_url or "https://api.openai.com/v1"
        )
        profile = load_profile(settings.application_pricing_file, base_url, role, model, payload)
        user_id = budget_owner()
        if user_id is None:
            raise BudgetError("budget_owner_missing")
        engine = budget_engine()
        amount = cost(
            profile, profile["input_token_upper_bound"], profile["output_token_upper_bound"]
        )
        identifier = ledger.reserve(engine, user_id, role, model, profile, amount)
        event.update(reservation_id=identifier, reserved_cny=str(amount), status="unknown")
    except BudgetError as exc:
        raise ProviderUnavailable(str(exc)) from None
    except Exception:
        # Includes uncertain commit receipts: never invoke before acknowledged commit.
        raise ProviderUnavailable("budget_unavailable") from None
    try:
        if role == "generation":
            tracing.generation_payload(payload)
        response = invoke()
    except Exception:
        try:
            # No provider error body/status is trusted as proof of zero billing.
            ledger.retain_unknown(engine, identifier, "provider_outcome_unknown")
        except Exception:
            raise ProviderUnavailable("budget_unavailable") from None
        warn("budget_charge_unknown")
        raise
    try:
        estimate = usage_cost(profile, role, response)
    except BudgetError as exc:
        try:
            ledger.retain_unknown(
                engine, identifier, str(exc), block=str(exc) == "budget_bound_violation"
            )
        except Exception:
            raise ProviderUnavailable("budget_unavailable") from None
        warn("budget_charge_unknown")
        if str(exc) == "budget_bound_violation":
            raise ProviderUnavailable("budget_bound_violation") from None
        # User chose to retain unknown holds but allow other calls within remaining
        # quota; the usable model result is not represented as zero-cost.
        return response
    raw = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        raw = dump()
    # usage_cost has already validated these counters; never retain arbitrary provider fields.
    keys = (
        ("total_tokens",)
        if role == "embedding"
        else ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if isinstance(raw, dict):
        event["usage"] = {key: raw[key] for key in keys}
    try:
        ledger.settle(engine, identifier, estimate)
    except Exception:
        # A successful provider call with uncertain settlement keeps its original
        # hold (or a committed idempotent settlement). Do not automatically retry.
        raise ProviderUnavailable("budget_unavailable") from None
    event.update(status="settled", estimated_cny=str(estimate))
    return response
