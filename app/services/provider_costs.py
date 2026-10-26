"""CNY usage/cost recording, without spend caps, reservations or budget locks."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar
from uuid import uuid4

T = TypeVar("T")


class ProviderCostError(ValueError):
    """Invalid price/model/currency configuration; never guess a rate."""


class ProviderRequestError(RuntimeError):
    """Provider failed; original credential-bearing error body is not exposed."""


def rate(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception:
        raise ProviderCostError("Invalid CNY rate") from None
    if not result.is_finite() or result < 0:
        raise ProviderCostError("CNY rates must be finite and nonnegative")
    return result


class CostTracker:
    """Append start/result events; unknown usage is not zero and does not poison calls.

    The lock protects JSONL writes/summary consistency, NOT a spend allocation.
    Default logs contain usage, not private prompts. Public benchmark capture can
    explicitly request payloads/responses. Provider failures still fail normally.
    """

    def __init__(
        self,
        pricing_file: str | Path,
        *,
        log_path: str | Path | None = None,
        secret: str = "",
        capture_payloads: bool = False,
        context: Callable[[], dict[str, Any]] | None = None,
        on_result: Callable[[dict[str, Any]], None] | None = None,
    ):
        try:
            self.config = json.loads(Path(pricing_file).read_text())
        except (OSError, ValueError):
            raise ProviderCostError("Cannot read pricing configuration") from None
        if (
            not isinstance(self.config, dict)
            or self.config.get("schema_version") != 2
            or self.config.get("ledger_currency") != "CNY"
        ):
            raise ProviderCostError("CNY pricing schema v2 required")
        self.log_path = Path(log_path) if log_path is not None else None
        self._secret = secret
        self.capture_payloads = capture_payloads
        self.context = context or (lambda: {})
        self.on_result = on_result
        self.records: list[dict[str, Any]] = []
        self.failed_requests = 0
        self.logging_failed = False
        self.lock = RLock()

    def profile(self, role: str, model: str) -> dict[str, Any]:
        try:
            p = self.config["roles"][role]
            if p["model"] != model or p.get("price_verified_from_screenshot") is not True:
                raise ProviderCostError("Exact model and confirmed price required")
            if p["currency"] != "CNY" or p["unit"] != "per_million_tokens":
                raise ProviderCostError("CNY per-million-token prices required")
            for key in ("source", "checked_at"):
                if not isinstance(p[key], str) or not p[key].strip():
                    raise ProviderCostError("Price provenance required")
            rate(p["input_rate"])
            output_rate = rate(p["output_rate"])
            if role in {"embedding", "native_rerank"} and output_rate != 0:
                raise ProviderCostError("Encoder output price is not applicable")
            return p
        except (KeyError, TypeError):
            raise ProviderCostError("Incomplete model price configuration") from None

    def safe(self, value: Any) -> Any:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if self._secret:
                encoded = encoded.replace(self._secret, "[REDACTED]")
            return json.loads(encoded)
        except (TypeError, ValueError):
            raise ProviderCostError("Nonserializable provider record") from None

    def emit(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            with self.lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError:
            self.logging_failed = True
            raise RuntimeError("Cannot persist model cost record") from None

    def estimate(self, p: dict[str, Any], usage: Any, role: str) -> Decimal:
        if not isinstance(usage, dict):
            raise ValueError("missing_usage")
        chat = role in {"llm_rerank", "generation"}
        keys = ("prompt_tokens", "completion_tokens") if chat else ("total_tokens",)
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in keys):
            raise ValueError("invalid_usage")
        inputs = usage[keys[0]]
        outputs = usage["completion_tokens"] if chat else 0
        if (
            chat
            and "total_tokens" in usage
            and (
                type(usage["total_tokens"]) is not int or usage["total_tokens"] != inputs + outputs
            )
        ):
            raise ValueError("inconsistent_total_tokens")
        # completion_tokens already includes reported reasoning: never add it again.
        if chat and isinstance(usage.get("completion_tokens_details"), dict):
            reasoning = usage["completion_tokens_details"].get("reasoning_tokens")
            if reasoning is not None and (
                type(reasoning) is not int or not 0 <= reasoning <= outputs
            ):
                raise ValueError("invalid_reasoning_tokens")
        return (inputs * rate(p["input_rate"]) + outputs * rate(p["output_rate"])) / 1_000_000

    def call(self, role: str, model: str, invoke: Callable[[], T], *, request: Any = None) -> T:
        p = self.profile(role, model)
        started = time.perf_counter()
        row = {
            **self.context(),
            "request_id": uuid4().hex,
            "role": role,
            "kind": "rerank" if role == "llm_rerank" else role,
            "model": model,
            "currency": "CNY",
            "usage": None,
            "estimated_cost_cny": None,
            "actual_cost_cny": None,
            "price_input_per_million_cny": str(rate(p["input_rate"])),
            "price_output_per_million_cny": str(rate(p["output_rate"])),
            "pricing_policy": "higher_full_input_no_cache_discount",
            "billing_audited": False,
        }
        if self.capture_payloads:
            row["request"] = self.safe(request)
        self.emit({**row, "event": "started", "status": "pending"})
        try:
            response = invoke()
        except Exception as exc:
            row.update(status="error", cost_status="unknown", error_type=type(exc).__name__)
            self.failed_requests += 1
            self.finish(row, started)
            raise ProviderRequestError("Provider request failed; inspect cost record") from None
        raw: Any = response
        try:
            usage: Any = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            row["usage"] = self.safe(usage)
            row["estimated_cost_cny"] = str(self.estimate(p, usage, role))
            row["cost_status"] = "estimated"
        except (ValueError, TypeError) as exc:
            row.update(cost_status="unknown", usage_error_type=type(exc).__name__)
        if self.capture_payloads:
            try:
                payload = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
                row["response"] = self.safe(payload)
            except (AttributeError, TypeError, ValueError):
                row["response_capture_error"] = True
        row["status"] = "ok"
        self.finish(row, started)
        return response

    def finish(self, row: dict[str, Any], started: float) -> None:
        row["elapsed_ms"] = (time.perf_counter() - started) * 1000
        self.emit({**row, "event": "finished"})
        with self.lock:
            self.records.append(row)
        if self.on_result is not None:
            self.on_result(row)

    @property
    def spent(self) -> Decimal:
        return sum(
            (
                Decimal(r["estimated_cost_cny"])
                for r in self.records
                if r["estimated_cost_cny"] is not None
            ),
            Decimal(0),
        )

    def group_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        known = sum(
            (Decimal(r["estimated_cost_cny"]) for r in rows if r["estimated_cost_cny"] is not None),
            Decimal(0),
        )
        unknown = sum(r["estimated_cost_cny"] is None for r in rows)
        return {
            "calls": len(rows),
            "known_estimated_cost_cny": str(known),
            "estimated_cost_cny": None if unknown else str(known),
            "actual_cost_cny": None,
            "unpriced_calls": unknown,
        }

    def summary(self) -> dict[str, Any]:
        with self.lock:
            unknown = sum(r["estimated_cost_cny"] is None for r in self.records)
            roles = {}
            for role in sorted({r["role"] for r in self.records}):
                rows = [r for r in self.records if r["role"] == role]
                roles[role] = self.group_summary(rows)
            models = {
                model: self.group_summary([r for r in self.records if r["model"] == model])
                for model in sorted({r["model"] for r in self.records})
            }
            return {
                "currency": "CNY",
                "calls": len(self.records),
                "unpriced_calls": unknown,
                "known_estimated_cost_cny": str(self.spent),
                "estimated_cost_cny": None if unknown else str(self.spent),
                "by_role": roles,
                "by_model": models,
                "actual_cost_cny": None,
                "budget_enforced": False,
                "billing_audited": False,
            }
