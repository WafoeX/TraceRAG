"""Explicitly authorized, one-shot wire probe; no database or production guard changes.

Default is a no-call plan. --execute sends at most one embedding and two chat
requests. Only fixed public fixture text is sent. Production pricing profiles
remain unverified: this probe observes their usage contract, not general limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from app.config import get_settings
from app.services.openai_client import get_client
from app.services.siliconflow import chat_content, ordered_embeddings, validate_llm_scores

# Overreserve twice the screenshot's 1M context (covering decimal/binary labels).
# This is a probe-only policy, NOT a verified API/tokenizer or global model limit.
INPUT_RESERVATION = 2 * 1024 * 1024
OUTPUT_CAP = 64
INPUT_RATE = Decimal("2")
OUTPUT_RATE = Decimal("8")
CHAT_RESERVATION = (INPUT_RESERVATION * INPUT_RATE + OUTPUT_CAP * OUTPUT_RATE) / 1_000_000
ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, state: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def settle(usage: Any, role: str) -> tuple[dict[str, int], Decimal]:
    if usage is None:
        raise ValueError("Missing usage")
    data = usage if isinstance(usage, dict) else usage.model_dump()
    keys = ("total_tokens",) if role == "embedding" else ("prompt_tokens", "completion_tokens")
    if any(type(data.get(k)) is not int or data[k] < 0 for k in keys):
        raise ValueError("Invalid usage")
    if role == "embedding":
        return {k: data[k] for k in keys}, Decimal(0)
    inputs, outputs = data["prompt_tokens"], data["completion_tokens"]
    details = data.get("completion_tokens_details")
    details = {} if details is None else details
    if not isinstance(details, dict):
        raise ValueError("Invalid completion token details")
    reasoning = details.get("reasoning_tokens")
    reasoning = 0 if reasoning is None else reasoning
    if type(reasoning) is not int or not 0 <= reasoning <= outputs:
        raise ValueError("Invalid reasoning token count")
    visible = outputs - reasoning
    # Observed on this exact DeepSeek endpoint: max_tokens capped visible output,
    # while completion_tokens also included reasoning. Bill ALL completion tokens.
    if inputs > INPUT_RESERVATION or visible > OUTPUT_CAP:
        raise ValueError("Usage exceeded probe reservation")
    if type(data.get("total_tokens")) is not int or data["total_tokens"] != inputs + outputs:
        raise ValueError("Inconsistent total usage")
    cost = (inputs * INPUT_RATE + outputs * OUTPUT_RATE) / 1_000_000
    if cost > CHAT_RESERVATION:
        raise ValueError("Total CNY usage exceeded probe reservation")
    return {
        **{k: data[k] for k in (*keys, "total_tokens")},
        "reasoning_tokens": reasoning,
        "visible_completion_tokens": visible,
    }, cost


def run(
    output: Path,
    execute: bool = False,
    client_factory=None,
    *,
    only_role: str | None = None,
    prior_cost_cny: Decimal = Decimal("0"),
) -> dict[str, Any]:
    roles = ("embedding", "llm_rerank", "generation")
    if only_role is not None:
        if only_role not in roles:
            raise ValueError("Unknown probe role")
        roles = (only_role,)
    if not prior_cost_cny.is_finite() or prior_cost_cny < 0:
        raise ValueError("Prior reported cost must be finite and nonnegative")
    settings = get_settings()
    if (
        settings.model_provider != "siliconflow"
        or settings.openai_base_url != "https://api.siliconflow.cn/v1"
        or settings.openai_embedding_model != "BAAI/bge-m3"
        or settings.embedding_dim != 1024
        or settings.openai_chat_model != "deepseek-ai/DeepSeek-V4.1-Flash"
        or (settings.llm_rerank_model or settings.openai_chat_model) != settings.openai_chat_model
    ):
        raise ValueError("Probe model/endpoint differs from screenshot-authorized plan")
    try:
        prices = json.loads((ROOT / settings.siliconflow_pricing_file).read_text())
    except (OSError, ValueError):
        raise ValueError("Cannot read probe pricing; no calls made") from None
    if prices.get("schema_version") != 2 or prices.get("ledger_currency") != "CNY":
        raise ValueError("CNY screenshot price configuration required")
    for role, model, inp, out in (
        ("embedding", settings.openai_embedding_model, "0", "0"),
        ("llm_rerank", settings.openai_chat_model, "2", "8"),
        ("generation", settings.openai_chat_model, "2", "8"),
    ):
        profile = prices["roles"][role]
        if (
            profile["model"] != model
            or profile["currency"] != "CNY"
            or profile["unit"] != "per_million_tokens"
            or profile.get("price_verified_from_screenshot") is not True
            or Decimal(profile["input_rate"]) != Decimal(inp)
            or Decimal(profile["output_rate"]) != Decimal(out)
        ):
            raise ValueError("Probe screenshot prices changed; review first")
    configured_budget = settings.siliconflow_max_cost_cny
    budget = configured_budget - prior_cost_cny
    reservation_total = sum((CHAT_RESERVATION for r in roles if r != "embedding"), Decimal(0))
    if execute and (budget <= 0 or budget < reservation_total):
        raise ValueError("Budget does not cover conservative probe plan; no calls made")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    state: dict[str, Any] = {
        "status": "planned",
        "currency": "CNY",
        "configured_budget_cny": str(configured_budget),
        "prior_user_reported_cost_cny": str(prior_cost_cny),
        "prior_cost_source": "operator report; not independently audited",
        "available_budget_cny": str(budget),
        "selected_roles": list(roles),
        "worst_case_probe_reservation_cny": str(reservation_total),
        "input_reservation_policy": (
            "2 * 1048576 tokens per tiny chat request; not verified provider limit"
        ),
        "requested_max_tokens": OUTPUT_CAP,
        "output_cap_scope": "visible text in observed DeepSeek response; reasoning billed too",
        "sdk_max_retries": 0,
        "database_operations": 0,
        "production_profiles_modified": False,
        "model_attempts": 0,
        "known_estimated_cost_cny": "0",
        "estimated_cost_cny": "0",
        "billing_audit": False,
        "calls": [],
    }
    save(path, state)
    if not execute:
        return state
    factory = client_factory or get_client
    client = factory().with_options(max_retries=0, timeout=60)
    spent = Decimal(0)
    state["status"] = "running"
    try:
        for role in roles:
            reservation = Decimal(0) if role == "embedding" else CHAT_RESERVATION
            if spent + reservation > budget:
                raise ValueError("Remaining budget insufficient")
            row: dict[str, Any] = {
                "role": role,
                "status": "pending",
                "stage": "request",
                "reserved_cost_cny": str(reservation),
                "usage": None,
            }
            state["calls"].append(row)
            state["model_attempts"] += 1
            state["estimated_cost_cny"] = None
            save(path, state)  # Pending/unknown billing survives an interrupted call.
            if role == "embedding":
                response = client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=["refund within thirty days", "data encryption"],
                    encoding_format="float",
                )
            else:
                prompt = (
                    (
                        "Question: When can I request a refund?\n"
                        "Passages: [0] Refunds are available within thirty days. "
                        "[1] Data is encrypted.\nReturn only a JSON array of two relevance "
                        "scores from 0 to 10 in passage order."
                    )
                    if role == "llm_rerank"
                    else "Reply with exactly OK."
                )
                response = client.chat.completions.create(
                    model=settings.openai_chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=OUTPUT_CAP,
                )
            # Preserve usage BEFORE validating it so an anomalous response is
            # diagnosable without repeating a potentially billed request.
            if role != "embedding":
                # Public fixture only. Keep the response body, not request headers,
                # and redact the exact configured key in case a provider echoes it.
                serialized = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
                row["response"] = json.loads(
                    serialized.replace(settings.openai_api_key, "[REDACTED]")
                )
            row.update(
                stage="usage_validation",
                raw_usage=(
                    row["response"].get("usage")
                    if role != "embedding"
                    else (response.usage.model_dump() if response.usage else None)
                ),
            )
            save(path, state)
            usage, cost = settle(response.usage, role)
            spent += cost
            row.update(usage=usage, estimated_cost_cny=str(cost), stage="response_validation")
            state.update(known_estimated_cost_cny=str(spent), estimated_cost_cny=str(spent))
            save(path, state)
            if role == "embedding":
                if not isinstance(response, CreateEmbeddingResponse):
                    raise ValueError("Expected typed embedding response")
                vectors = ordered_embeddings(response.data, 2)
                row.update(
                    model=response.model,
                    dimensions=[len(v) for v in vectors],
                    response_indices=[item.index for item in response.data],
                    vectors_sha256=hashlib.sha256(json.dumps(vectors).encode()).hexdigest(),
                )
            else:
                if not isinstance(response, ChatCompletion):
                    raise ValueError("Expected typed chat response")
                row.update(
                    model=response.model,
                    content=row["response"]["choices"][0]["message"].get("content"),
                    finish_reason=response.choices[0].finish_reason,
                )
                content = chat_content(response)
                if role == "llm_rerank":
                    validate_llm_scores(content, 2)
                elif content.strip() != "OK":
                    raise ValueError("Unexpected generation probe answer")
            row["status"] = "ok"
            save(path, state)
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="partial", error_type=type(exc).__name__)
        if state["calls"]:
            state["calls"][-1]["status"] = "error"
        # Do not log exception bodies, response headers, request objects or keys.
        code = getattr(exc, "status_code", None)
        if type(code) is int:
            state["http_status"] = code
    finally:
        client.close()
        save(path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only-role", choices=("embedding", "llm_rerank", "generation"))
    parser.add_argument("--prior-cost-cny", type=Decimal, default=Decimal("0"))
    args = parser.parse_args()
    try:
        state = run(
            args.output, args.execute, only_role=args.only_role, prior_cost_cny=args.prior_cost_cny
        )
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "model_attempts": state["model_attempts"],
                    "estimated_cost_cny": state["estimated_cost_cny"],
                }
            )
        )
        return 0 if state["status"] in {"planned", "complete"} else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "stopped",
                    "error_type": type(exc).__name__,
                    "note": "Inspect report for attempts; do not infer zero calls",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
