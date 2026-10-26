"""SiliconFlow wire adapters. Native rerank is opt-in, never used by search."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

import httpx

from app.config import get_settings

from .provider_costs import CostTracker

T = TypeVar("T")


class ProviderResponseError(ValueError):
    """Provider payload violates the selected model contract."""


@lru_cache
def get_cost_tracker() -> CostTracker:
    settings = get_settings()
    return CostTracker(
        settings.siliconflow_pricing_file,
        log_path=settings.siliconflow_cost_log,
        secret=settings.openai_api_key,
    )


def guarded_call(role: str, model: str, invoke: Callable[[], T], payload: dict[str, Any]) -> T:
    """Reserve before each application wire attempt; retain historical offline accounting."""
    from .application_budget import priced_call

    return priced_call(role, model, payload, lambda: _recorded_call(role, model, invoke, payload))


def _recorded_call(role: str, model: str, invoke: Callable[[], T], payload: dict[str, Any]) -> T:
    from .provider_costs import ProviderRequestError
    from .resilience import ProviderUnavailable, application_active, failure_code

    if not application_active():
        return get_cost_tracker().call(role, model, invoke, request=payload)
    code = None

    def observed():
        nonlocal code
        try:
            return invoke()
        except Exception as exc:
            # The cost tracker deliberately hides the original exception. Preserve
            # only its safe classification, never its body, for application retries.
            code = failure_code(exc)
            raise

    try:
        return get_cost_tracker().call(role, model, observed, request=payload)
    except ProviderRequestError:
        if code is not None:
            raise ProviderUnavailable(code) from None
        raise


def ordered_embeddings(data, count: int, dimension: int = 1024) -> list[list[float]]:
    if len(data) != count:
        raise ProviderResponseError("Embedding batch count mismatch")
    ordered: dict[int, list[float]] = {}
    for item in data:
        index, vector = item.index, item.embedding
        if type(index) is not int or not 0 <= index < count or index in ordered:
            raise ProviderResponseError("Invalid/duplicate embedding index")
        if len(vector) != dimension or any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
            for v in vector
        ):
            raise ProviderResponseError("Embedding dimension/value mismatch")
        ordered[index] = vector
    return [ordered[i] for i in range(count)]


def chat_content(response) -> str:
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise ProviderResponseError("Missing, truncated or refused chat response")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ProviderResponseError("Missing chat content (reasoning_content is not an answer)")
    return content


def validate_llm_scores(content: str, n: int) -> None:
    # Keep the original prompt/scoring/fallback algorithm. Reject unsafe numeric
    # values only on this provider path, before the existing lenient parser.
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end < start:
        raise ProviderResponseError("No JSON score array")
    try:
        data = json.loads(content[start : end + 1])
        if not isinstance(data, list) or len(data) != n:
            raise ValueError
        for value in data:
            if isinstance(value, dict):
                value = value.get("score", value.get("relevance"))
            if isinstance(value, bool) or not isinstance(value, float | int):
                raise ValueError
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError
    except (ValueError, TypeError):
        raise ProviderResponseError("Invalid LLM score JSON") from None


def native_rerank(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """Separate experimental API: returns original document indices, not chat scores.

    No fallback/substitution into LLM baseline. Missing usage records unknown cost.
    Native scores remain on the provider's scale, never multiplied by ten.
    """
    if not documents:
        return []
    if type(top_n) is not int or not 1 <= top_n <= len(documents):
        raise ValueError("top_n outside document range")
    settings = get_settings()
    if settings.model_provider != "siliconflow":
        raise ValueError("Native adapter requires explicit SiliconFlow configuration")
    payload = {
        "model": settings.native_rerank_model,
        "query": question,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }

    def invoke():
        with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
            response = client.post(
                "https://api.siliconflow.cn/v1/rerank",
                json=payload,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
            response.raise_for_status()
            return response.json()

    response = guarded_call("native_rerank", settings.native_rerank_model, invoke, payload)
    results = response.get("results")
    if not isinstance(results, list) or len(results) != top_n:
        raise ProviderResponseError("Invalid native rerank results count")
    output: list[tuple[int, float]] = []
    seen: set[int] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ProviderResponseError("Invalid native result")
        index, score = result.get("index"), result.get("relevance_score")
        if type(index) is not int or index in seen or not 0 <= index < len(documents):
            raise ProviderResponseError("Invalid native document index")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ProviderResponseError("Invalid native relevance score")
        seen.add(index)
        output.append((index, float(score)))
    return sorted(output, key=lambda pair: pair[1], reverse=True)
