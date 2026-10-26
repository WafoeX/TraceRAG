"""LLM-based re-ranking of retrieved candidates.

Retrieval is fast but coarse. A re-ranker reads each candidate against the
question and scores true relevance — catching passages that matched on vectors
or keywords but don't actually answer the question. Provider-agnostic: uses the
same chat model as generation, so it works on OpenAI, Gemini, or Ollama.
"""

from __future__ import annotations

import json

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam

from app.config import Settings, get_settings
from app.services.application_budget import priced_call
from app.services.openai_client import get_chat_client as get_client
from app.services.resilience import (
    ProviderUnavailable,
    answer_content,
    application_active,
    call_provider,
    failure_code,
    provider_client,
    warn,
)
from app.services.retrieval import RetrievedChunk

SYSTEM = "You are a precise search-relevance judge. Respond with JSON only."


def _parse_scores(content: str, n: int) -> list[float] | None:
    """Parse a JSON array of n relevance scores (one per passage, in order).

    Tolerant of bare numbers ([8, 2, 5]) and objects ([{"score": 8}, ...]).
    """
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(content[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None

    scores: list[float] = []
    for x in data:
        if isinstance(x, dict):
            x = x.get("score", x.get("relevance", 0))
        try:
            scores.append(float(x))
        except (TypeError, ValueError):
            scores.append(0.0)
    scores += [0.0] * (n - len(scores))
    return scores[:n]


def thinking_controls(settings: Settings) -> dict[str, bool]:
    """Explicit SiliconFlow rerank-only override; omit by default, not true."""
    if (
        settings.model_provider == "siliconflow"
        and not settings.deepseek_base_url
        and settings.llm_rerank_enable_thinking is not None
    ):
        return {"enable_thinking": settings.llm_rerank_enable_thinking}
    return {}


def build_messages(question: str, documents: list[str]) -> list[ChatCompletionMessageParam]:
    """Shared frozen prompt; documents already include title and passage coverage."""
    listing = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents))
    prompt = (
        f"Question: {question}\n\n"
        f"Passages:\n{listing}\n\n"
        f"Rate how well each passage answers the question, from 0 (irrelevant) to "
        f"10 (directly answers it). Return ONLY a JSON array of {len(documents)} "
        f"numbers, in the same order as the passages — e.g. [8, 2, 5, 0]."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rerank(question: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score candidates by true relevance and return the top_k.

    Falls back to the original order if the model's response can't be parsed, so
    a flaky rerank never breaks the request.
    """
    if not candidates:
        return []

    settings = get_settings()
    # Full chunk is opt-in. Do not silently change the frozen prefix baseline.
    documents = [
        f'(from "{c.document_title}") '
        + (c.content if settings.rerank_passage_mode == "full_chunk" else c.content[:600])
        for c in candidates
    ]

    def invoke():
        model = settings.llm_rerank_model or settings.openai_chat_model
        if settings.rerank_prompt_version == "baseline":
            messages = build_messages(question, documents)
        else:
            from app.services.rerank_prompts import build_messages as build_versioned_messages

            messages = build_versioned_messages(question, documents, settings.rerank_prompt_version)
        client = provider_client(get_client())
        if settings.model_provider == "siliconflow" and not settings.deepseek_base_url:
            from app.services.siliconflow import chat_content, guarded_call, validate_llm_scores

            max_tokens = settings.siliconflow_llm_max_tokens
            if application_active():
                max_tokens = min(max_tokens, settings.application_rerank_max_tokens)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            controls = thinking_controls(settings)
            payload.update(controls)
            response = guarded_call(
                "llm_rerank",
                model,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body=controls or None,
                ),
                payload,
            )
            if application_active():
                if response.choices and response.choices[0].finish_reason != "stop":
                    raise ValueError("Incomplete rerank")
                answer_content(
                    response.choices[0].message.content or "" if response.choices else ""
                )
            content = chat_content(response)
            if settings.rerank_prompt_version == "baseline":
                validate_llm_scores(content, len(candidates))
        else:
            payload = {"model": model, "messages": messages, "temperature": 0}
            if application_active():
                payload["max_tokens"] = settings.application_rerank_max_tokens
            response = priced_call(
                "llm_rerank",
                model,
                payload,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=settings.application_rerank_max_tokens
                    if application_active()
                    else NOT_GIVEN,
                ),
            )
            if application_active() and not response.choices:
                answer_content("")
            if application_active() and response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete rerank")
            content = response.choices[0].message.content or ""
        answer_content(content)
        if application_active() and settings.rerank_prompt_version == "baseline":
            from app.services.siliconflow import validate_llm_scores

            validate_llm_scores(content, len(candidates))
        if settings.rerank_prompt_version == "baseline":
            scores = _parse_scores(content, len(candidates))
        else:
            from app.services.rerank_prompts import parse_scores

            scores = parse_scores(content, len(candidates))
        return scores

    try:
        scores = call_provider(invoke)
    except Exception as exc:
        code = exc.code if isinstance(exc, ProviderUnavailable) else failure_code(exc)
        warn("rerank_" + code)
        scores = None

    # If re-ranking is unavailable, keep retrieval order and scores (never all-zero).
    if not scores or not any(s > 0 for s in scores):
        warn("rerank_retrieval_order")
        return candidates[:top_k]

    ranked = sorted(zip(candidates, scores, strict=False), key=lambda cs: cs[1], reverse=True)
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            content=c.content,
            score=round(float(s), 2),
        )
        for c, s in ranked[:top_k]
    ]
