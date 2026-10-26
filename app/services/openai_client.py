"""Shared, lazily-constructed OpenAI client."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_client() -> OpenAI:
    settings = get_settings()
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    if settings.model_provider == "siliconflow":
        # A failed billed request must never be retried invisibly.
        kwargs.update(max_retries=0, timeout=60.0)
    return OpenAI(**kwargs)


@lru_cache
def get_chat_client() -> OpenAI:
    """Route both generation and LLM reranking without moving BGE requests."""
    settings = get_settings()
    if not settings.deepseek_base_url:
        return get_client()
    if not settings.deepseek_api_key.strip():
        raise ValueError("Set DEEPSEEK_API_KEY in the local .env file")
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        max_retries=0,
        timeout=60.0,
    )
