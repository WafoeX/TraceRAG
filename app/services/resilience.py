"""Application-only provider policy; offline experiments retain their frozen behavior."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.config import get_settings

T = TypeVar("T")


@dataclass
class CallState:
    warnings: list[str] = field(default_factory=list)
    user_id: int | None = None


_state: ContextVar[CallState | None] = ContextVar("application_calls", default=None)


@contextmanager
def application_calls(user_id: int | None = None) -> Iterator[CallState]:
    state = CallState(user_id=user_id)
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


def application_active() -> bool:
    return _state.get() is not None


def set_budget_owner(user_id: int) -> None:
    state = _state.get()
    if state is not None:
        state.user_id = user_id


def budget_owner() -> int | None:
    state = _state.get()
    return state.user_id if state else None


def warn(code: str) -> None:
    from app.services.tracing import warning

    warning(code)
    state = _state.get()
    if state is not None and code not in state.warnings:
        state.warnings.append(code)


def warnings() -> list[str]:
    state = _state.get()
    return list(state.warnings) if state else []


class EmptyResponse(ValueError):
    """No usable answer; safe generic code only."""


class ProviderUnavailable(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def failure_code(exc: Exception) -> str:
    if isinstance(exc, ProviderUnavailable):
        return exc.code
    if isinstance(exc, APITimeoutError | TimeoutError):
        return "provider_timeout"
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return "provider_rate_limited"
        if exc.status_code >= 500:
            return "provider_server_error"
        return "provider_rejected"
    if isinstance(exc, APIConnectionError):
        return "provider_connection_error"
    if isinstance(exc, EmptyResponse):
        return "provider_empty_response"
    return "provider_invalid_response"


def provider_client(client):
    if not application_active():
        return client
    # Disable SDK retries: one visible policy, no multiplicative retry layers.
    return client.with_options(max_retries=0, timeout=get_settings().provider_timeout_seconds)


def call_provider(invoke: Callable[[], T]) -> T:
    if not application_active():
        return invoke()
    from app.services.application_limits import provider_capacity

    with provider_capacity():
        return _retry(invoke)


def _retry(invoke: Callable[[], T]) -> T:
    settings = get_settings()
    for attempt in range(settings.provider_max_attempts):
        try:
            return invoke()
        except Exception as exc:
            code = failure_code(exc)
            retryable = code in {
                "provider_timeout",
                "provider_rate_limited",
                "provider_server_error",
                "provider_connection_error",
                "provider_empty_response",
            }
            if not retryable or attempt + 1 == settings.provider_max_attempts:
                raise ProviderUnavailable(code) from None
            # Do not trust an arbitrarily long Retry-After header. Bounded local backoff.
            time.sleep(settings.provider_retry_backoff_seconds * (2**attempt))
    raise AssertionError("unreachable")


def answer_content(content: str) -> str:
    if application_active() and not content.strip():
        raise EmptyResponse()
    return content
