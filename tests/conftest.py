"""Offline test bootstrap, isolated from developer credentials and databases."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config import Settings
from scripts.prepare_test_cache import CACHE_DIR, verify_cache

if not verify_cache():
    raise pytest.UsageError(
        "Missing or corrupt cl100k_base tokenizer cache. Prepare it explicitly with "
        "`uv run --frozen python -m scripts.prepare_test_cache` before offline tests."
    )
os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)

# model_construct uses declared defaults without reading .env or process credentials.
# Only this pytest process is affected; production Settings validation is unchanged.
_TEST_SETTINGS = Settings.model_construct(
    openai_api_key="test-key",
    database_url="postgresql+psycopg://test:test@127.0.0.1:1/budgetrag_test",
    run_db_init=False,
    rate_limit_enabled=False,
)
_settings_patch = patch("app.config.get_settings", return_value=_TEST_SETTINGS)


def pytest_sessionstart(session: pytest.Session) -> None:
    _settings_patch.start()
    # Cover collection-time imports as well as tests. AF_UNIX is needed by the
    # local ASGI event loop; TCP/UDP remain blocked by pytest-socket.
    plugin = session.config.pluginmanager.get_plugin("socket")
    if plugin is None:
        raise pytest.UsageError("pytest-socket is required; run `uv sync --frozen --group dev`")
    plugin.disable_socket(allow_unix_socket=True)


@pytest.fixture(autouse=True)
def historical_protocol_boundary(request):
    """Old protocol/unit tests stub auth explicitly, not a production opt-out.

    Authorization acceptance lives in test_auth_isolation.py with real sessions.
    Existing response/order assertions remain unchanged.
    """
    old_modules = {"test_search.py", "test_application_lifecycle.py", "test_document_versions.py"}
    if request.path.name not in old_modules and "contracts" not in request.path.parts:
        yield
        return
    from app.auth import require_library
    from app.main import app

    previous = dict(app.dependency_overrides)
    app.dependency_overrides[require_library] = lambda: None
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _settings_patch.stop()
    plugin = session.config.pluginmanager.get_plugin("socket")
    if plugin is not None:
        plugin.enable_socket()
