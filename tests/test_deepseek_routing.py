"""Offline provider isolation; no credentials or paid API calls."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.services import generation, openai_client, reranking


def settings() -> Settings:
    return Settings.model_construct(
        model_provider="siliconflow",
        openai_api_key="sf-test",
        openai_base_url="https://api.siliconflow.cn/v1",
        deepseek_api_key="ds-test",
        deepseek_base_url="https://api.deepseek.com/v1",
        openai_chat_model="deepseek-flash",
        llm_rerank_model="deepseek-flash",
        llm_rerank_enable_thinking=False,
    )


def test_clients_are_isolated():
    openai_client.get_client.cache_clear()
    openai_client.get_chat_client.cache_clear()
    try:
        with (
            patch.object(openai_client, "get_settings", return_value=settings()),
            patch.object(openai_client, "OpenAI") as factory,
        ):
            openai_client.get_client()
            openai_client.get_chat_client()
            shared, chat = factory.call_args_list
            assert shared.kwargs["api_key"] == "sf-test"
            assert shared.kwargs["base_url"] == "https://api.siliconflow.cn/v1"
            assert chat.kwargs["api_key"] == "ds-test"
            assert chat.kwargs["base_url"] == "https://api.deepseek.com/v1"
            assert chat.kwargs["max_retries"] == 0
    finally:
        openai_client.get_client.cache_clear()
        openai_client.get_chat_client.cache_clear()


def test_missing_key_does_not_fall_back():
    config = settings()
    config.deepseek_api_key = ""
    openai_client.get_chat_client.cache_clear()
    with (
        patch.object(openai_client, "get_settings", return_value=config),
        pytest.raises(ValueError, match="DEEPSEEK_API_KEY"),
    ):
        openai_client.get_chat_client()


def test_generation_routes_to_official_model():
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "answer"
    with (
        patch.object(generation, "get_settings", return_value=settings()),
        patch.object(generation, "get_client", return_value=client),
    ):
        assert generation.generate_answer("question", []) == "answer"
    assert client.chat.completions.create.call_args.kwargs["model"] == "deepseek-flash"
    assert reranking.thinking_controls(settings()) == {}
