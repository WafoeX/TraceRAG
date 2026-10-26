"""Answer generation grounded in retrieved context (with citations)."""

from __future__ import annotations

from collections.abc import Iterator

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam

from app.config import get_settings
from app.services.application_budget import priced_call
from app.services.openai_client import get_chat_client as get_client
from app.services.resilience import (
    answer_content,
    application_active,
    call_provider,
    provider_client,
)
from app.services.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context. If the context does not contain the answer, say you "
    "don't know rather than guessing. Cite sources inline as [n], where n is "
    "the context number."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f'[{i}] (from "{c.document_title}")\n{c.content}' for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks) or "No context available."


def _messages(question: str, chunks: list[RetrievedChunk]) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}",
        },
    ]


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    return call_provider(lambda: answer_content(_generate_answer(question, chunks)))


def _generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    settings = get_settings()
    if settings.model_provider == "siliconflow" and not settings.deepseek_base_url:
        from app.services.siliconflow import chat_content, guarded_call

        model = settings.openai_chat_model
        max_tokens = settings.siliconflow_generation_max_tokens
        if application_active():
            max_tokens = min(max_tokens, settings.application_generation_max_tokens)
        messages = _messages(question, chunks)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = guarded_call(
            "generation",
            model,
            lambda: provider_client(get_client()).chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
            ),
            payload,
        )
        if application_active():
            if response.choices and response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete answer")
            answer_content(response.choices[0].message.content or "" if response.choices else "")
        return chat_content(response)
    messages = _messages(question, chunks)
    payload = {"model": settings.openai_chat_model, "messages": messages, "temperature": 0.2}
    if application_active():
        payload["max_tokens"] = settings.application_generation_max_tokens
    response = priced_call(
        "generation",
        settings.openai_chat_model,
        payload,
        lambda: provider_client(get_client()).chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.2,
            max_tokens=settings.application_generation_max_tokens
            if application_active()
            else NOT_GIVEN,
        ),
    )
    if application_active() and not response.choices:
        answer_content("")
    if application_active() and response.choices[0].finish_reason != "stop":
        raise ValueError("Incomplete answer")
    return response.choices[0].message.content or ""


def stream_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    if application_active() or (
        get_settings().model_provider == "siliconflow" and not get_settings().deepseek_base_url
    ):
        # Application requests buffer so retries never replay partial output.
        # SiliconFlow also needs a complete response for usage accounting.
        yield generate_answer(question, chunks)
        return
    stream = get_client().chat.completions.create(
        model=get_settings().openai_chat_model,
        messages=_messages(question, chunks),
        temperature=0.2,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
