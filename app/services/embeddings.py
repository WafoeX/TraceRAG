"""OpenAI embedding helpers (batched)."""

from __future__ import annotations

from openai.types import CreateEmbeddingResponse

from app.config import get_settings
from app.services.application_budget import priced_call
from app.services.openai_client import get_client
from app.services.resilience import (
    EmptyResponse,
    application_active,
    call_provider,
    provider_client,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return call_provider(lambda: _embed_texts(texts))


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single API call. Order is preserved.

    Requests ``embedding_dim`` dimensions so models that support Matryoshka
    truncation (OpenAI text-embedding-3-*, Gemini gemini-embedding-001) return
    vectors that match the database column.
    """
    if not texts:
        return []
    settings = get_settings()
    client = provider_client(get_client())
    if settings.model_provider == "siliconflow":
        from .siliconflow import guarded_call, ordered_embeddings

        # BGE-M3 dense is fixed 1024: dimensions is not sent to this endpoint.
        if settings.embedding_dim != 1024:
            raise ValueError("BGE-M3 requires 1024 dimensions")
        payload = {
            "model": settings.openai_embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        response: CreateEmbeddingResponse = guarded_call(
            "embedding",
            settings.openai_embedding_model,
            lambda: client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
                encoding_format="float",
            ),
            payload,
        )
        if application_active() and not response.data:
            raise EmptyResponse()
        return ordered_embeddings(response.data, len(texts))
    payload = {
        "model": settings.openai_embedding_model,
        "input": texts,
        "dimensions": settings.embedding_dim,
    }
    response = priced_call(
        "embedding",
        settings.openai_embedding_model,
        payload,
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
            dimensions=settings.embedding_dim,
        ),
    )
    if application_active():
        from .siliconflow import ordered_embeddings

        if not response.data:
            raise EmptyResponse()
        return ordered_embeddings(response.data, len(texts), settings.embedding_dim)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
