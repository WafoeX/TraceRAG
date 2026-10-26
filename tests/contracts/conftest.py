"""Real FastAPI routes with database/model boundaries replaced, never real services."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import get_db
from app.main import app
from app.services.retrieval import RetrievedChunk


@pytest.fixture
def db() -> Mock:
    session = Mock(spec=Session)
    session.execute.return_value.all.return_value = []
    return session


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, db: Mock) -> Iterator[TestClient]:
    previous = dict(app.dependency_overrides)

    def fake_db() -> Iterator[Mock]:
        yield db

    def unexpected_init() -> None:
        pytest.fail("Offline contract test attempted database bootstrap")

    app.dependency_overrides[get_db] = fake_db
    monkeypatch.setattr("app.main.init_db", unexpected_init)
    try:
        # Enter lifespan: run_db_init=False must prevent real database setup.
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


@pytest.fixture
def document() -> SimpleNamespace:
    return SimpleNamespace(
        id=17, title="Refund Policy", source="refund.txt", chunks=[object(), object()]
    )


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=101,
            document_id=17,
            document_title="Refund Policy",
            chunk_index=0,
            content="Customers may request a full refund within 30 days.",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id=102,
            document_id=23,
            document_title="Processing Times",
            chunk_index=2,
            content="Refunds are processed in 5-7 business days.",
            score=0.8,
        ),
    ]
