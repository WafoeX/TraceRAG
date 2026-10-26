"""Freeze existing document list, text ingestion and upload wire behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.routers import documents


def test_empty_document_list(client: TestClient, db: Mock) -> None:
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []
    db.execute.assert_called_once()


def test_document_list_preserves_integer_ids_and_counts(
    client: TestClient, db: Mock, document: SimpleNamespace
) -> None:
    db.execute.return_value.all.return_value = [(document, 2)]
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == [
        {"id": 17, "title": "Refund Policy", "source": "refund.txt", "chunks": 2}
    ]


def test_text_ingestion_delegates_without_changing_request(
    client: TestClient, db: Mock, document: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest = Mock(return_value=document)
    monkeypatch.setattr(documents, "ingest_document", ingest)
    response = client.post(
        "/documents",
        json={"title": "Refund Policy", "content": "30 days", "source": "refund.txt"},
    )
    assert response.status_code == 201
    assert response.json() == {"document_id": 17, "title": "Refund Policy", "chunks_created": 2}
    ingest.assert_called_once_with(db, "Refund Policy", "30 days", "refund.txt")


@pytest.mark.parametrize("payload", [{"title": "Empty", "content": ""}, {"content": "text"}])
def test_invalid_text_request_never_ingests(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    ingest = Mock()
    monkeypatch.setattr(documents, "ingest_document", ingest)
    assert client.post("/documents", json=payload).status_code == 422
    ingest.assert_not_called()


def test_ingestion_value_error_becomes_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents, "ingest_document", Mock(side_effect=ValueError("No chunks")))
    response = client.post("/documents", json={"title": "Blank", "content": "   "})
    assert response.status_code == 422
    assert response.json() == {"detail": "No chunks"}


@pytest.mark.parametrize("filename", ["refund.txt", "refund.md"])
def test_upload_uses_real_text_extractor(
    client: TestClient,
    db: Mock,
    document: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    ingest = Mock(return_value=document)
    monkeypatch.setattr(documents, "ingest_document", ingest)
    response = client.post(
        "/documents/upload",
        files={"file": (filename, "Refund: 30 days. 退款".encode(), "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["document_id"] == 17
    ingest.assert_called_once_with(db, filename, "Refund: 30 days. 退款", filename)


def test_upload_real_bundled_pdf_extraction(
    client: TestClient, db: Mock, document: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolve from the project root, not the caller's working directory.
    path = Path(__file__).resolve().parents[2] / "examples" / "onboarding-guide.pdf"
    data = path.read_bytes()
    ingest = Mock(return_value=document)
    monkeypatch.setattr(documents, "ingest_document", ingest)
    response = client.post(
        "/documents/upload", files={"file": (path.name, data, "application/pdf")}
    )
    assert response.status_code == 201
    ingest.assert_called_once()
    actual_db, title, content, source = ingest.call_args.args
    assert (actual_db, title, source) == (db, path.name, path.name)
    # Freeze real extracted output instead of recalculating it with the same parser.
    assert hashlib.sha256(content.encode()).hexdigest() == (
        "8a4f4df66eb4cf491ec3d5b63a13d7beeb1630a0c5119baf2b7dbbe6a6d32722"
    )


def test_blank_pdf_rejected_by_real_ingestion(client: TestClient, db: Mock) -> None:
    data = (Path(__file__).resolve().parents[1] / "fixtures" / "blank.pdf").read_bytes()
    response = client.post(
        "/documents/upload", files={"file": ("blank.pdf", data, "application/pdf")}
    )
    assert response.status_code == 422
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_upload_requires_file(client: TestClient) -> None:
    assert client.post("/documents/upload").status_code == 422


@pytest.mark.parametrize("endpoint", ["/documents", "/documents/upload", "/query", "/query/stream"])
def test_rate_limit_stops_expensive_post_routes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    from fastapi import HTTPException

    from app.main import app
    from app.services.application_limits import require_capacity

    def denied():
        raise HTTPException(429, "Account call limit exceeded")

    app.dependency_overrides[require_capacity] = denied
    try:
        response = client.post(endpoint, json={})
        assert response.status_code == 429
        assert response.json() == {"detail": "Account call limit exceeded"}
    finally:
        app.dependency_overrides.pop(require_capacity, None)
