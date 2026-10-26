"""Application increment: offline contracts + opt-in real PostgreSQL regression.

TRACERAG_INTEGRATION=1 enables ONLY the fixed disposable local test DB.
The fixed test schema is dropped afterwards; do not run via pytest-xdist.
Developer DATABASE_URL/.env are never used.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database import Base, get_db
from app.main import app
from app.models import Chunk, Document, IngestionJob
from app.routers import query as query_module
from app.services import ingestion, retrieval
from app.services.retrieval import RetrievedChunk
from app.services.search import SearchResult

TEST_URL = (
    "postgresql+psycopg://tracerag_test:local_test_only@127.0.0.1:55439/tracerag_application_test"
)


def vectors(pieces: list[str]) -> list[list[float]]:
    return [[1.0] + [0.0] * (get_settings().embedding_dim - 1) for _ in pieces]


@pytest.fixture
def database():
    admin = create_engine(TEST_URL)
    with admin.begin() as conn:
        assert conn.scalar(text("SELECT current_database()")) == "tracerag_application_test"
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        conn.exec_driver_sql("CREATE SCHEMA tracerag_lifecycle_test")
    engine = create_engine(
        TEST_URL, connect_args={"options": "-csearch_path=tracerag_lifecycle_test,public"}
    )
    try:
        Base.metadata.create_all(engine)
        yield sessionmaker(bind=engine, expire_on_commit=False)
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA tracerag_lifecycle_test CASCADE")
        admin.dispose()


@pytest.fixture
def api(database):
    def override():
        with database() as db:
            yield db

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def add_task(database, content=b"Scientists observed liver cells during the experiment."):
    with database() as db:
        return ingestion.enqueue(db, "Paper", "paper.txt", content).id


def take(database, seconds=900):
    with database() as db:
        item = ingestion.claim(db, seconds)
    assert item is not None
    return item


def expire(database, job_id):
    with database() as db:
        job = db.get(IngestionJob, job_id)
        assert job is not None
        job.lease_until = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()


@pytest.mark.parametrize(
    "answer,count,expected",
    [
        ("Evidence [1] and [2].", 2, []),
        ("Invalid [0] [3] [3].", 2, [0, 3]),
        ("No source claimed.", 0, []),
        ("[1]", 0, [1]),
        ("[10000000000]", 2, [10000000000]),
        ("[" + "9" * 5000 + "]", 2, [-1]),
    ],
)
def test_citation_range_only(answer, count, expected):
    assert query_module.invalid_citations(answer, count) == expected


@pytest.mark.parametrize("pages,encrypted", [(201, False), (1, True)])
def test_pdf_page_and_encryption_limits(pages, encrypted):
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    if encrypted:
        writer.encrypt("password")
    data = io.BytesIO()
    writer.write(data)
    item = ingestion.ClaimedJob(1, "token", "PDF", "paper.pdf", data.getvalue())
    with pytest.raises(ValueError):
        ingestion.extract_payload(item)


def test_citation_is_exact_full_context():
    content = "x" * 900 + " important final evidence"
    chunk = RetrievedChunk(7, 8, "Paper", 2, content, 0.8)
    citation = query_module._context_citations([chunk])[0]
    assert citation.content == content
    assert len(citation.snippet) == 240
    assert citation.context_number == 1
    assert citation.evidence_url == "/documents/8/chunks/2"


@pytest.mark.parametrize(
    "filename,data",
    [
        ("paper.exe", b"text"),
        ("paper.txt", b""),
        ("paper.txt", b"x" * (ingestion.MAX_UPLOAD_BYTES + 1)),
    ],
)
def test_input_validation_before_db(filename, data):
    with pytest.raises(ValueError):
        ingestion.enqueue(MagicMock(spec=Session), "Paper", filename, data)


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_api_upload_dedup_publish_evidence_delete(api, database, monkeypatch):
    raw = b"Scientists observed liver cells during the experiment."
    res = api.post("/documents/jobs/upload", files={"file": ("paper.txt", raw)})
    assert res.status_code == 202
    job_id = res.json()["id"]
    assert res.json()["state"] == "queued"
    duplicate = api.post("/documents/jobs/upload", files={"file": ("renamed.txt", raw)})
    assert duplicate.json()["id"] == job_id
    assert api.get("/documents").json() == []
    assert ingestion.process(database, take(database), vectors)
    job = api.get(f"/documents/jobs/{job_id}").json()
    assert job["state"] == "completed" and job["attempts"] == 1
    document_id = job["document_id"]
    assert api.post(f"/documents/jobs/{job_id}/retry").status_code == 409
    listed = api.get("/documents/catalog").json()
    assert len(listed) == 1 and listed[0]["created_at"]
    evidence_url = f"/documents/{document_id}/chunks/0"
    assert api.get(evidence_url).json()["content"] == raw.decode()
    with database() as db:
        stored = db.get(IngestionJob, job_id)
        assert stored is not None and stored.payload is None
        monkeypatch.setattr(retrieval, "embed_query", lambda q: vectors([q])[0])
        assert retrieval.vector_search(db, "liver", 4)
        assert retrieval.keyword_search(db, "liver", 4)
    assert api.delete(f"/documents/{document_id}").status_code == 204
    assert api.get(evidence_url).status_code == 404
    assert api.get(f"/documents/jobs/{job_id}").status_code == 404
    assert api.delete(f"/documents/{document_id}").status_code == 404
    with database() as db:
        assert retrieval.vector_search(db, "liver", 4) == []
        assert retrieval.keyword_search(db, "liver", 4) == []
        assert db.scalar(select(func.count(Chunk.id))) == 0
    # Re-upload after deletion is allowed; no tombstone leaks into dedup.
    assert (
        api.post("/documents/jobs/upload", files={"file": ("paper.txt", raw)}).json()["id"]
        != job_id
    )


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_concurrent_dedup_and_skip_locked(database):
    def upload(_):
        return add_task(database)

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(upload, range(6)))
    assert len(set(ids)) == 1
    with database() as locked:
        locked.scalar(select(IngestionJob).with_for_update())
        with database() as other:
            assert ingestion.claim(other) is None
    item = take(database)
    with database() as other:
        assert ingestion.claim(other) is None
    assert ingestion.process(database, item, vectors)
    assert not ingestion.process(database, item, vectors)
    with database() as db:
        assert db.scalar(select(func.count(Document.id))) == 1


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_expired_worker_cannot_publish_after_embedding(database):
    job_id = add_task(database)
    stale = take(database)
    replacement = []

    def slow_embed(pieces):
        expire(database, job_id)
        replacement.append(take(database))
        return vectors(pieces)

    assert not ingestion.process(database, stale, slow_embed)
    with database() as db:
        assert db.scalar(select(func.count(Document.id))) == 0
    assert ingestion.process(database, replacement[0], vectors)
    with database() as db:
        assert db.scalar(select(func.count(Document.id))) == 1


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_crash_recovery_stops_after_three_attempts(api, database):
    job_id = add_task(database)
    for _ in range(3):
        take(database)
        expire(database, job_id)
    with database() as db:
        assert ingestion.claim(db) is None
    job = api.get(f"/documents/jobs/{job_id}").json()
    assert job["state"] == "failed" and job["error_code"] == "lease_attempts_exhausted"
    assert api.post(f"/documents/jobs/{job_id}/retry").json()["attempts"] == 0
    assert ingestion.process(database, take(database), vectors)


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_failure_retry_is_atomic_and_error_is_redacted(api, database):
    job_id = add_task(database)

    def failure(_):
        raise RuntimeError("429 secret-key private-paper-text")

    assert not ingestion.process(database, take(database), failure)
    failed = api.get(f"/documents/jobs/{job_id}")
    assert "secret" not in failed.text and "private" not in failed.text
    assert failed.json()["error_code"] == "embedding_failed"
    assert api.get("/documents").json() == []
    assert api.post(f"/documents/jobs/{job_id}/retry").status_code == 202
    assert ingestion.process(database, take(database), vectors)
    with database() as db:
        assert db.scalar(select(func.count(Document.id))) == 1
        assert db.scalar(select(func.count(Chunk.id))) == 1


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
@pytest.mark.parametrize("kind", ["count", "dimension", "nan"])
def test_invalid_embedding_never_publishes(database, kind):
    job_id = add_task(database)

    def invalid(pieces):
        if kind == "count":
            return []
        if kind == "dimension":
            return [[1.0] for _ in pieces]
        return [[float("nan")] * get_settings().embedding_dim for _ in pieces]

    assert not ingestion.process(database, take(database), invalid)
    with database() as db:
        job = db.get(IngestionJob, job_id)
        assert job is not None and job.error_code == "embedding_failed"
        assert db.scalar(select(func.count(Document.id))) == 0


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_publication_rollback_keeps_no_partial_document(database):
    job_id = add_task(database)

    def reject_document(session, *_):
        if any(isinstance(obj, Document) for obj in session.new):
            raise RuntimeError("injected publication failure")

    event.listen(database.class_, "before_flush", reject_document)
    try:
        assert not ingestion.process(database, take(database), vectors)
    finally:
        event.remove(database.class_, "before_flush", reject_document)
    with database() as db:
        job = db.get(IngestionJob, job_id)
        assert job is not None and job.error_code == "publication_failed"
        assert db.scalar(select(func.count(Document.id))) == 0
        assert db.scalar(select(func.count(Chunk.id))) == 0


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
@pytest.mark.parametrize(
    "filename,raw",
    [
        ("bad.pdf", b"not a pdf"),
        ("bad.txt", b"\xff"),
        ("empty.md", b"   "),
    ],
)
def test_invalid_parse_can_be_discarded(api, database, filename, raw):
    res = api.post("/documents/jobs/upload", files={"file": (filename, raw)})
    job_id = res.json()["id"]
    assert not ingestion.process(database, take(database), vectors)
    assert api.get(f"/documents/jobs/{job_id}").json()["error_code"] == "invalid_document"
    assert api.delete(f"/documents/jobs/{job_id}").status_code == 204
    assert api.get(f"/documents/jobs/{job_id}").status_code == 404


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_upload_limits_and_task_contract(api):
    assert api.post("/documents/jobs/upload", files={"file": ("x.exe", b"text")}).status_code == 422
    assert (
        api.post(
            "/documents/jobs/upload",
            files={"file": ("x.txt", b"x" * (ingestion.MAX_UPLOAD_BYTES + 1))},
        ).status_code
        == 413
    )
    response = api.post("/documents/jobs", json={"title": "Text", "content": "Some text"})
    assert response.status_code == 202
    job = response.json()
    assert "payload" not in job and "lease_token" not in job
    assert api.post(f"/documents/jobs/{job['id']}/retry").status_code == 409
    assert api.delete(f"/documents/jobs/{job['id']}").status_code == 409
    assert api.get("/documents/jobs/999999999").status_code == 404


@pytest.mark.enable_socket
@pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB")
def test_query_evidence_audit_preserves_historical_stream(api, monkeypatch):
    chunk = RetrievedChunk(1, 2, "Paper", 0, "Complete original evidence.", 0.9)
    monkeypatch.setattr(query_module, "search", lambda *_: SearchResult([chunk], 0.9))
    monkeypatch.setattr(query_module, "generate_answer", lambda *_: "Valid [1], invalid [99].")
    monkeypatch.setattr(
        query_module, "stream_answer", lambda *_: iter(["Valid [1],", " invalid [99]."])
    )
    body = api.post("/query/evidence", json={"question": "Evidence?"}).json()
    assert body["invalid_citation_numbers"] == [99]
    assert body["citations"][0]["content"] == chunk.content
    import json

    frame = api.post("/query/stream", json={"question": "Evidence?"}).text.split("\x1f")[1]
    assert "invalid_citation_numbers" not in json.loads(frame)
    assert "content" not in json.loads(frame)["citations"][0]
    monkeypatch.setattr(query_module, "search", lambda *_: SearchResult([], 0.0))
    refused = api.post("/query/evidence", json={"question": "Unknown?"}).json()
    assert refused == {
        "answer": query_module.NO_ANSWER,
        "citations": [],
        "invalid_citation_numbers": [],
        "generation_status": "skipped",
        "warnings": [],
    }
