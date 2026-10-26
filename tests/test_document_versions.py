"""Opt-in PostgreSQL version lifecycle, migration and query-snapshot regression."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import event, func, inspect, select

from app import database as database_module
from app.main import app
from app.models import (
    Chunk,
    ChunkLocation,
    Document,
    DocumentVersion,
    IngestionJob,
    LogicalDocument,
)
from app.routers import query as query_module
from app.services import ingestion, retrieval
from app.services import search as search_module
from app.services.search import SearchResult
from scripts import migrate_application

from .test_application_lifecycle import api as _api_fixture
from .test_application_lifecycle import database as _database_fixture
from .test_application_lifecycle import take, vectors

api = _api_fixture
database = _database_fixture
upgrade = migrate_application.upgrade

pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in test DB"),
]


def publish(database, data=b"liver oldmarker evidence", base=None, filename="paper.txt"):
    with database() as db:
        job_id = ingestion.enqueue(db, filename, filename, data, base).id
    assert ingestion.process(database, take(database), vectors)
    with database() as db:
        job = db.get(IngestionJob, job_id)
        assert job is not None and job.document_id is not None
        return job.document_id


def stage(database, data, base, filename="update.txt"):
    with database() as db:
        return ingestion.enqueue(db, filename, filename, data, base).id


def test_successful_update_preserves_old_evidence_and_switches_search(api, database, monkeypatch):
    old = publish(database)
    with database() as db:
        first = db.get(DocumentVersion, old)
        assert first is not None
        logical_id = first.logical_id
    job_id = stage(database, b"kidney newmarker evidence", old)
    assert api.get("/documents/catalog").json()[0]["id"] == old
    assert ingestion.process(database, take(database), vectors)
    job = api.get(f"/documents/jobs/{job_id}").json()
    new = job["document_id"]
    assert new != old
    catalog = api.get("/documents/catalog").json()
    assert len(catalog) == 1 and catalog[0]["id"] == new
    assert catalog[0]["logical_id"] == logical_id and catalog[0]["version_number"] == 2
    assert catalog[0]["created_at"] == job["created_at"]
    assert len(api.get("/documents").json()) == 1
    assert api.get(f"/documents/{old}/chunks/0").json()["content"] == "liver oldmarker evidence"
    assert api.get(f"/documents/{old}/original").content == b"liver oldmarker evidence"
    history = api.get(f"/documents/{old}/versions").json()
    assert [(v["version_number"], v["active"]) for v in history] == [(2, True), (1, False)]
    assert all(v["original_available"] for v in history)
    with database() as db:
        monkeypatch.setattr(retrieval, "embed_query", lambda q: vectors([q])[0])
        assert {c.document_id for c in retrieval.vector_search(db, "liver", 10)} == {new}
        assert retrieval.keyword_search(db, "oldmarker", 10) == []
        assert {c.document_id for c in retrieval.keyword_search(db, "newmarker", 10)} == {new}
    assert (
        api.post(
            f"/documents/{old}/updates", json={"title": "stale", "content": "obsolete"}
        ).status_code
        == 409
    )
    # Re-uploading the current original is a no-op, even after base identity changes.
    with database() as db:
        same = ingestion.enqueue(db, "same", "renamed.txt", b"kidney newmarker evidence", new)
        assert same.id == job_id
        assert db.scalar(select(func.count(IngestionJob.id))) == 2
        assert db.scalar(select(func.count(DocumentVersion.document_id))) == 2


def test_failed_update_keeps_previous_active_and_retry_publishes_once(api, database):
    old = publish(database)
    job_id = stage(database, b"new replacement", old)

    def fail(_):
        raise RuntimeError("synthetic timeout not a real model")

    assert not ingestion.process(database, take(database), fail)
    assert api.get("/documents/catalog").json()[0]["id"] == old
    assert len(api.get(f"/documents/{old}/versions").json()) == 1
    assert api.post(f"/documents/jobs/{job_id}/retry").status_code == 202
    item = take(database)
    assert ingestion.process(database, item, vectors)
    assert not ingestion.process(database, item, vectors)
    assert len(api.get(f"/documents/{old}/versions").json()) == 2
    with database() as db:
        assert db.scalar(select(func.count(Chunk.id))) == 2


def test_atomic_activation_rollback_restores_previous_active(api, database):
    old = publish(database)
    job_id = stage(database, b"replacement that fails publication", old)

    def reject_new_document(session, *_):
        if any(isinstance(obj, Document) for obj in session.new):
            raise RuntimeError("failure after previous version was deactivated")

    event.listen(database.class_, "before_flush", reject_new_document)
    try:
        assert not ingestion.process(database, take(database), vectors)
    finally:
        event.remove(database.class_, "before_flush", reject_new_document)
    assert api.get(f"/documents/jobs/{job_id}").json()["error_code"] == "publication_failed"
    assert api.get("/documents/catalog").json()[0]["id"] == old
    with database() as db:
        assert db.scalar(select(func.count(DocumentVersion.document_id))) == 1
        assert db.scalar(select(func.count(Chunk.id))) == 1
        original = db.get(DocumentVersion, old)
        assert original is not None and original.active


def test_concurrent_updates_have_one_winner_and_no_silent_overwrite(api, database):
    old = publish(database)
    jobs = [stage(database, b"replacement one", old), stage(database, b"replacement two", old)]
    items = [take(database), take(database)]
    barrier = threading.Barrier(2)

    def slow_embed(pieces):
        barrier.wait(timeout=10)
        return vectors(pieces)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda item: ingestion.process(database, item, slow_embed), items))
    assert sorted(outcomes) == [False, True]
    results = [api.get(f"/documents/jobs/{j}").json() for j in jobs]
    loser = next(j for j in results if j["state"] == "failed")
    assert loser["error_code"] == "version_conflict"
    assert api.post(f"/documents/jobs/{loser['id']}/retry").status_code == 409
    history = api.get(f"/documents/{old}/versions").json()
    assert len(history) == 2 and sum(v["active"] for v in history) == 1


def test_delete_during_update_prevents_resurrection_and_removes_all_versions(api, database):
    first = publish(database)
    second = publish(database, b"updated once", first)
    stage(database, b"inflight replacement", second)
    item = take(database)

    def deleting_embed(pieces):
        assert api.delete(f"/documents/{first}").status_code == 204
        return vectors(pieces)

    assert not ingestion.process(database, item, deleting_embed)
    for doc in (first, second):
        assert api.get(f"/documents/{doc}/original").status_code == 404
        assert api.get(f"/documents/{doc}/chunks/0").status_code == 404
    with database() as db:
        for model in (
            Document,
            Chunk,
            DocumentVersion,
            LogicalDocument,
            IngestionJob,
            ChunkLocation,
        ):
            assert db.scalar(select(func.count()).select_from(model)) == 0


def pdf_bytes():
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for content in (b"First physical page evidence.", b"", b"Third physical page evidence."):
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        if content:
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 30 200 Td (" + content + b") Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_original_and_physical_page_locations(api, database, monkeypatch):
    raw = pdf_bytes()
    doc = publish(database, raw, filename="three-pages.pdf")
    original = api.get(f"/documents/{doc}/original")
    assert original.content == raw and original.headers["content-type"] == "application/pdf"
    assert original.headers["x-content-type-options"] == "nosniff"
    assert original.headers["cache-control"] == "no-store"
    for index, page in [(0, 1), (1, 3)]:
        evidence = api.get(f"/documents/{doc}/chunks/{index}").json()
        assert evidence["page_number"] == page
        assert evidence["original_url"] == f"/documents/{doc}/original#page={page}"
    monkeypatch.setattr(retrieval, "embed_query", lambda q: vectors([q])[0])
    monkeypatch.setattr(
        query_module,
        "search",
        lambda db, q, k: SearchResult(retrieval.vector_search(db, q, k), 0.9),
    )
    monkeypatch.setattr(query_module, "generate_answer", lambda *_: "Offline citation test [1].")
    response = api.post("/query/evidence", json={"question": "Evidence?"}).json()
    assert sorted(c["page_number"] for c in response["citations"]) == [1, 3]
    assert all(c["version_number"] == 1 for c in response["citations"])
    history = api.get(f"/documents/{doc}/versions").json()
    assert history[0]["file_hash"] == hashlib.sha256(raw).hexdigest()


def test_query_dependency_pins_versions_across_dense_and_fts(database, monkeypatch):
    old = publish(database, b"old sharedtoken evidence")
    stage(database, b"new sharedtoken evidence", old)
    item = take(database)
    monkeypatch.setattr(database_module, "SessionLocal", database)
    monkeypatch.setattr(retrieval, "embed_query", lambda q: vectors([q])[0])
    monkeypatch.setattr(query_module, "generate_answer", lambda *_: "Synthetic answer [1].")
    settings = ingestion.get_settings().model_copy(update={"use_rerank": False})
    monkeypatch.setattr(search_module, "get_settings", lambda: settings)
    actual_vector = retrieval.vector_search
    switched = []

    def vector_then_switch(db, question, n):
        rows = actual_vector(db, question, n)
        if not switched:
            assert db.connection().get_isolation_level() == "REPEATABLE READ"
            assert ingestion.process(database, item, vectors)
            switched.append(True)
        return rows

    monkeypatch.setattr(search_module, "vector_search", vector_then_switch)
    with TestClient(app) as client:
        first = client.post("/query/evidence", json={"question": "sharedtoken"}).json()
        assert {c["document_id"] for c in first["citations"]} == {old}
        next_answer = client.post("/query/evidence", json={"question": "sharedtoken"}).json()
        assert old not in {c["document_id"] for c in next_answer["citations"]}
        assert len({c["document_id"] for c in next_answer["citations"]}) == 1


@pytest.fixture
def legacy(database):
    with database() as db:
        doc = Document(title="Legacy", source="unknown.pdf")
        doc.chunks = [
            Chunk(chunk_index=0, content="preserved legacy evidence", embedding=vectors([""])[0])
        ]
        db.add(doc)
        db.commit()
        document_id, chunk_id = doc.id, doc.chunks[0].id
    engine = database.kw["bind"]
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DROP TABLE chunk_locations, document_versions, logical_documents CASCADE"
        )
        conn.exec_driver_sql("""
            ALTER TABLE ingestion_jobs DROP COLUMN dedup_key,
                DROP COLUMN target_logical_id, DROP COLUMN base_document_id,
                DROP COLUMN knowledge_base_id;
            ALTER TABLE ingestion_jobs ADD CONSTRAINT ingestion_jobs_file_hash_key
                UNIQUE(file_hash);
            INSERT INTO ingestion_jobs(file_hash,title,filename,payload,state,attempts,created_at)
                VALUES ('pending-hash','Pending','pending.txt',
                        convert_to('pending text','UTF8'),'queued',0,now());
        """)
        before = conn.exec_driver_sql(
            "SELECT id,document_id,chunk_index,content,embedding::text FROM chunks"
        ).all()
    return engine, document_id, chunk_id, before


def test_migration_backfills_legacy_ids_and_vectors_without_inventing_originals(
    api, database, legacy
):
    engine, doc, chunk, before = legacy
    assert upgrade(engine)
    assert not upgrade(engine)
    with engine.connect() as conn:
        assert (
            conn.exec_driver_sql(
                "SELECT id,document_id,chunk_index,content,embedding::text FROM chunks"
            ).all()
            == before
        )
        assert (
            conn.exec_driver_sql("SELECT dedup_key FROM ingestion_jobs").scalar_one()
            == "new:pending-hash"
        )
    from scripts.migrate_application_auth import upgrade as upgrade_auth

    assert upgrade_auth(engine, "legacy_owner", "legacy-password-123", "Imported")
    with engine.connect() as conn:
        kb_id = conn.exec_driver_sql("SELECT id FROM knowledge_bases").scalar_one()
    database.configure(info={"knowledge_base_id": kb_id})
    history = api.get(f"/documents/{doc}/versions").json()
    assert len(history) == 1 and history[0]["version_number"] == 1
    assert history[0]["logical_id"] == doc and not history[0]["original_available"]
    assert api.get(f"/documents/{doc}/original").status_code == 404
    assert api.get(f"/documents/{doc}/chunks/0").json()["page_number"] is None
    # A queued pre-upgrade job remains executable; it now retains its actual raw file.
    assert ingestion.process(database, take(database), vectors)
    with database() as db:
        assert db.scalar(select(func.count(DocumentVersion.document_id))) == 2
    replacement = publish(database, b"replacement for legacy", doc)
    assert replacement != doc
    assert api.get(f"/documents/{doc}/chunks/0").json()["content"] == "preserved legacy evidence"


def test_migration_rolls_back_all_ddl_on_failure(legacy):
    engine, _, _, before = legacy

    def inject(conn, cursor, statement, *_):
        if statement.lstrip().startswith("INSERT INTO application_migrations"):
            raise RuntimeError("injected commit boundary failure")

    event.listen(engine, "before_cursor_execute", inject)
    try:
        with pytest.raises(RuntimeError):
            upgrade(engine)
    finally:
        event.remove(engine, "before_cursor_execute", inject)
    assert not inspect(engine).has_table("document_versions")
    assert "dedup_key" not in {c["name"] for c in inspect(engine).get_columns("ingestion_jobs")}
    with engine.connect() as conn:
        assert (
            conn.exec_driver_sql(
                "SELECT id,document_id,chunk_index,content,embedding::text FROM chunks"
            ).all()
            == before
        )
    assert upgrade(engine)


def test_concurrent_migration_runners_are_serialized(legacy):
    engine, *_ = legacy
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(upgrade, [engine, engine])) == [False, True]
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM application_migrations").scalar_one() == 1


def test_bootstrap_refuses_implicit_upgrade(legacy, monkeypatch):
    engine, *_ = legacy
    monkeypatch.setattr(database_module, "engine", engine)
    with pytest.raises(RuntimeError, match="schema v1"):
        database_module.init_db()
    assert not inspect(engine).has_table("document_versions")


def test_native_versioned_schema_refuses_legacy_backfill(database):
    with pytest.raises(ValueError, match="already versioned"):
        upgrade(database.kw["bind"])
    assert not inspect(database.kw["bind"]).has_table("application_migrations")


def test_migration_detects_checksum_mismatch(legacy):
    engine, *_ = legacy
    assert upgrade(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE application_migrations SET sha256=%(hash)s", {"hash": "0" * 64}
        )
    with pytest.raises(ValueError, match="checksum"):
        upgrade(engine)
