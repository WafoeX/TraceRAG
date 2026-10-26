"""Real sessions + PostgreSQL isolation. No authorization dependency overrides."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, inspect

from app import database as database_module
from app.auth import COOKIE
from app.main import app
from app.models import AuthSession, IngestionJob
from app.routers import query as query_module
from app.security import hash_password, token_hash, verify_password
from app.services import ingestion, retrieval
from app.services import search as search_module
from scripts.manage_application_users import create_user
from scripts.migrate_application_auth import upgrade

from .test_application_lifecycle import database as _database_fixture
from .test_application_lifecycle import take, vectors

database = _database_fixture
PW = "test-password-1234"
pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in isolated DB"),
]


@pytest.fixture
def clients(database, monkeypatch):
    monkeypatch.setattr(database_module, "SessionLocal", database)
    monkeypatch.setattr(ingestion.get_settings(), "session_cookie_secure", False)
    create_user(database.kw["bind"], "alice", PW)
    create_user(database.kw["bind"], "bob", PW)
    with TestClient(app) as alice, TestClient(app) as bob:
        for client, name in [(alice, "alice"), (bob, "bob")]:
            response = client.post("/auth/login", json={"username": name, "password": PW})
            assert response.status_code == 200
            client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        a = alice.post("/knowledge-bases", json={"name": "Alice papers"}).json()["id"]
        b = bob.post("/knowledge-bases", json={"name": "Bob papers"}).json()["id"]
        yield alice, bob, a, b


def submit(client, kb, content="sharedtoken private evidence"):
    response = client.post(
        f"/documents/jobs?knowledge_base_id={kb}", json={"title": "Paper", "content": content}
    )
    assert response.status_code == 202
    return response.json()["id"]


def finish(database, job_id):
    with database() as db:
        item = ingestion.claim(db, require_owned=True)
    assert item is not None and item.id == job_id
    assert ingestion.process(database, item, vectors)
    with database() as db:
        job = db.get(IngestionJob, job_id)
        assert job is not None
        return job.document_id


@pytest.mark.parametrize(
    "path,method",
    [
        ("/documents", "GET"),
        ("/documents/catalog", "GET"),
        ("/documents/jobs", "GET"),
        ("/documents/1/original", "GET"),
        ("/documents/1/versions", "GET"),
        ("/documents/1/chunks/0", "GET"),
        ("/documents/jobs/1", "GET"),
        ("/query", "POST"),
        ("/query/evidence", "POST"),
        ("/query/stream", "POST"),
        ("/documents", "POST"),
        ("/documents/1", "DELETE"),
        ("/knowledge-bases", "GET"),
        ("/knowledge-bases", "POST"),
    ],
)
def test_anonymous_has_no_rag_entrypoint(database, monkeypatch, path, method):
    monkeypatch.setattr(database_module, "SessionLocal", database)
    with TestClient(app) as client:
        response = client.request(
            method, path, json={"title": "x", "content": "x", "question": "x", "name": "x"}
        )
        assert response.status_code == 401


def test_session_flags_rotation_logout_and_expiry(clients, database):
    alice, _, a, _ = clients
    raw = alice.cookies.get(COOKIE)
    assert raw is not None
    with database() as db:
        saved = db.get(AuthSession, token_hash(raw))
        assert saved is not None and saved.token_hash != raw
    assert alice.get("/auth/me").status_code == 200
    assert alice.post("/auth/logout").status_code == 204
    alice.cookies.set(COOKIE, raw)
    assert alice.get(f"/documents?knowledge_base_id={a}").status_code == 401
    alice.cookies.clear()  # remove the deliberately injected domain-less replay cookie
    response = alice.post("/auth/login", json={"username": "ALICE", "password": PW})
    assert response.status_code == 200 and alice.cookies.get(COOKIE) != raw
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    current_raw = alice.cookies.get(COOKIE)
    assert current_raw is not None
    with database() as db:
        current = db.get(AuthSession, token_hash(current_raw))
        assert current is not None
        current.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    assert alice.get("/auth/me").status_code == 401


def test_csrf_and_origin_are_required_for_mutations(clients):
    alice, _, a, _ = clients
    token = alice.headers.pop("X-CSRF-Token")
    assert alice.post("/knowledge-bases", json={"name": "blocked"}).status_code == 403
    alice.headers["X-CSRF-Token"] = token
    assert (
        alice.post(
            "/knowledge-bases", json={"name": "blocked"}, headers={"Origin": "https://evil.invalid"}
        ).status_code
        == 403
    )
    assert (
        alice.post(
            f"/documents/jobs?knowledge_base_id={a}",
            json={"title": "x", "content": "x"},
            headers={"X-CSRF-Token": "wrong"},
        ).status_code
        == 403
    )
    assert (
        alice.post(
            "/auth/login", json={"username": "alice", "password": PW}, headers={"Origin": "null"}
        ).status_code
        == 403
    )


@pytest.mark.parametrize(
    "suffix,method",
    [
        ("/original", "GET"),
        ("/versions", "GET"),
        ("/chunks/0", "GET"),
        ("", "DELETE"),
        ("/updates", "POST"),
    ],
)
def test_guessed_foreign_document_id_is_not_accessible(clients, database, suffix, method):
    alice, bob, a, b = clients
    doc = finish(database, submit(alice, a, "SECRET_ALICE_ONLY"))
    response = bob.request(
        method,
        f"/documents/{doc}{suffix}?knowledge_base_id={b}",
        json={"title": "hack", "content": "replace"},
    )
    assert response.status_code == 404 and "SECRET_ALICE_ONLY" not in response.text
    assert bob.get(f"/documents/{doc}/original").status_code == 404


def test_job_update_upload_and_retry_scope(clients, database):
    alice, bob, a, b = clients
    job = submit(alice, a)
    for method, suffix in [("GET", ""), ("POST", "/retry"), ("DELETE", "")]:
        assert bob.request(method, f"/documents/jobs/{job}{suffix}").status_code == 404
    doc = finish(database, job)
    assert (
        bob.post(
            f"/documents/{doc}/updates/upload", files={"file": ("x.txt", b"changed")}
        ).status_code
        == 404
    )
    assert alice.get(f"/documents/{doc}/original").status_code == 200
    assert bob.get(f"/documents/jobs?knowledge_base_id={b}").json() == []
    assert bob.get(f"/documents/catalog?knowledge_base_id={b}").json() == []


def test_dedup_is_per_library_and_lists_are_owner_only(clients, database):
    alice, bob, a, b = clients
    one = submit(alice, a)
    assert submit(alice, a) == one
    two = submit(bob, b)
    assert two != one
    assert [x["id"] for x in alice.get("/knowledge-bases").json()] == [a]
    assert [x["id"] for x in bob.get("/knowledge-bases").json()] == [b]
    assert alice.get(f"/documents?knowledge_base_id={b}").status_code == 404
    assert (
        alice.post(
            f"/documents/jobs?knowledge_base_id={b}", json={"title": "x", "content": "x"}
        ).status_code
        == 404
    )
    assert alice.get("/documents").status_code == 400


def test_dense_fts_and_query_are_filtered_before_limit(clients, database, monkeypatch):
    alice, bob, a, b = clients
    other = finish(database, submit(bob, b, "sharedtoken " * 30 + "BOB_ONLY"))
    mine = finish(database, submit(alice, a, "sharedtoken ALICE_ONLY"))
    monkeypatch.setattr(retrieval, "embed_query", lambda q: vectors([q])[0])
    with database() as db:
        db.info["knowledge_base_id"] = a
        assert [c.document_id for c in retrieval.vector_search(db, "sharedtoken", 1)] == [mine]
        assert [c.document_id for c in retrieval.keyword_search(db, "sharedtoken", 1)] == [mine]
    settings = ingestion.get_settings().model_copy(update={"use_rerank": False})
    monkeypatch.setattr(search_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_module,
        "generate_answer",
        lambda q, chunks: "ALICE_ONLY [1]"
        if all(c.document_id == mine for c in chunks)
        else "LEAK",
    )
    response = alice.post(
        f"/query/evidence?knowledge_base_id={a}", json={"question": "sharedtoken"}
    )
    assert response.status_code == 200
    assert "BOB_ONLY" not in response.text and "LEAK" not in response.text
    assert {c["document_id"] for c in response.json()["citations"]} == {mine}
    assert other != mine
    assert (
        alice.post(f"/query?knowledge_base_id={b}", json={"question": "sharedtoken"}).status_code
        == 404
    )


def test_two_libraries_of_one_owner_are_separate(clients, database, monkeypatch):
    alice, bob, first, _ = clients
    second = alice.post("/knowledge-bases", json={"name": "Second", "owner_id": 2}).json()["id"]
    one = finish(database, submit(alice, first, "first private contents"))
    two = finish(database, submit(alice, second, "second private contents"))
    assert bob.get("/knowledge-bases").json()[0]["name"] == "Bob papers"
    assert len(bob.get("/knowledge-bases").json()) == 1
    assert {d["id"] for d in alice.get(f"/documents?knowledge_base_id={first}").json()} == {one}
    assert {d["id"] for d in alice.get(f"/documents?knowledge_base_id={second}").json()} == {two}
    assert alice.get(f"/documents/{one}/original?knowledge_base_id={second}").status_code == 404
    monkeypatch.setattr(retrieval, "embed_query", lambda q: vectors([q])[0])
    with database() as db:
        db.info["knowledge_base_id"] = second
        assert {c.document_id for c in retrieval.vector_search(db, "contents", 10)} == {two}
        assert {c.document_id for c in retrieval.keyword_search(db, "contents", 10)} == {two}


@pytest.mark.parametrize(
    "path", ["/documents/upload", "/documents/jobs/upload", "/documents/1/updates/upload"]
)
def test_anonymous_file_uploads_are_not_a_bypass(database, monkeypatch, path):
    monkeypatch.setattr(database_module, "SessionLocal", database)
    with TestClient(app) as client:
        assert client.post(path, files={"file": ("x.txt", b"not published")}).status_code == 401


def test_synchronous_legacy_routes_are_authenticated_and_scoped(clients, database, monkeypatch):
    alice, bob, a, b = clients
    monkeypatch.setattr(ingestion, "embed_texts", vectors)
    response = alice.post(
        f"/documents?knowledge_base_id={a}",
        json={"title": "Legacy text", "content": "legacy sentinel"},
    )
    assert response.status_code == 201
    doc = response.json()["document_id"]
    assert bob.get(f"/documents/{doc}/chunks/0").status_code == 404
    uploaded = alice.post(
        f"/documents/upload?knowledge_base_id={a}", files={"file": ("old.txt", b"legacy file text")}
    )
    assert uploaded.status_code == 201
    assert bob.get(f"/documents?knowledge_base_id={b}").json() == []


def test_login_throttle_is_persistent_and_forwarded_ip_not_trusted(clients):
    alice, _, _, _ = clients
    for i in range(18):
        assert (
            alice.post(
                "/auth/login",
                json={"username": "alice", "password": "wrong"},
                headers={"X-Forwarded-For": str(i)},
            ).status_code
            == 401
        )
    assert alice.post("/auth/login", json={"username": "alice", "password": PW}).status_code == 429


def test_secure_cookie_default_and_password_redaction(clients, monkeypatch):
    alice, _, _, _ = clients
    monkeypatch.setattr(ingestion.get_settings(), "session_cookie_secure", True)
    response = alice.post("/auth/login", json={"username": "alice", "password": PW})
    assert "secure" in response.headers["set-cookie"].lower()
    bad = alice.post("/auth/login", json={"username": "alice", "password": "SECRET" * 100})
    assert bad.status_code == 422 and "SECRET" not in bad.text


def make_v2(database):
    with database() as db:
        job = ingestion.enqueue(db, "Legacy", "legacy.txt", b"legacy preserved original")
        job_id = job.id
    assert ingestion.process(database, take(database), vectors)
    engine = database.kw["bind"]
    with engine.begin() as conn:
        before = conn.exec_driver_sql(
            "SELECT id,document_id,content,embedding::text FROM chunks"
        ).all()
        conn.exec_driver_sql("""
            DROP TABLE application_request_feedback;
            DROP TABLE application_request_traces;
            DROP TABLE application_budget_events, application_budget_reservations,
                       application_budget_accounts;
            ALTER TABLE logical_documents DROP COLUMN knowledge_base_id;
            ALTER TABLE ingestion_jobs DROP COLUMN knowledge_base_id;
            DROP TABLE auth_sessions, auth_login_windows, knowledge_bases, app_users;
        """)
    return engine, job_id, before


def test_migration_assigns_only_explicit_owner_and_preserves_data(database, monkeypatch):
    engine, job_id, before = make_v2(database)
    assert upgrade(engine, "chosen_owner", PW, "Imported library")
    assert not upgrade(engine, "different_owner", PW, "Not reassigned")
    with engine.connect() as conn:
        assert (
            conn.exec_driver_sql("SELECT id,document_id,content,embedding::text FROM chunks").all()
            == before
        )
        assert conn.exec_driver_sql("SELECT username FROM app_users").scalar_one() == "chosen_owner"
    monkeypatch.setattr(database_module, "SessionLocal", database)
    monkeypatch.setattr(ingestion.get_settings(), "session_cookie_secure", False)
    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "chosen_owner", "password": PW})
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        kb = client.get("/knowledge-bases").json()[0]["id"]
        docs = client.get(f"/documents/catalog?knowledge_base_id={kb}").json()
        assert len(docs) == 1
        assert (
            client.get(f"/documents/{docs[0]['id']}/original").content
            == b"legacy preserved original"
        )
        assert client.get(f"/documents/jobs/{job_id}").status_code == 200


def test_auth_migration_is_atomic_and_bootstrap_refuses_implicit_upgrade(database, monkeypatch):
    engine, _, before = make_v2(database)
    monkeypatch.setattr(database_module, "engine", engine)
    with pytest.raises(RuntimeError, match="schema v2"):
        database_module.init_db()

    def fail(conn, cursor, statement, *_):
        if statement.startswith("INSERT INTO application_migrations"):
            raise RuntimeError("inject ledger failure")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError):
            upgrade(engine, "owner_test", PW, "Imported")
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    assert "knowledge_base_id" not in {
        c["name"] for c in inspect(engine).get_columns("logical_documents")
    }
    assert not inspect(engine).has_table("app_users")
    with engine.connect() as conn:
        assert (
            conn.exec_driver_sql("SELECT id,document_id,content,embedding::text FROM chunks").all()
            == before
        )


def test_migration_checksum_rejects_revised_source(database):
    engine, _, _ = make_v2(database)
    assert upgrade(engine, "owner_test", PW, "Imported")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE application_migrations SET sha256=%(hash)s WHERE version=%(v)s",
            {"hash": "0" * 64, "v": "002_owner_isolation"},
        )
    with pytest.raises(ValueError, match="checksum"):
        upgrade(engine, "owner_test", PW, "Imported")


def test_worker_does_not_claim_unowned_jobs(database):
    with database() as db:
        ingestion.enqueue(db, "Quarantine", "x.txt", b"unowned")
    with database() as db:
        assert ingestion.claim(db, require_owned=True) is None


def test_account_case_collision_cannot_replace_credentials(clients, database):
    from sqlalchemy.exc import IntegrityError

    alice, _, _, _ = clients
    with pytest.raises(IntegrityError):
        create_user(database.kw["bind"], "ALICE", "new-password-5678")
    assert alice.post("/auth/login", json={"username": "alice", "password": PW}).status_code == 200
    assert (
        alice.post("/auth/register", json={"username": "mallory", "password": PW}).status_code
        == 404
    )


def test_password_scrypt_salts_and_validation():
    a, b = hash_password(PW), hash_password(PW)
    assert a != b and verify_password(PW, a) and not verify_password("wrong", a)
    assert not verify_password(PW, "malformed")
    with pytest.raises(ValueError):
        hash_password("short")
