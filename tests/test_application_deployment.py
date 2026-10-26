"""Explicit disposable PG only: schema readiness, migration chain and quarantine."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, event, inspect

from app import database as database_module
from app.config import get_settings
from app.services.application_schema import ROOT, DatabaseNotReady, check_guard, ready
from scripts.manage_application_database import initialize, migrate, release, version
from scripts.migrate_application_recovery import upgrade

from .test_application_lifecycle import TEST_URL
from .test_application_lifecycle import database as _database_fixture
from .test_auth_isolation import clients as _clients_fixture

database = _database_fixture
clients = _clients_fixture
pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in disposable DB"),
]


@pytest.fixture
def healthy(database, monkeypatch):
    engine = database.kw["bind"]
    monkeypatch.setattr(database_module, "engine", engine)
    database_module.init_db()
    assert ready(engine, get_settings().embedding_dim)
    return engine


@pytest.fixture
def empty_database():
    admin = create_engine(TEST_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        assert (
            conn.exec_driver_sql("SELECT current_database()").scalar_one()
            == "tracerag_application_test"
        )
        conn.exec_driver_sql("CREATE DATABASE tracerag_deployment_bootstrap_test")
    engine = create_engine(
        TEST_URL.replace("/tracerag_application_test", "/tracerag_deployment_bootstrap_test")
    )
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.exec_driver_sql("DROP DATABASE tracerag_deployment_bootstrap_test WITH (FORCE)")
        admin.dispose()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_guard",
        "missing_column",
        "wrong_dimension",
        "missing_fts",
        "missing_version_index",
        "missing_fk",
    ],
)
def test_readiness_rejects_damaged_schema_without_ddl(healthy, damage):
    statements = {
        "missing_guard": "DROP TABLE application_recovery_state",
        "missing_column": "ALTER TABLE application_request_feedback DROP COLUMN outdated CASCADE",
        "missing_fts": "DROP INDEX chunks_content_fts",
        "missing_version_index": "DROP INDEX uq_document_version_active",
        "missing_fk": "ALTER TABLE auth_sessions DROP CONSTRAINT auth_sessions_user_id_fkey",
    }
    if damage != "wrong_dimension":
        with healthy.begin() as conn:
            conn.exec_driver_sql(statements[damage])
    assert not ready(healthy, 2 if damage == "wrong_dimension" else get_settings().embedding_dim)
    if damage == "missing_guard":
        assert not inspect(healthy).has_table("application_recovery_state")


def test_quarantine_blocks_login_private_routes_and_worker(clients, database):
    from app.services.ingestion import claim
    from app.worker import process_owned_job

    from .test_auth_isolation import submit

    alice, _, kb, _ = clients
    submit(alice, kb)
    with database() as db:
        item = claim(db, require_owned=True)
    assert item is not None
    with database.kw["bind"].begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO application_recovery_state(id,quarantined) VALUES(1,true)"
        )
    assert alice.get("/auth/me").status_code == 503
    assert alice.get(f"/documents/catalog?knowledge_base_id={kb}").status_code == 503
    assert (
        alice.post(
            "/auth/login", json={"username": "alice", "password": "test-password-1234"}
        ).status_code
        == 503
    )
    with pytest.raises(DatabaseNotReady, match="recovery review"):
        process_owned_job(item)


def test_v6_requires_explicit_migration_and_checksums(healthy, monkeypatch):
    with healthy.begin() as conn:
        conn.exec_driver_sql("DROP TABLE application_recovery_state")
    with pytest.raises(RuntimeError, match="v6"):
        database_module.init_db()
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: upgrade(healthy), range(2))) == [False, True]
    assert ready(healthy, get_settings().embedding_dim)
    with healthy.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE application_migrations SET sha256='tampered' WHERE version='006_recovery_guard'"
        )
    assert not ready(healthy, get_settings().embedding_dim)
    with pytest.raises(ValueError, match="checksum"):
        upgrade(healthy)


def test_native_guard_is_not_blindly_adopted(healthy):
    with pytest.raises(ValueError, match="Native"):
        upgrade(healthy)


def test_006_rolls_back_ddl_and_ledger(healthy):
    with healthy.begin() as conn:
        conn.exec_driver_sql("DROP TABLE application_recovery_state")

    def fail(conn, cursor, statement, *args):
        if "CREATE TABLE application_recovery_state" in statement:
            raise RuntimeError("injected")

    event.listen(healthy, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            upgrade(healthy)
    finally:
        event.remove(healthy, "before_cursor_execute", fail)
    assert not inspect(healthy).has_table("application_recovery_state")
    assert not inspect(healthy).has_table("application_migrations")


def test_explicit_empty_bootstrap_and_no_overwrite(empty_database):
    engine = empty_database
    initialize(engine, "tracerag_deployment_bootstrap_test", 3)
    assert ready(engine, 3) and not ready(engine, 1536)
    with pytest.raises(ValueError, match="empty"):
        initialize(engine, "tracerag_deployment_bootstrap_test", 3)
    with pytest.raises(ValueError, match="confirmation"):
        initialize(engine, "wrong_database", 3)


@pytest.mark.parametrize("kind", ["view", "sequence"])
def test_bootstrap_rejects_nonempty_relations_without_tables(empty_database, kind):
    with empty_database.begin() as conn:
        if kind == "view":
            conn.exec_driver_sql("CREATE VIEW preserved_view AS SELECT 1 AS marker")
        else:
            conn.exec_driver_sql("CREATE SEQUENCE preserved_sequence")
    with pytest.raises(ValueError, match="empty"):
        initialize(empty_database, "tracerag_deployment_bootstrap_test", 3)
    assert not inspect(empty_database).has_table("documents")


def test_bootstrap_transaction_rolls_back_on_index_failure(empty_database):
    def fail(conn, cursor, statement, *args):
        if "CREATE INDEX IF NOT EXISTS chunks_content_fts" in statement:
            raise RuntimeError("injected after tables")

    event.listen(empty_database, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            initialize(empty_database, "tracerag_deployment_bootstrap_test", 3)
    finally:
        event.remove(empty_database, "before_cursor_execute", fail)
    assert inspect(empty_database).get_table_names() == []


def legacy_v1(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE EXTENSION vector;
            CREATE TABLE documents(id SERIAL PRIMARY KEY,title VARCHAR(512) NOT NULL,
                source VARCHAR(1024),created_at TIMESTAMPTZ NOT NULL);
            CREATE TABLE chunks(id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,content TEXT NOT NULL,embedding vector(3) NOT NULL);
            CREATE TABLE ingestion_jobs(id SERIAL PRIMARY KEY,file_hash VARCHAR(64) NOT NULL UNIQUE,
                title VARCHAR(512) NOT NULL,filename VARCHAR(512) NOT NULL,payload BYTEA,
                state VARCHAR(20) NOT NULL,attempts INTEGER NOT NULL,lease_token VARCHAR(64),
                lease_until TIMESTAMPTZ,error_code VARCHAR(64),
                document_id INTEGER REFERENCES documents(id),
                created_at TIMESTAMPTZ NOT NULL);
            INSERT INTO documents(title,source,created_at) VALUES('Legacy','legacy.pdf',now());
            INSERT INTO chunks(document_id,chunk_index,content,embedding)
                VALUES(1,0,'preserved original evidence','[1,0,0]');
            INSERT INTO ingestion_jobs(file_hash,title,filename,payload,state,attempts,created_at)
                VALUES('pending','Pending','p.txt',convert_to('payload','UTF8'),'queued',0,now());
        """)


def test_full_frozen_migration_chain_preserves_legacy_data(empty_database):
    legacy_v1(empty_database)
    with empty_database.connect() as conn:
        assert version(conn) == 1
        before = conn.exec_driver_sql(
            "SELECT id,document_id,content,embedding::text FROM chunks"
        ).all()
    migrate(
        empty_database,
        "tracerag_deployment_bootstrap_test",
        1,
        3,
        "legacyowner",
        "test-password-1234",
        "Legacy papers",
    )
    assert ready(empty_database, 3)
    with empty_database.connect() as conn:
        assert version(conn) == 7
        assert (
            conn.exec_driver_sql("SELECT id,document_id,content,embedding::text FROM chunks").all()
            == before
        )
        assert conn.exec_driver_sql("SELECT count(*) FROM application_migrations").scalar_one() == 6
        assert (
            conn.exec_driver_sql(
                "SELECT count(*) FROM logical_documents WHERE knowledge_base_id IS NULL"
            ).scalar_one()
            == 0
        )
    with pytest.raises(ValueError, match="source version"):
        migrate(empty_database, "tracerag_deployment_bootstrap_test", 1, 3, None, None, None)
    # Explicit latest-version resume verifies ledger/schema and repairs only known indexes.
    migrate(empty_database, "tracerag_deployment_bootstrap_test", 7, 3, None, None, None)
    assert ready(empty_database, 3)


def test_quarantine_and_review_never_refund_or_automatically_retry(empty_database):
    from sqlalchemy.orm import Session

    initialize(empty_database, "tracerag_deployment_bootstrap_test", 3)
    with empty_database.begin() as conn:
        conn.exec_driver_sql("""
            INSERT INTO app_users(username,password_hash,created_at)
                VALUES('restored','hash',now());
            INSERT INTO knowledge_bases(owner_id,name,created_at) VALUES(1,'KB',now());
            INSERT INTO auth_sessions(token_hash,user_id,csrf_token,expires_at)
                VALUES('old',1,'csrf',now()+interval '1 day');
            INSERT INTO application_budget_accounts(key,limit_cny,spent_cny,held_cny,blocked)
                VALUES('global',10,1,2,false);
            INSERT INTO application_budget_reservations
                (id,user_id,role,model,profile_json,reserved_cny,status)
                VALUES('unknown',1,'generation','fixture','{}',2,'unknown');
            INSERT INTO ingestion_jobs(knowledge_base_id,file_hash,dedup_key,title,
                filename,payload,state,attempts,lease_token,created_at)
                VALUES(1,'h','kb:1:new:h','Pending','p.txt',convert_to('private','UTF8'),
                       'embedding',1,'old-lease',now());
            INSERT INTO application_request_traces(id,user_id,knowledge_base_id,
                created_at,expires_at,content_consent,metadata_json,snapshot_json)
                VALUES('expired',1,1,now()-interval '8 days',now()-interval '1 day',true,'{}','{}');
            INSERT INTO application_request_feedback
                (trace_id,unhelpful,unsupported_citation,outdated)
                VALUES('expired',true,false,false);
        """)
        conn.exec_driver_sql((ROOT / "deploy/restore-quarantine.sql").read_text())
    assert not ready(empty_database, 3)
    with Session(empty_database) as db, pytest.raises(DatabaseNotReady):
        check_guard(db)
    release(empty_database, "tracerag_deployment_bootstrap_test", 3)
    assert ready(empty_database, 3)
    with empty_database.connect() as conn:
        account = conn.exec_driver_sql(
            "SELECT spent_cny,held_cny,blocked FROM application_budget_accounts"
        ).one()
        assert tuple(account) == (1, 2, True)
        assert (
            conn.exec_driver_sql("SELECT status FROM application_budget_reservations").scalar_one()
            == "unknown"
        )
        job = conn.exec_driver_sql(
            "SELECT state,error_code,lease_token,payload FROM ingestion_jobs"
        ).one()
        assert (
            job.state == "failed"
            and job.error_code == "restore_review_required"
            and job.lease_token is None
        )
        assert job.payload == b"private"
        for table in (
            "auth_sessions",
            "application_request_traces",
            "application_request_feedback",
        ):
            assert conn.exec_driver_sql("SELECT count(*) FROM " + table).scalar_one() == 0
