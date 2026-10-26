"""Offline deployment guards: no Docker daemon/database/provider needed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.main import app
from app.services.application_schema import DatabaseNotReady, check_guard

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("available", [True, False])
def test_liveness_is_not_readiness_and_response_is_private(monkeypatch, available):
    monkeypatch.setattr(main_module, "ready", lambda *args: available)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        result = client.get("/ready")
    assert result.status_code == (200 if available else 503)
    assert result.json() == {"status": "ready" if available else "not_ready"}
    assert result.headers["cache-control"] == "no-store"


def test_startup_ddl_failure_is_fatal_and_redacted(monkeypatch):
    monkeypatch.setattr(get_settings(), "run_db_init", True)

    def fail():
        raise RuntimeError("SECRET DATABASE URL")

    monkeypatch.setattr(main_module, "init_db", fail)
    with pytest.raises(RuntimeError, match="startup failed") as exc, TestClient(app):
        pass
    assert "SECRET" not in str(exc.value) and exc.value.__suppress_context__


def test_strict_api_and_worker_refuse_unready_database(monkeypatch):
    from app import worker

    monkeypatch.setattr(get_settings(), "application_startup_check", True)
    monkeypatch.setattr(main_module, "ready", lambda *args: False)
    monkeypatch.setattr(worker, "ready", lambda *args: False)
    with pytest.raises(RuntimeError, match="startup failed"), TestClient(app):
        pass
    with pytest.raises(RuntimeError, match="startup failed"):
        worker.main()


@pytest.mark.parametrize("blocked", [False, True])
def test_recovery_guard_blocks_without_exception_content(blocked):
    db = MagicMock()
    db.connection.return_value.exec_driver_sql.return_value.scalar_one_or_none.return_value = (
        blocked
    )
    if blocked:
        with pytest.raises(DatabaseNotReady, match="recovery review"):
            check_guard(db)
    else:
        check_guard(db)
    db.connection.side_effect = RuntimeError("SECRET")
    with pytest.raises(DatabaseNotReady) as exc:
        check_guard(db)
    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize("args", [[], ["init"], ["migrate"], ["release"]])
def test_plan_never_loads_env_or_connects(args):
    env = {**os.environ, "OPENAI_API_KEY": "", "DATABASE_URL": "bad://SECRET"}
    result = subprocess.run(
        [sys.executable, "-m", "scripts.manage_application_database", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "PLAN ONLY" in result.stdout
    assert "SECRET" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "operation,flags",
    [
        ("init", []),
        ("migrate", ["--confirm-services-stopped", "--confirm-database", "test"]),
        ("release", ["--confirm-services-stopped", "--confirm-database", "test"]),
    ],
)
def test_mutations_require_explicit_confirmations(operation, flags):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.manage_application_database",
            operation,
            "--apply",
            "--database-url",
            "bad://SECRET",
            *flags,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "SECRET" not in result.stdout + result.stderr


def test_cli_imports_do_not_load_provider_settings():
    code = (
        "import sys; import scripts.manage_application_database; "
        "assert 'app.config' not in sys.modules; assert 'openai' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    result = subprocess.run(
        [sys.executable, "-m", "scripts.migrate_application_recovery"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "PLAN ONLY" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["backup", "../bad.dump", "--confirm-services-stopped", "--confirm-database", "wrong"],
        ["restore", "x.dump", "--confirm-services-stopped", "--confirm-database", "wrong"],
    ],
)
def test_backup_plan_or_rejection_before_any_connection(args):
    env = {**os.environ, "PGHOST": "127.0.0.1", "PGUSER": "test", "PGDATABASE": "unused"}
    result = subprocess.run(
        ["bash", "scripts/application_backup.sh", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (not args)


def test_image_and_compose_have_no_private_build_inputs_or_implicit_ddl():
    ignore = (ROOT / ".dockerignore").read_text()
    assert "**/.env*" in ignore and "**/*.dump" in ignore
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "USER 10001:10001" in dockerfile and "TIKTOKEN_CACHE_DIR" in dockerfile
    compose = (ROOT / "compose.application.yml").read_text()
    assert 'RUN_DB_INIT: "false"' in compose and 'APPLICATION_STARTUP_CHECK: "true"' in compose
    assert "127.0.0.1:" in compose and "/ready" in compose
    assert "service_healthy" in compose and "app.worker" in compose


def test_schema_manifest_tracks_runtime_models():
    from sqlalchemy.dialects import postgresql

    from app import models  # noqa: F401
    from app.database import Base

    manifest = json.loads((ROOT / "deploy/schema-manifest.json").read_text())
    assert set(manifest) == set(Base.metadata.tables)
    for name, table in Base.metadata.tables.items():
        assert set(manifest[name]["columns"]) == set(table.columns.keys())
        for column in table.columns:
            kind = str(column.type.compile(dialect=postgresql.dialect()))
            expected = manifest[name]["columns"][column.name]
            assert (
                expected["type"].replace("__EMBEDDING_DIM__", str(get_settings().embedding_dim))
                == kind
            )
            assert expected["nullable"] == column.nullable
