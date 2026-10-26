"""Migration CLI plans never open a database or import model/provider code."""

from __future__ import annotations

import sys

import pytest

from scripts import migrate_application


def test_plan_does_not_open_database(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A plan must never connect")

    monkeypatch.setattr(migrate_application, "create_engine", forbidden)
    monkeypatch.setattr(sys, "argv", ["migrate_application"])
    migrate_application.main()
    assert "PLAN ONLY" in capsys.readouterr().out


def test_apply_requires_explicit_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "must-not-use-developer-env")
    monkeypatch.setattr(sys, "argv", ["migrate_application", "--apply"])
    with pytest.raises(SystemExit) as error:
        migrate_application.main()
    assert error.value.code == 2


def test_engine_creation_failure_does_not_expose_credentials(monkeypatch):
    def fail(_):
        raise RuntimeError("secret-password-inside-error")

    monkeypatch.setattr(migrate_application, "create_engine", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_application",
            "--apply",
            "--database-url",
            "invalid-url-with-secret",
        ],
    )
    with pytest.raises(SystemExit) as error:
        migrate_application.main()
    assert "secret" not in str(error.value)
    assert "verify the ledger" in str(error.value)
