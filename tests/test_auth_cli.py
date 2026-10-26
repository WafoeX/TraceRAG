"""Offline administrator entrypoint and password-policy checks."""

from __future__ import annotations

import io
import sys

import pytest

from app.security import hash_password, normalize_username, verify_password
from scripts import manage_application_users, migrate_application_auth


def test_auth_migration_plan_never_connects(monkeypatch, capsys):
    def forbidden(*_, **__):
        raise AssertionError("Plan must never connect")

    monkeypatch.setattr(migrate_application_auth, "create_engine", forbidden)
    monkeypatch.setattr(sys, "argv", ["migration"])
    migrate_application_auth.main()
    assert "PLAN ONLY" in capsys.readouterr().out


def test_legacy_owner_must_be_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["migration", "--apply", "--database-url", "not-used"])
    with pytest.raises(SystemExit) as error:
        migrate_application_auth.main()
    assert error.value.code == 2


def test_admin_cli_never_echoes_password_or_connection_error(monkeypatch, capsys):
    def fail(*_, **__):
        raise RuntimeError("PRIVATE_CONNECTION_SECRET")

    monkeypatch.setattr(manage_application_users, "create_engine", fail)
    monkeypatch.setattr(sys, "stdin", io.StringIO("PRIVATE_ACCOUNT_PASSWORD\n"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["users", "--database-url", "not-used", "--username", "alice", "--password-stdin"],
    )
    with pytest.raises(SystemExit) as error:
        manage_application_users.main()
    assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in str(capsys.readouterr())


def test_offline_scrypt_and_username_policy():
    encoded = hash_password("test-password-1234")
    assert encoded.startswith("scrypt-32768-8-3$")
    assert verify_password("test-password-1234", encoded)
    assert not verify_password("x" * 129, encoded)
    assert not verify_password("password", "broken")
    assert normalize_username("Alice") == "alice"
    with pytest.raises(ValueError):
        hash_password("short")
    with pytest.raises(ValueError):
        normalize_username("user@example.com")
