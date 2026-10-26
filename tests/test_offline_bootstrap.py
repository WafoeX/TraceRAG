"""Prove tests cannot silently call external services or download tokenizers."""

from __future__ import annotations

import hashlib
import io
import socket
from unittest.mock import Mock

import pytest

from app.config import get_settings
from scripts import prepare_test_cache


@pytest.mark.parametrize("kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM])
def test_inet_sockets_are_blocked(kind) -> None:
    with pytest.raises(RuntimeError, match="socket.socket"):
        socket.socket(socket.AF_INET, kind)


def test_settings_do_not_use_developer_credentials() -> None:
    settings = get_settings()
    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url is None
    assert settings.database_url == "postgresql+psycopg://test:test@127.0.0.1:1/budgetrag_test"
    assert settings.run_db_init is False
    assert settings.rate_limit_enabled is False


def test_verified_cache_never_downloads(monkeypatch) -> None:
    download = Mock(side_effect=AssertionError("Verified cache must be offline"))
    monkeypatch.setattr(prepare_test_cache.urllib.request, "urlopen", download)
    assert prepare_test_cache.verify_cache()
    prepare_test_cache.main()
    download.assert_not_called()


def test_cache_hash_mismatch_does_not_replace_existing_file(tmp_path, monkeypatch) -> None:
    target = tmp_path / "tokenizer"
    target.write_bytes(b"existing corrupt cache")
    monkeypatch.setattr(prepare_test_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prepare_test_cache, "CACHE_FILE", target)
    monkeypatch.setattr(
        prepare_test_cache.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_test_cache.main()
    assert target.read_bytes() == b"existing corrupt cache"


def test_cache_preparation_writes_verified_bytes_atomically(tmp_path, monkeypatch) -> None:
    data = b"test tokenizer bytes"
    target = tmp_path / "tokenizer"
    monkeypatch.setattr(prepare_test_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prepare_test_cache, "CACHE_FILE", target)
    monkeypatch.setattr(prepare_test_cache, "TOKENIZER_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(
        prepare_test_cache.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(data)
    )
    prepare_test_cache.main()
    assert target.read_bytes() == data
    assert prepare_test_cache.verify_cache()
    assert not list(tmp_path.glob(".download-*"))
