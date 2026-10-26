"""Smoke test for the health route — no DB or network required."""
from app.main import health


def test_health_returns_ok():
    assert health() == {"status": "ok"}
