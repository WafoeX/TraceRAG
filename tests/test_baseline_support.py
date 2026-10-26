"""Offline safety, metric and budget tests; never connect to the real database."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.baseline_support import (
    DEFAULT_DATABASE_URL,
    GUARD_MARKER,
    BaselineError,
    CostLedger,
    corpus_manifest,
    digest,
    existing_schema_url,
    load_cases,
    prepare_schema,
    read_json,
    summarize,
    validate_database_url,
    write_json,
)


@pytest.fixture
def prices():
    return {
        kind: {
            "input_per_million_usd": "1",
            "output_per_million_usd": "2",
            "input_token_upper_bound": 100,
            "output_token_upper_bound": 50 if kind == "rerank" else 0,
        }
        for kind in ("embedding", "rerank")
    }


@pytest.mark.parametrize("change", ["content", "file_set", "symlink"])
def test_frozen_corpus_rejects_drift_and_symlinks(tmp_path, change) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    source = examples / "fixture.txt"
    source.write_text("frozen public fixture")
    manifest_path = tmp_path / "docs/refactor/baseline/source-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    write_json(
        manifest_path, {"files": [{"path": "examples/fixture.txt", "sha256": digest(source)}]}
    )
    assert len(corpus_manifest(tmp_path)) == 1
    if change == "content":
        source.write_text("changed")
    elif change == "file_set":
        (examples / "new.txt").write_text("not part of baseline")
    else:
        target = tmp_path / "outside.txt"
        target.write_text("frozen public fixture")
        source.unlink()
        source.symlink_to(target)
    with pytest.raises(BaselineError):
        corpus_manifest(tmp_path)


def test_only_isolated_database_is_accepted() -> None:
    validate_database_url(DEFAULT_DATABASE_URL)
    validate_database_url(DEFAULT_DATABASE_URL.replace("127.0.0.1", "localhost"))


@pytest.mark.parametrize(
    "url",
    [
        DEFAULT_DATABASE_URL.replace("127.0.0.1", "db.example.com"),
        DEFAULT_DATABASE_URL.replace(":55432/", ":5432/"),
        DEFAULT_DATABASE_URL.replace("/budgetrag_baseline", "/production"),
        DEFAULT_DATABASE_URL.replace("//budgetrag_baseline:", "//rag:"),
        DEFAULT_DATABASE_URL + "?options=-csearch_path=public",
        DEFAULT_DATABASE_URL + "#fragment",
        "sqlite:///data.db",
        "postgresql+psycopg://[invalid",
    ],
)
def test_unsafe_urls_fail_before_creating_engine(url) -> None:
    with patch("sqlalchemy.create_engine") as create:
        with pytest.raises(BaselineError):
            prepare_schema(url, "infra_" + "a" * 32)
        create.assert_not_called()


@pytest.mark.parametrize("schema", ["public", "infra_bad;DROP TABLE chunks", "infra_" + "a" * 31])
def test_schema_names_are_fully_allowlisted(schema) -> None:
    with patch("sqlalchemy.create_engine") as create:
        with pytest.raises(BaselineError):
            prepare_schema(DEFAULT_DATABASE_URL, schema)
        create.assert_not_called()


def test_marker_mismatch_prevents_all_ddl() -> None:
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.scalar_one.return_value = "not-an-isolated-baseline"
    with (
        patch("sqlalchemy.create_engine", return_value=engine),
        pytest.raises(BaselineError, match="guard mismatch"),
    ):
        prepare_schema(DEFAULT_DATABASE_URL, "infra_" + "a" * 32)
    assert connection.execute.call_count == 1
    assert connection.execute.call_args.args[1] == {"flag": True}
    engine.dispose.assert_called_once()


def test_guarded_schema_uses_quoted_identifier_api_and_bound_marker() -> None:
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.scalar_one.return_value = GUARD_MARKER
    schema = "baseline_" + "a" * 32
    with patch("sqlalchemy.create_engine", return_value=engine):
        url = prepare_schema(DEFAULT_DATABASE_URL, schema)
    statements = connection.execute.call_args_list
    assert len(statements) == 3
    assert statements[0].args[1] == {"flag": True}
    assert str(statements[2].args[0]) == f"CREATE SCHEMA {schema}"
    assert parse_qs(urlsplit(url).query)["options"] == [f"-csearch_path={schema},public"]
    engine.dispose.assert_called_once()


def test_reopen_url_is_infrastructure_only() -> None:
    schema = "infra_" + "f" * 32
    url = existing_schema_url(DEFAULT_DATABASE_URL, schema)
    assert parse_qs(urlsplit(url).query) == {"options": [f"-csearch_path={schema},public"]}
    with pytest.raises(BaselineError):
        existing_schema_url(DEFAULT_DATABASE_URL, "public")
    with pytest.raises(BaselineError):
        existing_schema_url(DEFAULT_DATABASE_URL, "baseline_" + "a" * 32)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        [{"id": "a", "question": "q"}],
        [{"id": "a", "question": "q", "answer": " "}],
        [{"id": "a", "question": "q", "answer": "a"}] * 2,
    ],
)
def test_invalid_cases_fail_closed(tmp_path, payload) -> None:
    path = tmp_path / "cases.json"
    write_json(path, payload)
    with pytest.raises(BaselineError):
        load_cases(path)


def test_missing_and_invalid_json_are_controlled_errors(tmp_path) -> None:
    path = tmp_path / "input.json"
    with pytest.raises(BaselineError):
        read_json(path)
    path.write_text("not JSON")
    with pytest.raises(BaselineError):
        read_json(path)


def test_smoke_statistics_do_not_mislabel_recall_or_count_failures_as_misses() -> None:
    rows = [
        {"status": "ok", "answer_rank": rank, "elapsed_ms": elapsed}
        for rank, elapsed in [(1, 10), (2, 20), (0, 30)]
    ] + [{"status": "error", "elapsed_ms": 10000}]
    result = summarize(rows, 5)
    assert result["phrase_hit_rate_at_k"] == pytest.approx(2 / 3)
    assert result["phrase_mrr_at_k"] == 0.5
    assert result["latency_ms_p50"] == 20
    assert result["latency_ms_p95"] == 29
    assert result["failed_queries"] == 1
    assert "recall_at_k" not in result
    assert summarize([], 5)["phrase_mrr_at_k"] is None
    assert summarize([], 5)["latency_ms_p50"] is None


def test_budget_settlement_records_actual_usage_and_decimal_estimates(prices) -> None:
    ledger = CostLedger(Decimal("1"), prices)
    assert ledger.reserve("rerank") == Decimal("0.0002")
    ledger.settle("rerank", {"prompt_tokens": 10, "completion_tokens": 5})
    assert ledger.spent == Decimal("0.00002")
    assert ledger.reserved is None
    assert ledger.costs[0]["usage"]["prompt_tokens"] == 10


def test_budget_reservation_blocks_before_an_api_call(prices) -> None:
    ledger = CostLedger(Decimal("0.00001"), prices)
    with pytest.raises(BaselineError, match="remaining budget"):
        ledger.reserve("rerank")
    assert ledger.stop_reason == "budget_reservation_exceeded"
    assert ledger.reserved is None
    assert ledger.costs == []
    assert ledger.blocked is False  # stopped before sending, not unknown billing


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": True}, {"total_tokens": -1}])
def test_unknown_or_invalid_usage_prevents_all_further_calls(prices, usage) -> None:
    ledger = CostLedger(Decimal("1"), prices)
    ledger.reserve("embedding")
    with pytest.raises(BaselineError):
        ledger.settle("embedding", usage)
    assert ledger.blocked
    with pytest.raises(BaselineError):
        ledger.reserve("rerank")


def test_actual_usage_exceeding_operator_bounds_stops_run(prices) -> None:
    ledger = CostLedger(Decimal("1"), prices)
    ledger.reserve("embedding")
    with pytest.raises(BaselineError, match="exceeded declared model bounds"):
        ledger.settle("embedding", {"total_tokens": 101})
    assert ledger.blocked


@pytest.mark.parametrize("limit", ["0", "-1", "1.01", "NaN", "Infinity"])
def test_explicit_budget_is_positive_finite_and_at_most_one_dollar(prices, limit) -> None:
    with pytest.raises(BaselineError):
        CostLedger(Decimal(limit), prices)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_per_million_usd", "NaN"),
        ("output_per_million_usd", "-1"),
        ("input_token_upper_bound", 0),
        ("output_token_upper_bound", True),
    ],
)
def test_invalid_price_profiles_are_rejected(prices, field, value) -> None:
    prices["embedding"][field] = value
    with pytest.raises(BaselineError):
        CostLedger(Decimal("1"), prices)


def test_database_connection_errors_do_not_disclose_error_body() -> None:
    engine = MagicMock()
    engine.begin.side_effect = RuntimeError("SECRET_DATABASE_DETAILS")
    with (
        patch("sqlalchemy.create_engine", return_value=engine),
        pytest.raises(BaselineError) as error,
    ):
        prepare_schema(DEFAULT_DATABASE_URL, "infra_" + "a" * 32)
    assert "SECRET_DATABASE_DETAILS" not in str(error.value)
    engine.dispose.assert_called_once()


@pytest.mark.parametrize("payload", [None, {}, {"status": "failed"}])
def test_invalid_restart_report_never_creates_schema_or_output(tmp_path, payload) -> None:
    from scripts import verify_baseline_database

    source = tmp_path / "report.json"
    write_json(source, payload)
    output = tmp_path / "must-not-exist"
    with (
        patch.object(verify_baseline_database, "prepare_schema") as prepare,
        pytest.raises(BaselineError, match="Invalid successful infrastructure report"),
    ):
        verify_baseline_database.verify(DEFAULT_DATABASE_URL, output, source)
    prepare.assert_not_called()
    assert not output.exists()
