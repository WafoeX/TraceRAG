"""Explicit disposable PG ledger + real SDK MockTransport; never paid APIs."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from sqlalchemy import inspect

from app import database as database_module
from app.config import get_settings
from app.models import BudgetReservation
from app.services import application_budget, embeddings, generation, reranking
from app.services import budget_ledger as ledger
from app.services.budget_policy import BudgetError, cost, load_profile
from app.services.resilience import ProviderUnavailable, application_calls
from app.services.retrieval import RetrievedChunk
from scripts.manage_application_users import create_user
from scripts.migrate_application_budget import upgrade

from .test_application_lifecycle import TEST_URL
from .test_application_lifecycle import database as _database_fixture
from .test_application_resilience import completion
from .test_auth_isolation import clients as _clients_fixture

PRICE = str(Path(__file__).parent / "fixtures/application-budget-pricing.json")
database = _database_fixture
clients = _clients_fixture
pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in disposable DB"),
]


@pytest.fixture
def config(database, monkeypatch):
    monkeypatch.setattr(database_module, "SessionLocal", database)
    settings = get_settings()
    for key, value in {
        "application_pricing_file": PRICE,
        "openai_base_url": "https://offline.invalid/v1",
        "openai_chat_model": "fixture-chat",
        "llm_rerank_model": "fixture-chat",
        "openai_embedding_model": "fixture-embedding",
        "embedding_dim": 3,
        "provider_retry_backoff_seconds": 0,
    }.items():
        monkeypatch.setattr(settings, key, value)
    return database.kw["bind"]


@pytest.fixture
def bank(config):
    a = create_user(config, "budgetalice", "test-password-1234")
    b = create_user(config, "budgetbob", "test-password-1234")
    for user in (None, a, b):
        ledger.set_limit(config, user, Decimal("100"), "offline fixture allocation")
    return config, a, b


def raw_rows(engine):
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.exec_driver_sql(
                "SELECT * FROM application_budget_reservations ORDER BY created_at,id"
            ).mappings()
        ]


def reserve(engine, user, amount="1"):
    return ledger.reserve(engine, user, "generation", "fixture-chat", {}, Decimal(amount))


@pytest.fixture
def sdk(monkeypatch):
    calls, outcomes = [], []

    def handle(request):
        calls.append(request)
        outcome = outcomes.pop(0)
        if outcome == "timeout":
            raise httpx.ReadTimeout("private provider response", request=request)
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"error": {"message": "private"}}, request=request)
        return httpx.Response(200, json=outcome, request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as http,
        OpenAI(
            api_key="offline-key",
            base_url="https://offline.invalid/v1",
            http_client=http,
            max_retries=5,
        ) as client,
    ):
        for module in (generation, embeddings, reranking):
            monkeypatch.setattr(module, "get_client", lambda: client)
        yield calls, outcomes


def test_parallel_reservations_cannot_overshoot_either_account(bank):
    engine, a, _ = bank
    ledger.set_limit(engine, a, Decimal("3"), "parallel cap")

    def attempt(_):
        try:
            return reserve(engine, a)
        except BudgetError as exc:
            assert str(exc) == "budget_exhausted"
            return None

    with ThreadPoolExecutor(max_workers=8) as workers:
        identifiers = list(workers.map(attempt, range(12)))
    assert sum(i is not None for i in identifiers) == 3
    own = ledger.snapshot(engine, a)["account"]
    assert Decimal(own["held_cny"]) == 3 and Decimal(own["available_cny"]) == 0
    assert len(raw_rows(engine)) == 3


def test_global_and_user_allocation_are_one_atomic_transaction(bank):
    engine, a, b = bank
    ledger.set_limit(engine, None, Decimal("1"), "global cap")

    def attempt(user):
        try:
            return reserve(engine, user)
        except BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sum(i is not None for i in workers.map(attempt, [a, b])) == 1
    assert len(raw_rows(engine)) == 1
    assert sum(Decimal(ledger.snapshot(engine, u)["account"]["held_cny"]) for u in (a, b)) == 1


def test_settlement_release_and_repeated_receipts_are_idempotent(bank):
    engine, a, _ = bank
    identifier = reserve(engine, a)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(
            workers.map(lambda _: ledger.settle(engine, identifier, Decimal("0.2")), range(8))
        )
    assert sum(results) == 1
    assert not ledger.settle(engine, identifier, Decimal("0.2"))
    with pytest.raises(BudgetError, match="budget_settlement_conflict"):
        ledger.settle(engine, identifier, Decimal("0.1"))
    own = ledger.snapshot(engine, a)["account"]
    assert Decimal(own["estimated_spent_cny"]) == Decimal("0.2")
    assert Decimal(own["held_cny"]) == 0
    other = reserve(engine, a)
    ledger.retain_unknown(engine, other, "provider_outcome_unknown")
    assert ledger.resolve(
        engine, other, Decimal(0), "billing audit confirms no charge; worker stopped"
    )
    assert not ledger.resolve(engine, other, Decimal(0), "same audit receipt")
    ledger.retain_unknown(engine, identifier, "late failure")
    assert raw_rows(engine)[0]["status"] == "settled"
    ledger.set_limit(engine, a, Decimal("100"), "absolute limit again, not a top-up")
    ledger.set_limit(engine, a, Decimal("100"), "repeated absolute assignment")
    assert Decimal(ledger.snapshot(engine, a)["account"]["estimated_spent_cny"]) == Decimal("0.2")


def test_unknown_and_crash_holds_do_not_expire_or_mint_credit(bank):
    engine, a, _ = bank
    identifier = reserve(engine, a)
    ledger.retain_unknown(engine, identifier, "provider_outcome_unknown")
    ledger.retain_unknown(engine, identifier, "provider_outcome_unknown")
    with pytest.raises(BudgetError, match="budget_audit_required"):
        ledger.settle(engine, identifier, Decimal("0"))
    reserve(engine, a)  # simulate death after commit and before recording response
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE application_budget_reservations SET created_at=now()-interval '30 days'"
        )
    view = ledger.snapshot(engine, a)
    assert Decimal(view["account"]["held_cny"]) == 2
    assert all(r["estimated_cny"] is None for r in view["reservations"])
    reserve(engine, a)  # remaining allocation is still usable
    with pytest.raises(BudgetError, match="budget_limit_below_committed"):
        ledger.set_limit(engine, a, Decimal("1"), "cannot refund holds by lowering limit")


@pytest.mark.parametrize("role", ["generation", "embedding", "llm_rerank"])
def test_every_model_role_has_precommitted_reservation_and_settlement(bank, sdk, role):
    engine, a, _ = bank
    calls, outcomes = sdk
    chunk = RetrievedChunk(1, 1, "fixture", 0, "private evidence", 0.8)
    if role == "embedding":
        outcomes.append(
            {
                "object": "list",
                "model": "fixture-embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        )
    else:
        outcomes.append(completion("[8]" if role == "llm_rerank" else "answer [1]"))
    # Real SDK requests with synthetic responses; all three roles use the ledger.
    with application_calls(user_id=a):
        if role == "generation":
            assert generation.generate_answer("q", [chunk]) == "answer [1]"
        elif role == "embedding":
            assert embeddings.embed_query("q") == [1.0, 0.0, 0.0]
        else:
            assert reranking.rerank("q", [chunk], 1)[0].score == 8
    assert len(calls) == 1
    row = raw_rows(engine)[0]
    assert row["role"] == role and row["status"] == "settled"
    assert row["estimated_cny"] < row["reserved_cny"]
    assert "private evidence" not in row["profile_json"]


def test_provider_observes_committed_hold_not_a_future_settlement(bank):
    engine, a, _ = bank

    def invoke():
        assert raw_rows(engine)[0]["status"] == "held"
        assert Decimal(ledger.snapshot(engine, a)["account"]["held_cny"]) > 0
        return completion()

    with application_calls(user_id=a):
        application_budget.priced_call("generation", "fixture-chat", {"max_tokens": 2048}, invoke)
    assert raw_rows(engine)[0]["status"] == "settled"


def test_timeout_retry_reserves_twice_and_keeps_first_unknown(bank, sdk):
    engine, a, _ = bank
    calls, outcomes = sdk
    outcomes.extend(["timeout", completion()])
    with application_calls(user_id=a) as state:
        assert generation.generate_answer("q", [])
    rows = raw_rows(engine)
    assert len(calls) == 2 and len(rows) == 2
    assert [r["status"] for r in rows] == ["unknown", "settled"]
    assert rows[0]["estimated_cny"] is None
    assert Decimal(ledger.snapshot(engine, a)["account"]["held_cny"]) == rows[0]["reserved_cny"]
    assert "budget_charge_unknown" in state.warnings


def test_insufficient_remaining_budget_prevents_retry_wire(bank, sdk):
    engine, a, _ = bank
    profile = load_profile(
        PRICE, "https://offline.invalid/v1", "generation", "fixture-chat", {"max_tokens": 2048}
    )
    cap = cost(profile, profile["input_token_upper_bound"], profile["output_token_upper_bound"])
    ledger.set_limit(engine, a, cap, "one attempt only")
    calls, outcomes = sdk
    outcomes.extend(["timeout", completion()])
    with application_calls(user_id=a), pytest.raises(ProviderUnavailable, match="budget_exhausted"):
        generation.generate_answer("q", [])
    assert len(calls) == 1 and len(raw_rows(engine)) == 1


def test_usage_missing_is_unknown_and_bound_violation_blocks_until_audited(bank, sdk):
    engine, a, b = bank
    calls, outcomes = sdk
    missing = completion()
    missing["usage"] = None
    outcomes.append(missing)
    with application_calls(user_id=a) as state:
        assert generation.generate_answer("q", [])
    assert "budget_charge_unknown" in state.warnings
    assert raw_rows(engine)[0]["estimated_cny"] is None
    over = completion()
    over["usage"] = {"prompt_tokens": 10001, "completion_tokens": 1, "total_tokens": 10002}
    outcomes.append(over)
    with (
        application_calls(user_id=a),
        pytest.raises(ProviderUnavailable, match="budget_bound_violation"),
    ):
        generation.generate_answer("q", [])
    with pytest.raises(BudgetError, match="budget_blocked"):
        reserve(engine, b)
    identifier = raw_rows(engine)[1]["id"]
    ledger.retain_unknown(engine, identifier, "late provider outcome")
    assert raw_rows(engine)[1]["reason"] == "budget_bound_violation"
    with pytest.raises(BudgetError, match="budget_audit_required"):
        ledger.unblock(engine, None, "not yet audited")
    ledger.resolve(engine, identifier, Decimal("0.1"), "audited amount; stopped caller")
    ledger.unblock(engine, a, "audit complete")
    ledger.unblock(engine, None, "audit complete")
    reserve(engine, b)


def test_database_or_pricing_failure_never_invokes_provider(bank, monkeypatch):
    from unittest.mock import Mock

    engine, a, _ = bank
    invoke = Mock()
    monkeypatch.setattr(get_settings(), "application_pricing_file", "missing-unverified-file")
    with (
        application_calls(user_id=a),
        pytest.raises(ProviderUnavailable, match="budget_pricing_unverified"),
    ):
        application_budget.priced_call("generation", "fixture-chat", {"max_tokens": 2048}, invoke)
    invoke.assert_not_called()
    monkeypatch.setattr(get_settings(), "application_pricing_file", PRICE)
    monkeypatch.setattr(
        ledger, "reserve", lambda *args: (_ for _ in ()).throw(RuntimeError("private DB URL"))
    )
    with (
        application_calls(user_id=a),
        pytest.raises(ProviderUnavailable, match="budget_unavailable"),
    ):
        application_budget.priced_call("generation", "fixture-chat", {"max_tokens": 2048}, invoke)
    invoke.assert_not_called()
    assert raw_rows(engine) == []


def test_commit_receipt_and_settlement_failure_never_release_hold(bank, monkeypatch):
    engine, a, _ = bank
    original = ledger.reserve

    def receipt_lost(*args):
        original(*args)
        raise RuntimeError("commit receipt lost")

    monkeypatch.setattr(ledger, "reserve", receipt_lost)
    invoked = []
    with (
        application_calls(user_id=a),
        pytest.raises(ProviderUnavailable, match="budget_unavailable"),
    ):
        application_budget.priced_call(
            "generation", "fixture-chat", {"max_tokens": 2048}, lambda: invoked.append(True)
        )
    assert not invoked and len(raw_rows(engine)) == 1
    monkeypatch.setattr(ledger, "reserve", original)
    monkeypatch.setattr(
        ledger, "settle", lambda *args: (_ for _ in ()).throw(RuntimeError("DB down"))
    )
    with (
        application_calls(user_id=a),
        pytest.raises(ProviderUnavailable, match="budget_unavailable"),
    ):
        application_budget.priced_call(
            "generation", "fixture-chat", {"max_tokens": 2048}, completion
        )
    assert len(raw_rows(engine)) == 2
    assert all(row["status"] == "held" for row in raw_rows(engine))


@pytest.mark.parametrize("existing_hold", [True, False])
def test_independent_process_and_crash_preserve_committed_holds(bank, existing_hold):
    engine, a, _ = bank
    ledger.set_limit(engine, None, Decimal("1"), "global cap")
    if existing_hold:
        reserve(engine, a)
    script = """
import os, sys
from decimal import Decimal
from sqlalchemy import create_engine
from app.services.budget_ledger import reserve
from app.services.budget_policy import BudgetError
engine=create_engine(sys.argv[1],
    connect_args={'options':'-csearch_path=tracerag_lifecycle_test,public'})
try:
    reserve(engine,int(sys.argv[2]),'generation','fixture',{},Decimal('1'))
    print('reserved-before-crash', flush=True)
    os._exit(0)
except BudgetError as exc:
    print(str(exc))
finally:
    engine.dispose()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, TEST_URL, str(a)], capture_output=True, text=True, timeout=10
    )
    expected = "budget_exhausted" if existing_hold else "reserved-before-crash"
    assert result.returncode == 0 and result.stdout.strip() == expected
    assert raw_rows(engine)[0]["status"] == "held"
    with pytest.raises(BudgetError, match="budget_exhausted"):
        reserve(engine, a)


def drop_budget(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DROP TABLE application_budget_events, application_budget_reservations, "
            "application_budget_accounts"
        )


def test_migration_is_explicit_atomic_idempotent_and_credits_nothing(bank):
    engine, a, _ = bank
    drop_budget(engine)
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sum(workers.map(lambda _: upgrade(engine), range(2))) == 1
    assert not upgrade(engine)
    assert ledger.snapshot(engine, a)["account"] is None
    assert {
        c["name"] for c in inspect(engine).get_columns("application_budget_reservations")
    } == set(BudgetReservation.__table__.columns.keys())
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE application_migrations SET sha256='changed' "
            "WHERE version='003_application_budget'"
        )
    with pytest.raises(ValueError, match="checksum"):
        upgrade(engine)


def test_native_v4_rejected_and_startup_cannot_implicitly_migrate_v3(bank, monkeypatch):
    engine, _, _ = bank
    with pytest.raises(ValueError, match="Native v4"):
        upgrade(engine)
    drop_budget(engine)
    monkeypatch.setattr(database_module, "engine", engine)
    with pytest.raises(RuntimeError, match="schema v3"):
        database_module.init_db()
    assert not inspect(engine).has_table("application_budget_accounts")


def test_budget_display_is_owner_only_and_has_no_http_mutations(clients, config):
    alice, bob, _, _ = clients
    with config.connect() as conn:
        a = conn.exec_driver_sql("SELECT id FROM app_users WHERE username='alice'").scalar_one()
        b = conn.exec_driver_sql("SELECT id FROM app_users WHERE username='bob'").scalar_one()
    for user in (None, a, b):
        ledger.set_limit(config, user, Decimal("2"), "fixture")
    identifier = reserve(config, a)
    own = alice.get("/budget")
    assert own.status_code == 200 and own.headers["cache-control"] == "no-store"
    assert own.json()["amount_kind"] == "estimate" and own.json()["actual_cost_cny"] is None
    assert own.json()["reservations"][0]["id"] == identifier
    foreign = bob.get(f"/budget?user_id={a}")
    assert foreign.json()["reservations"] == []
    assert "global" not in foreign.json()
    assert bob.post("/budget", json={"limit_cny": "9999"}).status_code == 405
    assert alice.post("/auth/logout").status_code == 204
    assert alice.get("/budget").status_code == 401


def configure_clients(clients, engine, amount="2"):
    with engine.connect() as conn:
        a = conn.exec_driver_sql("SELECT id FROM app_users WHERE username='alice'").scalar_one()
        b = conn.exec_driver_sql("SELECT id FROM app_users WHERE username='bob'").scalar_one()
    for user in (None, a, b):
        ledger.set_limit(engine, user, Decimal(amount), "offline allocation")
    return a, b


def test_http_budget_denial_keeps_evidence_and_charges_only_authenticated_owner(
    clients, config, sdk, monkeypatch
):
    from app.routers import query
    from app.services.search import SearchResult

    alice, bob, a_kb, b_kb = clients
    a, b = configure_clients(clients, config)
    ledger.set_limit(config, a, Decimal(0), "no paid quota")
    chunk = RetrievedChunk(1, 1, "evidence", 0, "full private evidence", 0.8)
    monkeypatch.setattr(query, "search", lambda *args: SearchResult([chunk], 0.8))
    monkeypatch.setattr(query, "evidence_metadata", lambda *args: {})
    calls, outcomes = sdk
    outcomes.append(completion())
    denied = alice.post(
        f"/query/evidence?knowledge_base_id={a_kb}", json={"question": "q", "user_id": b}
    )
    assert denied.status_code == 200 and denied.json()["generation_status"] == "failed"
    assert "generation_budget_exhausted" in denied.json()["warnings"]
    assert denied.json()["citations"][0]["content"] == "full private evidence"
    assert not calls
    assert (
        bob.post(f"/query/evidence?knowledge_base_id={b_kb}", json={"question": "q"}).status_code
        == 200
    )
    assert len(calls) == 1 and raw_rows(config)[0]["user_id"] == b
    assert ledger.snapshot(config, a)["reservations"] == []


def test_worker_charges_task_library_owner_and_retries_only_after_allocation(
    clients, config, sdk, monkeypatch
):
    from app import worker
    from app.services import ingestion

    alice, _, kb, _ = clients
    a, b = configure_clients(clients, config)
    ledger.set_limit(config, a, Decimal(0), "budget denied")
    monkeypatch.setattr(get_settings(), "embedding_dim", 1536)
    job_id = alice.post(
        f"/documents/jobs?knowledge_base_id={kb}",
        json={"title": "paper", "content": "Scientists observed liver cells."},
    ).json()["id"]
    with database_module.SessionLocal() as db:
        item = ingestion.claim(db, require_owned=True)
    assert item is not None and item.id == job_id
    assert not worker.process_owned_job(item)
    calls, outcomes = sdk
    assert not calls
    saved = alice.get(f"/documents/jobs/{job_id}").json()
    assert saved["error_code"] == "budget_exhausted"
    ledger.set_limit(config, a, Decimal("1"), "explicit allocation")
    assert alice.post(f"/documents/jobs/{job_id}/retry").status_code == 202
    with database_module.SessionLocal() as db:
        item = ingestion.claim(db, require_owned=True)
    assert item is not None
    outcomes.append(
        {
            "object": "list",
            "model": "fixture-embedding",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0] + [0.0] * 1535}],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
    )
    assert worker.process_owned_job(item)
    assert len(calls) == 1 and raw_rows(config)[0]["user_id"] == a
    assert ledger.snapshot(config, b)["reservations"] == []


def test_siliconflow_cost_log_and_budget_each_track_every_retry(bank, sdk, monkeypatch, tmp_path):
    import json

    from app.services import siliconflow
    from app.services.provider_costs import CostTracker
    from tests.test_siliconflow import price_config

    engine, a, _ = bank
    prices = price_config()
    prices["roles"]["generation"]["model"] = "fixture-chat"
    path = tmp_path / "sf-fixture.json"
    path.write_text(json.dumps(prices))
    tracker = CostTracker(path)
    monkeypatch.setattr(siliconflow, "get_cost_tracker", lambda: tracker)
    monkeypatch.setattr(get_settings(), "model_provider", "siliconflow")
    calls, outcomes = sdk
    outcomes.extend(["timeout", completion()])
    with application_calls(user_id=a):
        assert generation.generate_answer("q", [])
    assert len(calls) == len(raw_rows(engine)) == len(tracker.records) == 2
    assert tracker.records[0]["estimated_cost_cny"] is None
    assert raw_rows(engine)[0]["status"] == "unknown"


def test_migration_ddl_failure_rolls_back_without_touching_users(bank):
    from sqlalchemy import event

    engine, a, _ = bank
    drop_budget(engine)

    def fail(conn, cursor, statement, parameters, context, executemany):
        if "CREATE TABLE application_budget_accounts" in statement:
            raise RuntimeError("synthetic DDL failure")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError):
            upgrade(engine)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    assert not inspect(engine).has_table("application_budget_accounts")
    with engine.connect() as conn:
        assert (
            conn.exec_driver_sql(
                "SELECT count(*) FROM app_users WHERE id=%(id)s", {"id": a}
            ).scalar_one()
            == 1
        )
    assert upgrade(engine)


def test_admin_cli_absolute_limit_and_confirmed_reconciliation(bank):
    import json

    engine, a, _ = bank
    identifier = reserve(engine, a)
    ledger.settle(engine, identifier, Decimal("0.2"))
    held = reserve(engine, a)
    url = TEST_URL + "?options=-csearch_path%3Dtracerag_lifecycle_test%2Cpublic"

    def run(*args):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.manage_application_budget",
                "--database-url",
                url,
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

    for _ in range(2):
        result = run(
            "set-limit", "--username", "budgetalice", "--cny", "5", "--reason", "offline CLI test"
        )
        assert result.returncode == 0
    view = run("show", "--username", "budgetalice")
    assert view.returncode == 0
    account = json.loads(view.stdout)["account"]
    assert Decimal(account["limit_cny"]) == 5
    assert Decimal(account["estimated_spent_cny"]) == Decimal("0.2")
    assert Decimal(account["held_cny"]) == 1
    rejected = run("resolve", "--reservation", held, "--cny", "0", "--reason", "not confirmed")
    assert rejected.returncode != 0
    accepted = run(
        "resolve",
        "--reservation",
        held,
        "--cny",
        "0",
        "--reason",
        "worker stopped, audited no charge",
        "--confirm-audited",
    )
    assert accepted.returncode == 0
    assert Decimal(ledger.snapshot(engine, a)["account"]["held_cny"]) == 0
