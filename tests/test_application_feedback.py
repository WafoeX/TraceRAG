"""Disposable PG: real sessions, fixed-label feedback, safe explicit exports."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import event, func, inspect, select

from app import database as database_module
from app.models import BudgetReservation, RequestFeedback, RequestTrace
from app.services.feedback_cases import CandidateOutput, RegressionFile, regression_checks
from scripts.manage_application_traces import purge_expired
from scripts.migrate_application_feedback import upgrade

from .test_application_lifecycle import database as _database_fixture
from .test_application_traces import ask
from .test_application_traces import prepared as _prepared_fixture
from .test_auth_isolation import clients as _clients_fixture

database = _database_fixture
clients = _clients_fixture
prepared = _prepared_fixture
pytestmark = [
    pytest.mark.enable_socket,
    pytest.mark.skipif(os.getenv("TRACERAG_INTEGRATION") != "1", reason="opt-in disposable DB"),
]


def case_payload():
    return json.loads(
        (Path(__file__).parent / "fixtures/tracerag-feedback-sanitized.json").read_text()
    )["case"]


@pytest.fixture
def trace(prepared):
    alice, bob, kb, _, doc = prepared
    identifier, detail = ask(alice, kb, True)
    return alice, bob, identifier, detail, doc, kb


def feedback_url(identifier):
    return f"/traces/{identifier}/feedback"


def export_url(identifier):
    return f"/traces/{identifier}/export"


def save(alice, identifier, categories=None, citation=None):
    result = alice.put(
        feedback_url(identifier),
        json={"categories": categories or ["unhelpful"], "citation_number": citation},
    )
    assert result.status_code == 200, result.text
    return result.json()


@pytest.mark.parametrize("category", ["unhelpful", "unsupported_citation", "outdated"])
def test_labels_idempotent_replace_and_retract(trace, database, category):
    alice, _, identifier, detail, _, _ = trace
    assert alice.get(feedback_url(identifier)).json() == {"feedback": None}
    first = save(alice, identifier, [category], 1)
    second = save(alice, identifier, [category], 1)
    assert first == second and first["expires_at"] == detail["expires_at"]
    changed = save(alice, identifier, ["unhelpful", "outdated"])
    assert changed["feedback"]["categories"] == ["unhelpful", "outdated"]
    with database() as db:
        assert db.scalar(select(func.count()).select_from(RequestFeedback)) == 1
    assert alice.delete(feedback_url(identifier)).status_code == 204
    assert alice.delete(feedback_url(identifier)).status_code == 204
    assert alice.get(feedback_url(identifier)).json() == {"feedback": None}
    assert alice.post(export_url(identifier), json={}).status_code == 409
    assert alice.get("/traces/" + identifier).status_code == 200


@pytest.mark.parametrize(
    "method,suffix,payload",
    [
        ("GET", "feedback", None),
        ("PUT", "feedback", {"categories": ["outdated"]}),
        ("DELETE", "feedback", None),
        ("POST", "export", {}),
    ],
)
def test_owner_and_csrf_boundaries(trace, method, suffix, payload):
    alice, bob, identifier, _, _, _ = trace
    url = f"/traces/{identifier}/{suffix}"
    assert bob.request(method, url, json=payload).status_code == 404
    if method != "GET":
        token = alice.headers.pop("X-CSRF-Token")
        assert alice.request(method, url, json=payload).status_code == 403
        alice.headers["X-CSRF-Token"] = token
        assert (
            alice.request(
                method, url, json=payload, headers={"Origin": "https://other.invalid"}
            ).status_code
            == 403
        )
    assert alice.request(method, f"/traces/invalid/{suffix}", json=payload).status_code == 404
    alice.cookies.clear()
    assert alice.request(method, url, json=payload).status_code == 401


def test_citation_validation_and_fixed_private_errors(trace):
    alice, _, identifier, _, _, _ = trace
    for payload in [
        {"categories": ["unhelpful"], "citation_number": 2},
        {"categories": ["unhelpful"], "citation_number": True},
        {"categories": ["unhelpful"], "note": "SECRET COMMENT"},
        {"categories": ["unhelpful"], "user_id": 999},
    ]:
        result = alice.put(feedback_url(identifier), json=payload)
        assert result.status_code == 422
        assert "SECRET" not in result.text
        assert result.headers["cache-control"] == "no-store"
    save(alice, identifier, ["unsupported_citation"], 1)


def test_metadata_export_has_no_private_identity_content_or_budget(trace):
    alice, _, identifier, _, _, _ = trace
    save(alice, identifier, ["unsupported_citation"], 1)
    result = alice.post(export_url(identifier), json={})
    assert (
        result.status_code == 200
        and result.json()["schema_version"] == "tracerag-feedback-metadata-v1"
    )
    assert result.headers["cache-control"] == "no-store" and result.headers["vary"] == "Cookie"
    assert (
        result.headers["content-disposition"]
        == 'attachment; filename="tracerag-feedback-metadata.json"'
    )
    assert "SECRET" not in result.text and identifier not in result.text
    for key in (
        "trace_id",
        "user_id",
        "document_id",
        "chunk_id",
        "reservation_id",
        "snapshot",
        "model",
        "created_at",
        "estimated_cny",
    ):
        assert f'"{key}"' not in result.text
    assert result.json()["run"]["context_count"] == 1


def test_reviewed_case_export_is_ephemeral_and_replayable(trace, database):
    alice, _, identifier, _, _, _ = trace
    save(alice, identifier, ["unsupported_citation"], 1)
    with database() as db:
        row = db.get(RequestTrace, identifier)
        assert row is not None
        before = (row.metadata_json, row.snapshot_json, row.expires_at)
        budget = db.scalar(select(func.count()).select_from(BudgetReservation))
    result = alice.post(
        export_url(identifier), json={"mode": "regression", "reviewed_case": case_payload()}
    )
    assert result.status_code == 200 and "SECRET" not in result.text
    exported = RegressionFile.model_validate(result.json())
    assert regression_checks(exported.case)["passed"]
    assert not regression_checks(
        exported.case, CandidateOutput(answer="[99]", generation_status="completed")
    )["passed"]
    assert set(result.json()) == {"schema_version", "feedback", "case"}
    assert result.json()["case"] == case_payload()
    with database() as db:
        row = db.get(RequestTrace, identifier)
        assert row is not None and before == (row.metadata_json, row.snapshot_json, row.expires_at)
        assert db.scalar(select(func.count()).select_from(BudgetReservation)) == budget
        feedback = db.get(RequestFeedback, identifier)
        assert feedback is not None
        assert "question" not in RequestFeedback.__table__.columns


@pytest.mark.parametrize("mutation", ["missing", "unreviewed", "extra", "oversized"])
def test_export_review_and_preparse_body_limit(trace, mutation):
    alice, _, identifier, _, _, _ = trace
    save(alice, identifier)
    case = case_payload()
    if mutation == "missing":
        payload = {"mode": "regression"}
    elif mutation == "unreviewed":
        case["reviewed"] = False
        payload = {"mode": "regression", "reviewed_case": case}
    elif mutation == "extra":
        case["trace_id"] = "SECRET"
        payload = {"mode": "regression", "reviewed_case": case}
    else:
        response = alice.post(export_url(identifier), content=b"x" * (1024 * 1024 + 1))
        assert response.status_code == 413
        response = alice.put(feedback_url(identifier), content=b"x" * (1024 * 1024 + 1))
        assert response.status_code == 413
        return
    response = alice.post(export_url(identifier), json=payload)
    assert response.status_code == 422 and "SECRET" not in response.text


def test_trace_expiry_hides_feedback_and_cleanup_cascades(trace, database):
    alice, _, identifier, _, _, _ = trace
    save(alice, identifier)
    with database() as db:
        row = db.get(RequestTrace, identifier)
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    for method, suffix, payload in [
        ("GET", "feedback", None),
        ("PUT", "feedback", {"categories": ["outdated"]}),
        ("DELETE", "feedback", None),
        ("POST", "export", {}),
    ]:
        assert (
            alice.request(method, f"/traces/{identifier}/{suffix}", json=payload).status_code == 404
        )
    assert purge_expired(database.kw["bind"]) == 1
    with database() as db:
        assert db.get(RequestFeedback, identifier) is None


def test_delete_trace_cascades_but_delete_document_retains_only_fixed_labels(trace, database):
    alice, _, identifier, _, doc, kb = trace
    save(alice, identifier, ["outdated"], 1)
    assert alice.delete(f"/documents/{doc}?knowledge_base_id={kb}").status_code == 204
    assert alice.get("/traces/" + identifier).json()["snapshot"] is None
    assert alice.get(feedback_url(identifier)).json()["feedback"]["categories"] == ["outdated"]
    assert alice.post(export_url(identifier), json={}).status_code == 200
    assert alice.delete("/traces/" + identifier).status_code == 204
    with database() as db:
        assert db.get(RequestFeedback, identifier) is None


def test_concurrent_put_and_trace_deletion_cannot_resurrect_feedback(trace, database):
    alice, _, identifier, detail, _, _ = trace

    def put(i):
        return alice.put(
            feedback_url(identifier), json={"categories": [["unhelpful"], ["outdated"]][i % 2]}
        ).status_code

    with ThreadPoolExecutor(6) as pool:
        assert all(status == 200 for status in pool.map(put, range(8)))
    assert alice.get("/traces/" + identifier).json()["expires_at"] == detail["expires_at"]
    with ThreadPoolExecutor(6) as pool:
        pending = [pool.submit(put, i) for i in range(6)]
        deletion = pool.submit(alice.delete, "/traces/" + identifier)
        assert all(f.result() in {200, 404} for f in pending)
        assert deletion.result().status_code == 204
    with database() as db:
        assert db.get(RequestFeedback, identifier) is None


def make_v5(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE application_request_feedback")


def test_005_concurrent_idempotent_checksum_and_no_backfill(database):
    engine = database.kw["bind"]
    make_v5(engine)
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: upgrade(engine), range(2))) == [False, True]
    with engine.begin() as conn:
        assert (
            conn.exec_driver_sql("SELECT count(*) FROM application_request_feedback").scalar_one()
            == 0
        )
        conn.exec_driver_sql(
            "UPDATE application_migrations SET sha256='tampered' "
            "WHERE version='005_application_feedback'"
        )
    with pytest.raises(ValueError, match="checksum"):
        upgrade(engine)


def test_005_native_schema_is_not_adopted(database):
    with pytest.raises(ValueError, match="Native"):
        upgrade(database.kw["bind"])


def test_init_v5_requires_explicit_005(database, monkeypatch):
    engine = database.kw["bind"]
    make_v5(engine)
    monkeypatch.setattr(database_module, "engine", engine)
    with pytest.raises(RuntimeError, match="v5"):
        database_module.init_db()
    assert not inspect(engine).has_table("application_request_feedback")


def test_005_ddl_failure_rolls_back(database):
    engine = database.kw["bind"]
    make_v5(engine)

    def fail(conn, cursor, statement, *args):
        if "CREATE TABLE application_request_feedback" in statement:
            raise RuntimeError("injected")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            upgrade(engine)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    assert not inspect(engine).has_table("application_request_feedback")
    assert not inspect(engine).has_table("application_migrations")
