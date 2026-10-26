"""Collector fidelity and fail-closed controls, using only fake model/search calls."""

from __future__ import annotations

import json
from contextlib import suppress
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.retrieval import RetrievedChunk
from app.services.search import SearchResult
from scripts import capture_baseline as capture
from scripts.baseline_support import BaselineError, CostLedger, digest, load_cases, read_json
from scripts.eval import CASES


def prices():
    return {
        kind: {
            "input_per_million_usd": "1",
            "output_per_million_usd": "2",
            "input_token_upper_bound": 100,
            "output_token_upper_bound": 50,
        }
        for kind in ("embedding", "rerank")
    }


def fake_chunk(content="Refund within 30 days.", chunk_id=1):
    return RetrievedChunk(chunk_id, 1, "Fixture", 0, content, 0.8)


def complete_config():
    config = capture.load_config(capture.DEFAULT_CONFIG, False)
    config.update(
        provider="fixture-only",
        base_url="https://models.example.test/v1",
        embedding_model="fixture-dense",
        chat_model="fixture-rerank",
        embedding_dim=3,
        pricing=prices(),
        pricing_and_bounds_source="unit-test fixture; not real pricing",
    )
    return config


def test_frozen_cases_exactly_match_all_16_original_questions_and_phrases() -> None:
    frozen = load_cases(capture.CASES_PATH)
    assert len(frozen) == len(CASES) == 16
    assert [(c["question"], c["answer"]) for c in frozen] == [(c.question, c.answer) for c in CASES]


def test_plan_copies_sources_and_corpus_without_database_models_or_env_secrets(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "plan"
    monkeypatch.setenv("OPENAI_API_KEY", "SENTINEL-DO-NOT-LOG")
    prepare = Mock(side_effect=AssertionError("Plan must not access database"))
    execute = Mock(side_effect=AssertionError("Plan must not execute models"))
    monkeypatch.setattr(capture, "prepare_schema", prepare)
    monkeypatch.setattr(capture, "execute", execute)
    assert capture.main(["--output", str(output)]) == 0
    state = read_json(output / "status.json")
    assert state["status"] == "planned"
    assert state["expected_queries"] == 32
    assert state["model_calls"] == 0
    assert state["metrics"] is None
    prepare.assert_not_called()
    execute.assert_not_called()
    for entry in read_json(output / "corpus_manifest.json"):
        assert digest(output / "corpus_sources" / entry["path"].split("/")[-1]) == entry["sha256"]
    sources = read_json(output / "source_manifest.json")
    assert digest(output / "collector_sources/capture_baseline.py") == sources["collector_sha256"]
    assert len(sources["runtime"]) == 22
    assert all(
        b"SENTINEL-DO-NOT-LOG" not in p.read_bytes() for p in output.rglob("*") if p.is_file()
    )


def test_existing_run_directory_is_never_overwritten(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "status.json"
    marker.write_text('{"status":"original"}')
    assert capture.main(["--output", str(output)]) == 2
    assert marker.read_text() == '{"status":"original"}'


def test_incomplete_template_cannot_execute_paid_calls(tmp_path, monkeypatch) -> None:
    execute = Mock()
    prepare = Mock()
    monkeypatch.setattr(capture, "execute", execute)
    monkeypatch.setattr(capture, "prepare_schema", prepare)
    output = tmp_path / "not-created"
    assert capture.main(["--output", str(output), "--execute-models", "--max-cost-usd", "1"]) == 2
    assert not output.exists()
    execute.assert_not_called()
    prepare.assert_not_called()


@pytest.mark.parametrize("budget", [None, "0", "1.01", "NaN"])
def test_missing_or_invalid_budget_prevents_run_before_any_db_access(
    tmp_path, monkeypatch, budget
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(complete_config()))
    prepare = Mock()
    monkeypatch.setattr(capture, "prepare_schema", prepare)
    args = ["--config", str(path), "--output", str(tmp_path / "run"), "--execute-models"]
    if budget is not None:
        args += ["--max-cost-usd", budget]
    assert capture.main(args) == 2
    prepare.assert_not_called()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_url", "https://user:secret@example.test/v1"),
        ("base_url", "https://example.test/v1?api_key=secret"),
        ("base_url", "http://example.test/v1"),
        ("chunk_overlap", 800),
        ("top_k", 21),
        ("api_key", "do-not-store"),
    ],
)
def test_invalid_config_and_credentials_rejected_even_in_plan(tmp_path, field, value) -> None:
    config = capture.load_config(capture.DEFAULT_CONFIG, False)
    config[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(BaselineError):
        capture.load_config(path, False)


def test_stage_traces_keep_original_candidate_ids_and_scores(tmp_path) -> None:
    recorder = capture.Recorder(tmp_path)
    chunks = [fake_chunk()]
    wrapped = recorder.stage("vector_search", lambda: chunks)
    assert wrapped() is chunks
    trace = json.loads((tmp_path / "stages.jsonl").read_text())
    assert trace["status"] == "ok"
    assert trace["candidates"][0]["chunk_id"] == 1
    assert trace["candidates"][0]["score"] == 0.8


def test_rerank_stage_keeps_both_input_and_output_order(tmp_path) -> None:
    recorder = capture.Recorder(tmp_path)
    chunks = [fake_chunk(chunk_id=1), fake_chunk(chunk_id=2)]
    wrapped = recorder.stage("rerank", lambda q, items, k: list(reversed(items))[:k])
    wrapped("question", chunks, 1)
    trace = json.loads((tmp_path / "stages.jsonl").read_text())
    assert [c["chunk_id"] for c in trace["input_candidates"]] == [1, 2]
    assert [c["chunk_id"] for c in trace["candidates"]] == [2]
    assert trace["fallback_reason"] is None
    assert trace["top_k"] == 1


def test_failed_stage_is_saved_without_error_body(tmp_path) -> None:
    recorder = capture.Recorder(tmp_path)

    def failure():
        raise RuntimeError("SECRET_ERROR_BODY")

    with pytest.raises(RuntimeError):
        recorder.stage("vector_search", failure)()
    trace = (tmp_path / "stages.jsonl").read_text()
    assert "SECRET_ERROR_BODY" not in trace
    assert json.loads(trace)["status"] == "error"


def test_model_usage_and_raw_response_are_recorded(tmp_path) -> None:
    ledger = CostLedger(Decimal("1"), prices())
    recorder = capture.Recorder(tmp_path, ledger)
    usage = {"total_tokens": 10}
    response = SimpleNamespace(
        usage=Mock(model_dump=lambda: usage),
        model_dump=lambda **kwargs: {"data": ["fake"], "usage": usage},
    )
    api = Mock(return_value=response)
    assert recorder.call("embedding", api, model="fixture", input=["test"]) is response
    api.assert_called_once_with(model="fixture", input=["test"])
    trace = json.loads((tmp_path / "model_calls.jsonl").read_text())
    assert trace["usage"] == usage
    assert trace["response"]["data"] == ["fake"]
    assert trace["attempt"] == 1
    assert trace["estimated_cost_usd"] == "0.00001"


def test_provider_failure_preserves_failure_and_prohibits_retry(tmp_path) -> None:
    ledger = CostLedger(Decimal("1"), prices())
    recorder = capture.Recorder(tmp_path, ledger)
    api = Mock(side_effect=RuntimeError("SECRET_PROVIDER_BODY"))
    with pytest.raises(BaselineError):
        recorder.call("embedding", api, model="fixture")
    with pytest.raises(BaselineError):
        recorder.call("embedding", api, model="fixture")
    api.assert_called_once()
    trace = (tmp_path / "model_calls.jsonl").read_text()
    assert "SECRET_PROVIDER_BODY" not in trace
    assert json.loads(trace)["status"] == "error"
    assert ledger.blocked


def test_two_modes_use_injected_original_search_and_preserve_rankings(tmp_path) -> None:
    recorder = capture.Recorder(tmp_path)
    cases = [{"id": "fixture", "question": "refund?", "answer": "30 days"}]
    retrieve = Mock(return_value=SearchResult([fake_chunk()], 0.8))
    configure = Mock()
    rows = capture.capture_queries(cases, retrieve, configure, recorder, 5)
    assert [row["mode"] for row in rows] == list(capture.MODES)
    assert all(row["answer_rank"] == 1 for row in rows)
    assert all(row["model_calls"] == 0 for row in rows)  # fake search, not a benchmark
    assert configure.call_count == retrieve.call_count == 2
    metrics = read_json(tmp_path / "metrics.json")
    assert all(m["complete"] for m in metrics.values())


def test_query_failure_stops_run_and_partial_metrics_are_explicit(tmp_path) -> None:
    recorder = capture.Recorder(tmp_path)
    cases = [{"id": str(i), "question": "q", "answer": "a"} for i in range(2)]
    retrieve = Mock(side_effect=RuntimeError("SECRET_QUERY_ERROR"))
    rows = capture.capture_queries(cases, retrieve, lambda mode: None, recorder, 5)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    metrics = read_json(tmp_path / "metrics.json")
    assert metrics["dense_only"]["complete"] is False
    assert metrics["dense_only"]["failed_queries"] == 1
    assert metrics["dense_only"]["phrase_mrr_at_k"] is None
    assert metrics["fts_rrf_llm_rerank"]["queries"] == 0
    assert "SECRET_QUERY_ERROR" not in (tmp_path / "rankings.jsonl").read_text()


def test_budget_exception_swallowed_by_old_reranker_cannot_fake_success(tmp_path) -> None:
    ledger = CostLedger(Decimal("0.00001"), prices())
    recorder = capture.Recorder(tmp_path, ledger)
    api = Mock()

    def old_fallback_search(question, k):
        # Simulate the original reranker's catch-all fallback to original order.
        with suppress(BaselineError):
            recorder.call("rerank", api, model="fixture")
        return SearchResult([fake_chunk()], 0.8)

    rows = capture.capture_queries(
        [{"id": "fixture", "question": "q", "answer": "30 days"}],
        old_fallback_search,
        lambda mode: None,
        recorder,
        5,
    )
    api.assert_not_called()
    assert rows[0]["status"] == "error"
    assert ledger.stop_reason == "budget_reservation_exceeded"
    assert not ledger.blocked  # no request sent, no unpriced billing


def test_rerank_parser_fallback_is_recorded(tmp_path) -> None:
    recorder = capture.Recorder(tmp_path)
    assert recorder.scores(lambda content, n: None)("not JSON", 2) is None
    assert recorder.fallback == "unparseable_or_nonpositive_rerank_scores"
