"""Coverage option and bounded read-only comparison contracts, offline only."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.services import reranking
from scripts.baseline_support import ROOT, BaselineError, load_cases, read_json, write_json
from scripts.capture_siliconflow_baseline import resolved_config
from scripts.compare_rerank_coverage import (
    ComparisonCalls,
    compare_rows,
    load_source,
    main,
)
from tests.test_siliconflow import chunk, completion, sf_settings


@pytest.mark.parametrize("mode", ["prefix_600", "full_chunk"])
@pytest.mark.parametrize("provider", ["openai", "siliconflow"])
def test_only_passage_coverage_changes(mode, provider, monkeypatch):
    settings = sf_settings(model_provider=provider, rerank_passage_mode=mode)
    monkeypatch.setattr(reranking, "get_settings", lambda: settings)
    create = Mock(return_value=completion("[9]"))
    monkeypatch.setattr(
        reranking,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr("app.services.siliconflow.guarded_call", lambda r, m, f, p: f())
    candidate = chunk(1)
    candidate.content = "中" * 600 + " late answer 🚀"
    assert reranking.rerank("question", [candidate], 1)[0].chunk_id == 1
    body = candidate.content if mode == "full_chunk" else candidate.content[:600]
    expected = (
        f'Question: question\n\nPassages:\n[0] (from "fixture") {body}\n\n'
        "Rate how well each passage answers the question, from 0 (irrelevant) to "
        "10 (directly answers it). Return ONLY a JSON array of 1 "
        "numbers, in the same order as the passages — e.g. [8, 2, 5, 0]."
    )
    assert create.call_args.kwargs["messages"][1]["content"] == expected
    assert create.call_args.kwargs["messages"][0]["content"] == reranking.SYSTEM
    assert create.call_args.kwargs["temperature"] == 0


def test_default_prefix_invalid_mode_and_full_fallback(monkeypatch):
    assert sf_settings().rerank_passage_mode == "prefix_600"
    with pytest.raises(ValidationError):
        sf_settings(rerank_passage_mode="silently-truncate")
    monkeypatch.setattr(
        reranking, "get_settings", lambda: sf_settings(rerank_passage_mode="full_chunk")
    )
    monkeypatch.setattr(reranking, "get_client", Mock(side_effect=RuntimeError("offline")))
    candidates = [chunk(1), chunk(2)]
    assert reranking.rerank("q", candidates, 1) == candidates[:1]
    assert reranking.rerank("q", [], 1) == []


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Local synthetic source metadata; no dependency on ignored real-run artifacts.
    import scripts.compare_rerank_coverage as compare

    monkeypatch.setattr(compare, "ROOT", tmp_path)
    folder = tmp_path / "runs/source"
    folder.mkdir(parents=True)
    cases = load_cases(ROOT / "configs/baseline/smoke_cases.json")
    settings = sf_settings(llm_rerank_model="deepseek-ai/DeepSeek-V4.1-Flash")
    write_json(folder / "resolved_config.json", resolved_config(settings))
    write_json(
        folder / "status.json",
        {
            "status": "complete",
            "completed_queries": 32,
            "old_data_unchanged": True,
            "schema": "baseline_" + "a" * 32,
        },
    )
    write_json(folder / "cases.json", cases)
    write_json(folder / "metrics.json", {})
    price_path = ROOT / "configs/providers/siliconflow.pricing.json"
    (folder / "pricing.json").write_bytes(price_path.read_bytes())
    rows = [
        {
            "case_id": c["id"],
            "mode": "fts_rrf_llm_rerank",
            "answer_rank": 2,
            "status": "ok",
            "elapsed_ms": 10,
            "rerank_fallback": None,
        }
        for c in cases
    ]
    import json

    (folder / "rankings.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (folder / "stages.jsonl").write_text("")
    (folder / "indexed_chunks.jsonl").write_text("")
    return folder, settings


def test_plan_no_requests_and_no_overwrite(source, tmp_path, monkeypatch):
    folder, settings = source
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    # Fake ROOT only contains source; collector snapshots come from actual repo.
    import scripts.compare_rerank_coverage as compare

    original_plan = compare.plan

    def wrapped(output, config):
        # main's source-copy loop uses ROOT/scripts, supply explicit fixture copies.
        scripts = tmp_path / "scripts"
        scripts.mkdir(exist_ok=True)
        for name in (
            "compare_rerank_coverage.py",
            "capture_siliconflow_baseline.py",
            "siliconflow_http_smoke.py",
            "siliconflow_e2e.py",
        ):
            (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
        return original_plan(output, config)

    monkeypatch.setattr(compare, "plan", wrapped)
    output = tmp_path / "plan"
    args = ["--output", str(output), "--source-run", str(folder)]
    assert main(args) == 0
    state = read_json(output / "status.json")
    assert state["model_calls"] == 0 and state["expected_queries"] == 16
    assert state["modes"] == ["fts_rrf_llm_rerank"]
    assert main(args) == 2
    assert read_json(output / "status.json") == state


@pytest.mark.parametrize("change", ["partial", "schema", "cases", "chunk", "rows"])
def test_source_negative_controls(source, change):
    folder, _ = source
    state = read_json(folder / "status.json")
    if change == "partial":
        state["status"] = "partial"
    elif change == "schema":
        state["schema"] = "public; DROP TABLE chunks"
    elif change == "cases":
        cases = read_json(folder / "cases.json")
        cases[0]["answer"] = "changed"
        write_json(folder / "cases.json", cases)
    elif change == "chunk":
        config = read_json(folder / "resolved_config.json")
        config["chunk_size"] = 100
        write_json(folder / "resolved_config.json", config)
    else:
        (folder / "rankings.jsonl").write_text("")
    write_json(folder / "status.json", state)
    with pytest.raises((BaselineError, ValueError)):
        load_source(folder)


def test_authorized_order_inputs_and_exact_maximum():
    cases = [{"question": f"q{i}"} for i in range(16)]
    tracker = Mock()
    calls = ComparisonCalls(tracker, cases)
    with pytest.raises(BaselineError):
        calls("generation", "model", Mock(), {})
    with pytest.raises(BaselineError):
        calls("embedding", "model", Mock(), {"input": ["unexpected"]})
    for case in cases:
        calls("embedding", "model", Mock(), {"input": [case["question"]]})
        calls(
            "llm_rerank",
            "model",
            Mock(),
            {"messages": [{}, {"content": "Question: " + case["question"] + "\n\nPassages:\n"}]},
        )
    with pytest.raises(BaselineError):
        calls("embedding", "model", Mock(), {"input": ["q0"]})
    assert calls.attempts == tracker.call.call_count == 32


def test_paired_results_partial_and_case_mismatch():
    old = [
        {
            "case_id": "a",
            "answer_rank": 2,
            "status": "ok",
            "elapsed_ms": 10,
            "rerank_fallback": "zero",
        }
    ]
    new = [{**old[0], "answer_rank": 1, "rerank_fallback": None}]
    result = compare_rows(old, new, 5)
    assert result["before"]["phrase_mrr_at_k"] == 0.5
    assert result["after"]["phrase_mrr_at_k"] == 1
    assert result["before_fallbacks"] == 1 and result["after_fallbacks"] == 0
    assert compare_rows(old, [], 5)["after"]["successful_queries"] == 0
    with pytest.raises(BaselineError):
        compare_rows(old, [{**new[0], "case_id": "b"}], 5)
