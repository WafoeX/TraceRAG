"""Small functional checks for the real baseline runner; models remain offline."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import scripts.evaluate_scifact as evaluation
import scripts.run_scifact_baseline as baseline
from tests.test_evaluate_scifact import prepared  # noqa: F401 (shared fixture)
from tests.test_siliconflow import sf_settings


def test_rerank_head_keeps_tail_and_recall_candidate_set():
    candidates = list(range(200))
    calls = []

    def rerank(q, head, k):
        calls.append((q, head, k))
        return list(reversed(head))

    ranked = baseline.rerank_with_tail("q", candidates, rerank)
    assert calls == [("q", list(range(10)), 10)]
    assert ranked[:10] == list(reversed(range(10)))
    assert ranked[10:] == candidates[10:] and set(ranked) == set(candidates)


def test_empty_head_calls_no_models_via_existing_reranker():
    def rerank(q, head, k):
        assert head == [] and k == 0
        return []

    assert baseline.rerank_with_tail("q", [], rerank) == []


def test_percentiles_and_empty():
    assert baseline.latency_summary([]) == {"p50_ms": None, "p95_ms": None}
    assert baseline.latency_summary([1]) == {"p50_ms": 1, "p95_ms": 1}
    assert baseline.latency_summary([0, 10])["p95_ms"] == 9.5


def test_concurrent_context_is_per_query_not_shared():
    def task(i):
        baseline.CALL_CONTEXT.set({"query_id": str(i)})
        baseline.FALLBACK.set("fallback" if i % 2 else None)
        return baseline.CALL_CONTEXT.get(), baseline.FALLBACK.get()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(task, range(40)))
    assert all(context == {"query_id": str(i)} for i, (context, _) in enumerate(results))
    assert all(reason == ("fallback" if i % 2 else None) for i, (_, reason) in enumerate(results))


def test_plan_no_execution_and_default_protocol(request, tmp_path, monkeypatch):
    source = request.getfixturevalue("prepared")
    chunk_plan = tmp_path / "chunk-plan"
    evaluation.run(source, chunk_plan)
    monkeypatch.setattr("app.config.get_settings", lambda: sf_settings())
    execute = SimpleNamespace(called=False)

    def forbidden(*args):
        execute.called = True
        raise AssertionError("No execution in default plan")

    monkeypatch.setattr(baseline, "execute", forbidden)
    output = tmp_path / "baseline-plan"
    assert (
        baseline.main(
            ["--prepared", str(source), "--chunk-plan", str(chunk_plan), "--output", str(output)]
        )
        == 0
    )
    state = evaluation.read_object(output / "resolved_config.json")
    assert state["queries"] == 2 and state["full_development"]
    assert state["test_queries_used"] == 0 and state["retriever_depth_chunks"] == 200
    assert state["llm_rerank_head_chunks"] == 10
    assert state["rerank_passage_mode"] == "prefix_600" and not state["budget_enforced"]
    assert not execute.called
    with pytest.raises(ValueError):
        baseline.main(
            ["--prepared", str(source), "--chunk-plan", str(chunk_plan), "--output", str(output)]
        )
