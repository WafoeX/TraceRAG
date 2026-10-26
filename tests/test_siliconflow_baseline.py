"""Adapted collector plan/protocol; no DB/network during offline tests."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.baseline_support import BaselineError, read_json
from scripts.capture_baseline import capture_queries
from scripts.capture_siliconflow_baseline import CostRecorder, main, resolved_config


def settings():
    return SimpleNamespace(
        model_provider="siliconflow",
        openai_base_url="https://api.siliconflow.cn/v1",
        openai_embedding_model="BAAI/bge-m3",
        openai_chat_model="deepseek-ai/DeepSeek-V4.1-Flash",
        llm_rerank_model=None,
        embedding_dim=1024,
        siliconflow_llm_max_tokens=1024,
        siliconflow_pricing_file="configs/providers/siliconflow.pricing.json",
    )


def test_config_frozen_and_roles_separate():
    config = resolved_config(settings())
    assert config["top_k"] == 5
    assert config["chunk_size"] == 800
    assert config["generation_calls"] == 0
    assert not config["budget_enforced"]
    wrong = settings()
    wrong.model_provider = "openai"
    with pytest.raises(BaselineError):
        resolved_config(wrong)


def test_plan_only_and_cannot_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.get_settings", settings)
    target = tmp_path / "plan"
    assert main(["--output", str(target)]) == 0
    state = read_json(target / "status.json")
    assert state["expected_queries"] == 32
    assert state["model_calls"] == 0
    assert not state["budget_enforced"]
    assert (target / "pricing.json").exists()
    assert (target / "collector_sources/capture_siliconflow_baseline.py").exists()
    assert main(["--output", str(target)]) == 2
    assert read_json(target / "status.json") == state


def test_rejected_response_fallback_and_valid_parse(tmp_path):
    recorder = CostRecorder(tmp_path)
    recorder.stage("rerank", lambda *a: [])("q", [], 5)
    assert recorder.fallback == "provider_response_rejected_or_request_failed"
    recorder.scores(lambda content, n: [1])("[1]", 1)
    assert recorder.fallback is None
    recorder.scores(lambda content, n: [0])("[0]", 1)
    assert recorder.fallback == "unparseable_or_nonpositive_rerank_scores"


def test_two_modes_32_rows_and_failure_stops(tmp_path):
    @dataclass
    class Chunk:
        content: str = "answer"

    cases = [{"id": str(i), "question": "q", "answer": "answer"} for i in range(16)]
    recorder = CostRecorder(tmp_path)
    result = SimpleNamespace(chunks=[Chunk()], confidence=0.8)
    rows = capture_queries(cases, lambda q, k: result, lambda m: None, recorder, 5)
    assert len(rows) == 32
    assert all(row["answer_rank"] == 1 for row in rows)
    metrics = read_json(tmp_path / "metrics.json")
    assert all(m["complete"] for m in metrics.values())

    def fail(q, k):
        raise RuntimeError("provider failure")

    assert len(capture_queries(cases, fail, lambda m: None, recorder, 5)) == 1
