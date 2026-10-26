"""Offline plan/scoring integration with tiny self-contained prepared fixtures."""

import json
from pathlib import Path

import pytest

import scripts.evaluate_scifact as evaluation
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.retrieval_metrics import EvaluationError


def refresh_manifest(folder):
    report = json.loads((folder / "report.json").read_text())
    report["files"] = {
        p.relative_to(folder).as_posix(): {"sha256": hash_file(p), "bytes": p.stat().st_size}
        for p in folder.rglob("*")
        if p.is_file() and p.name != "report.json"
    }
    write_json(folder / "report.json", report)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    folder = tmp_path / "prepared"
    folder.mkdir()
    write_json(folder / "dataset_config.json", {"dataset": "beir-scifact", "fixture": True})
    write_rows(
        folder / "corpus.jsonl",
        [
            {"document_id": "10", "title": "First", "text": "first abstract"},
            {"document_id": "20", "title": "Second", "text": "second abstract"},
        ],
    )
    for split, queries in (("development", ["q1", "q2"]), ("test", ["qt"])):
        path = folder / "splits" / split
        path.mkdir(parents=True)
        write_rows(path / "queries.jsonl", [{"query_id": q, "text": q} for q in queries])
        write_rows(
            path / "qrels.jsonl",
            [
                {"query_id": q, "document_id": "10" if i == 0 else "20", "relevance": 1}
                for i, q in enumerate(queries)
            ],
        )
    write_json(
        folder / "report.json",
        {
            "status": "prepared",
            "dataset": "beir-scifact",
            "counts": {"corpus": 2, "test_queries": 1},
            "development_queries": 2,
            "files": {},
        },
    )
    refresh_manifest(folder)
    # No tokenizer/models/DB needed for unit-level integration.
    monkeypatch.setattr(
        evaluation, "load_chunker", lambda: lambda text, size, overlap: [text[:5], text[5:]]
    )
    return folder


def test_plan_stable_ids_multiple_chunks_and_no_overwrite(prepared, tmp_path):
    output = tmp_path / "plan"
    state = evaluation.run(prepared, output)
    assert state["status"] == "planned" and state["metrics"] is None
    assert state["chunks"] == 4 and state["documents"] == state["queries"] == 2
    assert state["model_calls_in_this_step"] == state["database_operations_in_this_step"] == 0
    chunks = evaluation.read_rows(output / "chunks.jsonl")
    assert [c["document_id"] for c in chunks] == ["10", "10", "20", "20"]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 0, 1]
    second = tmp_path / "plan2"
    evaluation.run(prepared, second)
    assert (output / "chunks.jsonl").read_bytes() == (second / "chunks.jsonl").read_bytes()
    for name, info in state["manifest"].items():
        assert hash_file(output / name) == info["sha256"]
    with pytest.raises(EvaluationError, match="overwrite"):
        evaluation.run(prepared, output)


def predictions_for_plan(folder, plan, *, missing=False):
    chunks = evaluation.read_rows(plan / "chunks.jsonl")
    path = folder / "predictions.jsonl"
    # Two chunks for first document; dedup must not consume document rank 2.
    hits = [{"chunk_id": c["chunk_id"], "score": 1} for c in chunks]
    write_rows(
        path,
        [{"query_id": "q1", "chunks": hits}]
        + ([] if missing else [{"query_id": "q2", "chunks": []}]),
    )
    return path


def test_offline_score_empty_queries_included_and_fixture_not_quality(prepared, tmp_path):
    plan = tmp_path / "plan"
    evaluation.run(prepared, plan)
    predictions = predictions_for_plan(tmp_path, plan)
    output = tmp_path / "score"
    state = evaluation.run(prepared, output, predictions=predictions, fixture=True)
    assert state["prediction_kind"] == "fixture" and state["quality_claim"] == "none"
    assert all(v == 0.5 for v in state["metrics"].values())
    scores = json.loads((output / "scores.json").read_text())
    assert scores["document_rankings"]["q1"] == ["10", "20"]
    assert scores["empty_rankings"] == 1 and scores["queries"] == 2
    assert state["upstream_prediction_generation_cost_cny"] is None
    assert (output / "predictions.jsonl").read_bytes() == predictions.read_bytes()


@pytest.mark.parametrize(
    "bad",
    ["missing", "duplicate", "unknown_query", "unknown_chunk", "forged_document", "invalid_json"],
)
def test_prediction_negative_controls(prepared, tmp_path, bad):
    plan = tmp_path / "plan"
    evaluation.run(prepared, plan)
    path = predictions_for_plan(tmp_path, plan, missing=bad == "missing")
    rows = evaluation.read_rows(path)
    if bad == "duplicate":
        rows.append(rows[0])
    elif bad == "unknown_query":
        rows[0]["query_id"] = "other"
    elif bad == "unknown_chunk":
        rows[0]["chunks"][0]["chunk_id"] = "database-integer-id-is-not-a-dataset-id"
    elif bad == "forged_document":
        rows[0]["chunks"][0]["document_id"] = "wrong"
    write_rows(path, rows)
    if bad == "invalid_json":
        path.write_text("not JSON")
    output = tmp_path / "rejected"
    with pytest.raises(EvaluationError):
        evaluation.run(prepared, output, predictions=path)
    assert not output.exists()


def test_test_scoring_explicit_ack_required_before_loading(prepared, tmp_path):
    with pytest.raises(EvaluationError, match="acknowledgement"):
        evaluation.run(prepared, tmp_path / "denied", split="test", predictions=Path("missing"))
    plan = tmp_path / "test-plan"
    evaluation.run(prepared, plan, split="test")
    chunks = evaluation.read_rows(plan / "chunks.jsonl")
    predictions = tmp_path / "test-predictions.jsonl"
    write_rows(
        predictions,
        [{"query_id": "qt", "chunks": [{"chunk_id": chunks[0]["chunk_id"], "score": 1}]}],
    )
    state = evaluation.run(
        prepared,
        tmp_path / "explicit",
        split="test",
        predictions=predictions,
        acknowledge_test_scoring=True,
        fixture=True,
    )
    assert state["metrics"]["MRR@10"] == 1 and state["quality_claim"] == "none"


@pytest.mark.parametrize("bad", ["hash", "escape", "dangling", "zero", "count", "duplicate_doc"])
def test_source_manifest_and_semantic_negative_controls(prepared, bad):
    if bad == "hash":
        (prepared / "corpus.jsonl").write_text("changed")
    elif bad == "escape":
        report = json.loads((prepared / "report.json").read_text())
        report["files"]["../outside"] = {"sha256": "x", "bytes": 0}
        write_json(prepared / "report.json", report)
    elif bad == "count":
        report = json.loads((prepared / "report.json").read_text())
        report["development_queries"] = 100
        write_json(prepared / "report.json", report)
    elif bad == "duplicate_doc":
        path = prepared / "corpus.jsonl"
        path.write_text(path.read_text() * 2)
        refresh_manifest(prepared)
    else:
        path = prepared / "splits/development/qrels.jsonl"
        rows = evaluation.read_rows(path)
        rows[0]["document_id"] = "missing" if bad == "dangling" else "10"
        rows[0]["relevance"] = 0 if bad == "zero" else 1
        write_rows(path, rows)
        refresh_manifest(prepared)
    with pytest.raises(EvaluationError):
        evaluation.load_prepared(prepared, "development")


def test_chunk_namespace_and_protocol_negative_controls(prepared):
    data = evaluation.load_prepared(prepared, "development")
    protocol = evaluation.read_object(evaluation.PROTOCOL)

    def chunker(text, size, overlap):
        return [text]

    first = evaluation.build_chunks(data["corpus"], protocol, "corpus-a", chunker)
    second = evaluation.build_chunks(data["corpus"], protocol, "corpus-b", chunker)
    assert first[0]["chunk_id"] != second[0]["chunk_id"]
    with pytest.raises(EvaluationError):
        evaluation.build_chunks(data["corpus"], {**protocol, "chunk_size": 1}, "x", chunker)
    with pytest.raises(EvaluationError):
        evaluation.build_chunks(data["corpus"], protocol, "x", lambda *a: [])


def test_cli_default_plan_and_rejected_args(prepared, tmp_path):
    assert evaluation.main(["--prepared", str(prepared), "--output", str(tmp_path / "plan")]) == 0
    assert (
        evaluation.main(
            ["--prepared", str(prepared), "--output", str(tmp_path / "bad"), "--fixture"]
        )
        == 1
    )
