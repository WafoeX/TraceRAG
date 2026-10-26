"""Pure schemas, privacy allowlists and offline regression checks, no DB/models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.feedback_cases import (
    CandidateOutput,
    ExportInput,
    FeedbackInput,
    RegressionFile,
    ReviewedCase,
    metadata_export,
    regression_checks,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tracerag-feedback-sanitized.json"


def reviewed_case():
    return RegressionFile.model_validate_json(FIXTURE.read_text()).case


@pytest.mark.parametrize(
    "payload",
    [
        {"categories": []},
        {"categories": ["other"]},
        {"categories": ["unhelpful", "unhelpful"]},
        {"categories": ["unhelpful"], "note": "SECRET"},
        {"categories": ["unhelpful"], "user_id": 1},
        {"categories": ["unhelpful"], "citation_number": True},
        {"categories": ["unhelpful"], "citation_number": "1"},
        {"categories": ["unhelpful"], "citation_number": 0},
        {"categories": ["unhelpful"], "citation_number": 21},
    ],
)
def test_feedback_rejects_free_text_and_spoofed_or_invalid_fields(payload):
    with pytest.raises(ValueError):
        FeedbackInput.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "regression"},
        {"mode": "raw"},
        {"trace_id": "SECRET"},
        {"mode": "metadata", "reviewed_case": json.loads(FIXTURE.read_text())["case"]},
    ],
)
def test_export_never_falls_back_to_raw_snapshots(payload):
    with pytest.raises(ValueError):
        ExportInput.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewed", False),
        ("reviewed", "true"),
        ("question", " "),
        ("expected_semantics", ""),
        ("contexts", [" "]),
        ("contexts", ["x"] * 21),
        ("contexts", ["x" * 32769]),
        ("contexts", ["字" * 32768] * 20),
    ],
)
def test_review_is_explicit_and_bounded(field, value):
    raw = reviewed_case().model_dump()
    raw[field] = value
    with pytest.raises(ValueError):
        ReviewedCase.model_validate(raw)


def test_metadata_export_rebuilds_allowlist_without_content_identity_or_costs():
    raw = {
        "trace_id": "SECRET",
        "question": "SECRET",
        "contexts": [{"content": "SECRET", "document_id": 99}],
        "calls": [{"reservation_id": "SECRET", "model": "SECRET", "estimated_cny": "SECRET"}],
        "config": {
            "use_hybrid": True,
            "use_rerank": False,
            "top_k": 4,
            "retrieval_candidates": 10,
            "min_relevance": 0.12,
            "rerank_prompt_version": "baseline",
            "rerank_passage_mode": "prefix_600",
            "openai_chat_model": "SECRET",
            "endpoint": "SECRET",
            "api_key": "SECRET",
        },
        "generation_status": "completed",
        "http_status": 200,
    }
    result = metadata_export(raw, FeedbackInput(categories=["unhelpful"], citation_number=None))
    assert "SECRET" not in json.dumps(result) and "document_id" not in json.dumps(result)
    assert result["run"]["context_count"] == 1 and result["run"]["config"]["top_k"] == 4
    raw["config"] = {
        "use_hybrid": "SECRET",
        "top_k": True,
        "min_relevance": float("nan"),
        "rerank_prompt_version": ["SECRET"],
    }
    assert (
        metadata_export(raw, FeedbackInput(categories=["outdated"], citation_number=None))["run"][
            "config"
        ]
        == {}
    )


def test_structural_replay_reproduces_failure_and_candidate_fix_without_semantic_claims():
    case = reviewed_case()
    assert regression_checks(case)["passed"]
    bad = CandidateOutput(answer="Invalid citation [99]", generation_status="completed")
    result = regression_checks(case, bad)
    assert not result["passed"] and not result["checks"]["citation_range_valid"]
    assert result["manual_semantic_review_required"]
    assert regression_checks(
        case, CandidateOutput(answer="Correct range [1]", generation_status="completed")
    )["passed"]
    assert not regression_checks(
        case, CandidateOutput(answer="[" + "9" * 1000 + "]", generation_status="completed")
    )["passed"]


def test_cli_is_offline_private_and_does_not_load_settings(tmp_path):
    env = {**os.environ, "OPENAI_API_KEY": "", "DATABASE_URL": "bad://SECRET"}
    result = subprocess.run(
        [sys.executable, "-m", "scripts.check_feedback_regression", "--case", str(FIXTURE)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and json.loads(result.stdout)["passed"]
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"answer": "SECRET [99]", "generation_status": "completed"}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_feedback_regression",
            "--case",
            str(FIXTURE),
            "--candidate",
            str(candidate),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "SECRET" not in result.stdout + result.stderr
    candidate.write_text("SECRET invalid JSON")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.check_feedback_regression", "--case", str(candidate)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "SECRET" not in result.stdout + result.stderr
    code = (
        "import sys; import scripts.check_feedback_regression; "
        "assert 'app.config' not in sys.modules; assert 'openai' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0


def test_migration_plan_is_offline_and_apply_errors_redacted():
    for args, valid in [([], True), (["--apply", "--database-url", "bad://SECRET"], False)]:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.migrate_application_feedback", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == valid
        assert "SECRET" not in result.stdout + result.stderr
        if valid:
            assert "PLAN ONLY" in result.stdout
