"""Synthetic A-only execution contracts; all model calls mocked and sockets disabled."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

import scripts.run_scifact_generation_pilot_a as run
from scripts import evaluate_scifact_answers as ev
from scripts import prepare_scifact_generation_pilot as prep
from scripts.prepare_scifact import hash_file, write_json, write_rows


@pytest.fixture
def plan(tmp_path, monkeypatch):
    folder = tmp_path / "plan"
    folder.mkdir()
    suffix = folder / "candidate_prompt.txt"
    suffix.write_text("Synthetic contract")
    monkeypatch.setattr(prep, "PROMPT", suffix)
    monkeypatch.setattr(ev, "verify_pack", lambda _: None)
    context = [
        {
            "document_id": "d1",
            "chunk_id": "c1",
            "chunk_index": 0,
            "title": "Fixture",
            "text": "Synthetic evidence.",
        }
    ]
    cases = [
        {"query_id": q, "claim": "Synthetic claim " + q, "contexts": {"B": context}}
        for q in run.IDS
    ]
    casebook = [
        {
            "query_id": c["query_id"],
            "claim": c["claim"],
            "historical_answers": {"B": {"contexts": context}},
        }
        for c in cases
    ]
    future = [
        {"query_id": "B_NEVER_EXECUTE", "claim": "Must not execute", "contexts": {"B": context}}
    ]
    previews = prep.make_previews(cases, future, "Synthetic contract")
    write_rows(folder / "cases.jsonl", casebook)
    write_rows(folder / "requests.plan.jsonl", previews)
    (folder / "protocol.md").write_text("Synthetic test protocol")
    write_json(
        folder / "manifest.json",
        {
            "generation_config": ev.CONFIG,
            "stages": {"A": {"query_ids": run.IDS, "calls": 24}},
            "only_experimental_factor": "system_prompt_suffix",
            "source_hashes": {},
            "output_hashes": {
                name: hash_file(folder / name) for name in ["cases.jsonl", "requests.plan.jsonl"]
            },
        },
    )
    return folder


def mock_sdk(monkeypatch, fail_first=False):
    calls, options = [], []

    def create(**payload):
        calls.append(payload)
        if fail_first and len(calls) == 1:
            raise TimeoutError("do-not-log-test-secret")
        return ChatCompletion.model_validate(
            {
                "id": "fixture",
                "created": 0,
                "object": "chat.completion",
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Supported [1]."},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    def client(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None
        )

    monkeypatch.setattr("openai.OpenAI", client)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(
            deepseek_base_url=ev.CONFIG["base_url"],
            openai_chat_model=ev.CONFIG["model"],
            deepseek_api_key="do-not-log-test-secret",
        ),
    )
    return calls, options


def test_plan_only_never_initializes_client(plan, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No client in default planning")

    monkeypatch.setattr("openai.OpenAI", forbidden)
    run.run(plan, tmp_path / "out")
    assert not (tmp_path / "out/answers.jsonl").exists()
    assert ev.read_object(tmp_path / "out/status.json")["B_calls"] == 0


def test_exactly_24_no_B_no_secret_and_offline_replay(plan, tmp_path, monkeypatch):
    calls, options = mock_sdk(monkeypatch)
    out = tmp_path / "out"
    run.run(plan, out, execute=True)
    assert len(calls) == 24 and options[0]["max_retries"] == 0 and options[0]["timeout"] == 60
    assert "B_NEVER_EXECUTE" not in str(calls)
    assert all("do-not-log-test-secret" not in p.read_text() for p in out.iterdir())
    rows = run.verify_run(plan, out)
    assert len(rows) == 24 and all(r["estimated_cost_cny"] is None for r in rows)
    summary = ev.read_object(out / "generation_summary.json")
    assert summary["B_calls"] == summary["judge_API_calls"] == 0
    assert summary["summaries"]["baseline_generation"]["usage"]["prompt_tokens"] == 120
    with pytest.raises(ValueError, match="fresh"):
        run.run(plan, out, execute=True)
    assert len(calls) == 24


def test_error_is_unknown_cost_retained_and_not_retried(plan, tmp_path, monkeypatch):
    calls, _ = mock_sdk(monkeypatch, fail_first=True)
    out = tmp_path / "out"
    run.run(plan, out, execute=True)
    rows = run.verify_run(plan, out)
    assert len(calls) == 24
    assert rows[0]["status"] == "error" and rows[0]["estimated_cost_cny"] is None
    assert rows[0]["error_type"] == "TimeoutError"
    assert len(ev.read_rows(out / "events.jsonl")) == 48
    assert "do-not-log-test-secret" not in (out / "answers.jsonl").read_text()


@pytest.mark.parametrize("change", ["payload", "order", "missing"])
def test_preflight_refuses_changed_A_before_client(plan, tmp_path, monkeypatch, change):
    rows = ev.read_rows(plan / "requests.plan.jsonl")
    if change == "payload":
        rows[0]["request"]["temperature"] = 1.0
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows.pop(0)
    write_rows(plan / "requests.plan.jsonl", rows)
    m = ev.read_object(plan / "manifest.json")
    m["output_hashes"]["requests.plan.jsonl"] = hash_file(plan / "requests.plan.jsonl")
    write_json(plan / "manifest.json", m)

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must fail before client")

    monkeypatch.setattr("openai.OpenAI", forbidden)
    with pytest.raises(ValueError):
        run.run(plan, tmp_path / "never", execute=True)
    assert not (tmp_path / "never").exists()


def test_event_id_tampering_not_hidden_by_other_matching_id(plan, tmp_path, monkeypatch):
    mock_sdk(monkeypatch)
    out = tmp_path / "out"
    run.run(plan, out, execute=True)
    events = ev.read_rows(out / "events.jsonl")
    events[1]["request_id"] = "wrong-finished-id"
    write_rows(out / "events.jsonl", events)
    m = ev.read_object(out / "run_manifest.json")
    m["files"]["events.jsonl"] = hash_file(out / "events.jsonl")
    write_json(out / "run_manifest.json", m)
    with pytest.raises(ValueError, match="event"):
        run.verify_run(plan, out)


def test_empty_or_truncated_response_preserved(monkeypatch):
    item = {"query_id": "0", "variant": prep.VARIANTS[0], "request": {"model": "fixture"}}
    response = ChatCompletion.model_validate(
        {
            "id": "fixture",
            "created": 0,
            "object": "chat.completion",
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "Partial"},
                }
            ],
        }
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    before = copy.deepcopy(item)
    row = run.collect_one(item, [], client, None)
    assert row["status"] == "invalid" and row["finish_reason"] == "length"
    assert row["usage"] is None and row["estimated_cost_cny"] is None
    assert item == before


@pytest.fixture
def gate_data():
    rows, reviews, cases = [], [], {}
    for i, q in enumerate(run.IDS):
        cases[q] = {"pilot_role": ["target", "control", "ambiguity"][i // 4]}
        for variant in prep.VARIANTS:
            bad = variant == "baseline_generation" and i < 2
            row = {
                "query_id": q,
                "variant": variant,
                "answer": "Supported [1].",
                "status": "ok",
                "finish_reason": "stop",
                "contexts": [{}],
                "citations": [{"offset": 10, "source": 1, "token": "[1]"}],
            }
            rows.append(row)
            reviews.append(
                {
                    "query_id": q,
                    "variant": variant,
                    "answer_sha256": ev.digest(row),
                    "review_kind": "main_assistant_ai",
                    "human_verified": False,
                    "independent_blind_review": False,
                    "acceptability": "unacceptable" if bad else "acceptable",
                    "scope_requirements_pass": not bad,
                    "new_unsupported_key_assertion": False,
                    "unsupported_key_spans": ["Supported"] if bad else [],
                    "context_sufficient": True,
                    "abstained": False,
                    "format_compliant": True,
                    "notes": "SYNTHETIC TEST ONLY; no actual scientific judgment",
                    "condition_checks": [{"condition": "fixture condition", "addressed": not bad}],
                    "fragments": [
                        {"span": "Supported", "supported": not bad, "citation_offsets": [10]}
                    ],
                    "citation_assessments": [
                        {
                            "offset": 10,
                            "source": 1,
                            "token": "[1]",
                            "statement": "Supported",
                            "supported": not bad,
                        }
                    ],
                    "joint_citation_assessments": [
                        {"offset": 10, "token": "[1]", "sources": [1], "supported": not bad}
                    ],
                }
            )
    return rows, reviews, cases


def test_frozen_gate_positive_fixture_not_a_scientific_result(gate_data):
    from scripts.review_scifact_generation_pilot_a import evaluate_gate

    rows, reviews, cases = gate_data
    result = evaluate_gate(rows, reviews, cases)
    assert result["gate_pass"] is True
    assert result["paired_changes"]["improved"] == 2
    assert result["B_executed"] is False and result["human_validated"] is False


def test_truncation_alone_blocks_gate_without_retry(gate_data):
    from scripts.review_scifact_generation_pilot_a import evaluate_gate

    rows, reviews, cases = gate_data
    row, review = rows[1], reviews[1]
    row.update(answer="", citations=[], status="invalid", finish_reason="length")
    review.update(
        answer_sha256=ev.digest(row),
        acceptability="unacceptable",
        scope_requirements_pass=False,
        fragments=[],
        citation_assessments=[],
        joint_citation_assessments=[],
        unsupported_key_spans=[],
    )
    review["condition_checks"][0]["addressed"] = False
    result = evaluate_gate(rows, reviews, cases)
    assert result["gate_pass"] is False
    assert [k for k, passed in result["criteria"].items() if not passed] == [
        "all_24_success_nontruncated"
    ]
    assert result["B_executed"] is False


def test_uncertain_does_not_count_as_target_repair_or_win(gate_data):
    from scripts.review_scifact_generation_pilot_a import evaluate_gate

    rows, reviews, cases = gate_data
    for i in [1, 3]:
        reviews[i]["acceptability"] = "uncertain"
        reviews[i]["scope_requirements_pass"] = None
        reviews[i]["condition_checks"][0]["addressed"] = None
    result = evaluate_gate(rows, reviews, cases)
    assert result["paired_changes"]["improved"] == 0
    assert result["paired_changes"]["uncertain"] == 2
    assert not result["criteria"]["at_least_3_of_4_targets"]


@pytest.mark.parametrize("kind", ["hash", "human", "citation", "condition", "fragment"])
def test_AI_review_binding_and_internal_consistency(gate_data, kind):
    from scripts.review_scifact_generation_pilot_a import evaluate_gate

    rows, reviews, cases = gate_data
    review = reviews[1]
    if kind == "hash":
        review["answer_sha256"] = "wrong"
    elif kind == "human":
        review["human_verified"] = True
    elif kind == "citation":
        review["citation_assessments"] = []
    elif kind == "condition":
        review["condition_checks"][0]["addressed"] = None
    else:
        review["fragments"][0]["span"] = "Not present in actual answer"
    with pytest.raises(ValueError):
        evaluate_gate(rows, reviews, cases)


def test_new_unsupported_key_assertion_blocks_gate(gate_data):
    from scripts.review_scifact_generation_pilot_a import evaluate_gate

    rows, reviews, cases = gate_data
    reviews[1]["new_unsupported_key_assertion"] = True
    reviews[1]["unsupported_key_spans"] = ["Supported"]
    result = evaluate_gate(rows, reviews, cases)
    assert not result["criteria"]["no_new_candidate_unsupported_key_assertion"]
    assert not result["gate_pass"]


def test_actual_control_regression_blocks_gate(gate_data):
    from scripts.review_scifact_generation_pilot_a import evaluate_gate

    rows, reviews, cases = gate_data
    review = reviews[9]  # candidate for first control in this synthetic fixture
    review["acceptability"] = "unacceptable"
    review["scope_requirements_pass"] = False
    review["condition_checks"][0]["addressed"] = False
    result = evaluate_gate(rows, reviews, cases)
    assert not result["criteria"]["all_4_controls_preserved"]
    assert not result["criteria"]["zero_clear_paired_regressions"]
