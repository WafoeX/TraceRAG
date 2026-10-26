"""Synthetic annotations below are TEST FIXTURES, never human gold for SciFact."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from scripts import evaluate_scifact_answers as ev
from scripts.prepare_scifact import hash_file, write_json, write_rows


def approved(name="fixture-reviewer"):
    return {
        "status": "approved",
        "reviewer": name,
        "reviewed_at": "2026-10-20",
        "notes": "SYNTHETIC UNIT TEST ONLY, not a real scientific review",
    }


@pytest.fixture
def pack(tmp_path):
    folder = tmp_path / "pack"
    folder.mkdir()
    cases = []
    for i in range(100):
        contexts = [
            {
                "chunk_id": f"c{j}",
                "document_id": str(j),
                "chunk_index": 0,
                "title": "Test paper",
                "text": "A treatment improved survival.",
            }
            for j in range(4)
        ]
        cases.append(
            {
                "query_id": str(i),
                "claim": "A treatment improved survival.",
                "contexts": {"A": contexts, "B": contexts},
                "reference_documents": [],
            }
        )
    gold = [
        {
            "query_id": c["query_id"],
            "case_sha256": ev.digest(c),
            "label": "SUPPORT",
            "required_points": [
                {
                    "id": "p1",
                    "text": "Treatment improved survival.",
                    "basis": "Synthetic test evidence",
                }
            ],
            "context_sufficient": {"A": True, "B": True},
            "review": approved(),
        }
        for c in cases
    ]
    write_rows(folder / "cases.jsonl", cases)
    write_rows(folder / "gold.draft.jsonl", gold)
    manifest = {
        "schema_version": 1,
        "split": "development",
        "generation": ev.CONFIG,
        "arms": ev.ARMS,
        "query_ids": [c["query_id"] for c in cases],
        "source_hashes": {},
        "cases_sha256": hash_file(folder / "cases.jsonl"),
        "messages_sha256": ev.digest([ev.context_messages(c, a) for c in cases for a in ev.ARMS]),
    }
    write_json(folder / "manifest.json", manifest)
    return folder, cases, gold


def test_context_uses_full_chunk_and_no_gold(pack):
    _, cases, _ = pack
    messages = ev.context_messages(cases[0], "A")
    assert "SUPPORTED GOLD" not in json.dumps(messages)
    content = messages[1].get("content")
    assert isinstance(content, str)
    assert '[1] (from "Test paper")' in content
    assert content.endswith("Claim: A treatment improved survival.")
    from app.services.generation import SYSTEM_PROMPT

    assert messages[0].get("content") == SYSTEM_PROMPT


@pytest.mark.parametrize(
    "change",
    [
        "pending",
        "no_reviewer",
        "missing_context",
        "wrong_case",
        "invalid_label",
        "no_points",
        "duplicate_points",
        "placeholder",
    ],
)
def test_gold_refuses_unverified_or_invalid(pack, change):
    _, cases, gold = pack
    g = copy.deepcopy(gold)
    if change == "pending":
        g[0]["review"] = ev.pending()
    elif change == "no_reviewer":
        g[0]["review"]["reviewer"] = ""
    elif change == "missing_context":
        g[0]["context_sufficient"]["A"] = None
    elif change == "wrong_case":
        g[0]["case_sha256"] = "other"
    elif change == "invalid_label":
        g[0]["label"] = "relevant"
    elif change == "no_points":
        g[0]["required_points"] = []
    elif change == "duplicate_points":
        g[0]["required_points"] *= 2
    else:
        g[0]["required_points"][0]["text"] = "HUMAN: replace this"
    with pytest.raises(ValueError):
        ev.validate_gold(cases, g)


def test_unsealed_generation_no_output_no_model(pack, tmp_path, monkeypatch):
    folder, _, _ = pack

    def forbidden(*args, **kwargs):
        pytest.fail("Must not create a model client")

    monkeypatch.setattr("openai.OpenAI", forbidden)
    with pytest.raises(ValueError):
        ev.generate(folder, tmp_path / "never-created", execute=True)
    assert not (tmp_path / "never-created").exists()


def test_seal_and_default_plan_are_offline(pack, tmp_path, monkeypatch):
    folder, cases, _ = pack
    ev.seal(folder, folder / "gold.draft.jsonl")
    assert len(ev.sealed_gold(folder, cases)) == 100

    def forbidden(*args, **kwargs):
        pytest.fail("Plan must not create a model client")

    monkeypatch.setattr("openai.OpenAI", forbidden)
    p = ev.generate(folder, tmp_path / "plan")
    assert len(p["order"]) == 200
    assert [x["arm"] for x in p["order"][:4]] == ["A", "B", "B", "A"]
    assert not (tmp_path / "plan/answers.jsonl").exists()
    with pytest.raises(ValueError, match="fresh output"):
        ev.seal(folder, folder / "gold.draft.jsonl")
    gold_file = folder / "sealed/gold.jsonl"
    gold_file.write_text(gold_file.read_text() + "\n")
    with pytest.raises(ValueError, match="Sealed"):
        ev.generate(folder, tmp_path / "changed-gold")


def test_frozen_context_change_rejected(pack):
    folder, cases, _ = pack
    cases[0]["claim"] = "Changed"
    write_rows(folder / "cases.jsonl", cases)
    with pytest.raises(ValueError, match="Frozen cases"):
        ev.verify_pack(folder)


def test_cost_unknown_and_cache_split():
    pricing = {
        "input_cache_hit_per_million": 0.2,
        "input_cache_miss_per_million": 2,
        "output_per_million": 8,
    }
    usage = {
        "prompt_tokens": 100,
        "prompt_cache_hit_tokens": 20,
        "prompt_cache_miss_tokens": 80,
        "completion_tokens": 10,
    }
    assert ev.estimate_cost(usage, pricing) == pytest.approx(0.000244)
    assert ev.estimate_cost(usage, None) is None
    assert ev.estimate_cost({"prompt_tokens": 100, "completion_tokens": 10}, pricing) is None
    usage["prompt_tokens"] = 101
    assert ev.estimate_cost(usage, pricing) is None


def test_citation_occurrences_not_just_unique_ids():
    assert ev.citations("A[1] B[1] C[9]") == [
        {"offset": 1, "source": 1, "token": "[1]"},
        {"offset": 6, "source": 1, "token": "[1]"},
        {"offset": 11, "source": 9, "token": "[9]"},
    ]


def test_zero_denominator_not_perfect_score():
    m = {
        "conclusion_accuracy": 1,
        "fact_support_rate": 1,
        "citation_accuracy": None,
        "completeness": 1,
        "false_refusal_rate": 0,
        "request_success_rate": 1,
    }
    assert ev.quality_gate(m) == "not_evaluable"
    m["citation_accuracy"] = 1
    assert ev.quality_gate(m) == "pass"
    m["request_success_rate"] = 0.99
    assert ev.quality_gate(m) == "fail"


@pytest.fixture
def fake_generated(pack, tmp_path, monkeypatch, request):
    folder, cases, gold = pack
    ev.seal(folder, folder / "gold.draft.jsonl")
    calls = []

    mode = getattr(request, "param", "ok")

    def create(**payload):
        calls.append(payload)
        if mode == "failures" and len(calls) == 1:
            raise TimeoutError("synthetic-secret-never-log must be redacted")
        return ChatCompletion.model_validate(
            {
                "id": "synthetic",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length"
                        if mode == "failures" and len(calls) == 2
                        else "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Supported: treatment improved survival [1].",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            }
        )

    opts = []

    def client(**kwargs):
        opts.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None
        )

    monkeypatch.setattr("openai.OpenAI", client)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(
            deepseek_base_url=ev.CONFIG["base_url"],
            openai_chat_model="deepseek-flash",
            deepseek_api_key="synthetic-secret-never-log",
        ),
    )
    answers = tmp_path / "answers"
    ev.generate(folder, answers, execute=True)
    assert len(calls) == 200
    assert opts[0]["max_retries"] == 0
    assert opts[0]["timeout"] == 60
    assert "synthetic-secret-never-log" not in "".join(p.read_text() for p in answers.iterdir())
    reviews = ev.read_rows(answers / "answer_reviews.draft.jsonl")
    for r in reviews:
        r.update(
            {
                "conclusion": "SUPPORT",
                "abstained": False,
                "facts": [
                    {
                        "span": "treatment improved survival",
                        "supported": True,
                        "citation_offsets": [38],
                    }
                ],
                "covered_point_ids": ["p1"],
                "review": approved(),
                "second_review": approved("fixture-reviewer-2"),
            }
        )
        for c in r["citations"]:
            c.update({"statement": "treatment improved survival", "supported": True})
        r["facts"][0]["citation_offsets"] = [r["citations"][0]["offset"]] if r["citations"] else []
    path = answers / "reviews.fixture.jsonl"
    write_rows(path, reviews)
    return folder, answers, path, cases, gold


def test_full_mocked_200_answer_score_and_no_improvement(fake_generated, tmp_path):
    folder, answers, path, _, _ = fake_generated
    report = ev.score(folder, answers, path, tmp_path / "score")
    assert report["status"] == "pass"
    assert report["paired_conclusion_delta"] == 0
    assert report["paired_95ci"] == [0, 0]
    assert report["improvement_evidence"] is False
    assert report["tied"] == 100
    for m in report["metrics"].values():
        assert m["conclusion_accuracy"] == m["fact_support_rate"] == m["citation_accuracy"] == 1
        assert m["estimated_cost_cny"] is None and m["unknown_cost_calls"] == 100
        assert m["proper_refusal_rate"] is None  # no insufficient contexts in fixture


def test_independent_second_review_gate(fake_generated, tmp_path):
    folder, answers, path, _, _ = fake_generated
    reviews = ev.read_rows(path)
    for r in reviews:
        r["second_review"] = ev.pending()
    write_rows(path, reviews)
    report = ev.score(folder, answers, path, tmp_path / "missing-second")
    assert report["status"] == "awaiting_second_review"
    assert len(report["pending_second_reviews"]) == 20


@pytest.mark.parametrize(
    "change", ["missing", "wrong_binding", "missing_fact", "bad_citation", "bad_point"]
)
def test_review_integrity(fake_generated, tmp_path, change):
    folder, answers, path, _, _ = fake_generated
    reviews = ev.read_rows(path)
    if change == "missing":
        reviews.pop()
    elif change == "wrong_binding":
        reviews[0]["answer_sha256"] = "other"
    elif change == "missing_fact":
        reviews[0]["facts"] = []
    elif change == "bad_citation":
        reviews[0]["citations"] = []
    else:
        reviews[0]["covered_point_ids"] = ["p999"]
    write_rows(path, reviews)
    with pytest.raises(ValueError):
        ev.score(folder, answers, path, tmp_path / change)


def test_raw_response_replay(fake_generated):
    _, answers, _, _, _ = fake_generated
    row = ev.read_rows(answers / "answers.jsonl")[0]
    ev.replay_response(row)
    row["answer"] = "Tampered answer"
    with pytest.raises(ValueError, match="Raw response"):
        ev.replay_response(row)


@pytest.mark.parametrize("fake_generated", ["failures"], indirect=True)
def test_api_failure_truncation_no_retry_no_secret(fake_generated, tmp_path):
    folder, answers, path, _, _ = fake_generated
    rows = ev.read_rows(answers / "answers.jsonl")
    assert len(rows) == 200
    assert rows[0]["status"] == "error" and rows[0]["error_type"] == "TimeoutError"
    assert rows[0]["estimated_cost_cny"] is None
    assert rows[1]["status"] == "invalid" and rows[1]["finish_reason"] == "length"
    assert len(ev.read_rows(answers / "events.jsonl")) == 400
    ev.replay_response(rows[0])
    ev.replay_response(rows[1])
    with pytest.raises(ValueError, match="Failed/truncated"):
        ev.score(folder, answers, path, tmp_path / "must-not-pass")


def test_failure_and_safe_refusal_have_different_denominators(pack):
    _, _, gold = pack
    g = gold[0]
    g["context_sufficient"] = {"A": False, "B": True}
    row = {"query_id": "0", "arm": "A", "status": "ok"}
    review = {
        "conclusion": "INSUFFICIENT",
        "abstained": True,
        "facts": [{"supported": True, "citation_offsets": []}],
        "citations": [],
        "covered_point_ids": [],
    }
    m = ev.summarize_quality([row], [review], {"0": g})
    assert m["conclusion_accuracy"] == 0
    assert m["proper_refusal_rate"] == 1
    assert m["false_refusal_rate"] is None
    row["arm"] = "B"
    assert ev.summarize_quality([row], [review], {"0": g})["false_refusal_rate"] == 1


def test_html_escapes_claim_and_never_marks_reviewed(pack, tmp_path):
    _, cases, gold = pack
    cases[0]["claim"] = "<script>alert(1)</script>"
    target = tmp_path / "review.html"
    ev.render_cases(target, cases[:1], gold[:1])
    content = target.read_text()
    assert "&lt;script&gt;" in content and "<script>alert" not in content
    assert "NOT HUMAN VERIFIED" in content


def test_render_real_format_without_inventing_quality(fake_generated, tmp_path):
    import scripts.render_scifact_answers as view

    folder, answers, _, _, _ = fake_generated
    report = view.render(folder, answers, tmp_path / "rendered")
    assert report["answers"] == 200
    assert report["quality_gate"] == "not_yet_evaluated"
    assert report["conclusion_accuracy"] is None and report["quality_improved"] is None
    assert report["usage_total"]["prompt_tokens"] == 20000
    assert report["human_answer_reviews"] == 0
    assert all(v["estimated_total_cost_cny"] is None for v in report["summaries"].values())
    page = (tmp_path / "rendered/answer_review.html").read_text()
    assert page.count('<textarea class="review"') == 200
    assert "尚未评分" in page and "answer_reviews.reviewed.jsonl" in page


@pytest.mark.parametrize("change", ["derived_answer", "event", "template"])
def test_render_replay_refuses_tamper(fake_generated, tmp_path, change):
    import scripts.render_scifact_answers as view

    folder, answers, _, _, _ = fake_generated
    if change == "derived_answer":
        rows = ev.read_rows(answers / "answers.jsonl")
        rows[0]["answer"] = "Changed derived answer"
        write_rows(answers / "answers.jsonl", rows)
        manifest = ev.read_object(answers / "answers_manifest.json")
        manifest["answers_sha256"] = hash_file(answers / "answers.jsonl")
        write_json(answers / "answers_manifest.json", manifest)
    elif change == "event":
        rows = ev.read_rows(answers / "events.jsonl")
        rows[0]["arm"] = "B"
        write_rows(answers / "events.jsonl", rows)
        manifest = ev.read_object(answers / "answers_manifest.json")
        manifest["events_sha256"] = hash_file(answers / "events.jsonl")
        write_json(answers / "answers_manifest.json", manifest)
    else:
        rows = ev.read_rows(answers / "answer_reviews.draft.jsonl")
        rows[0]["answer_sha256"] = "wrong"
        write_rows(answers / "answer_reviews.draft.jsonl", rows)
    with pytest.raises(ValueError):
        view.render(folder, answers, tmp_path / "tampered-render")


def test_partial_usage_stays_unknown():
    import scripts.render_scifact_answers as view

    result = view.usage_summary([{"usage": {"prompt_tokens": 10}}, {"usage": None}])
    assert result["calls"] == 2 and result["calls_with_usage"] == 1
    assert result["completion_tokens"] is None
    assert result["reasoning_tokens_if_reported"] is None
