"""Synthetic AI audit contracts, not scientific gold or human signatures."""

from __future__ import annotations

import copy

import pytest

from scripts import audit_scifact_ai_reviews as audit
from scripts import evaluate_scifact_answers as ev


@pytest.mark.parametrize(
    "token,numbers",
    [("[1]", [1]), ("[1,2]", [1, 2]), ("[2–4]", [2, 3, 4]), ("[1, 3-4]", [1, 3, 4])],
)
def test_grouped_citation_offsets_and_sources(token, numbers):
    found = audit.expanded_citations("X " + token)
    assert [c["source"] for c in found] == numbers
    assert all(c["offset"] == 2 and c["token"] == token for c in found)


@pytest.mark.parametrize("token", ["[4-2]", "[1–9999]", "[1,1]"])
def test_invalid_ranges_not_silently_repaired(token):
    with pytest.raises(ValueError):
        audit.expanded_citations(token)


@pytest.fixture
def data():
    cases = [{"query_id": "q"}]
    gold = {
        "q": {
            "label": "SUPPORT",
            "required_points": [{"id": "p1"}],
            "context_sufficient": {"A": True, "B": True},
        }
    }
    answers, reviews = [], []
    for arm in ev.ARMS:
        row = {
            "query_id": "q",
            "arm": arm,
            "answer": "Supported [1,2].",
            "contexts": [{"document_id": "d1"}, {"document_id": "d2"}],
        }
        answers.append(row)
        reviews.append(
            {
                "query_id": "q",
                "arm": arm,
                "answer_sha256": ev.digest(row),
                "gold_sha256": "gold",
                "conclusion": "SUPPORT",
                "abstained": False,
                "facts": [
                    {
                        "span": "Supported",
                        "start": 0,
                        "end": 9,
                        "supported": True,
                        "asserted": True,
                        "citation_offsets": [10],
                    }
                ],
                "citations": [
                    {
                        **c,
                        "statement": "Supported",
                        "supported": True,
                        "evidence_document_id": "d" + str(c["source"]),
                    }
                    for c in audit.expanded_citations(row["answer"])
                ],
                "covered_point_ids": ["p1"],
                "point_coverage": [{"id": "p1", "status": "covered"}],
                "review": {
                    "status": "ai_reviewed",
                    "human_verified": False,
                    "reviewer": "fixture AI (not human)",
                    "notes": "TEST ONLY",
                },
                "ai_assessment": {
                    "conclusion_supported_by_context": True,
                    "context_sufficient": True,
                    "gold_disputed": False,
                    "flags": [],
                },
            }
        )
    return cases, gold, answers, reviews


def test_ai_does_not_become_human_and_denominator_is_explicit(data):
    original = copy.deepcopy(data)
    report = audit.audit_data(data[0], data[1], data[2], data[3], "gold")
    assert data == original
    assert report["human_gate_evaluated"] is False
    assert report["status"] == "ai_audited_not_human_validated"
    assert report["v1_group_parser_affected_answers"] == 2
    for m in report["summaries"].values():
        assert m["citation_support_ai_grouped_v2"]["n"] == 2
        assert m["citation_support_ai_legacy_v1_sensitivity"]["n"] == 0
        assert m["citation_support_ai_legacy_v1_sensitivity"]["lower_rate"] is None


def test_uncertain_is_not_correct_or_incorrect(data):
    data[3][0]["facts"][0]["supported"] = None
    data[3][0]["ai_assessment"]["conclusion_supported_by_context"] = None
    report = audit.audit_data(data[0], data[1], data[2], data[3], "gold")
    m = report["summaries"]["baseline"]
    assert m["fact_fragment_support_ai"]["lower_rate"] == 0
    assert m["fact_fragment_support_ai"]["upper_rate"] == 1
    assert m["context_conclusion_support_ai"]["uncertain"] == 1
    assert report["v1_boolean_schema_affected_answers"] == 1


@pytest.mark.parametrize(
    "kind",
    [
        "hash",
        "invented_citation",
        "fact_offset",
        "span",
        "human",
        "source",
        "coverage",
        "duplicate_review",
    ],
)
def test_review_tamper_or_schema_mismatch_rejected(data, kind):
    r = data[3][0]
    if kind == "hash":
        r["answer_sha256"] = "changed"
    elif kind == "invented_citation":
        r["citations"][0]["offset"] += 1
    elif kind == "fact_offset":
        r["facts"][0]["citation_offsets"] = [999]
    elif kind == "span":
        r["facts"][0]["span"] = "not from answer"
    elif kind == "human":
        r["review"]["status"] = "approved"
    elif kind == "source":
        r["citations"][0]["evidence_document_id"] = "wrong"
    elif kind == "coverage":
        r["point_coverage"][0]["status"] = "partial"
    else:
        data[3].append(copy.deepcopy(r))
    with pytest.raises(ValueError):
        audit.audit_data(data[0], data[1], data[2], data[3], "gold")


def test_unknown_context_sufficiency_is_not_sufficient_or_insufficient():
    rows = [
        {"abstained": True, "enough": True},
        {"abstained": True, "enough": None},
        {"abstained": False, "enough": False},
    ]
    m = audit.refusal_stats(rows, lambda r: r["enough"])
    assert m["sufficient_n"] == m["insufficient_n"] == m["uncertain_sufficiency_n"] == 1
    assert m["false_refusal_rate"] == 1 and m["proper_refusal_rate"] == 0


def test_invalid_support_not_coerced():
    with pytest.raises(ValueError):
        audit.support_counts([1])
