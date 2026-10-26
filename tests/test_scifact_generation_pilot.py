"""Offline planning contracts; no generated answers or scientific verdict fixtures."""

from __future__ import annotations

import copy

import pytest

from scripts import evaluate_scifact_answers as ev
from scripts import prepare_scifact_generation_pilot as pilot


@pytest.fixture
def case():
    return {
        "query_id": "q1",
        "claim": "A bounded test claim.",
        "desired_behavior": "DIAGNOSIS_MUST_NOT_ENTER_REQUEST",
        "contexts": {
            "B": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "chunk_index": 0,
                    "title": "Source",
                    "text": "Literal evidence.",
                }
            ]
        },
    }


def test_candidate_changes_only_system_suffix_and_not_inputs(case):
    original = copy.deepcopy(case)
    control = pilot.make_payload(case, "baseline_generation", "contract")
    candidate = pilot.make_payload(case, "evidence_contract_v1", "contract")
    assert case == original
    assert control["messages"] == ev.context_messages(case, "B")
    assert candidate["messages"][1:] == control["messages"][1:]
    a, b = candidate["messages"][0].get("content"), control["messages"][0].get("content")
    assert isinstance(a, str) and isinstance(b, str) and a == b + "\n\ncontract"
    assert {k: v for k, v in candidate.items() if k != "messages"} == {
        k: v for k, v in control.items() if k != "messages"
    }
    assert "DIAGNOSIS_MUST_NOT_ENTER_REQUEST" not in str(candidate)
    assert "api_key" not in candidate


@pytest.mark.parametrize("variant,suffix", [("unknown", "suffix"), ("evidence_contract_v1", " ")])
def test_unknown_or_empty_contract_rejected(case, variant, suffix):
    with pytest.raises(ValueError):
        pilot.make_payload(case, variant, suffix)


def test_interleaving_and_conditional_phase_previews(case):
    second, fresh = copy.deepcopy(case), copy.deepcopy(case)
    second["query_id"], fresh["query_id"] = "q2", "q3"
    rows = pilot.make_previews([case, second], [fresh], "contract")
    assert len(rows) == 6
    assert [r["variant"] for r in rows[:4]] == [
        "baseline_generation",
        "evidence_contract_v1",
        "evidence_contract_v1",
        "baseline_generation",
    ]
    assert rows[-1]["phase"] == "B_prospective_development"
    assert all(r["context_arm"] == "B_relevance_v1" for r in rows)


def test_overlapping_future_cohort_rejected(case):
    with pytest.raises(ValueError, match="overlap"):
        pilot.make_previews([case], [case], "contract")


def test_anchor_must_exist_in_both_actual_contexts(case):
    case["contexts"]["A"] = copy.deepcopy(case["contexts"]["B"])
    spec = {"anchors": [{"document_id": "d1", "quote": "evidence"}]}
    found = pilot.evidence_bindings(spec, case)
    assert len(found) == 2
    assert found[0]["matches"][0]["source_number"] == 1
    case["contexts"]["A"][0]["text"] = "Missing information."
    with pytest.raises(ValueError, match="absent"):
        pilot.evidence_bindings(spec, case)


def test_duplicate_document_does_not_imply_quote_in_each_chunk(case):
    original = copy.deepcopy(case["contexts"]["B"][0])
    blank = {**original, "chunk_id": "c0", "text": "different part"}
    case["contexts"] = {"A": [blank, original], "B": [original, blank]}
    found = pilot.evidence_bindings({"anchors": [{"document_id": "d1", "quote": "evidence"}]}, case)
    assert found[0]["matches"][0]["source_number"] == 2
    assert found[1]["matches"][0]["source_number"] == 1


def test_probe_distinguishes_corpus_content_from_context_truncation(case):
    case["contexts"]["A"] = copy.deepcopy(case["contexts"]["B"])
    spec = {
        "corpus_probe": {
            "document_id": "d1",
            "expected_chunk_count": 1,
            "full_document_in_both_contexts": True,
        }
    }
    corpus = {"d1": {"text": "Literal evidence."}}
    chunks = {"c1": case["contexts"]["B"][0]}
    result = pilot.corpus_probe(spec, case, corpus, chunks)
    assert result is not None
    assert result["full_local_text_sent_to_both_arms"] is True
    assert result["does_not_prove_entire_corpus_has_no_answer"] is True
    case["contexts"]["A"][0]["text"] = "Only part"
    with pytest.raises(ValueError, match="coverage"):
        pilot.corpus_probe(spec, case, corpus, chunks)


@pytest.mark.parametrize("change", ["version", "duplicate", "roles", "no_quote"])
def test_fixed_twelve_case_spec_validation(change):
    rows = [
        {
            "query_id": str(i),
            "pilot_role": ["target", "control", "ambiguity"][i // 4],
            "anchors": [{"quote": "evidence"}],
        }
        for i in range(12)
    ]
    specs = {"schema_version": 1, "cases": rows}
    assert len(pilot.validate_specs(specs)) == 12
    if change == "version":
        specs["schema_version"] = 2
    elif change == "duplicate":
        rows[1]["query_id"] = rows[0]["query_id"]
    elif change == "roles":
        rows[0]["pilot_role"] = "control"
    else:
        rows[0]["anchors"][0]["quote"] = ""
    with pytest.raises(ValueError):
        pilot.validate_specs(specs)
