"""Offline casebook + preregistered generation pilot previews. No executor/API calls.

Only system-prompt suffix varies. Historical answers are diagnosis evidence, not
concurrent controls. Prospective development queries are selected by fixed order.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from scripts import evaluate_scifact_answers as ev
from scripts.evaluate_scifact import DEFAULT_PREPARED, load_prepared, read_rows
from scripts.prepare_scifact import ROOT, hash_file, write_json, write_rows
from scripts.render_scifact_answers import load_verified

SPECS = ROOT / "configs/evaluation/scifact.answer_failure_cases.v1.json"
PROMPT = ROOT / "configs/prompts/answer_evidence_contract_v1.txt"
PROTOCOL = ROOT / "docs/refactor/SCIFACT_GENERATION_PILOT_PROTOCOL.md"
PACK = ROOT / "runs/scifact-answer-review-v1"
ANSWERS = ROOT / "runs/scifact-answers-paired-v1"
VARIANTS = ("baseline_generation", "evidence_contract_v1")


def validate_specs(specs):
    if specs.get("schema_version") != 1:
        raise ValueError("Unknown casebook schema")
    rows = specs["cases"]
    if len(rows) != 12 or len({s["query_id"] for s in rows}) != 12:
        raise ValueError("Exactly 12 distinct diagnostic cases required")
    if Counter(s["pilot_role"] for s in rows) != {"target": 4, "control": 4, "ambiguity": 4}:
        raise ValueError("Freeze four targets, controls and ambiguity probes each")
    for s in rows:
        if not s["anchors"] or any(not ev.text(a.get("quote")) for a in s["anchors"]):
            raise ValueError("Each diagnosis needs literal source evidence anchors")
    return rows


def make_payload(case, variant, suffix):
    if variant not in VARIANTS or not ev.text(suffix):
        raise ValueError("Unknown generation variant or empty candidate contract")
    # Copy the application's wire messages; never mutate the frozen baseline.
    messages = copy.deepcopy(ev.context_messages(case, "B"))
    if variant == "evidence_contract_v1":
        content = messages[0].get("content")
        if not isinstance(content, str):
            raise ValueError("Expected the frozen textual system prompt")
        messages[0]["content"] = content + "\n\n" + suffix
    return {
        **{k: ev.CONFIG[k] for k in ("model", "temperature", "max_tokens", "stream")},
        "messages": messages,
    }


def make_previews(diagnostic, prospective, suffix):
    all_cases = diagnostic + prospective
    if len({c["query_id"] for c in all_cases}) != len(all_cases):
        raise ValueError("Diagnostic/prospective cohort overlap")
    result = []
    for phase, cases in (
        ("A_diagnostic_regression", diagnostic),
        ("B_prospective_development", prospective),
    ):
        for index, case in enumerate(cases):
            order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
            for variant in order:
                result.append(
                    {
                        "phase": phase,
                        "query_id": case["query_id"],
                        "variant": variant,
                        "context_arm": "B_relevance_v1",
                        "context_sha256": ev.digest(case["contexts"]["B"]),
                        "request": make_payload(case, variant, suffix),
                    }
                )
    return result


def evidence_bindings(spec, case):
    result = []
    for arm in ev.ARMS:
        for anchor in spec["anchors"]:
            matches = []
            for n, context in enumerate(case["contexts"][arm], 1):
                if context["document_id"] == anchor["document_id"]:
                    start = context["text"].find(anchor["quote"])
                    if start >= 0:
                        matches.append(
                            {
                                "source_number": n,
                                "chunk_id": context["chunk_id"],
                                "start": start,
                                "end": start + len(anchor["quote"]),
                            }
                        )
            if not matches:
                raise ValueError(
                    f"Evidence quote absent from actual context: {case['query_id']}/{arm}"
                )
            result.append({"arm": arm, **anchor, "matches": matches})
    return result


def corpus_probe(spec, case, corpus, chunks):
    probe = spec.get("corpus_probe")
    if probe is None:
        return None
    did = probe["document_id"]
    doc = corpus[did]
    pieces = [c for c in chunks.values() if c["document_id"] == did]
    if len(pieces) != probe["expected_chunk_count"]:
        raise ValueError("Corpus probe chunk count changed")
    covered = all(
        any(c["document_id"] == did and doc["text"] in c["text"] for c in case["contexts"][arm])
        for arm in ev.ARMS
    )
    if covered != probe["full_document_in_both_contexts"]:
        raise ValueError("Corpus coverage probe disagrees with actual sent context")
    return {
        "document_id": did,
        "local_document_chars": len(doc["text"]),
        "chunks_in_frozen_manifest": len(pieces),
        "full_local_text_sent_to_both_arms": covered,
        "does_not_prove_entire_corpus_has_no_answer": True,
    }


def prepare(output):
    specs = ev.read_object(SPECS)
    selected = validate_specs(specs)
    cases, gold, answers, _, _ = load_verified(PACK, ANSWERS)
    by_case = ev.unique(cases, "query_id")
    by_answer = {(a["query_id"], a["arm"]): a for a in answers}
    data = load_prepared(DEFAULT_PREPARED, "development")
    chunks = ev.unique(read_rows(ev.CHUNKS), "chunk_id")
    ranked = ev.unique(read_rows(ev.NEW), "query_id")
    prospective_ids = list(data["queries"])[100:120]
    if (
        len(prospective_ids) != 20
        or set(prospective_ids) & set(by_case)
        or set(by_case) != set(list(data["queries"])[:100])
    ):
        raise ValueError("Frozen development cohort/order mismatch")
    diagnostic, casebook = [], []
    for spec in selected:
        q = spec["query_id"]
        case = by_case[q]
        if [r["chunk_id"] for r in ranked[q]["chunks"][:4]] != [
            r["chunk_id"] for r in case["contexts"]["B"]
        ]:
            raise ValueError("Diagnostic new-rerank context changed")
        diagnostic.append(case)
        a, b = by_answer[(q, "A")], by_answer[(q, "B")]
        casebook.append(
            {
                **copy.deepcopy(spec),
                "claim": case["claim"],
                "historical_reference_label_not_scientific_truth": gold[q]["label"],
                "evidence_bindings": evidence_bindings(spec, case),
                "corpus_probe_result": corpus_probe(spec, case, data["corpus"], chunks),
                "same_generation_request_A_B": a["request"] == b["request"],
                "same_context_chunk_set_A_B": {c["chunk_id"] for c in a["contexts"]}
                == {c["chunk_id"] for c in b["contexts"]},
                "historical_answers": {
                    arm: {
                        "answer": by_answer[(q, arm)]["answer"],
                        "answer_sha256": ev.digest(by_answer[(q, arm)]),
                        "contexts": by_answer[(q, arm)]["contexts"],
                    }
                    for arm in ev.ARMS
                },
                "reviewer_kind": "main_assistant_ai",
                "human_verified": False,
                "independent_blind_review": False,
                "outcome": "diagnosis_and_hypothesis_only_not_candidate_result",
            }
        )
    prospective = []
    for q in prospective_ids:
        if ranked[q]["mode"] != "relevance_v1":
            raise ValueError("Prospective input must reuse frozen relevance_v1")
        contexts = [
            {
                **chunks[r["chunk_id"]],
                "title": data["corpus"][chunks[r["chunk_id"]]["document_id"]]["title"],
            }
            for r in ranked[q]["chunks"][:4]
        ]
        if len(contexts) != 4:
            raise ValueError("Missing prospective top4")
        prospective.append(
            {"query_id": q, "claim": data["queries"][q], "contexts": {"B": contexts}}
        )
    suffix = PROMPT.read_text().strip()
    previews = make_previews(diagnostic, prospective, suffix)
    if len(previews) != 64:
        raise ValueError("Exactly 24+40 conditional request previews expected")
    ev.fresh(output)
    write_rows(output / "cases.jsonl", casebook)
    write_rows(output / "requests.plan.jsonl", previews)
    # This file is NOT an executed generation run or a new answer gold file.
    source_paths = [
        SPECS,
        PROMPT,
        PROTOCOL,
        Path(__file__),
        PACK / "manifest.json",
        PACK / "sealed/gold.jsonl",
        ANSWERS / "answers.jsonl",
        ev.NEW,
        ev.CHUNKS,
        DEFAULT_PREPARED / "report.json",
        ROOT / "docs/answer_reviews.reviewed.jsonl",
    ]
    manifest = {
        "status": "planned_not_executed",
        "schema_version": 1,
        "model_calls": 0,
        "generated_candidate_answers": 0,
        "application_judge_calls": 0,
        "planned_max_generation_calls": 64,
        "stages": {
            "A": {"query_ids": [c["query_id"] for c in diagnostic], "calls": 24},
            "B": {"query_ids": prospective_ids, "calls": 40, "requires_A_gate": True},
        },
        "context_policy": "fixed relevance_v1 top4 full chunks for BOTH generation variants",
        "generation_config": ev.CONFIG,
        "only_experimental_factor": "system_prompt_suffix",
        "variants": list(VARIANTS),
        "requires_explicit_future_execution": True,
        "is_independent_test": False,
        "historical_answers_are_not_concurrent_controls": True,
        "source_hashes": {str(p.relative_to(ROOT)): hash_file(p) for p in source_paths},
        "output_hashes": {n: hash_file(output / n) for n in ("cases.jsonl", "requests.plan.jsonl")},
        "main_session_review_cost_cny": None,
    }
    for name, path in (
        ("candidate_prompt.txt", PROMPT),
        ("protocol.md", PROTOCOL),
        ("preparation_source.py", Path(__file__)),
    ):
        (output / name).write_bytes(path.read_bytes())
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    m = prepare(args.output)
    print(
        json.dumps(
            {
                "status": m["status"],
                "model_calls": 0,
                "diagnostic_queries": 12,
                "prospective_queries": 20,
                "conditional_request_previews": m["planned_max_generation_calls"],
            }
        )
    )


if __name__ == "__main__":
    main()
