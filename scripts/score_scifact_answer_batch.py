"""Validate saved AI answer annotations and aggregate frozen-protocol rubric scores.

No semantic judge, application imports, model calls, or database access. Fact quotes
are exact answer spans; evidence quotes are exact spans of ACTUALLY SENT chunks.
Citation groups follow existing v2 token/range expansion. Compact annotation facts:
[answer_quote, support_label, [[source, evidence_quote], ...], reason, kind].
Kinds: fact, derived, scope_audit (absence/limits checked against all four chunks).
Citation group judgments: [per_source_labels, joint_label, fact_ids_1based, reason].
Only explicit AI judgments are aggregated; successful validation is not expert truth
or proof that semantic atomization is exhaustive. Repeated paraphrases of the same
proposition need not be counted twice; verdict is separately assessed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from scripts.complete_scifact_six_metrics import ROOT, canonical_hash, digest, require, rows

CITATIONS_V2 = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
LABELS = {"supported", "unsupported", "uncertain"}
CITATION_LABELS = LABELS | {"partial"}


def span(text: str, quote: str) -> dict:
    require(bool(quote), "Empty quote")
    start = text.find(quote)
    require(start >= 0, "Quote absent from exact source/answer")
    require(text.find(quote, start + 1) < 0, "Ambiguous quote; provide longer exact text")
    return {"start": start, "end": start + len(quote), "quote": quote}


def citation_tokens(answer: str) -> list[dict]:
    result = []
    for m in CITATIONS_V2.finditer(answer):
        numbers = []
        for part in m.group(1).split(","):
            limits = re.split(r"\s*[-–]\s*", part.strip())
            try:
                a, b = int(limits[0]), int(limits[-1])
            except ValueError as exc:
                raise ValueError("Non-numeric citation range") from exc
            require(1 <= a <= b <= 4, "Invalid citation source/range")
            numbers.extend(range(a, b + 1))
        require(len(numbers) == len(set(numbers)), "Duplicate citation source")
        result.append({"offset": m.start(), "end": m.end(), "token": m.group(), "sources": numbers})
    return result


def ratio(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def support_summary(labels: list[str]) -> dict:
    c = Counter(labels)
    n = len(labels)
    return {
        "n": n,
        **{k: c[k] for k in sorted(LABELS)},
        "lower": ratio(c["supported"], n),
        "upper": ratio(c["supported"] + c["uncertain"], n),
    }


def assess(pack: Path, context_run: Path, annotations: dict) -> list[dict]:
    require(annotations["protocol"] == "SCIFACT_SIX_METRICS_PROTOCOL_v1", "Wrong protocol")
    require(annotations["reviewer"] == "main_assistant_AI", "AI provenance required")
    require(
        annotations["human_verified"] is False and annotations["independent_blind_review"] is False,
        "False provenance",
    )
    start, stop = annotations["start"], annotations["stop"]
    require(type(start) is int and type(stop) is int and 0 <= start < stop <= 100, "Bad range")
    pending = [
        r for r in rows(pack / "answers.pending.jsonl") if start <= r["query_position"] < stop
    ]
    source = {(r["query_id"], r["arm"]): r for r in pending}
    contexts = {
        (r["query_id"], r["chunk_id"]): r for r in rows(context_run / "contexts.reviewed.jsonl")
    }
    sufficiency = {
        (r["query_id"], r["arm"]): r["context_sufficiency"]
        for r in rows(context_run / "metrics.jsonl")
    }
    originals = {
        (r["query_id"], r["arm"]): r
        for r in rows(ROOT / "runs/scifact-answers-paired-v1/answers.jsonl")
    }
    expected = [(r["query_id"], r["arm"]) for r in pending]
    actual = [(r["query_id"], r["arm"]) for r in annotations["reviews"]]
    require(
        actual == expected and len(set(actual)) == len(actual),
        "Missing/duplicate/unordered reviews",
    )
    results = []
    for review in annotations["reviews"]:
        key = (review["query_id"], review["arm"])
        saved = source[key]
        original = originals[key]
        require(
            canonical_hash(original) == saved["source_answer_row_sha256"], "Original answer changed"
        )
        answer = saved["answer"]
        group = [contexts[key[0], c["chunk_id"]] for c in saved["contexts"]]
        require(len(group) == 4, "Four actual contexts required")
        for r in group:
            require(digest(r["text"].encode()) == r["text_sha256"], "Context hash mismatch")
        relevance = review["answer_relevance"]
        require(
            relevance is None or (type(relevance) is int and relevance in (0, 1, 2)),
            "Bad relevance",
        )
        require(bool(review["answer_relevance_reason"].strip()), "Missing relevance reason")
        require(
            type(review["refusal"]) is bool and bool(review["refusal_reason"]),
            "Missing refusal judgment",
        )
        conclusion = review["conclusion"]
        require(conclusion[1] in LABELS and bool(conclusion[2]), "Invalid conclusion")
        conclusion_span = span(answer, conclusion[0])
        facts = []
        for i, (quote, label, evidence, reason, kind) in enumerate(review["facts"], 1):
            require(label in LABELS and bool(reason), "Invalid fact judgment")
            require(kind in {"fact", "derived", "scope_audit"}, "Invalid fact kind")
            es = []
            for number, evidence_quote in evidence:
                require(type(number) is int and 1 <= number <= 4, "Invalid evidence source")
                ctx = group[number - 1]
                es.append(
                    {
                        "source": number,
                        "chunk_id": ctx["chunk_id"],
                        "text_sha256": ctx["text_sha256"],
                        **span(ctx["text"], evidence_quote),
                    }
                )
            require(
                label != "supported" or bool(es) or kind == "scope_audit",
                "Supported fact lacks evidence",
            )
            facts.append(
                {
                    "id": i,
                    "answer_span": span(answer, quote),
                    "label": label,
                    "reason": reason,
                    "kind": kind,
                    "evidence_spans": es,
                    "scope_context_hashes": [c["text_sha256"] for c in group]
                    if kind == "scope_audit"
                    else [],
                }
            )
        tokens = citation_tokens(answer)
        require(len(tokens) == len(review["citation_groups"]), "Citation group coverage mismatch")
        citations = []
        for token, (labels, joint, ids, reason) in zip(
            tokens, review["citation_groups"], strict=True
        ):
            require(
                len(labels) == len(token["sources"])
                and all(label in CITATION_LABELS for label in labels),
                "Invalid citation labels",
            )
            require(joint in LABELS and bool(reason) and bool(ids), "Invalid joint citation")
            require(
                len(ids) == len(set(ids))
                and all(type(i) is int and 1 <= i <= len(facts) for i in ids),
                "Bad cited fact IDs",
            )
            require(
                all(
                    not facts[i - 1]["evidence_spans"]
                    or any(e["source"] in token["sources"] for e in facts[i - 1]["evidence_spans"])
                    for i in ids
                ),
                "Cited evidence identity mismatch",
            )
            citations.append(
                {
                    **token,
                    "fact_ids": ids,
                    "joint_support": joint,
                    "reason": reason,
                    "source_support": [
                        {"source": n, "label": label, "chunk_id": group[n - 1]["chunk_id"]}
                        for n, label in zip(token["sources"], labels, strict=True)
                    ],
                }
            )
        delivered = (
            saved["status"] == "ok" and bool(answer.strip()) and saved["finish_reason"] == "stop"
        )
        require(delivered or not facts, "Failed delivery facts must not receive a success score")
        summary = support_summary([f["label"] for f in facts])
        cited_ids = {
            i for c in citations for i in c["fact_ids"] if c["joint_support"] == "supported"
        }
        eligible = [f for f in facts if f["kind"] != "scope_audit"]
        results.append(
            {
                "query_id": key[0],
                "arm": key[1],
                "query_position": saved["query_position"],
                "query": saved["query"],
                "answer_sha256": digest(answer.encode()),
                "source_answer_row_sha256": saved["source_answer_row_sha256"],
                "context_hashes": [c["text_sha256"] for c in group],
                "context_sufficiency": sufficiency[key],
                "answer_relevance": relevance,
                "answer_relevance_reason": review["answer_relevance_reason"],
                "conclusion": {
                    "answer_span": conclusion_span,
                    "label": conclusion[1],
                    "reason": conclusion[2],
                },
                "assertions": facts,
                "faithfulness": summary if delivered else None,
                "citation_groups": citations,
                "citation_coverage": {
                    "eligible_non_scope_assertions": len(eligible),
                    "joint_supported_cited_assertions": sum(f["id"] in cited_ids for f in eligible),
                    "rate": ratio(sum(f["id"] in cited_ids for f in eligible), len(eligible)),
                },
                "refusal": review["refusal"],
                "refusal_reason": review["refusal_reason"],
                "delivery_ok": delivered,
                "status": saved["status"],
                "finish_reason": saved["finish_reason"],
                "generation_ms": original["generation_ms"],
                "usage": original["usage"],
                "estimated_cost_cny": original["estimated_cost_cny"],
                "actual_cost_cny": original["actual_cost_cny"],
                "reviewer": "main_assistant_AI_nonblind",
                "human_verified": False,
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    summaries = {}
    for arm in ("A", "B"):
        rs = [r for r in results if r["arm"] == arm]
        faithful = [r for r in rs if r["faithfulness"] is not None and r["faithfulness"]["n"]]
        facts = [f["label"] for r in faithful for f in r["assertions"]]
        suff = [r for r in rs if r["context_sufficiency"] == "sufficient"]
        insuff = [r for r in rs if r["context_sufficiency"] == "insufficient"]
        appropriate = [
            r
            for r in insuff
            if r["refusal"]
            and r["delivery_ok"]
            and r["answer_relevance"] == 2
            and r["conclusion"]["label"] == "supported"
        ]
        citations = [c for r in rs for c in r["citation_groups"]]
        source_labels = [s["label"] for c in citations for s in c["source_support"]]
        grades = Counter(str(r["answer_relevance"]) for r in rs)
        summaries[arm] = {
            "answers": len(rs),
            "delivery_ok": sum(r["delivery_ok"] for r in rs),
            "faithfulness_micro": support_summary(facts),
            "faithfulness_non_scope_micro_sensitivity": support_summary(
                [
                    f["label"]
                    for r in faithful
                    for f in r["assertions"]
                    if f["kind"] != "scope_audit"
                ]
            ),
            "faithfulness_macro_lower": math.fsum(r["faithfulness"]["lower"] for r in faithful)
            / len(faithful)
            if faithful
            else None,
            "faithfulness_macro_upper": math.fsum(r["faithfulness"]["upper"] for r in faithful)
            / len(faithful)
            if faithful
            else None,
            "faithfulness_scored_answers": len(faithful),
            "answer_relevance_counts": dict(grades),
            "answer_relevance_grade2_lower": sum(r["answer_relevance"] == 2 for r in rs) / len(rs),
            "answer_relevance_grade2_upper": sum(r["answer_relevance"] in (2, None) for r in rs)
            / len(rs),
            "answer_relevance_normalized_lower": sum(r["answer_relevance"] or 0 for r in rs)
            / (2 * len(rs)),
            "answer_relevance_normalized_upper": sum(
                2 if r["answer_relevance"] is None else r["answer_relevance"] for r in rs
            )
            / (2 * len(rs)),
            "conclusion_support": support_summary([r["conclusion"]["label"] for r in rs]),
            "citation_source_counts_v2": dict(Counter(source_labels)),
            "citation_source_full_support_rate": ratio(
                source_labels.count("supported"), len(source_labels)
            ),
            "citation_joint_support": support_summary([c["joint_support"] for c in citations]),
            "citation_eligible_assertions": sum(
                r["citation_coverage"]["eligible_non_scope_assertions"] for r in rs
            ),
            "citation_joint_supported_assertions": sum(
                r["citation_coverage"]["joint_supported_cited_assertions"] for r in rs
            ),
            "refusals": sum(r["refusal"] for r in rs),
            "sufficient_n": len(suff),
            "false_refusals": sum(r["refusal"] for r in suff),
            "false_refusal_rate": ratio(sum(r["refusal"] for r in suff), len(suff)),
            "insufficient_n": len(insuff),
            "appropriate_refusals": len(appropriate),
            "appropriate_refusal_rate": ratio(len(appropriate), len(insuff)),
            "uncertain_sufficiency_n": sum(r["context_sufficiency"] == "uncertain" for r in rs),
            "original_cost_unknown_answers": sum(r["actual_cost_cny"] is None for r in rs),
            "usage_preserved_not_resummed": True,
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "runs") and not output.exists(), "Fresh run required")
    blocked = []

    def audit(event, values):
        if event in {
            "socket.connect",
            "socket.connect_ex",
            "socket.getaddrinfo",
            "socket.bind",
            "sqlite3.connect",
        }:
            blocked.append(event)
            raise RuntimeError("No network/database allowed")
        if event == "open" and isinstance(values[0], str | bytes):
            name = str(values[0])
            if Path(name).name.startswith(".env") or "/splits/test/" in name:
                blocked.append("secret_or_test_open")
                raise RuntimeError("No secrets/test files allowed")

    sys.addaudithook(audit)
    try:
        manifest = json.loads((args.pack / "manifest.json").read_text())
        context_verification = json.loads((args.contexts / "verification.json").read_text())
        annotation = json.loads(args.annotations.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read valid evaluation input JSON") from exc
    for name, expected in manifest["outputs_sha256"].items():
        require(digest((args.pack / name).read_bytes()) == expected, "Frozen pack changed")
    for name, expected in manifest["inputs_sha256"].items():
        require(digest((ROOT / name).read_bytes()) == expected, "Frozen source changed")
    for name, expected in context_verification["outputs_sha256"].items():
        require(
            digest((args.contexts / name).read_bytes()) == expected, "Frozen context run changed"
        )
    inputs = [
        args.annotations,
        args.pack / "manifest.json",
        args.contexts / "verification.json",
        ROOT / "docs/refactor/SCIFACT_SIX_METRICS_PROTOCOL.md",
        Path(__file__),
    ]
    annotation_source = args.annotations.with_name("annotation_source.py")
    if annotation_source.is_file():
        inputs.append(annotation_source)
    hashes = {str(p.resolve().relative_to(ROOT)): digest(p.read_bytes()) for p in inputs}
    results = assess(args.pack, args.contexts, annotation)
    report = {
        "status": "partial_answer_evaluation",
        "protocol": annotation["protocol"],
        "start": annotation["start"],
        "stop": annotation["stop"],
        "answer_count": len(results),
        "summaries": summarize(results),
        "reviewer": "main_assistant_AI_nonblind",
        "human_verified": False,
        "independent_blind_review": False,
        "framework": "stdlib explicit-rubric aggregation, not built-in DeepEval judge metrics",
        "new_model_API_calls": 0,
        "main_session_AI_cost_cny": None,
        "inputs_sha256": hashes,
        "forbidden_runtime_events": blocked,
        "scope_audit_facts_in_faithfulness": True,
        "limits": [
            "Partial fixed development cohort",
            "AI atomization not expert truth or machine-proven exhaustive",
            "Scope-audit statements check all four texts; citation coverage excludes them",
            "Unsupported means not grounded in sent context, not necessarily false in reality",
            "Uncertain labels not forced binary; bounds are not confidence intervals",
        ],
    }
    require(not blocked, "Forbidden event")
    require(all(digest((ROOT / p).read_bytes()) == h for p, h in hashes.items()), "Input changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "answers.reviewed.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    verification = {
        "status": "passed",
        "inputs_sha256": hashes,
        "outputs_sha256": {p.name: digest(p.read_bytes()) for p in output.iterdir()},
        "all_answer_and_evidence_spans_verified": True,
        "all_v2_citation_groups_and_actual_source_ids_verified": True,
        "forbidden_runtime_events": blocked,
    }
    (output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"answers": len(results), "summaries": report["summaries"]}))


if __name__ == "__main__":
    main()
