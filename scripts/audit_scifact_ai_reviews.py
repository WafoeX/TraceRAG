"""Offline AI-review audit v2; never turn AI metadata into human approval.

Preserves frozen v1 artifacts. Reports grouped citations and tri-state judgments
explicitly, with reference-label agreement separate from context-based AI opinion.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from scripts import evaluate_scifact_answers as ev
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.render_scifact_answers import load_verified

CITATIONS_V2 = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")


def expanded_citations(answer):
    result = []
    for match in CITATIONS_V2.finditer(answer):
        numbers = []
        for part in match.group(1).split(","):
            limits = re.split(r"\s*[-–]\s*", part.strip())
            start = int(limits[0])
            stop = int(limits[-1])
            if stop < start or stop - start > 100:
                raise ValueError("Invalid/excessive citation range")
            numbers.extend(range(start, stop + 1))
        if len(numbers) != len(set(numbers)):
            raise ValueError("Duplicate source within citation token")
        result.extend(
            {"offset": match.start(), "source": n, "token": match.group()} for n in numbers
        )
    return result


def citation_key(c):
    return c["offset"], c["source"], c["token"]


def support_counts(items):
    counts = {"supported": 0, "unsupported": 0, "uncertain": 0}
    for value in items:
        if value is True:
            counts["supported"] += 1
        elif value is False:
            counts["unsupported"] += 1
        elif value is None:
            counts["uncertain"] += 1
        else:
            raise ValueError("Support judgment must be bool or null")
    n = sum(counts.values())
    return {
        **counts,
        "n": n,
        "lower_rate": ev.ratio(counts["supported"], n),
        "upper_rate": ev.ratio(counts["supported"] + counts["uncertain"], n),
    }


def refusal_stats(rows, enough):
    certain = [r for r in rows if type(enough(r)) is bool]
    sufficient = [r for r in certain if enough(r)]
    insufficient = [r for r in certain if not enough(r)]
    return {
        "sufficient_n": len(sufficient),
        "insufficient_n": len(insufficient),
        "uncertain_sufficiency_n": len(rows) - len(certain),
        "refusal_rate": ev.ratio(sum(r["abstained"] for r in rows), len(rows)),
        "false_refusal_rate": ev.ratio(sum(r["abstained"] for r in sufficient), len(sufficient)),
        "proper_refusal_rate": ev.ratio(
            sum(r["abstained"] for r in insufficient), len(insufficient)
        ),
    }


def audit_data(cases, gold, answers, reviews, gold_sha):
    expected = {(c["query_id"], a) for c in cases for a in ev.ARMS}
    by_answer = {(r["query_id"], r["arm"]): r for r in answers}
    by_review = {(r["query_id"], r["arm"]): r for r in reviews}
    if len(reviews) != len(expected) or set(by_review) != expected or set(by_answer) != expected:
        raise ValueError("Distinct paired answer/review cohort mismatch")
    findings, per_answer = [], []
    group_count = old_type_count = 0
    for case in cases:
        q = case["query_id"]
        for arm in ev.ARMS:
            row, review = by_answer[(q, arm)], by_review[(q, arm)]
            if review["answer_sha256"] != ev.digest(row) or review["gold_sha256"] != gold_sha:
                raise ValueError(f"Review binding mismatch: {q}/{arm}")
            if (
                review["conclusion"] not in ev.LABELS
                or type(review["abstained"]) is not bool
                or review["review"].get("status") != "ai_reviewed"
            ):
                raise ValueError(f"Expected AI review (not invented human approval): {q}/{arm}")
            if review["review"].get("human_verified") is not False:
                raise ValueError("AI provenance must explicitly deny human verification")
            legacy = ev.citations(row["answer"])
            expanded = expanded_citations(row["answer"])
            judged = review["citations"]
            keys = [citation_key(c) for c in judged]
            if len(keys) != len(set(keys)) or sorted(keys) != sorted(map(citation_key, expanded)):
                raise ValueError(f"Missing, duplicated or invented citation occurrence: {q}/{arm}")
            grouped = len(expanded) != len(legacy)
            group_count += grouped
            if grouped:
                findings.append(
                    {
                        "query_id": q,
                        "arm": arm,
                        "kind": "v1_parser_omits_group",
                        "legacy_count": len(legacy),
                        "expanded_count": len(expanded),
                    }
                )
            offsets = {c["offset"] for c in expanded}
            unknown = False
            for index, fact in enumerate(review["facts"]):
                span = fact.get("span")
                if (
                    not ev.text(span)
                    or span not in row["answer"]
                    or not set(fact["citation_offsets"]) <= offsets
                ):
                    raise ValueError(f"Invalid fact span/citation binding: {q}/{arm}/{index}")
                if "start" in fact and row["answer"][fact["start"] : fact["end"]] != span:
                    raise ValueError(f"Fact character offsets mismatch: {q}/{arm}/{index}")
                support_counts([fact["supported"]])
                unknown |= fact["supported"] is None
            old_type_count += unknown
            for c in judged:
                support_counts([c["supported"]])
                if not ev.text(c["statement"]):
                    raise ValueError("Citation needs corresponding statement")
                if not 1 <= c["source"] <= len(row["contexts"]) and c["supported"] is True:
                    raise ValueError("Out-of-range citation cannot be supported")
                if (
                    1 <= c["source"] <= len(row["contexts"])
                    and c.get("evidence_document_id")
                    and c["evidence_document_id"] != row["contexts"][c["source"] - 1]["document_id"]
                ):
                    raise ValueError("Citation evidence points to a different source")
            point_ids = {p["id"] for p in gold[q]["required_points"]}
            covered = review["covered_point_ids"]
            if len(covered) != len(set(covered)) or not set(covered) <= point_ids:
                raise ValueError("Unknown/duplicate covered point")
            pc = review.get("point_coverage", [])
            if pc and (
                len(pc) != len(point_ids)
                or {p["id"] for p in pc} != point_ids
                or {p["id"] for p in pc if p["status"] == "covered"} != set(covered)
            ):
                raise ValueError("Point coverage annotation disagrees with covered IDs")
            assessment = review["ai_assessment"]
            support_counts([assessment["conclusion_supported_by_context"]])
            support_counts([assessment["context_sufficient"]])
            if unknown:
                findings.append(
                    {
                        "query_id": q,
                        "arm": arm,
                        "kind": "tri_state_fact_judgment",
                        "unknown_facts": sum(f["supported"] is None for f in review["facts"]),
                    }
                )
            per_answer.append(
                {
                    "query_id": q,
                    "arm": arm,
                    "review_conclusion": review["conclusion"],
                    "reference_label": gold[q]["label"],
                    "reference_agreement": review["conclusion"] == gold[q]["label"],
                    "context_conclusion_support": assessment["conclusion_supported_by_context"],
                    "gold_disputed": assessment["gold_disputed"],
                    "flags": assessment.get("flags", []),
                    "ai_notes": review["review"]["notes"],
                }
            )
    summaries = {}
    for arm, name in ev.ARMS.items():
        rs = [by_review[(c["query_id"], arm)] for c in cases]
        facts = [f for r in rs for f in r["facts"]]
        cs = [c for r in rs for c in r["citations"]]
        legacy_keys = {
            (r["query_id"], citation_key(c))
            for r in answers
            if r["arm"] == arm
            for c in ev.citations(r["answer"])
        }
        legacy_cs = [
            c for r in rs for c in r["citations"] if (r["query_id"], citation_key(c)) in legacy_keys
        ]
        summaries[name] = {
            "n": len(rs),
            "reference_label_agreement": ev.ratio(
                sum(r["conclusion"] == gold[r["query_id"]]["label"] for r in rs), len(rs)
            ),
            "context_conclusion_support_ai": support_counts(
                [r["ai_assessment"]["conclusion_supported_by_context"] for r in rs]
            ),
            "fact_fragment_support_ai": support_counts([f["supported"] for f in facts]),
            "asserted_fragment_support_sensitivity": support_counts(
                [f["supported"] for f in facts if f.get("asserted") is not False]
            ),
            "nonasserted_fragments": sum(f.get("asserted") is False for f in facts),
            "citation_support_ai_grouped_v2": support_counts([c["supported"] for c in cs]),
            "citation_support_ai_legacy_v1_sensitivity": support_counts(
                [c["supported"] for c in legacy_cs]
            ),
            "reference_point_coverage": statistics.mean(
                len(r["covered_point_ids"]) / len(gold[r["query_id"]]["required_points"])
                for r in rs
            ),
            "refusal_using_reference_context_flags": refusal_stats(
                rs, lambda r, arm=arm: gold[r["query_id"]]["context_sufficient"][arm]
            ),
            "refusal_using_ai_context_flags_sensitivity": refusal_stats(
                rs, lambda r: r["ai_assessment"]["context_sufficient"]
            ),
            "disputed_reference_answers": sum(r["ai_assessment"]["gold_disputed"] for r in rs),
            "context_sufficiency_disagreements": sum(
                r["ai_assessment"]["context_sufficient"]
                != gold[r["query_id"]]["context_sufficient"][arm]
                for r in rs
            ),
        }
    deltas = [
        int(by_review[(c["query_id"], "B")]["conclusion"] == gold[c["query_id"]]["label"])
        - int(by_review[(c["query_id"], "A")]["conclusion"] == gold[c["query_id"]]["label"])
        for c in cases
    ]
    return {
        "status": "ai_audited_not_human_validated",
        "protocol": "ai_review_audit_v2",
        "quality_improvement_claim_supported": False,
        "human_gate_evaluated": False,
        "reviewed_answers": len(reviews),
        "reviewers": dict(Counter(r["review"]["reviewer"] for r in reviews)),
        "source_review_semantic_rechecks": (
            "See separate main-assistant sample record; not all 200 rejudged"
        ),
        "structural_audit_v2": "passed",
        "v1_group_parser_affected_answers": group_count,
        "v1_boolean_schema_affected_answers": old_type_count,
        "summaries": summaries,
        "findings": findings,
        "per_answer": per_answer,
        "paired_reference_agreement": {
            "delta": statistics.mean(deltas),
            "improved": deltas.count(1),
            "worsened": deltas.count(-1),
            "tied": deltas.count(0),
            "descriptive_95ci": ev.paired_interval(deltas),
        },
        "api_judge_calls_added": 0,
        "limits": [
            "AI reviews are not human or biomedical expert validation",
            "Earlier reviewer saw both arms and reference labels; not blind or independent",
            "Reference-label agreement is not established scientific conclusion accuracy",
            "Fact spans are sentence/clause fragments, not proven exhaustive atomic facts",
            "Uncertainty preserved as lower/upper bounds; bounds are not confidence intervals",
            "V2 grouped-citation denominator differs from frozen V1; both reported",
            "Used development cohort; no independent test or online generalization claim",
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--reviews", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    cases, gold, answers, plan, _ = load_verified(args.pack, args.answers)
    report = audit_data(cases, gold, answers, ev.read_rows(args.reviews), plan["gold_sha256"])
    report["inputs"] = {
        str(args.reviews): hash_file(args.reviews),
        str(args.pack / "sealed/gold.jsonl"): plan["gold_sha256"],
        str(args.answers / "answers.jsonl"): hash_file(args.answers / "answers.jsonl"),
    }
    ev.fresh(args.output)
    write_json(args.output / "audit.json", report)
    write_rows(args.output / "per_answer.jsonl", report["per_answer"])
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["per_answer", "findings"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
