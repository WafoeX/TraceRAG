"""Recompute A gates from explicit main-assistant AI annotations; no judge/API/B calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import evaluate_scifact_answers as ev
from scripts import prepare_scifact_generation_pilot as prep
from scripts.audit_scifact_ai_reviews import support_counts
from scripts.prepare_scifact import hash_file, write_json
from scripts.run_scifact_generation_pilot_a import load_inputs, summarize, verify_run

GRADES = {"acceptable", "unacceptable", "uncertain"}


def pair_change(baseline, candidate):
    if "uncertain" in (baseline, candidate):
        return "uncertain"
    if baseline == candidate:
        return "tied"
    return "improved" if candidate == "acceptable" else "regressed"


def check_review(answer, review):
    if (
        review["answer_sha256"] != ev.digest(answer)
        or review["review_kind"] != "main_assistant_ai"
        or review["human_verified"] is not False
        or review["independent_blind_review"] is not False
        or review["acceptability"] not in GRADES
    ):
        raise ValueError("AI provenance/grade/answer binding mismatch")
    for field in ("scope_requirements_pass", "new_unsupported_key_assertion", "context_sufficient"):
        if review[field] is not None and type(review[field]) is not bool:
            raise ValueError("Preserve tri-state decisions")
    if any(type(review[field]) is not bool for field in ("abstained", "format_compliant")):
        raise ValueError("Abstention and format annotations must be booleans")
    if not ev.text(review["notes"]):
        raise ValueError("AI judgments need a reason")
    if answer["status"] != "ok" and review["acceptability"] != "unacceptable":
        raise ValueError("Failed/truncated generation cannot count as acceptable")
    expected = {(c["offset"], c["source"], c["token"]) for c in answer["citations"]}
    judged = review["citation_assessments"]
    keys = [(c["offset"], c["source"], c["token"]) for c in judged]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("Every citation-source occurrence must be assessed exactly once")
    offsets = {c["offset"] for c in answer["citations"]}
    groups = {}
    for c in answer["citations"]:
        groups.setdefault((c["offset"], c["token"]), []).append(c["source"])
    joint = review["joint_citation_assessments"]
    if len(joint) != len(groups) or {(g["offset"], g["token"]) for g in joint} != set(groups):
        raise ValueError("Every citation group needs a separate joint-support assessment")
    for c in judged:
        support_counts([c["supported"]])
        if not ev.text(c["statement"]) or c["statement"] not in answer["answer"]:
            raise ValueError("Citation judgment must point to answer text")
        if c["supported"] is True and not 1 <= c["source"] <= len(answer["contexts"]):
            raise ValueError("Out-of-range source cannot support a statement")
    for g in joint:
        support_counts([g["supported"]])
        if sorted(g["sources"]) != sorted(groups[(g["offset"], g["token"])]):
            raise ValueError("Joint-citation sources changed")
    if answer["status"] == "ok" and not review["fragments"]:
        raise ValueError("Review material assertions, not just headlines or cited fragments")
    for f in review["fragments"]:
        if not ev.text(f["span"]) or f["span"] not in answer["answer"]:
            raise ValueError("Fragment must be a literal answer span")
        support_counts([f["supported"]])
        if not set(f["citation_offsets"]) <= offsets:
            raise ValueError("Fragment references an invented citation offset")
    if not review["condition_checks"]:
        raise ValueError("Record the material conditions and missing-evidence qualifications")
    for c in review["condition_checks"]:
        support_counts([c["addressed"]])
        if not ev.text(c["condition"]):
            raise ValueError("Condition description required")
    if review["scope_requirements_pass"] and not all(
        c["addressed"] for c in review["condition_checks"]
    ):
        raise ValueError("Scope cannot pass with a failed or uncertain material condition")
    for f in review["unsupported_key_spans"]:
        if f not in answer["answer"]:
            raise ValueError("Unsupported key assertion needs a literal span")
    if not review["unsupported_key_spans"] and review["new_unsupported_key_assertion"] is True:
        raise ValueError("A new unsupported key assertion requires evidence")


def evaluate_gate(rows, reviews, cases):
    keys = {(q, v) for q in cases for v in prep.VARIANTS}
    by_answer = {(r["query_id"], r["variant"]): r for r in rows}
    by_review = {(r["query_id"], r["variant"]): r for r in reviews}
    if (
        len(rows) != 24
        or len(reviews) != 24
        or len(cases) != 12
        or set(by_answer) != keys
        or set(by_review) != keys
        or Counter(c["pilot_role"] for c in cases.values())
        != {"target": 4, "control": 4, "ambiguity": 4}
    ):
        raise ValueError("Exactly the 12-case, 24-answer A cohort is required")
    for key in keys:
        check_review(by_answer[key], by_review[key])
    pairs = []
    for q, case in cases.items():
        a, b = (by_review[(q, v)] for v in prep.VARIANTS)
        pairs.append(
            {
                "query_id": q,
                "role": case["pilot_role"],
                "baseline": a["acceptability"],
                "candidate": b["acceptability"],
                "change": pair_change(a["acceptability"], b["acceptability"]),
                "candidate_scope_pass": b["scope_requirements_pass"],
                "candidate_new_unsupported_key_assertion": b["new_unsupported_key_assertion"],
            }
        )
    changes = Counter(p["change"] for p in pairs)
    targets = [p for p in pairs if p["role"] == "target"]
    controls = [p for p in pairs if p["role"] == "control"]
    criteria = {
        "at_least_3_of_4_targets": sum(
            bool(p["candidate_scope_pass"]) and p["candidate"] == "acceptable" for p in targets
        )
        >= 3,
        "all_4_controls_preserved": all(
            bool(p["candidate_scope_pass"])
            and p["candidate"] == "acceptable"
            and p["change"] != "regressed"
            for p in controls
        ),
        "at_least_1_clear_paired_improvement": changes["improved"] >= 1,
        "zero_clear_paired_regressions": changes["regressed"] == 0,
        "all_24_success_nontruncated": all(
            r["status"] == "ok" and r["finish_reason"] == "stop" for r in rows
        ),
        "no_new_candidate_unsupported_key_assertion": all(
            p["candidate_new_unsupported_key_assertion"] is False for p in pairs
        ),
    }
    metrics = {}
    for variant in prep.VARIANTS:
        rs = [by_review[(q, variant)] for q in cases]
        facts = [f for r in rs for f in r["fragments"]]
        supported_citations = {
            (r["query_id"], c["offset"])
            for r in rs
            for c in r["citation_assessments"]
            if c["supported"] is True
        }
        metrics[variant] = {
            "acceptability": dict(Counter(r["acceptability"] for r in rs)),
            "fragment_support_AI": support_counts([f["supported"] for f in facts]),
            "source_citation_support_AI": support_counts(
                [c["supported"] for r in rs for c in r["citation_assessments"]]
            ),
            "joint_citation_support_AI": support_counts(
                [c["supported"] for r in rs for c in r["joint_citation_assessments"]]
            ),
            "necessary_condition_addressing_AI": support_counts(
                [c["addressed"] for r in rs for c in r["condition_checks"]]
            ),
            "fragment_supported_citation_coverage": ev.ratio(
                sum(
                    any(
                        (r["query_id"], off) in supported_citations for off in f["citation_offsets"]
                    )
                    for r in rs
                    for f in r["fragments"]
                ),
                len(facts),
            ),
            "abstentions": sum(r["abstained"] for r in rs),
            "format_compliant": sum(r["format_compliant"] for r in rs),
        }
        for enough in (True, False, None):
            cohort = [r for r in rs if r["context_sufficient"] is enough]
            name = {True: "sufficient", False: "insufficient", None: "uncertain_sufficiency"}[
                enough
            ]
            metrics[variant][name] = {
                "n": len(cohort),
                "refusals": sum(r["abstained"] for r in cohort),
                "refusal_rate": ev.ratio(sum(r["abstained"] for r in cohort), len(cohort)),
            }
    return {
        "status": "AI_screen_gate_passed" if all(criteria.values()) else "AI_screen_gate_failed",
        "criteria": criteria,
        "gate_pass": all(criteria.values()),
        "pairs": pairs,
        "paired_changes": {k: changes[k] for k in ["improved", "regressed", "tied", "uncertain"]},
        "metrics": metrics,
        "B_executed": False,
        "human_validated": False,
        "independent_blind_review": False,
        "quality_generalization_or_resume_improvement_proven": False,
        "application_judge_API_calls": 0,
        "main_session_AI_review_cost_cny": None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reviews", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    _, cases, _ = load_inputs(args.plan)
    rows = verify_run(args.plan, args.run)
    result = evaluate_gate(rows, ev.read_rows(args.reviews), cases)
    result["generation"] = summarize(rows)
    result["review_sha256"] = hash_file(args.reviews)
    result["source_run_manifest_sha256"] = hash_file(args.run / "run_manifest.json")
    result["source_plan_manifest_sha256"] = hash_file(args.plan / "manifest.json")
    ev.fresh(args.output)
    write_json(args.output / "assessment.json", result)
    print(json.dumps({k: result[k] for k in ["status", "criteria", "paired_changes"]}))


if __name__ == "__main__":
    main()
