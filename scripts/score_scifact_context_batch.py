"""Validate explicit AI context annotations; aggregate a bounded batch without a judge API."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .complete_scifact_six_metrics import (
    ROOT,
    digest,
    parse_object,
    require,
    rows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack, annotations, output = (
        args.pack.resolve(),
        args.annotations.resolve(),
        args.output.resolve(),
    )
    require(0 <= args.start < args.stop <= 100, "Invalid cohort slice")
    require(
        output.is_relative_to(ROOT / "runs") and not output.exists(), "Fresh run output required"
    )
    manifest = parse_object((pack / "manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        require(digest((pack / name).read_bytes()) == expected, "Frozen pack hash mismatch")
    protocol_hash = digest((pack / "protocol.md").read_bytes())
    source = parse_object(annotations.read_text())
    require(source["reviewer"] == "main_assistant_AI", "Explicit AI provenance required")
    require(
        source["human_verified"] is False and source["independent_blind_review"] is False,
        "Do not invent independent/human review",
    )
    pending = [
        r
        for r in rows(pack / "contexts.pending.jsonl")
        if args.start <= r["query_position"] < args.stop
    ]
    expected = {(r["query_id"], r["document_id"]): r for r in pending}
    require(len(expected) == len(pending), "Multiple chunks per query-document: use chunk IDs")
    labels = {(r["query_id"], r["document_id"]): r for r in source["context_reviews"]}
    require(
        len(labels) == len(source["context_reviews"]) and set(labels) == set(expected),
        "Batch must cover every unique query-chunk exactly once",
    )
    reviewed = []
    for key, row in expected.items():
        label = labels[key]
        grade = label["grade"]
        require(grade is None or (type(grade) is int and grade in (0, 1, 2)), "Invalid grade")
        require(
            isinstance(label["reason"], str) and bool(label["reason"].strip()), "Reason required"
        )
        quote = label["quote"]
        require(isinstance(quote, str) and bool(quote), "Nonempty evidence/topic quote required")
        start = row["text"].find(quote)
        require(start >= 0, "Quote not in actual source text")
        reviewed.append(
            {
                **row,
                "review_status": "ai_reviewed",
                "grade": grade,
                "reason": label["reason"],
                "reviewer": source["reviewer"],
                "protocol_sha256": protocol_hash,
                "evidence_spans": [{"start": start, "end": start + len(quote), "quote": quote}],
            }
        )
    cohort = list(dict.fromkeys(r["query_id"] for r in pending))
    require(len(cohort) == args.stop - args.start, "Missing batch queries")
    sufficiency = {}
    for label in source["sufficiency_reviews"]:
        require(
            label["query_id"] in cohort
            and label["label"] in ("sufficient", "insufficient", "uncertain"),
            "Invalid sufficiency label",
        )
        require(
            isinstance(label["reason"], str) and bool(label["reason"].strip()), "Reason required"
        )
        for arm in label["arms"]:
            key = (label["query_id"], arm)
            require(arm in ("A", "B") and key not in sufficiency, "Invalid/duplicate arm")
            sufficiency[key] = label
    require(
        set(sufficiency) == {(q, arm) for q in cohort for arm in ("A", "B")},
        "Incomplete sufficiency coverage",
    )
    per_query = []
    for q in cohort:
        for arm in ("A", "B"):
            group = [
                r
                for r in reviewed
                if r["query_id"] == q and any(p["arm"] == arm for p in r["positions"])
            ]
            require(len(group) == 4, "Need actual four chunks, not four unique documents")
            grades = [r["grade"] for r in group]
            unknown = sum(g is None for g in grades)
            related = sum(g is not None and g >= 1 for g in grades)
            useful = sum(g == 2 for g in grades)
            total = sum(g for g in grades if g is not None)
            per_query.append(
                {
                    "query_id": q,
                    "arm": arm,
                    "chunks": 4,
                    "metrics": {
                        "related_precision_lower": related / 4,
                        "related_precision_upper": (related + unknown) / 4,
                        "useful_precision_lower": useful / 4,
                        "useful_precision_upper": (useful + unknown) / 4,
                        "normalized_grade_lower": total / 8,
                        "normalized_grade_upper": (total + 2 * unknown) / 8,
                    },
                    "context_sufficiency": sufficiency[q, arm]["label"],
                    "context_sufficiency_reason": sufficiency[q, arm]["reason"],
                }
            )
    summaries = {}
    for arm in ("A", "B"):
        group = [r for r in per_query if r["arm"] == arm]
        counts = Counter(r["context_sufficiency"] for r in group)
        summaries[arm] = {
            "queries_reviewed": len(group),
            "metrics": {
                m: math.fsum(r["metrics"][m] for r in group) / len(group)
                for m in group[0]["metrics"]
            },
            "sufficiency_counts": {
                k: counts[k] for k in ("sufficient", "insufficient", "uncertain")
            },
            "sufficiency_lower": counts["sufficient"] / len(group),
            "sufficiency_upper": (counts["sufficient"] + counts["uncertain"]) / len(group),
        }
    inputs = {
        str(annotations.relative_to(ROOT)): digest(annotations.read_bytes()),
        str((pack / "manifest.json").relative_to(ROOT)): digest(
            (pack / "manifest.json").read_bytes()
        ),
        str(Path(__file__).relative_to(ROOT)): digest(Path(__file__).read_bytes()),
    }
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "partial_context_AI_review",
        "start": args.start,
        "stop": args.stop,
        "query_ids": cohort,
        "total_cohort_queries": 100,
        "queries_reviewed": len(cohort),
        "unique_contexts_reviewed": len(reviewed),
        "context_positions_reviewed": 8 * len(cohort),
        "summaries": summaries,
        "answer_reviews_completed": 0,
        "reviewer": source["reviewer"],
        "human_verified": False,
        "independent_blind_review": False,
        "protocol_sha256": protocol_hash,
        "new_model_calls": 0,
        "database_requests": 0,
        "inputs_sha256": inputs,
        "limitation": "Bounded development batch only; not full100 or generalization evidence",
    }
    write_json(output / "report.json", result)
    for name, data in (("contexts.reviewed.jsonl", reviewed), ("per_query.jsonl", per_query)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
    (output / "annotations.json").write_bytes(annotations.read_bytes())
    (output / "aggregator_source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        output / "verification.json",
        {
            "status": "passed",
            "batch_coverage_exact": True,
            "all_quotes_match": True,
            "source_pack_hashes_verified": True,
            "reviewed_query_count": len(cohort),
            "reviewed_query_chunk_pairs": len(reviewed),
            "protocol_sha256": protocol_hash,
            "outputs_sha256": {
                p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()
            },
            "inputs_sha256": inputs,
        },
    )
    print(json.dumps({"queries": len(cohort), "contexts": len(reviewed), "summaries": summaries}))


if __name__ == "__main__":
    main()
