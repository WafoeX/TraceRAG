"""Offline verification, paired development statistics and application raw-response replay."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.provider_costs import CostTracker
from scripts import reduce_scifact_fallbacks as experiment
from scripts.evaluate_scifact import read_object, read_rows
from scripts.prepare_scifact import hash_file, write_json
from scripts.retrieval_metrics import aggregate_chunks


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_stats(before, after, ids, iterations=10000, seed=20261020):
    """Whole-query pairs, frozen order, no sorting IDs; positive fallback delta = reduction."""
    old, new = before["per_query"], after["per_query"]
    quality = [new[q]["nDCG@10"] - old[q]["nDCG@10"] for q in ids]
    fallback = [float(q in before["fallback_ids"]) - float(q in after["fallback_ids"]) for q in ids]
    rng = random.Random(seed)
    q_samples, f_samples = [], []
    for _ in range(iterations):
        selected = rng.choices(range(len(ids)), k=len(ids))
        q_samples.append(math.fsum(quality[i] for i in selected) / len(ids))
        f_samples.append(math.fsum(fallback[i] for i in selected) / len(ids))
    q_interval = [percentile(q_samples, p) for p in (0.025, 0.975)]
    f_interval = [percentile(f_samples, p) * 100 for p in (0.025, 0.975)]
    return {
        "queries": len(ids),
        "seed": seed,
        "iterations": iterations,
        "interval_method": "paired-query bootstrap percentile, linear interpolation",
        "nDCG_delta": math.fsum(quality) / len(ids),
        "nDCG_delta_95pct": q_interval,
        "quality_point_guard_passed": math.fsum(quality) / len(ids) >= -0.01,
        "quality_interval_lower_above_minus_001": q_interval[0] > -0.01,
        "fallback_reduction_pp": math.fsum(fallback) / len(ids) * 100,
        "fallback_reduction_pp_95pct": f_interval,
        "ndcg_improved": sum(v > 0 for v in quality),
        "ndcg_worsened": sum(v < 0 for v in quality),
        "ndcg_tied": sum(v == 0 for v in quality),
        "fallback_recovered": sum(v > 0 for v in fallback),
        "fallback_new": sum(v < 0 for v in fallback),
        "scope": "descriptive, selected on used development; not held-out/test inference",
    }


def replay_application(rows, cases, data, chunks, pricing):
    from openai.types.chat import ChatCompletion

    from app.config import get_settings
    from app.services import reranking, siliconflow
    from app.services.retrieval import RetrievedChunk

    by_id = {c["query_id"]: c for c in cases}
    tracker = CostTracker(pricing)
    for row in rows:
        case = by_id[row["query_id"]]
        settings = get_settings().model_copy(
            update={
                "rerank_prompt_version": row["mode"],
                "llm_rerank_enable_thinking": False,
                "rerank_passage_mode": "prefix_600",
            }
        )
        candidates = []
        for i, item in enumerate(case["fused"][:10]):
            chunk = chunks[item["chunk_id"]]
            title = data["corpus"][chunk["document_id"]]["title"]
            candidates.append(RetrievedChunk(i, 1, title, i, chunk["text"], item["score"]))
        seen = []

        def create(_seen=seen, _case=case, _row=row, **kwargs):
            _seen.append(kwargs)
            expected = {
                **experiment.payload_for(_case, _row["mode"]),
                "extra_body": {"enable_thinking": False},
            }
            assert kwargs == expected
            if _row["call"]["status"] != "ok":
                raise TimeoutError("saved_request_failure")
            return ChatCompletion.model_validate(_row["call"]["response"])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with (
            patch.object(reranking, "get_settings", return_value=settings),
            patch.object(siliconflow, "get_settings", return_value=settings),
            patch.object(reranking, "get_client", return_value=client),
            patch.object(siliconflow, "get_cost_tracker", return_value=tracker),
        ):
            ranked = reranking.rerank(case["question"], candidates, 10)
        actual_ids = [case["fused"][c.chunk_id]["chunk_id"] for c in ranked]
        assert actual_ids == [c["chunk_id"] for c in row["chunks"][:10]]
        assert len(seen) == 1
    return len(rows)


def verify_run(folder, data, chunks, all_cases, baseline, reused=None):
    report = read_object(folder / "report.json")
    cases = all_cases[:100] if report["stage"] == "screen" else all_cases
    ids = [c["query_id"] for c in cases]
    assert ids == report["cohort_query_ids"]
    rows = read_rows(folder / "query_results.jsonl")
    pairs = {(r["query_id"], r["mode"]) for r in rows}
    assert len(rows) == len(pairs) == len(ids) * len(report["versions"])
    assert pairs == {(q, v) for q in ids for v in report["versions"]}
    events = read_rows(folder / "cost_events.jsonl")
    starts = {e["request_id"]: e for e in events if e["event"] == "started"}
    ends = {
        e["request_id"]: {k: v for k, v in e.items() if k != "event"}
        for e in events
        if e["event"] == "finished"
    }
    assert len(events) == 2 * len(ends)
    assert len(starts) == len(ends) == report["new_calls_planned"]
    assert starts.keys() == ends.keys()
    reuse_by_pair = {(r["query_id"], r["mode"]): r for r in (reused or [])}
    used_reuse = set()
    by_id = {c["query_id"]: c for c in cases}
    for row in rows:
        case, call = by_id[row["query_id"]], row["call"]
        request = {**experiment.payload_for(case, row["mode"]), "enable_thinking": False}
        assert call["request"] == request
        if call["request_id"] in ends:
            assert call == ends[call["request_id"]]
            assert starts[call["request_id"]]["request"] == request
        else:
            key = (row["query_id"], row["mode"])
            assert row == reuse_by_pair[key]
            used_reuse.add(key)
        scores, reason = experiment.decode(call, 10)
        expected = (
            case["fused"] if reason else experiment.full.experiment.reorder(case["fused"], scores)
        )
        assert reason == row["fallback"]
        assert expected == row["chunks"]
        assert row["chunks"][10:] == case["fused"][10:]
        assert {r["chunk_id"] for r in row["chunks"][:10]} == {
            r["chunk_id"] for r in case["fused"][:10]
        }
        if call.get("usage") is None:
            assert call["estimated_cost_cny"] is None
        else:
            usage = call["usage"]
            cost = (
                Decimal(usage["prompt_tokens"]) * Decimal(call["price_input_per_million_cny"])
                + Decimal(usage["completion_tokens"])
                * Decimal(call["price_output_per_million_cny"])
            ) / Decimal(1000000)
            assert cost == Decimal(call["estimated_cost_cny"])
    assert len(used_reuse) == report["reused_calls"]
    tracker = CostTracker(folder / "pricing.json")
    tracker.records = list(ends.values())
    assert tracker.summary() == report["new_costs"]
    for version in report["versions"]:
        score = experiment.summarize(
            [r for r in rows if r["mode"] == version], cases, data, chunks, folder / "pricing.json"
        )
        assert score == report["summaries"][version]
        assert score["fewer_than_100_unique_documents"] == 0
    assert (
        experiment.summarize(
            [r for r in baseline if r["query_id"] in set(ids)],
            cases,
            data,
            chunks,
            folder / "pricing.json",
        )
        == report["baseline"]
    )
    for p, digest in report["frozen"].items():
        assert (
            data["report_sha256"]
            if p == "preparation_report_sha256"
            else hash_file(experiment.ROOT / p)
        ) == digest
    for p, digest in report["source_hashes"].items():
        assert hash_file(folder / "sources" / p) == digest
        assert hash_file(experiment.ROOT / p) == digest
    replayed = replay_application(rows, cases, data, chunks, folder / "pricing.json")
    result = {
        "status": "passed",
        "new_started_finished": len(ends),
        "raw_response_rows_verified": len(rows),
        "reused_rows_verified": len(used_reuse),
        "application_replay_rows": replayed,
        "http_sql_blocked": True,
        "costs_metrics_head_tail_hashes_verified": True,
    }
    write_json(folder / "verification.json", result)
    return report, rows


def analyze(screen_folder, expanded_folder=None):
    data, chunks, cases, baseline = experiment.inputs()
    screen, screen_rows = verify_run(screen_folder, data, chunks, cases, baseline)
    selected = experiment.choose(screen["summaries"], screen["baseline"])
    assert selected == screen["selected_version"]
    result = {
        "screen_verification": "passed",
        "selected_version": selected,
        "test_queries_used": 0,
        "new_costs_screen": screen["new_costs"],
    }
    if expanded_folder is None:
        return result
    report, rows = verify_run(expanded_folder, data, chunks, cases, baseline, screen_rows)
    assert report["versions"] == [selected]
    cohorts = {"screen100": cases[:100], "remaining707": cases[100:], "full807": cases}
    for name, cohort in cohorts.items():
        ids = [c["query_id"] for c in cohort]
        old_rows = [r for r in baseline if r["query_id"] in set(ids)]
        new_rows = [r for r in rows if r["query_id"] in set(ids)]
        old = experiment.summarize(old_rows, cohort, data, chunks, expanded_folder / "pricing.json")
        new = experiment.summarize(new_rows, cohort, data, chunks, expanded_folder / "pricing.json")
        old["fallback_ids"] = [r["query_id"] for r in old_rows if r["fallback"]]
        new["fallback_ids"] = [r["query_id"] for r in new_rows if r["fallback"]]
        stats = paired_stats(old, new, ids)
        result[name] = {"baseline": old, "candidate": new, "paired_statistics": stats}
    old_all = {r["query_id"]: r for r in baseline}
    by_id = {c["query_id"]: c for c in cases}
    mapping = {k: c["document_id"] for k, c in chunks.items()}
    all_zero_ids = [q for q, row in old_all.items() if row["fallback"] == "ValueError"]
    evidence = sum(
        any(mapping[c["chunk_id"]] in data["qrels"][q] for c in by_id[q]["fused"][:10])
        for q in all_zero_ids
    )
    transitions = Counter(
        (old_all[r["query_id"]]["fallback"] or "none", r["fallback"] or "none") for r in rows
    )
    result["diagnosis"] = {
        "baseline_all_zero": len(all_zero_ids),
        "all_zero_with_judged_relevant_document_in_head": evidence,
        "limitation": "qrels are document-level, not prefix entailment labels",
        "fallback_transitions": {f"{a}->{b}": n for (a, b), n in transitions.items()},
    }
    changes = []
    for row in rows:
        q = row["query_id"]
        before = result["full807"]["baseline"]["per_query"][q]["nDCG@10"]
        after = result["full807"]["candidate"]["per_query"][q]["nDCG@10"]
        changes.append(
            {
                "query_id": q,
                "nDCG_delta": after - before,
                "before_fallback": old_all[q]["fallback"],
                "after_fallback": row["fallback"],
                "baseline_top3": aggregate_chunks(old_all[q]["chunks"], mapping)[:3],
                "candidate_top3": aggregate_chunks(row["chunks"], mapping)[:3],
            }
        )
    result["examples_by_delta"] = {
        "best5": sorted(changes, key=lambda c: -c["nDCG_delta"])[:5],
        "worst5": sorted(changes, key=lambda c: c["nDCG_delta"])[:5],
    }
    tracker = CostTracker(expanded_folder / "pricing.json")
    tracker.records = [r["call"] for r in screen_rows] + [
        r["call"] for r in rows if r["query_id"] not in {c["query_id"] for c in cases[:100]}
    ]
    result["all_new_costs"] = tracker.summary()
    result["new_calls"] = len(tracker.records)
    assert result["new_calls"] == 907
    write_json(expanded_folder / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--expanded", type=Path)
    args = parser.parse_args()
    with (
        patch("httpx.Client.send", side_effect=AssertionError("HTTP forbidden")),
        patch("sqlalchemy.engine.Engine.connect", side_effect=AssertionError("SQL forbidden")),
    ):
        result = analyze(args.screen, args.expanded)
    print(
        {
            "verification": "passed",
            "selected": result["selected_version"],
            "full_statistics": result.get("full807", {}).get("paired_statistics"),
            "new_costs": result.get("all_new_costs", result["new_costs_screen"]),
        }
    )


if __name__ == "__main__":
    main()
