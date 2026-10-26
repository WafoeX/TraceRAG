"""Bounded development-only prompt ablation: 200 screen calls, <=707 expansion calls.

Plan by default. Shared application prompts/parsers, fixed candidates and all-zero
fallback, no retries or label-aware requests. See SCIFACT_FALLBACK_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

from app.config import get_settings
from app.services.provider_costs import CostTracker
from app.services.rerank_prompts import VERSIONS, build_messages, parse_scores
from app.services.reranking import thinking_controls
from app.services.siliconflow import chat_content
from scripts import validate_scifact_nonthinking as full
from scripts.evaluate_scifact import read_object, read_rows
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.run_scifact_baseline import append

ROOT = full.ROOT
BASELINE = ROOT / "runs/scifact-nonthinking-development807-v1"
PROTOCOL = ROOT / "docs/refactor/SCIFACT_FALLBACK_PROTOCOL.md"


def inputs():
    data, chunks, cases, _ = full.full_inputs(
        full.experiment.SOURCE, full.experiment.CHUNKS, full.RETRIEVAL, full.REUSE
    )
    baseline = read_rows(BASELINE / "query_results.jsonl")
    if len(baseline) != 807 or {r["query_id"] for r in baseline} != set(data["queries"]):
        raise ValueError("Incomplete frozen non-thinking baseline")
    by_id = {r["query_id"]: r for r in baseline}
    for case in cases:
        row = by_id[case["query_id"]]
        if row["call"]["request"] != {**case["payload"], "enable_thinking": False}:
            raise ValueError("Frozen baseline request mismatch")
    return data, chunks, cases, baseline


def payload_for(case, version):
    return {
        **copy.deepcopy(case["payload"]),
        "messages": build_messages(case["question"], case["documents"], version),
    }


def decode(call, n):
    """Canonical replay path. No score padding, retries, coercion or forced positive values."""
    if call["status"] != "ok":
        return None, "request_error"
    from openai.types.chat import ChatCompletion

    from app.services.rerank_prompts import ScoreContractError
    from app.services.siliconflow import ProviderResponseError

    try:
        response = ChatCompletion.model_validate(call["response"])
        scores = parse_scores(chat_content(response), n)
    except (ValueError, KeyError, ScoreContractError, ProviderResponseError):
        return None, "score_contract_error"
    if not any(s > 0 for s in scores):
        return scores, "all_zero_scores"
    return scores, None


def measure(case, version, client, tracker):
    full.experiment.CONTEXT.set(
        {"query_id": case["query_id"], "mode": version, "phase": "rerank_only"}
    )
    payload = payload_for(case, version)
    started = time.perf_counter()
    # Tracker records sanitized errors and unknown costs; never omit a query.
    with suppress(Exception):
        tracker.call(
            "llm_rerank",
            payload["model"],
            lambda: client.chat.completions.create(
                **payload, extra_body={"enable_thinking": False}
            ),
            request={**payload, "enable_thinking": False},
        )
    elapsed = (time.perf_counter() - started) * 1000
    record = next(
        r
        for r in reversed(tracker.records)
        if r.get("query_id") == case["query_id"] and r.get("mode") == version
    )
    scores, fallback = decode(record, 10)
    predictions = case["fused"] if fallback else full.experiment.reorder(case["fused"], scores)
    return {
        "query_id": case["query_id"],
        "mode": version,
        "chunks": predictions,
        "rerank_ms": elapsed,
        "fallback": fallback,
        "call": record,
    }


def summarize(rows, cases, data, chunks, pricing):
    from collections import Counter

    score = full.summary(rows, cases, data, chunks, pricing)
    by_id = {c["query_id"]: c for c in cases}
    score["fallback_reasons"] = dict(Counter(r["fallback"] for r in rows if r["fallback"]))
    score["unchanged_head_order"] = sum(
        [x["chunk_id"] for x in r["chunks"][:10]]
        == [x["chunk_id"] for x in by_id[r["query_id"]]["fused"][:10]]
        for r in rows
    )
    score["valid_unchanged_head_order"] = sum(
        not r["fallback"]
        and [x["chunk_id"] for x in r["chunks"][:10]]
        == [x["chunk_id"] for x in by_id[r["query_id"]]["fused"][:10]]
        for r in rows
    )
    return score


def choose(scores, baseline):
    """Pre-registered screen: guard quality, then fewer fallbacks, then higher nDCG."""
    eligible = [
        v
        for v in VERSIONS
        if scores[v]["metrics"]["nDCG@10"] >= baseline["metrics"]["nDCG@10"] - 0.01
        and scores[v]["fallbacks"] < baseline["fallbacks"]
    ]
    return (
        min(eligible, key=lambda v: (scores[v]["fallbacks"], -scores[v]["metrics"]["nDCG@10"], v))
        if eligible
        else None
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "expand"], required=True)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    settings = get_settings()
    if settings.model_provider != "siliconflow" or thinking_controls(settings) != {
        "enable_thinking": False
    }:
        raise ValueError("Explicit SiliconFlow non-thinking configuration required")
    data, chunks, all_cases, baseline = inputs()
    if any(
        c["payload"]["model"] != (settings.llm_rerank_model or settings.openai_chat_model)
        for c in all_cases
    ):
        raise ValueError("Baseline model changed")
    paths = [
        BASELINE / "query_results.jsonl",
        BASELINE / "cost_events.jsonl",
        BASELINE / "pricing.json",
        full.RETRIEVAL / "fts_rrf.predictions.jsonl",
        full.experiment.CHUNKS / "chunks.jsonl",
        PROTOCOL,
        ROOT / ".env",
    ]
    frozen = {str(p.relative_to(ROOT)): hash_file(p) for p in paths}
    frozen["preparation_report_sha256"] = data["report_sha256"]
    rows = []
    if args.stage == "screen":
        cases, versions = all_cases[:100], list(VERSIONS)
    else:
        if args.screen is None:
            raise ValueError("Expansion must reference a completed screen")
        screen = read_object(args.screen / "report.json")
        if screen["stage"] != "screen" or screen["frozen"] != frozen:
            raise ValueError("Screen inputs changed")
        selected = choose(screen["summaries"], screen["baseline"])
        if selected is None or selected != screen["selected_version"]:
            raise ValueError("No pre-registered eligible version")
        cases, versions = all_cases, [selected]
        rows = [r for r in read_rows(args.screen / "query_results.jsonl") if r["mode"] == selected]
        if len(rows) != 100 or {r["query_id"] for r in rows} != {
            c["query_id"] for c in all_cases[:100]
        }:
            raise ValueError("Reuse must contain exactly the fixed first 100")
        by_id = {c["query_id"]: c for c in all_cases}
        for row in rows:
            expected = {**payload_for(by_id[row["query_id"]], selected), "enable_thinking": False}
            if row["call"]["request"] != expected:
                raise ValueError("Prompt changed after screen")
    reused = {(r["query_id"], r["mode"]) for r in rows}
    todo = [(c, v) for c in cases for v in versions if (c["query_id"], v) not in reused]
    args.output.mkdir(parents=True)
    pricing = args.output / "pricing.json"
    pricing.write_bytes((BASELINE / "pricing.json").read_bytes())
    resolved = {
        "stage": args.stage,
        "versions": versions,
        "split": "development",
        "cohort_query_ids": [c["query_id"] for c in cases],
        "frozen": frozen,
        "new_calls_planned": len(todo),
        "reused_calls": len(rows),
        "workers": 4,
        "sdk_retries": 0,
        "timeout_seconds": 60,
        "test_queries_used": 0,
        "database_requests": 0,
        "embedding_requests": 0,
        "generation_requests": 0,
        "rerank_depth": 10,
        "passage_mode": "prefix_600",
        "thinking": False,
        "application_defaults_changed": False,
        "screen_source": str(args.screen) if args.screen else None,
    }
    sources = [
        Path(__file__),
        ROOT / "app/services/rerank_prompts.py",
        ROOT / "app/services/reranking.py",
        ROOT / "app/config.py",
    ]
    resolved["source_hashes"] = {str(p.relative_to(ROOT)): hash_file(p) for p in sources}
    for p in sources:
        destination = args.output / "sources" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    write_json(args.output / "resolved_config.json", resolved)
    write_json(args.output / "status.json", {"status": "planned", **resolved})
    if not args.execute:
        print(f"Planned {len(todo)} calls; no API calls")
        return 0
    from openai import OpenAI

    tracker = CostTracker(
        pricing,
        log_path=args.output / "cost_events.jsonl",
        secret=settings.openai_api_key,
        capture_payloads=True,
        context=lambda: (full.experiment.CONTEXT.get() or {}).copy(),
    )
    tracker.profile("llm_rerank", cases[0]["payload"]["model"])
    write_rows(args.output / "query_results.jsonl", rows)
    started = time.perf_counter()
    with (
        OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_retries=0,
            timeout=60,
        ) as client,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):

        def run(item):
            return measure(item[0], item[1], client, tracker)

        for row in pool.map(run, todo):
            rows.append(row)
            append(args.output / "query_results.jsonl", row)
            write_json(args.output / "cost_summary_new_calls.json", tracker.summary())
            write_json(args.output / "status.json", {"status": "running", "rows": len(rows)})
            print(
                f"{len(rows)}/{len(cases) * len(versions)} {row['mode']} "
                f"fallback={row['fallback']}",
                flush=True,
            )
    ids = {c["query_id"] for c in cases}
    if len(rows) != len(cases) * len(versions) or any(
        {r["query_id"] for r in rows if r["mode"] == v} != ids for v in versions
    ):
        raise ValueError("Incomplete/duplicate cohort")
    summaries = {
        v: summarize([r for r in rows if r["mode"] == v], cases, data, chunks, pricing)
        for v in versions
    }
    base = summarize([r for r in baseline if r["query_id"] in ids], cases, data, chunks, pricing)
    for p, digest in frozen.items():
        if p != "preparation_report_sha256" and hash_file(ROOT / p) != digest:
            raise ValueError("Frozen inputs changed")
    selected = choose(summaries, base) if args.stage == "screen" else versions[0]
    report = {
        "status": "complete",
        **resolved,
        "summaries": summaries,
        "baseline": base,
        "selected_version": selected,
        "new_costs": tracker.summary(),
        "wall_seconds": time.perf_counter() - started,
        "latency_scope": "rerank only; historical baseline; not online E2E",
    }
    write_json(args.output / "report.json", report)
    write_json(args.output / "status.json", {"status": "complete", "rows": len(rows)})
    print(f"Complete: {args.output}; selected={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
