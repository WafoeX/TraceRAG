"""Fixed-candidate development rerank experiments; no DB, embedding or test scoring.

Plan by default. Replay original public-benchmark chat messages unchanged, with
only explicit thinking controls varied. Native reranker sees the same title and
600-character passage prefix. Keep the original tail and document metric policy.
"""

from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from scripts.baseline_support import ROOT
from scripts.evaluate_scifact import DEFAULT_PREPARED, load_prepared, read_object, read_rows
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.retrieval_metrics import aggregate_chunks, evaluate_rankings
from scripts.run_scifact_baseline import append, latency_summary

CONTROLS = {
    "llm_low": {"reasoning_effort": "low"},
    "llm_no_thinking": {"enable_thinking": False},
    "native": {},
}
CONTEXT: ContextVar[dict | None] = ContextVar("rerank_experiment", default=None)
SOURCE = ROOT / "runs/scifact-baseline-development-v1"
CHUNKS = ROOT / "runs/scifact-evaluation-plan-v1"


def inputs(source: Path, chunk_plan: Path, limit: int):
    data = load_prepared(DEFAULT_PREPARED, "development")
    plan = read_object(chunk_plan / "status.json")
    if (
        hash_file(chunk_plan / "chunks.jsonl") != plan["manifest"]["chunks.jsonl"]["sha256"]
        or plan["source_preparation_report_sha256"] != data["report_sha256"]
    ):
        raise ValueError("Frozen chunk/source mismatch")
    chunks = {r["chunk_id"]: r for r in read_rows(chunk_plan / "chunks.jsonl")}
    results = {r["query_id"]: r for r in read_rows(source / "query_results.jsonl")}
    events = read_rows(source / "cost_events.jsonl")
    starts = {
        r["query_id"]: r for r in events if r["event"] == "started" and r["role"] == "llm_rerank"
    }
    ends = {r["request_id"]: r for r in events if r["event"] == "finished"}
    selected = list(data["queries"])[:limit]
    cases = []
    for q in selected:
        original = results[q]
        fused = original["predictions"]["fts_rrf"]
        if len(fused) < 10:
            raise ValueError("Expected frozen top-10")
        documents = []
        for row in fused[:10]:
            chunk = chunks[row["chunk_id"]]
            title = data["corpus"][chunk["document_id"]]["title"]
            documents.append(f'(from "{title}") ' + chunk["text"][:600])
        payload = starts[q]["request"]
        listing = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents))
        if (
            f"Question: {data['queries'][q]}\n\nPassages:\n{listing}\n\n"
            not in payload["messages"][1]["content"]
        ):
            raise ValueError("Saved prompt and frozen candidate text mismatch")
        cases.append(
            {
                "query_id": q,
                "question": data["queries"][q],
                "fused": fused,
                "documents": documents,
                "payload": payload,
                "original": original,
                "baseline_call": ends[starts[q]["request_id"]],
            }
        )
    return data, chunks, cases


def reorder(fused, scores):
    """Stable score sort on the head only; never sort tail by incomparable scores."""
    head = [
        {"chunk_id": row["chunk_id"], "score": float(score)}
        for row, score in zip(fused[:10], scores, strict=True)
    ]
    return sorted(head, key=lambda row: row["score"], reverse=True) + fused[10:]


def measure(case, mode, client, tracker, settings):
    from app.services import reranking, siliconflow

    q, fused = case["query_id"], case["fused"]
    CONTEXT.set({"query_id": q, "mode": mode, "phase": "rerank_only"})
    started = time.perf_counter()
    fallback, predictions = None, fused
    try:
        if mode == "native":
            ordered = siliconflow.native_rerank(case["question"], case["documents"], 10)
            head = [{"chunk_id": fused[i]["chunk_id"], "score": score} for i, score in ordered]
            predictions = head + fused[10:]
        else:
            payload = copy.deepcopy(case["payload"])
            payload.update(CONTROLS[mode])
            base = case["payload"]
            response = tracker.call(
                "llm_rerank",
                base["model"],
                lambda: client.chat.completions.create(**base, extra_body=CONTROLS[mode]),
                request=payload,
            )
            content = siliconflow.chat_content(response)
            siliconflow.validate_llm_scores(content, 10)
            scores = reranking._parse_scores(content, 10)
            if not scores or not any(s > 0 for s in scores):
                raise ValueError("nonpositive_scores")
            predictions = reorder(fused, scores)
    except Exception as exc:
        fallback = type(exc).__name__
    elapsed = (time.perf_counter() - started) * 1000
    # Context is per worker; cost records contain no credential-bearing errors.
    record = next(
        r for r in reversed(tracker.records) if r.get("query_id") == q and r.get("mode") == mode
    )
    return {
        "query_id": q,
        "mode": mode,
        "chunks": predictions,
        "rerank_ms": elapsed,
        "fallback": fallback,
        "call": record,
    }


def summarize(rows, cases, data, chunks, pricing):
    from app.services.provider_costs import CostTracker

    selected = [c["query_id"] for c in cases]
    labels = {q: data["qrels"][q] for q in selected}
    mapping = {key: value["document_id"] for key, value in chunks.items()}
    summaries = {}
    base_by_mode = {
        "fts_rrf": [(c["query_id"], c["fused"], 0, None, None) for c in cases],
        "original_llm": [
            (
                c["query_id"],
                c["original"]["predictions"]["fts_rrf_llm"],
                c["original"]["timings"]["fts_rrf_llm"] - c["original"]["timings"]["fts_rrf"],
                c["original"]["rerank_fallback"],
                c["baseline_call"],
            )
            for c in cases
        ],
    }
    for mode in sorted({r["mode"] for r in rows}):
        mode_rows = [r for r in rows if r["mode"] == mode]
        if len(mode_rows) != len(selected) or {r["query_id"] for r in mode_rows} != set(selected):
            raise ValueError("Exactly one result per selected query/mode required")
        base_by_mode[mode] = [
            (r["query_id"], r["chunks"], r["rerank_ms"], r["fallback"], r["call"])
            for r in mode_rows
        ]
    for mode, items in base_by_mode.items():
        score = evaluate_rankings(
            {q: aggregate_chunks(pred, mapping) for q, pred, _, _, _ in items}, labels
        )
        tracker = CostTracker(pricing)
        tracker.records = [call for _, _, _, _, call in items if call is not None]
        usages = [r.get("usage") for r in tracker.records]
        known = [u for u in usages if isinstance(u, dict)]
        reasoning = [
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens") for u in known
        ]
        score.update(
            rerank_latency=latency_summary([ms for _, _, ms, _, _ in items]),
            fallbacks=sum(reason is not None for _, _, _, reason, _ in items),
            costs=tracker.summary(),
            usage={
                "reported_calls": len(known),
                "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in known),
                "completion_tokens": sum(u.get("completion_tokens", 0) for u in known),
                "reasoning_tokens_known": sum(v for v in reasoning if type(v) is int),
                "reasoning_usage_missing_calls": sum(v is None for v in reasoning),
            },
        )
        summaries[mode] = score
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--chunk-plan", type=Path, default=CHUNKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 101))
    parser.add_argument("--modes", nargs="+", choices=CONTROLS, default=["llm_low"])
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or len(args.modes) != len(set(args.modes)):
        raise ValueError("New output directory and unique modes required")
    data, chunks, cases = inputs(args.source, args.chunk_plan, args.limit)
    from app.config import get_settings
    from app.services.provider_costs import CostTracker

    settings = get_settings()
    if settings.model_provider != "siliconflow":
        raise ValueError("Explicit SiliconFlow configuration required")
    if any(
        c["payload"]["model"] != (settings.llm_rerank_model or settings.openai_chat_model)
        for c in cases
    ):
        raise ValueError("Baseline model changed")
    frozen = {
        "source_ledger_sha256": hash_file(args.source / "cost_events.jsonl"),
        "source_results_sha256": hash_file(args.source / "query_results.jsonl"),
        "chunk_manifest_sha256": hash_file(args.chunk_plan / "chunks.jsonl"),
        "source_preparation_sha256": data["report_sha256"],
        "native_model": settings.native_rerank_model,
    }
    rows = []
    for folder in args.reuse:
        old = read_object(folder / "resolved_config.json")
        if old["frozen"] != frozen or old["controls"] != CONTROLS:
            raise ValueError("Reuse inputs/controls changed")
        rows.extend(
            r
            for r in read_rows(folder / "query_results.jsonl")
            if r["mode"] in args.modes and r["query_id"] in {c["query_id"] for c in cases}
        )
    reused = {(r["query_id"], r["mode"]) for r in rows}
    if len(reused) != len(rows):
        raise ValueError("Duplicate reused results")
    args.output.mkdir(parents=True)
    pricing = args.output / "pricing.json"
    pricing.write_bytes((args.source / "pricing.json").read_bytes())
    resolved = {
        "frozen": frozen,
        "controls": CONTROLS,
        "modes": args.modes,
        "cohort_query_ids": [c["query_id"] for c in cases],
        "workers": args.workers,
        "split": "development",
        "test_queries_used": 0,
        "database_requests": 0,
        "embedding_requests": 0,
        "sdk_retries": 0,
        "rerank_depth": 10,
        "passage_mode": "prefix_600",
        "reused_calls": len(rows),
        "reuse_sources": [str(p) for p in args.reuse],
        "new_calls_planned": len(cases) * len(args.modes) - len(rows),
    }
    write_json(args.output / "resolved_config.json", resolved)
    (args.output / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.output / "status.json", {"status": "planned", **resolved})
    if not args.execute:
        print(f"Planned {resolved['new_calls_planned']} new calls; no model requests", flush=True)
        return 0
    from openai import OpenAI

    tracker = CostTracker(
        pricing,
        log_path=args.output / "cost_events.jsonl",
        secret=settings.openai_api_key,
        capture_payloads=True,
        context=lambda: (CONTEXT.get() or {}).copy(),
    )
    for mode in args.modes:
        role = "native_rerank" if mode == "native" else "llm_rerank"
        model = settings.native_rerank_model if mode == "native" else cases[0]["payload"]["model"]
        tracker.profile(role, model)
    write_rows(args.output / "query_results.jsonl", rows)
    started = time.perf_counter()
    from app.services import siliconflow

    with (
        OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_retries=0,
            timeout=60,
        ) as client,
        patch.object(siliconflow, "get_settings", return_value=settings),
        patch.object(siliconflow, "get_cost_tracker", return_value=tracker),
    ):
        for mode in args.modes:
            todo = [c for c in cases if (c["query_id"], mode) not in reused]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:

                def run_case(case, selected_mode=mode):
                    return measure(case, selected_mode, client, tracker, settings)

                for row in pool.map(run_case, todo):
                    rows.append(row)
                    append(args.output / "query_results.jsonl", row)
                    write_json(args.output / "cost_summary_new_calls.json", tracker.summary())
                    write_json(
                        args.output / "status.json", {"status": "running", "rows": len(rows)}
                    )
                    print(
                        f"{mode} q={row['query_id']} {row['rerank_ms']:.0f}ms "
                        f"fallback={row['fallback']}",
                        flush=True,
                    )
    summaries = summarize(rows, cases, data, chunks, pricing)
    for mode, score in summaries.items():
        write_json(args.output / f"{mode}.scores.json", score)
    if hash_file(args.source / "cost_events.jsonl") != frozen["source_ledger_sha256"]:
        raise ValueError("Baseline ledger changed")
    report = {
        "status": "complete",
        **resolved,
        "summaries": summaries,
        "new_costs": tracker.summary(),
        "wall_seconds": time.perf_counter() - started,
        "latency_scope": (
            "rerank only; baseline historical 4-worker timing vs configured workers; not E2E"
        ),
        "baseline_unchanged": True,
        "application_defaults_changed": False,
    }
    write_json(args.output / "report.json", report)
    write_json(args.output / "status.json", {"status": "complete", "rows": len(rows)})
    print(f"Complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
