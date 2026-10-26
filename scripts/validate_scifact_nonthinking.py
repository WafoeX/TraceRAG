"""Validate frozen top-10/prefix_600 non-thinking reranking on development only.

Plan by default; --execute opts into at most 707 new calls after reusing the
fixed first 100. No retrieval, embeddings, database, generation, or test queries.
"""

from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts import optimize_scifact_rerank as experiment
from scripts.evaluate_scifact import read_rows
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.retrieval_metrics import aggregate_chunks, evaluate_rankings
from scripts.run_scifact_baseline import append

ROOT = experiment.ROOT
RETRIEVAL = ROOT / "runs/scifact-baseline-quick-v1"
REUSE = ROOT / "runs/scifact-rerank-fast-first100-v1"


def full_inputs(source, chunk_plan, retrieval, reuse):
    from app.services.reranking import build_messages

    data, chunks, first = experiment.inputs(source, chunk_plan, 100)
    frozen = {
        row["query_id"]: row["chunks"] for row in read_rows(retrieval / "fts_rrf.predictions.jsonl")
    }
    if set(frozen) != set(data["queries"]):
        raise ValueError("Expected exactly the frozen development cohort")
    template = first[0]["payload"]
    cases = []
    first_by_id = {c["query_id"]: c for c in first}
    for q, question in data["queries"].items():
        fused = frozen[q]
        if len(fused) < 10:
            raise ValueError("Missing top-10 candidates")
        documents = [
            f'(from "{data["corpus"][chunks[r["chunk_id"]]["document_id"]]["title"]}") '
            + chunks[r["chunk_id"]]["text"][:600]
            for r in fused[:10]
        ]
        payload = {**copy.deepcopy(template), "messages": build_messages(question, documents)}
        case = {
            "query_id": q,
            "question": question,
            "fused": fused,
            "documents": documents,
            "payload": payload,
        }
        if q in first_by_id:
            original = first_by_id[q]
            if original["fused"] != fused or original["payload"] != payload:
                raise ValueError("Shared application prompt/candidates changed")
            case.update(original)
        cases.append(case)
    old = [r for r in read_rows(reuse / "query_results.jsonl") if r["mode"] == "llm_no_thinking"]
    if len(old) != 100 or {r["query_id"] for r in old} != set(first_by_id):
        raise ValueError("Reuse must contain exactly the original first 100")
    for row in old:
        expected = {**first_by_id[row["query_id"]]["payload"], "enable_thinking": False}
        if row["call"]["request"] != expected:
            raise ValueError("Reused request does not match frozen non-thinking request")
    return data, chunks, cases, old


def summary(rows, cases, data, chunks, pricing):
    # Full cohort has no historical slow-LLM results: never manufacture that comparison.
    labels = {c["query_id"]: data["qrels"][c["query_id"]] for c in cases}
    mapping = {k: c["document_id"] for k, c in chunks.items()}
    from app.services.provider_costs import CostTracker

    tracker = CostTracker(pricing)
    tracker.records = [r["call"] for r in rows]
    scores = evaluate_rankings(
        {r["query_id"]: aggregate_chunks(r["chunks"], mapping) for r in rows}, labels
    )
    scores.update(
        rerank_latency=experiment.latency_summary([r["rerank_ms"] for r in rows]),
        fallbacks=sum(r["fallback"] is not None for r in rows),
        costs=tracker.summary(),
    )
    scores["fallback_rate"] = scores["fallbacks"] / len(rows)
    scores["usage"] = {
        "reported_calls": sum(isinstance(r["call"].get("usage"), dict) for r in rows),
        "reasoning_zero_calls": sum(
            (r["call"].get("usage") or {})
            .get("completion_tokens_details", {})
            .get("reasoning_tokens")
            == 0
            for r in rows
        ),
    }
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    from app.config import get_settings
    from app.services.provider_costs import CostTracker
    from app.services.reranking import thinking_controls

    settings = get_settings()
    if settings.model_provider != "siliconflow" or thinking_controls(settings) != {
        "enable_thinking": False
    }:
        raise ValueError("Set LLM_RERANK_ENABLE_THINKING=false explicitly")
    data, chunks, cases, rows = full_inputs(experiment.SOURCE, experiment.CHUNKS, RETRIEVAL, REUSE)
    if len(cases) != 807 or any(
        c["payload"]["model"] != (settings.llm_rerank_model or settings.openai_chat_model)
        for c in cases
    ):
        raise ValueError("Frozen cohort/model changed")
    paths = [
        experiment.SOURCE / "cost_events.jsonl",
        experiment.SOURCE / "query_results.jsonl",
        RETRIEVAL / "fts_rrf.predictions.jsonl",
        REUSE / "query_results.jsonl",
        experiment.CHUNKS / "chunks.jsonl",
        ROOT / ".env",
    ]
    hashes = {str(p.relative_to(ROOT)): hash_file(p) for p in paths}
    reused = {r["query_id"] for r in rows}
    todo = [c for c in cases if c["query_id"] not in reused]
    args.output.mkdir(parents=True)
    pricing = args.output / "pricing.json"
    pricing.write_bytes((experiment.SOURCE / "pricing.json").read_bytes())
    resolved = {
        "split": "development",
        "test_queries_used": 0,
        "cohort_query_ids": [c["query_id"] for c in cases],
        "frozen_hashes": hashes,
        "reused_calls": len(rows),
        "new_calls_planned": len(todo),
        "workers": 4,
        "sdk_retries": 0,
        "database_requests": 0,
        "embedding_requests": 0,
        "generation_requests": 0,
        "rerank_depth": 10,
        "passage_mode": "prefix_600",
        "thinking_controls": thinking_controls(settings),
        "application_defaults_changed": False,
    }
    write_json(args.output / "resolved_config.json", resolved)
    (args.output / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "measure_source.py").write_bytes(Path(experiment.__file__).read_bytes())
    write_json(args.output / "status.json", {"status": "planned", **resolved})
    if not args.execute:
        print(f"Planned {len(todo)} new calls, reusing {len(rows)}; no API calls")
        return 0
    from openai import OpenAI

    tracker = CostTracker(
        pricing,
        log_path=args.output / "cost_events.jsonl",
        secret=settings.openai_api_key,
        capture_payloads=True,
        context=lambda: (experiment.CONTEXT.get() or {}).copy(),
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

        def run(case):
            return experiment.measure(case, "llm_no_thinking", client, tracker, settings)

        for row in pool.map(run, todo):
            rows.append(row)
            append(args.output / "query_results.jsonl", row)
            write_json(args.output / "cost_summary_new_calls.json", tracker.summary())
            write_json(args.output / "status.json", {"status": "running", "rows": len(rows)})
            print(f"{len(rows)}/807 fallback={row['fallback']}", flush=True)
    if {r["query_id"] for r in rows} != set(data["queries"]) or len(rows) != 807:
        raise ValueError("Incomplete/duplicate evaluation")
    full = summary(rows, cases, data, chunks, pricing)
    mapping = {k: c["document_id"] for k, c in chunks.items()}
    baseline = evaluate_rankings(
        {c["query_id"]: aggregate_chunks(c["fused"], mapping) for c in cases}, data["qrels"]
    )
    first_ids = {c["query_id"] for c in cases[:100]}
    paired = experiment.summarize(
        [r for r in rows if r["query_id"] in first_ids], cases[:100], data, chunks, pricing
    )
    if any(hash_file(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("Frozen evidence or .env changed")
    report = {
        "status": "complete",
        **resolved,
        "full_nonthinking": full,
        "full_fts_rrf": baseline,
        "first100": paired,
        "new_costs": tracker.summary(),
        "wall_seconds": time.perf_counter() - started,
        "latency_scope": "rerank only; 100 reused historical calls, 707 new; not online E2E",
    }
    write_json(args.output / "report.json", report)
    write_json(args.output / "status.json", {"status": "complete", "rows": len(rows)})
    print(f"Complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
