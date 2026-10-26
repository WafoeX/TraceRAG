"""Finish cached SciFact retrieval and fixed first-100 LLM comparison, no API calls.

Used after the user chose to stop the slow full LLM run. The source model ledger
is preserved; in-flight requests are explicitly unknown, never free or retried.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_scifact import DEFAULT_PREPARED, load_prepared, read_object, read_rows
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.retrieval_metrics import aggregate_chunks, evaluate_rankings
from scripts.run_scifact_baseline import DEPTH, MODES, latency_summary


def finish(source: Path, output: Path):
    if output.exists():
        raise ValueError("Use a new output directory")
    data = load_prepared(DEFAULT_PREPARED, "development")
    old_state = read_object(source / "status.json")
    results = {r["query_id"]: r for r in read_rows(source / "query_results.jsonl")}
    cohort = list(data["queries"])[:100]
    if not set(cohort) <= set(results):
        raise ValueError(
            "Fixed first 100 queries are incomplete; do not select by completion speed"
        )
    events = read_rows(source / "cost_events.jsonl")
    starts = {r["request_id"]: r for r in events if r["event"] == "started"}
    ends = {r["request_id"]: r for r in events if r["event"] == "finished"}
    pending = [rid for rid in starts if rid not in ends]
    from app.services.provider_costs import CostTracker

    tracker = CostTracker(source / "pricing.json")
    tracker.records = [
        ends.get(
            rid,
            {**row, "status": "interrupted", "cost_status": "unknown", "estimated_cost_cny": None},
        )
        for rid, row in starts.items()
    ]
    costs = tracker.summary()
    write_json(source / "status.last_checkpoint.json", old_state)
    write_json(source / "cost_summary.json", costs)
    write_json(
        source / "interruption.json",
        {
            "reason": (
                "User chose fast baseline: full cached retrieval, fixed first-100 LLM comparison"
            ),
            "llm_results_completed": len(results),
            "inflight_request_ids": pending,
            "inflight_fees_unknown": True,
            "no_retries": True,
        },
    )
    write_json(
        source / "status.json",
        {
            **old_state,
            "status": "stopped_by_user",
            "completed_queries": len(results),
            "costs": costs,
            "inflight_requests": len(pending),
        },
    )
    with gzip.open(source / "query_embeddings.jsonl.gz", "rt") as handle:
        vectors = {r["query_id"]: r["embedding"] for r in map(json.loads, handle)}
    if set(vectors) != set(data["queries"]):
        raise ValueError("Cached query embedding coverage mismatch")
    mapping = read_rows(source / "backend_mapping.jsonl")
    stable_by_backend = {r["backend_chunk_id"]: r["chunk_id"] for r in mapping}
    chunk_to_document = {r["chunk_id"]: r["document_id"] for r in mapping}
    from app.config import get_settings
    from scripts.siliconflow_http_smoke import readonly_url

    settings = get_settings().model_copy(
        update={
            "database_url": readonly_url(old_state["schema"]),
            "run_db_init": False,
            "rate_limit_enabled": False,
        }
    )
    output.mkdir(parents=True)
    start_time = time.perf_counter()
    cache_hits = len(results)
    with patch("app.config.get_settings", return_value=settings):
        from app import database
        from app.services import retrieval

        database.engine.update_execution_options(schema_translate_map={None: old_state["schema"]})
        try:
            vectors_by_text = {text: vectors[q] for q, text in data["queries"].items()}
            with (
                database.SessionLocal() as db,
                patch.object(
                    retrieval, "embed_query", side_effect=lambda question: vectors_by_text[question]
                ),
            ):
                if (
                    db.connection().exec_driver_sql("SHOW transaction_read_only", {}).scalar_one()
                    != "on"
                ):
                    raise ValueError("Expected read-only cached retrieval")
                for q, question in data["queries"].items():
                    if q in results:
                        continue
                    t0 = time.perf_counter()
                    vector = retrieval.vector_search(db, question, DEPTH)
                    t1 = time.perf_counter()
                    keyword = retrieval.keyword_search(db, question, DEPTH)
                    fused = retrieval.reciprocal_rank_fusion([vector, keyword])[:DEPTH]
                    t2 = time.perf_counter()

                    def serialized(items):
                        return [
                            {"chunk_id": stable_by_backend[c.chunk_id], "score": c.score}
                            for c in items
                        ]

                    results[q] = {
                        "query_id": q,
                        "status": "ok",
                        "predictions": {
                            "dense_only": serialized(vector),
                            "fts_rrf": serialized(fused),
                        },
                        "timings": {"dense_only": (t1 - t0) * 1000, "fts_rrf": (t2 - t0) * 1000},
                    }
        finally:
            database.engine.dispose()
    full = {}
    paired = {}
    for mode in MODES:
        selected = (
            data["queries"] if mode != "fts_rrf_llm" else {q: data["queries"][q] for q in cohort}
        )
        ranking = {
            q: aggregate_chunks(results[q]["predictions"][mode], chunk_to_document)
            for q in selected
        }
        labels = {q: data["qrels"][q] for q in selected}
        score = evaluate_rankings(ranking, labels)
        score["latency"] = latency_summary([results[q]["timings"][mode] for q in selected])
        write_json(output / (mode + ".scores.json"), score)
        write_rows(
            output / (mode + ".predictions.jsonl"),
            [{"query_id": q, "chunks": results[q]["predictions"][mode]} for q in selected],
        )
        if mode != "fts_rrf_llm":
            full[mode] = {
                "queries": score["queries"],
                "metrics": score["metrics"],
                "fewer_than_100_unique_documents": score["fewer_than_100_unique_documents"],
                "latency": score["latency"],
            }
        paired_score = evaluate_rankings(
            {q: ranking[q] for q in cohort}, {q: data["qrels"][q] for q in cohort}
        )
        paired_score["latency"] = latency_summary([results[q]["timings"][mode] for q in cohort])
        write_json(output / (mode + ".first100.scores.json"), paired_score)
        paired[mode] = {
            "queries": 100,
            "metrics": paired_score["metrics"],
            "latency": paired_score["latency"],
        }
    selected_calls = [
        r for r in tracker.records if r.get("query_id") in cohort and r["role"] == "llm_rerank"
    ]
    subset = CostTracker(source / "pricing.json")
    subset.records = selected_calls
    report = {
        "status": "complete_scoped_baseline",
        "split": "development",
        "full_retrieval": full,
        "fixed_first100_comparison": paired,
        "cohort_selection": (
            "First 100 prepared development queries; chosen for runtime before inspecting metrics"
        ),
        "cohort_query_ids": cohort,
        "full_llm_807_completed": False,
        "llm_results_preserved": cache_hits,
        "llm_inflight_unknown": len(pending),
        "first100_llm_fallbacks": sum(results[q]["rerank_fallback"] is not None for q in cohort),
        "costs_all_requests": costs,
        "costs_first100_llm": subset.summary(),
        "new_model_requests_in_completion": 0,
        "database_read_only": True,
        "schema": old_state["schema"],
        "test_queries_used": 0,
        "source_run": source.name,
        "source_model_events_sha256": hash_file(source / "cost_events.jsonl"),
        "completion_wall_seconds": time.perf_counter() - start_time,
        "latency_scope": (
            "Cached query vectors; mixed original 4-worker and completion sequential SQL timing, "
            "not E2E"
        ),
    }
    write_json(output / "cost_summary.json", costs)
    write_json(output / "report.json", report)
    (output / "completion_source.py").write_bytes(Path(__file__).read_bytes())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finish(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
