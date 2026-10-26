"""Run the SciFact development baseline, reusing existing RAG/cost/metric code.

Default: plan only. --execute indexes the frozen chunk manifest in a new local
PG schema and evaluates Dense, FTS+RRF, and FTS+RRF+current LLM rerank. No test
queries, generation, native reranker, automatic retries or monetary budget gate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from scripts.baseline_support import DEFAULT_DATABASE_URL, ROOT, prepare_schema
from scripts.evaluate_scifact import DEFAULT_PREPARED, load_prepared, read_object, read_rows
from scripts.prepare_scifact import hash_file, write_json, write_rows
from scripts.retrieval_metrics import aggregate_chunks, evaluate_rankings

MODES = ("dense_only", "fts_rrf", "fts_rrf_llm")
DEPTH = 200
RERANK_DEPTH = 10
BATCH = 32
CALL_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("scifact_call", default=None)
FALLBACK: ContextVar[str | None] = ContextVar("scifact_fallback", default=None)


def append(path: Path, row: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def rerank_with_tail(question, fused, rerank, depth=RERANK_DEPTH):
    """Only reorder the head; keep the tail for document-level Recall@100."""
    head = fused[:depth]
    return rerank(question, head, len(head)) + fused[depth:]


def latency_summary(values):
    values = sorted(values)

    def percentile(q):
        if not values:
            return None
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)

    return {"p50_ms": percentile(0.5), "p95_ms": percentile(0.95)}


def execute(output: Path, data: dict, chunks: list[dict], settings, workers: int) -> dict:
    if "app.database" in sys.modules:
        raise ValueError("Run in a fresh process")
    from openai import OpenAI
    from sqlalchemy import insert, text

    from app.services.provider_costs import CostTracker

    def context():
        return (CALL_CONTEXT.get() or {}).copy()

    tracker = CostTracker(
        output / "pricing.json",
        log_path=output / "cost_events.jsonl",
        secret=settings.openai_api_key,
        context=context,
    )
    tracker.profile("embedding", settings.openai_embedding_model)
    tracker.profile("llm_rerank", settings.llm_rerank_model or settings.openai_chat_model)
    schema = "baseline_" + uuid4().hex
    state: dict[str, Any] = {
        "status": "indexing",
        "schema": schema,
        "split": "development",
        "queries_expected": len(data["queries"]),
        "documents": len(data["corpus"]),
        "chunks": len(chunks),
        "completed_queries": 0,
        "workers": workers,
    }
    write_json(output / "status.json", state)
    client = engine = None
    started = time.perf_counter()
    try:
        scoped = settings.model_copy(
            update={
                "database_url": prepare_schema(DEFAULT_DATABASE_URL, schema),
                "run_db_init": False,
                "rate_limit_enabled": False,
                "rerank_passage_mode": "prefix_600",
            }
        )
        with ExitStack() as stack:
            stack.enter_context(patch("app.config.get_settings", return_value=scoped))
            from app import database
            from app.models import Chunk, Document
            from app.services import embeddings, reranking, retrieval, siliconflow

            engine = database.engine
            engine.update_execution_options(schema_translate_map={None: schema})
            # Exact vector scan at this corpus size; do not create a default-ef HNSW index
            # which can silently return fewer than 100 documents at ef_search=40.
            database.Base.metadata.create_all(engine)
            with engine.connect() as conn:
                if conn.exec_driver_sql("SELECT current_schema()", {}).scalar_one() != schema:
                    raise ValueError("Isolated schema mismatch")
            client = OpenAI(
                api_key=scoped.openai_api_key,
                base_url=scoped.openai_base_url,
                max_retries=0,
                timeout=60,
            )
            for module in (embeddings, reranking):
                stack.enter_context(patch.object(module, "get_client", return_value=client))
            stack.enter_context(patch.object(siliconflow, "get_cost_tracker", return_value=tracker))
            parse_scores = reranking._parse_scores

            def observe_scores(content, count):
                result = parse_scores(content, count)
                FALLBACK.set(
                    None if result and any(s > 0 for s in result) else "nonpositive_scores"
                )
                return result

            stack.enter_context(patch.object(reranking, "_parse_scores", observe_scores))
            document_ids = {doc_id: i for i, doc_id in enumerate(data["corpus"], 1)}
            stable_by_backend = {i: c["chunk_id"] for i, c in enumerate(chunks, 1)}
            chunk_to_document = {c["chunk_id"]: c["document_id"] for c in chunks}
            write_rows(
                output / "backend_mapping.jsonl",
                [
                    {
                        "backend_chunk_id": i,
                        "chunk_id": c["chunk_id"],
                        "document_id": c["document_id"],
                        "backend_document_id": document_ids[c["document_id"]],
                    }
                    for i, c in enumerate(chunks, 1)
                ],
            )
            with database.SessionLocal() as db:
                db.execute(
                    insert(Document),
                    [
                        {"id": document_ids[d], "title": row["title"], "source": "scifact:" + d}
                        for d, row in data["corpus"].items()
                    ],
                )
                db.commit()
                for offset in range(0, len(chunks), BATCH):
                    batch = chunks[offset : offset + BATCH]
                    CALL_CONTEXT.set(
                        {
                            "phase": "indexing",
                            "batch_offset": offset,
                            "chunk_ids": [c["chunk_id"] for c in batch],
                        }
                    )
                    vectors = embeddings.embed_texts([c["text"] for c in batch])
                    db.execute(
                        insert(Chunk),
                        [
                            {
                                "id": offset + i + 1,
                                "document_id": document_ids[c["document_id"]],
                                "chunk_index": c["chunk_index"],
                                "content": c["text"],
                                "embedding": vector,
                            }
                            for i, (c, vector) in enumerate(zip(batch, vectors, strict=True))
                        ],
                    )
                    db.commit()
                    state["indexed_chunks"] = offset + len(batch)
                    write_json(output / "status.json", state)
                    if offset % (BATCH * 20) == 0:
                        print(f"Indexed {state['indexed_chunks']}/{len(chunks)} chunks", flush=True)
                db.execute(
                    text(
                        "CREATE INDEX chunks_content_fts ON chunks "
                        "USING gin (to_tsvector('english', content))"
                    )
                )
                db.commit()
            state["indexing_ms"] = (time.perf_counter() - started) * 1000
            query_vectors = {}
            query_items = list(data["queries"].items())
            with gzip.open(output / "query_embeddings.jsonl.gz", "wt", compresslevel=1) as cache:
                for offset in range(0, len(query_items), BATCH):
                    batch = query_items[offset : offset + BATCH]
                    CALL_CONTEXT.set(
                        {"phase": "query_embedding", "query_ids": [q for q, _ in batch]}
                    )
                    vectors = embeddings.embed_texts([question for _, question in batch])
                    for (query_id, question), vector in zip(batch, vectors, strict=True):
                        query_vectors[question] = vector
                        cache.write(json.dumps({"query_id": query_id, "embedding": vector}) + "\n")
            # Both retrieval modes share exactly the same precomputed query vector.
            stack.enter_context(patch.object(retrieval, "embed_query", lambda q: query_vectors[q]))
            # Retain full chat payloads, not huge indexing vectors.
            tracker.capture_payloads = True
            state["status"] = "evaluating"
            state["query_embedding_ms"] = (time.perf_counter() - started) * 1000 - state[
                "indexing_ms"
            ]
            write_json(output / "status.json", state)
            log_lock = Lock()

            def predict(item):
                query_id, question = item
                CALL_CONTEXT.set({"phase": "query", "query_id": query_id, "mode": "fts_rrf_llm"})
                FALLBACK.set("response_rejected_or_request_failed")
                wall = time.perf_counter()
                result = {"query_id": query_id, "status": "ok"}
                try:
                    with database.SessionLocal() as db:
                        db.execute(text("SET TRANSACTION READ ONLY"))
                        t0 = time.perf_counter()
                        vector = retrieval.vector_search(db, question, DEPTH)
                        t1 = time.perf_counter()
                        keyword = retrieval.keyword_search(db, question, DEPTH)
                        fused = retrieval.reciprocal_rank_fusion([vector, keyword])[:DEPTH]
                        t2 = time.perf_counter()
                        reranked = rerank_with_tail(question, fused, reranking.rerank)
                        t3 = time.perf_counter()

                    def serialized(items):
                        return [
                            {"chunk_id": stable_by_backend[c.chunk_id], "score": c.score}
                            for c in items
                        ]

                    result.update(
                        predictions={
                            "dense_only": serialized(vector),
                            "fts_rrf": serialized(fused),
                            "fts_rrf_llm": serialized(reranked),
                        },
                        timings={
                            "dense_only": (t1 - t0) * 1000,
                            "fts_rrf": (t2 - t0) * 1000,
                            "fts_rrf_llm": (t3 - t0) * 1000,
                        },
                        rerank_fallback=FALLBACK.get(),
                        keyword_candidates=len(keyword),
                    )
                    with log_lock:
                        append(
                            output / "candidates.jsonl",
                            {
                                "query_id": query_id,
                                "vector": serialized(vector),
                                "keyword": serialized(keyword),
                                "fused": serialized(fused),
                            },
                        )
                except Exception as exc:
                    result.update(
                        status="error",
                        error_type=type(exc).__name__,
                        predictions={mode: [] for mode in MODES},
                        timings={mode: (time.perf_counter() - wall) * 1000 for mode in MODES},
                        rerank_fallback=FALLBACK.get(),
                    )
                return result

            results = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(predict, item) for item in query_items]
                for future in as_completed(futures):
                    result = future.result()
                    results[result["query_id"]] = result
                    append(output / "query_results.jsonl", result)
                    for mode in MODES:
                        append(
                            output / (mode + ".predictions.jsonl"),
                            {"query_id": result["query_id"], "chunks": result["predictions"][mode]},
                        )
                    state["completed_queries"] = len(results)
                    if len(results) % 25 == 0 or len(results) == len(query_items):
                        state["costs"] = tracker.summary()
                        write_json(output / "status.json", state)
                        print(
                            f"Queries {len(results)}/{len(query_items)}; "
                            f"estimated CNY {tracker.spent}",
                            flush=True,
                        )
            summaries = {}
            for mode in MODES:
                rankings = {
                    q: aggregate_chunks(results[q]["predictions"][mode], chunk_to_document)
                    for q in data["queries"]
                }
                evaluation = evaluate_rankings(rankings, data["qrels"])
                write_json(output / (mode + ".scores.json"), evaluation)
                summaries[mode] = {
                    "queries": evaluation["queries"],
                    "metrics": evaluation["metrics"],
                    "latency": latency_summary([r["timings"][mode] for r in results.values()]),
                    "fewer_than_100_unique_documents": evaluation[
                        "fewer_than_100_unique_documents"
                    ],
                }
            state.update(
                status="complete",
                summaries=summaries,
                failed_queries=sum(r["status"] != "ok" for r in results.values()),
                rerank_fallbacks=sum(r["rerank_fallback"] is not None for r in results.values()),
                failed_model_requests=tracker.failed_requests,
                measured_latency_scope="Retrieval/rerank; cached query vectors, not online E2E",
            )
            if state["failed_queries"] or tracker.failed_requests:
                state["status"] = "complete_with_errors"
            write_json(output / "metrics.json", summaries)
    except Exception as exc:
        state.update(status="partial", error_type=type(exc).__name__)
    finally:
        if client is not None:
            client.close()
        if engine is not None:
            engine.dispose()
        state.update(costs=tracker.summary(), wall_seconds=time.perf_counter() - started)
        write_json(output / "cost_summary.json", tracker.summary())
        write_json(output / "status.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--chunk-plan", type=Path, default=ROOT / "runs/scifact-evaluation-plan-v1")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new output directory")
    data = load_prepared(args.prepared, "development")
    plan = read_object(args.chunk_plan / "status.json")
    if hash_file(args.chunk_plan / "chunks.jsonl") != plan["manifest"]["chunks.jsonl"]["sha256"]:
        raise ValueError("Chunk manifest changed")
    if plan["source_preparation_report_sha256"] != data["report_sha256"]:
        raise ValueError("Chunk manifest/source preparation mismatch")
    chunks = read_rows(args.chunk_plan / "chunks.jsonl")
    if args.query_limit is not None:
        if not 1 <= args.query_limit <= len(data["queries"]):
            raise ValueError("Invalid query limit")
        data["queries"] = dict(list(data["queries"].items())[: args.query_limit])
        data["qrels"] = {q: data["qrels"][q] for q in data["queries"]}
    if not 1 <= args.workers <= 8:
        raise ValueError("Workers outside 1..8")
    from app.config import get_settings

    settings = get_settings()
    if settings.model_provider != "siliconflow":
        raise ValueError("SiliconFlow baseline configuration required")
    args.output.mkdir(parents=True)
    resolved = {
        "protocol": "scifact-development-baseline-v1",
        "split": "development",
        "queries": len(data["queries"]),
        "full_development": args.query_limit is None,
        "modes": MODES,
        "retriever_depth_chunks": DEPTH,
        "llm_rerank_head_chunks": RERANK_DEPTH,
        "rerank_passage_mode": "prefix_600",
        "vector_search": "exact cosine, no HNSW",
        "embedding_batch": BATCH,
        "workers": args.workers,
        "embedding_model": settings.openai_embedding_model,
        "llm_rerank_model": settings.llm_rerank_model or settings.openai_chat_model,
        "llm_max_tokens": settings.siliconflow_llm_max_tokens,
        "shared_query_embeddings": True,
        "sdk_retries": 0,
        "budget_enforced": False,
        "test_queries_used": 0,
        "source_preparation_sha256": data["report_sha256"],
        "chunk_manifest_sha256": hash_file(args.chunk_plan / "chunks.jsonl"),
    }
    write_json(args.output / "resolved_config.json", resolved)
    (args.output / "pricing.json").write_bytes(Path(settings.siliconflow_pricing_file).read_bytes())
    (args.output / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    for relative in (
        "app/services/reranking.py",
        "app/services/retrieval.py",
        "scripts/retrieval_metrics.py",
    ):
        (args.output / Path(relative).name).write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "source_chunk_plan.json", plan)
    state = {"status": "planned", **resolved}
    write_json(args.output / "status.json", state)
    if args.execute:
        state = execute(args.output, data, chunks, settings, args.workers)
    print(f"SciFact baseline: {state['status']} ({args.output})", flush=True)
    return 0 if state["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
