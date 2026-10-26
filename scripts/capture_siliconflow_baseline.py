"""Adapted SiliconFlow 16-case retrieval baseline; CNY costs only, no spend caps.

Default is plan-only (no DB/models). --execute explicitly runs the frozen
corpus/two modes in a fresh guarded schema. Not the unchanged-source baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from scripts.baseline_support import (
    DEFAULT_DATABASE_URL,
    BaselineError,
    digest,
    load_cases,
    read_json,
    write_json,
)
from scripts.capture_baseline import MODES, Recorder, capture_queries, plan
from scripts.prepare_test_cache import CACHE_DIR, verify_cache


class CostRecorder(Recorder):
    def scores(self, original):
        observed = super().scores(original)

        def parse(content, n):
            self.fallback = None
            return observed(content, n)

        return parse

    def stage(self, name, original):
        def checked(*args, **kwargs):
            if name == "rerank":
                self.fallback = "provider_response_rejected_or_request_failed"
            return original(*args, **kwargs)

        return super().stage(name, checked)


def resolved_config(settings) -> dict[str, Any]:
    if settings.model_provider != "siliconflow":
        raise BaselineError("Explicit SiliconFlow configuration required")
    return {
        "schema_version": 2,
        "provider": settings.model_provider,
        "base_url": settings.openai_base_url,
        "embedding_model": settings.openai_embedding_model,
        "chat_model": settings.openai_chat_model,
        "llm_rerank_model": settings.llm_rerank_model or settings.openai_chat_model,
        "embedding_dim": settings.embedding_dim,
        # Freeze the original collector protocol; do not inherit developer tuning.
        "chunk_size": 800,
        "chunk_overlap": 120,
        "retrieval_candidates": 10,
        "top_k": 5,
        "siliconflow_llm_max_tokens": settings.siliconflow_llm_max_tokens,
        "protocol": "siliconflow-adapted-16-case-retrieval-v1",
        "budget_enforced": False,
        "generation_calls": 0,
    }


def execute(output: Path, settings, config: dict) -> dict:
    if "app.database" in sys.modules:
        raise BaselineError("Fresh process required")
    if not verify_cache():
        raise BaselineError("Hash-verified tokenizer cache required")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    from openai import OpenAI

    from app.services.provider_costs import CostTracker
    from scripts.baseline_support import prepare_schema
    from scripts.siliconflow_e2e import database_snapshot

    recorder = CostRecorder(output)

    def record_cost(row: dict[str, Any]) -> None:
        recorder.calls.append(row)
        recorder.append("model_calls.jsonl", row)

    tracker = CostTracker(
        output / "pricing.json",
        log_path=output / "cost_events.jsonl",
        secret=settings.openai_api_key,
        capture_payloads=True,
        context=lambda: recorder.context.copy(),
        on_result=record_cost,
    )
    for role, model in (
        ("embedding", config["embedding_model"]),
        ("llm_rerank", config["llm_rerank_model"]),
    ):
        tracker.profile(role, model)
    before = database_snapshot()
    write_json(output / "old_tables_before.json", before)
    schema = "baseline_" + uuid4().hex
    state: dict[str, Any] = {
        "status": "running",
        "schema": schema,
        "protocol": config["protocol"],
        "generation_calls": 0,
    }
    write_json(output / "status.json", state)
    engine = client = None
    try:
        scoped_url = prepare_schema(DEFAULT_DATABASE_URL, schema)
        scoped = settings.model_copy(
            update={
                **{
                    key: config[key]
                    for key in ("chunk_size", "chunk_overlap", "top_k", "retrieval_candidates")
                },
                "database_url": scoped_url,
                "run_db_init": False,
                "rate_limit_enabled": False,
            }
        )
        with ExitStack() as stack:
            stack.enter_context(patch("app.config.get_settings", return_value=scoped))
            from app import database
            from app.services import embeddings, ingestion, reranking, siliconflow
            from app.services import search as search_module
            from scripts.ingest_examples import read_file

            engine = database.engine
            engine.update_execution_options(schema_translate_map={None: schema})
            database.init_db()
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                max_retries=0,
                timeout=60,
            )
            stack.enter_context(patch.object(siliconflow, "get_cost_tracker", return_value=tracker))
            for module in (embeddings, reranking):
                stack.enter_context(patch.object(module, "get_client", return_value=client))
            for stage in ("vector_search", "keyword_search", "reciprocal_rank_fusion", "rerank"):
                stack.enter_context(
                    patch.object(
                        search_module, stage, recorder.stage(stage, getattr(search_module, stage))
                    )
                )
            stack.enter_context(
                patch.object(reranking, "_parse_scores", recorder.scores(reranking._parse_scores))
            )
            with database.SessionLocal() as db:
                conn = db.connection()
                if conn.exec_driver_sql("SELECT current_schema()", {}).scalar_one() != schema:
                    raise BaselineError("Schema mismatch")
                write_json(
                    output / "database.json",
                    {
                        "schema": schema,
                        "postgres_version": conn.exec_driver_sql(
                            "SELECT version()", {}
                        ).scalar_one(),
                        "pgvector_version": conn.exec_driver_sql(
                            "SELECT extversion FROM pg_extension WHERE extname='vector'", {}
                        ).scalar_one(),
                    },
                )
                started = time.perf_counter()
                for entry in read_json(output / "corpus_manifest.json"):
                    path = output / "corpus_sources" / Path(entry["path"]).name
                    if digest(path) != entry["sha256"]:
                        raise BaselineError("Frozen corpus mismatch")
                    doc = ingestion.ingest_document(db, path.stem, read_file(path), path.name)
                    for chunk in sorted(doc.chunks, key=lambda c: c.chunk_index):
                        recorder.append(
                            "indexed_chunks.jsonl",
                            {
                                "id": chunk.id,
                                "document_id": doc.id,
                                "document_title": doc.title,
                                "source": doc.source,
                                "chunk_index": chunk.chunk_index,
                                "content": chunk.content,
                            },
                        )
                state["indexing_ms"] = (time.perf_counter() - started) * 1000

                def configure(mode):
                    scoped.use_hybrid = mode != "dense_only"
                    scoped.use_rerank = mode != "dense_only"

                def search(question, k):
                    failures = tracker.failed_requests
                    result = search_module.search(db, question, k)
                    if tracker.failed_requests != failures or tracker.logging_failed:
                        raise BaselineError("Provider failure swallowed by reranker; stop run")
                    return result

                cases = load_cases(output / "cases.json")
                rows = capture_queries(cases, search, configure, recorder, config["top_k"])
                state.update(
                    status="complete"
                    if len(rows) == len(cases) * len(MODES)
                    and all(r["status"] == "ok" for r in rows)
                    else "partial",
                    completed_queries=len(rows),
                )
    except Exception as exc:
        state.update(status="partial", error_type=type(exc).__name__)
    finally:
        if client is not None:
            client.close()
        if engine is not None:
            engine.dispose()
        try:
            after = database_snapshot(exclude_schema=schema)
            write_json(output / "old_tables_after.json", after)
            state["old_data_unchanged"] = before == after
            if before != after:
                state["status"] = "partial"
        except Exception as exc:
            state.update(status="partial", old_data_check_error=type(exc).__name__)
        summary = tracker.summary()
        write_json(output / "cost_summary.json", summary)
        state.update(model_calls=summary["calls"], costs=summary)
        write_json(output / "status.json", state)
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    owned = False
    try:
        from app.config import get_settings

        settings = get_settings()
        config = resolved_config(settings)
        state = plan(args.output, config)
        owned = True
        source = args.output / "collector_sources" / Path(__file__).name
        source.write_bytes(Path(__file__).read_bytes())
        price = Path(settings.siliconflow_pricing_file)
        (args.output / "pricing.json").write_bytes(price.read_bytes())
        state.update(
            protocol=config["protocol"],
            budget_enforced=False,
            note="Adapted baseline. Plan only: no DB/model calls.",
        )
        write_json(args.output / "status.json", state)
        if args.execute:
            state = execute(args.output, settings, config)
        print(f"SiliconFlow baseline: {state['status']} ({args.output})")
        return 0 if state["status"] in {"planned", "complete"} else 1
    except Exception as exc:
        if owned:
            state = read_json(args.output / "status.json")
            state.update(status="stopped", error_type=type(exc).__name__)
            write_json(args.output / "status.json", state)
        print(f"SiliconFlow baseline stopped: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
