"""Opt-in 16-case full-chunk rerank comparison against a frozen prefix baseline.

Plan by default. --execute reopens the source schema read-only; no indexing,
generation, native rerank or retries. At most 16 embeddings + 16 LLM requests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.baseline_support import (
    ROOT,
    BaselineError,
    digest,
    load_cases,
    read_json,
    summarize,
    write_json,
)
from scripts.capture_baseline import CASES_PATH, plan
from scripts.capture_siliconflow_baseline import CostRecorder
from scripts.prepare_test_cache import CACHE_DIR, verify_cache

DEFAULT_SOURCE = ROOT / "runs/siliconflow-baseline-7230f51350734c53a74eb7509f9771d5"
MODE = "fts_rrf_llm_rerank"


def load_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_relative_to(ROOT / "runs"):
        raise BaselineError("Source must be a local runs artifact")
    state = read_json(source / "status.json")
    config = read_json(source / "resolved_config.json")
    if (
        state.get("status") != "complete"
        or state.get("completed_queries") != 32
        or state.get("old_data_unchanged") is not True
        or config.get("protocol") != "siliconflow-adapted-16-case-retrieval-v1"
    ):
        raise BaselineError("Completed adapted baseline required")
    # Snapshot JSON formatting need not match, but all values/order must.
    if load_cases(source / "cases.json") != load_cases(CASES_PATH):
        raise BaselineError("Frozen cases changed")
    if (
        config.get("chunk_size"),
        config.get("chunk_overlap"),
        config.get("top_k"),
        config.get("retrieval_candidates"),
    ) != (800, 120, 5, 10):
        raise BaselineError("Unexpected source retrieval protocol")
    rows = read_rows(source / "rankings.jsonl")
    hybrid = [row for row in rows if row["mode"] == MODE]
    cases = load_cases(source / "cases.json")
    if len(cases) != 16 or [r["case_id"] for r in hybrid] != [c["id"] for c in cases]:
        raise BaselineError("Source comparison rows incomplete")
    if not all(row["status"] == "ok" for row in hybrid):
        raise BaselineError("Source has failed queries")
    # Validate schema allowlist even for plan; this helper does no DB work.
    from scripts.siliconflow_http_smoke import readonly_url

    readonly_url(state["schema"])
    return {"path": source, "state": state, "config": config, "rows": hybrid, "cases": cases}


def read_rows(path: Path) -> list[dict[str, Any]]:
    import json

    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError):
        raise BaselineError("Cannot read JSONL artifact") from None


class ComparisonCalls:
    """Authorization/request-count boundary, not a monetary budget mechanism."""

    def __init__(self, tracker, cases):
        self.tracker, self.cases, self.attempts = tracker, cases, 0

    def __call__(self, role, model, invoke, payload):
        if self.attempts >= 2 * len(self.cases):
            raise BaselineError("Authorized request count exhausted")
        case = self.cases[self.attempts // 2]
        expected = "embedding" if self.attempts % 2 == 0 else "llm_rerank"
        if role != expected:
            raise BaselineError("Unexpected model role/order")
        if role == "embedding" and payload.get("input") != [case["question"]]:
            raise BaselineError("Unexpected query embedding input")
        if role == "llm_rerank" and not payload["messages"][1]["content"].startswith(
            "Question: " + case["question"] + "\n\nPassages:\n"
        ):
            raise BaselineError("Unexpected rerank question")
        self.attempts += 1
        return self.tracker.call(role, model, invoke, request=payload)


def compare_rows(before, after, k):
    paired = []
    for old, new in zip(before, after, strict=False):
        if old["case_id"] != new["case_id"]:
            raise BaselineError("Case order mismatch")
        paired.append(
            {
                "case_id": old["case_id"],
                "before_rank": old["answer_rank"],
                "after_rank": new["answer_rank"],
                "before_fallback": old["rerank_fallback"],
                "after_fallback": new.get("rerank_fallback"),
                "status": new["status"],
            }
        )
    return {
        "scope": "exploratory same-set sequential comparison, not held-out evaluation",
        "before": summarize(before, k),
        "after": summarize(after, k),
        "before_fallbacks": sum(bool(r["rerank_fallback"]) for r in before),
        "after_fallbacks": sum(bool(r.get("rerank_fallback")) for r in after),
        "paired": paired,
    }


def execute(output, settings, source):
    if "app.database" in sys.modules or not verify_cache():
        raise BaselineError("Fresh process and verified tokenizer cache required")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    from openai import OpenAI

    from app.services.provider_costs import CostTracker
    from scripts.siliconflow_e2e import database_snapshot
    from scripts.siliconflow_http_smoke import readonly_url

    recorder = CostRecorder(output)

    def record_cost(row):
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
    config = source["config"]
    for role, model in (
        ("embedding", config["embedding_model"]),
        ("llm_rerank", config["llm_rerank_model"]),
    ):
        tracker.profile(role, model)
    calls = ComparisonCalls(tracker, source["cases"])
    before = database_snapshot()
    write_json(output / "tables_before.json", before)
    state = {
        "status": "running",
        "schema": source["state"]["schema"],
        "rerank_passage_mode": "full_chunk",
        "completed_queries": 0,
        "database_read_only": False,
    }
    write_json(output / "status.json", state)
    rows = []
    engine = client = None
    try:
        scoped = settings.model_copy(
            update={
                **{
                    key: config[key]
                    for key in ("chunk_size", "chunk_overlap", "top_k", "retrieval_candidates")
                },
                "database_url": readonly_url(state["schema"]),
                "run_db_init": False,
                "rate_limit_enabled": False,
                "rerank_passage_mode": "full_chunk",
                "use_hybrid": True,
                "use_rerank": True,
            }
        )
        with ExitStack() as stack:
            stack.enter_context(patch("app.config.get_settings", return_value=scoped))
            from app import database
            from app.models import Chunk, Document
            from app.services import embeddings, reranking, siliconflow
            from app.services import search as search_module

            engine = database.engine
            engine.update_execution_options(schema_translate_map={None: state["schema"]})
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                max_retries=0,
                timeout=60,
            )
            stack.enter_context(patch.object(siliconflow, "guarded_call", calls))
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
                if (
                    conn.exec_driver_sql("SHOW transaction_read_only", {}).scalar_one() != "on"
                    or conn.exec_driver_sql("SELECT current_schema()", {}).scalar_one()
                    != state["schema"]
                ):
                    raise BaselineError("Read-only schema mismatch")
                state["database_read_only"] = True
                actual = []
                for chunk, doc in db.query(Chunk, Document).join(Document).order_by(Chunk.id):
                    if len(chunk.embedding) != 1024:
                        raise BaselineError("Unexpected stored embedding dimension")
                    actual.append(
                        {
                            "id": chunk.id,
                            "document_id": doc.id,
                            "document_title": doc.title,
                            "source": doc.source,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                        }
                    )
                if actual != read_rows(source["path"] / "indexed_chunks.jsonl"):
                    raise BaselineError("Stored corpus no longer matches source")
                write_json(output / "stored_chunks.json", actual)
                for case in source["cases"]:
                    recorder.context = {"phase": "query", "mode": MODE, "case_id": case["id"]}
                    recorder.fallback = None
                    started, call_start = time.perf_counter(), len(recorder.calls)
                    row = {
                        **recorder.context,
                        "question": case["question"],
                        "expected_phrase": case["answer"],
                    }
                    try:
                        result = search_module.search(db, case["question"], config["top_k"])
                        if tracker.failed_requests or tracker.logging_failed:
                            raise BaselineError("Swallowed provider/log failure")
                        if len(recorder.calls) - call_start != 2:
                            raise BaselineError("Expected two model requests per case")
                        chunks = [asdict(c) for c in result.chunks]
                        rank = next(
                            (
                                i
                                for i, c in enumerate(chunks, 1)
                                if case["answer"].lower() in c["content"].lower()
                            ),
                            0,
                        )
                        row.update(
                            status="ok",
                            answer_rank=rank,
                            candidates=chunks,
                            confidence=result.confidence,
                            rerank_fallback=recorder.fallback,
                        )
                    except Exception as exc:
                        row.update(status="error", error_type=type(exc).__name__, answer_rank=0)
                    row.update(
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        model_calls=len(recorder.calls) - call_start,
                    )
                    rows.append(row)
                    recorder.append("rankings.jsonl", row)
                    write_json(output / "comparison.json", compare_rows(source["rows"], rows, 5))
                    if row["status"] != "ok":
                        break
                state.update(
                    status="complete"
                    if len(rows) == 16 and all(r["status"] == "ok" for r in rows)
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
            after = database_snapshot()
            write_json(output / "tables_after.json", after)
            state["database_data_unchanged"] = before == after
            if before != after:
                state["status"] = "partial"
        except Exception as exc:
            state.update(status="partial", data_check_error=type(exc).__name__)
        state.update(model_attempts=calls.attempts, costs=tracker.summary())
        write_json(output / "cost_summary.json", tracker.summary())
        write_json(output / "status.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    owned = False
    try:
        from app.config import get_settings

        settings = get_settings()
        source = load_source(args.source_run)
        config = source["config"]
        if settings.model_provider != "siliconflow":
            raise BaselineError("Explicit source provider required")
        if (
            settings.openai_base_url != config["base_url"]
            or settings.openai_embedding_model != config["embedding_model"]
            or settings.embedding_dim != config["embedding_dim"]
            or (settings.llm_rerank_model or settings.openai_chat_model)
            != config["llm_rerank_model"]
            or settings.siliconflow_llm_max_tokens != config["siliconflow_llm_max_tokens"]
        ):
            raise BaselineError("Models/limits must match frozen source")
        state = plan(
            args.output,
            {
                **config,
                "rerank_passage_mode": "full_chunk",
                "protocol": "rerank-full-chunk-comparison-v1",
            },
        )
        owned = True
        state.update(
            expected_queries=16,
            modes=[MODE],
            budget_enforced=False,
            note="Plan only; no DB/model requests. Same-set exploratory comparison.",
        )
        (args.output / "pricing.json").write_bytes((source["path"] / "pricing.json").read_bytes())
        source_dir = args.output / "source_baseline"
        source_dir.mkdir()
        hashes = {}
        for name in (
            "status.json",
            "resolved_config.json",
            "metrics.json",
            "rankings.jsonl",
            "stages.jsonl",
            "cases.json",
            "indexed_chunks.jsonl",
            "pricing.json",
        ):
            (source_dir / name).write_bytes((source["path"] / name).read_bytes())
            hashes[name] = digest(source_dir / name)
        write_json(
            args.output / "comparison_source.json",
            {
                "source_run": source["path"].name,
                "source_sha256": hashes,
                "scope": "same-set exploratory; no independent test claim",
            },
        )
        for name in (
            "compare_rerank_coverage.py",
            "capture_siliconflow_baseline.py",
            "siliconflow_http_smoke.py",
            "siliconflow_e2e.py",
        ):
            (args.output / "collector_sources" / name).write_bytes(
                (ROOT / "scripts" / name).read_bytes()
            )
        write_json(args.output / "status.json", state)
        if args.execute:
            state = execute(args.output, settings, source)
        print(f"Rerank coverage comparison: {state['status']} ({args.output})")
        return 0 if state["status"] in {"planned", "complete"} else 1
    except Exception as exc:
        if owned:
            state = read_json(args.output / "status.json")
            state.update(status="stopped", error_type=type(exc).__name__)
            write_json(args.output / "status.json", state)
        print(f"Rerank comparison stopped: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
