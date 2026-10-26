"""Structured capture of the unchanged PostgreSQL search baseline.

Default invocation only writes a plan: no database or model calls.
Real execution requires --execute-models, a complete operator-verified config,
an explicit <= USD 1 estimate budget, and the isolated database guard.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

from .baseline_support import (
    DEFAULT_DATABASE_URL,
    ROOT,
    BaselineError,
    CostLedger,
    corpus_manifest,
    digest,
    load_cases,
    prepare_schema,
    read_json,
    runtime_manifest,
    summarize,
    validate_database_url,
    write_json,
)

MODES = ("dense_only", "fts_rrf_llm_rerank")
DEFAULT_CONFIG = ROOT / "configs/baseline/capture.template.json"
CASES_PATH = ROOT / "configs/baseline/smoke_cases.json"


def load_config(path: Path, execute: bool) -> dict[str, Any]:
    config = read_json(path)
    expected = {
        "schema_version",
        "provider",
        "base_url",
        "embedding_model",
        "chat_model",
        "embedding_dim",
        "chunk_size",
        "chunk_overlap",
        "retrieval_candidates",
        "top_k",
        "pricing",
        "pricing_and_bounds_source",
    }
    if not isinstance(config, dict) or set(config) != expected or config["schema_version"] != 1:
        raise BaselineError("Unexpected config schema/fields; never put API keys in this file")
    for key in ("chunk_size", "retrieval_candidates", "top_k"):
        if type(config[key]) is not int or config[key] <= 0:
            raise BaselineError("Invalid positive integer baseline setting")
    if (
        type(config["chunk_overlap"]) is not int
        or not 0 <= config["chunk_overlap"] < config["chunk_size"]
        or config["top_k"] > 20
    ):
        raise BaselineError("Invalid chunk overlap/top_k")
    if config["base_url"] is not None:
        if not isinstance(config["base_url"], str):
            raise BaselineError("Endpoint must be a string or null")
        endpoint = urlsplit(config["base_url"])
        if (
            endpoint.scheme != "https"
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise BaselineError("Endpoint must be HTTPS without credentials/query/fragment")
    if execute:
        for key in ("provider", "embedding_model", "chat_model", "pricing_and_bounds_source"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise BaselineError("Models/provider and verified price/bounds provenance required")
        if type(config["embedding_dim"]) is not int or config["embedding_dim"] <= 0:
            raise BaselineError("Explicit positive embedding dimension required")
        if not isinstance(config["base_url"], str):
            raise BaselineError("Explicit HTTPS endpoint required")
        url = urlsplit(config["base_url"])
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise BaselineError("Only HTTPS endpoint without credentials/query/fragment allowed")
        if not isinstance(config["pricing"], dict):
            raise BaselineError("Explicit model prices and conservative request bounds required")
    return config


class Recorder:
    def __init__(self, output: Path, ledger: CostLedger | None = None):
        self.output = output
        self.ledger = ledger
        self.context: dict[str, Any] = {"phase": "indexing"}
        self.calls: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.fallback: str | None = None

    def append(self, filename: str, value: Any) -> None:
        # Open/close per record so a crash retains all completed records.
        with (self.output / filename).open("a") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def call(self, kind: str, original, **kwargs):
        if self.ledger is None:
            raise BaselineError("No model execution ledger; call not made")
        reserved = self.ledger.reserve(kind)
        started = time.perf_counter()
        record = {
            **self.context,
            "kind": kind,
            "request": kwargs,
            "attempt": 1,
            "reserved_cost_usd": str(reserved),
            "usage": None,
            "estimated_cost_usd": None,
        }
        try:
            response = original(**kwargs)
            usage = response.usage.model_dump() if response.usage is not None else None
            # Full public-example response retained, including embedding vectors and rerank text.
            record.update(response=response.model_dump(mode="json"), usage=usage, status="ok")
            self.ledger.settle(kind, usage)
            record["estimated_cost_usd"] = self.ledger.costs[-1]["estimated_cost_usd"]
            return response
        except Exception as exc:
            self.ledger.blocked = True
            self.fallback = "model_error_or_unknown_billing"
            record.update(status="error", error_type=type(exc).__name__)
            raise BaselineError("Model request failed/usage unknown; paid calls stopped") from None
        finally:
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            self.calls.append(record)
            self.append("model_calls.jsonl", record)

    def stage(self, name: str, original):
        def observed(*args, **kwargs):
            started = time.perf_counter()
            row: dict[str, Any] = {**self.context, "stage": name}
            try:
                if name == "rerank":
                    row["input_candidates"] = [asdict(item) for item in args[1]]
                    row["top_k"] = args[2]
                result = original(*args, **kwargs)
                row.update(status="ok", candidates=[asdict(item) for item in result])
                return result
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, candidates=None)
                raise
            finally:
                row["elapsed_ms"] = (time.perf_counter() - started) * 1000
                if name == "rerank":
                    row["fallback_reason"] = self.fallback
                self.stages.append(row)
                self.append("stages.jsonl", row)

        return observed

    def scores(self, original):
        def observed(content, n):
            scores = original(content, n)
            if not scores or not any(s > 0 for s in scores):
                self.fallback = "unparseable_or_nonpositive_rerank_scores"
            return scores

        return observed


def capture_queries(cases, search, configure_mode, recorder: Recorder, k: int) -> list[dict]:
    """Injectable orchestration; executes original search, never reimplements retrieval."""
    records: list[dict[str, Any]] = []
    for mode in MODES:
        configure_mode(mode)
        for case in cases:
            recorder.context = {"phase": "query", "mode": mode, "case_id": case["id"]}
            recorder.fallback = None
            started = time.perf_counter()
            call_start = len(recorder.calls)
            row: dict[str, Any] = {
                **recorder.context,
                "question": case["question"],
                "expected_phrase": case["answer"],
            }
            try:
                result = search(case["question"], k)
                if recorder.ledger is not None and (
                    recorder.ledger.blocked or recorder.ledger.stop_reason is not None
                ):
                    raise BaselineError("A swallowed model/budget failure requires stopping")
                chunks = [asdict(item) for item in result.chunks[:k]]
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
                    confidence=result.confidence,
                    candidates=chunks,
                    rerank_fallback=recorder.fallback,
                )
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, answer_rank=0)
            row.update(
                elapsed_ms=(time.perf_counter() - started) * 1000,
                model_calls=len(recorder.calls) - call_start,
            )
            records.append(row)
            recorder.append("rankings.jsonl", row)
            metrics = {
                name: {
                    **summarize([r for r in records if r["mode"] == name], k),
                    "expected_queries": len(cases),
                    "complete": (
                        len([r for r in records if r["mode"] == name]) == len(cases)
                        and all(r["status"] == "ok" for r in records if r["mode"] == name)
                    ),
                }
                for name in MODES
            }
            write_json(recorder.output / "metrics.json", metrics)
            if row["status"] != "ok":
                return records
    return records


def plan(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    cases = load_cases(CASES_PATH)
    # Plan may inventory adapted runtime; execute still enforces the original
    # byte guard until a separate provider-specific capture protocol is approved.
    runtime = runtime_manifest(allow_adapted=True)
    manifest = corpus_manifest()
    original_archive = ROOT / "docs/refactor/baseline/source.tar.gz"
    archive_hash = digest(original_archive)
    if (
        archive_hash
        != read_json(ROOT / "docs/refactor/baseline/source-manifest.json")["archive_sha256"]
    ):
        raise BaselineError("Original source archive hash mismatch")
    if not manifest:
        raise BaselineError("Examples corpus is empty")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "resolved_config.json", config)
    write_json(output / "cases.json", cases)
    write_json(output / "corpus_manifest.json", manifest)
    corpus_sources = output / "corpus_sources"
    corpus_sources.mkdir()
    for entry in manifest:
        source = ROOT / entry["path"]
        archived = corpus_sources / source.name
        archived.write_bytes(source.read_bytes())
        if digest(archived) != entry["sha256"]:
            raise BaselineError("Corpus changed while taking snapshot")
    sources = output / "collector_sources"
    sources.mkdir()
    for source in (
        Path(__file__),
        ROOT / "scripts/baseline_support.py",
        ROOT / "scripts/prepare_test_cache.py",
    ):
        (sources / source.name).write_bytes(source.read_bytes())
    environment_sources = output / "environment_sources"
    environment_sources.mkdir()
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        (environment_sources / name).write_bytes((ROOT / name).read_bytes())
    (output / "original_source.tar.gz").write_bytes(original_archive.read_bytes())
    runtime_sources = []
    for package in ("app", "api"):
        for source in sorted((ROOT / package).rglob("*.py")):
            relative = source.relative_to(ROOT)
            target = output / "runtime_sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            runtime_sources.append({"path": relative.as_posix(), "sha256": digest(target)})
    write_json(
        output / "source_manifest.json",
        {
            "runtime": runtime,
            "runtime_sources": runtime_sources,
            "runtime_matches_original": all(r["sha256"] == r["original_sha256"] for r in runtime),
            "collector_sha256": digest(Path(__file__)),
            "support_sha256": digest(ROOT / "scripts/baseline_support.py"),
            "uv_lock_sha256": digest(ROOT / "uv.lock"),
            "cases_sha256": digest(CASES_PATH),
            "original_source_archive_sha256": archive_hash,
            "cache_preparation_sha256": digest(ROOT / "scripts/prepare_test_cache.py"),
            "environment_sources": {
                name: digest(environment_sources / name)
                for name in ("uv.lock", "pyproject.toml", ".python-version")
            },
        },
    )
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_head": head,
            "working_tree_dirty": dirty,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("openai", "sqlalchemy", "psycopg", "pgvector", "tiktoken")
            },
            "latency_scope": "instrumented search; includes query encode/recall/fusion/rerank",
            "sdk_max_retries": 0,
            "generation_calls": 0,
        },
    )
    state = {
        "status": "planned",
        "expected_queries": len(cases) * len(MODES),
        "modes": list(MODES),
        "model_calls": 0,
        "metrics": None,
        "note": "Plan only. Adapted runtime inventoried; original execution byte guard retained.",
    }
    write_json(output / "status.json", state)
    return state


def execute(output: Path, config: dict[str, Any], database_url: str, limit: Decimal) -> dict:
    # This function is never entered by the default CLI or this phase's database smoke.
    if "app.database" in sys.modules:
        raise BaselineError("Execute the collector in a fresh process to avoid cached DB settings")
    ledger = CostLedger(limit, config["pricing"])
    from .prepare_test_cache import CACHE_DIR, verify_cache

    if not verify_cache():
        raise BaselineError("Prepare the hash-verified tokenizer cache before execution")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    runtime_manifest()
    saved_corpus = read_json(output / "corpus_manifest.json")
    for entry in saved_corpus:
        source = output / "corpus_sources" / Path(entry["path"]).name
        if digest(source) != entry["sha256"]:
            raise BaselineError("Archived corpus hash mismatch")
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key in {"test-key", "sk-your-key-here"}:
        raise BaselineError(
            "Configure a real API key in the local environment, not a file argument"
        )
    schema = "baseline_" + uuid4().hex
    scoped_url = prepare_schema(database_url, schema)
    from app.config import Settings

    fields = {
        name: field.default
        for name, field in Settings.model_fields.items()
        if not field.is_required()
    }
    fields.update(
        openai_api_key=key,
        openai_base_url=config["base_url"],
        openai_embedding_model=config["embedding_model"],
        openai_chat_model=config["chat_model"],
        embedding_dim=config["embedding_dim"],
        database_url=scoped_url,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        top_k=config["top_k"],
        retrieval_candidates=config["retrieval_candidates"],
        run_db_init=False,
        rate_limit_enabled=False,
    )
    fields["_env_file"] = None
    settings = Settings(**fields)
    recorder = Recorder(output, ledger)
    state: dict[str, Any] = {"status": "running", "schema": schema}
    write_json(output / "status.json", state)
    with ExitStack() as stack:
        stack.enter_context(patch("app.config.get_settings", return_value=settings))
        from app.database import SessionLocal, engine, init_db
        from app.services import embeddings, ingestion, openai_client, reranking
        from app.services import search as search_module
        from scripts.ingest_examples import read_file

        client = openai_client.get_client().with_options(max_retries=0)
        proxy = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kw: recorder.call("embedding", client.embeddings.create, **kw)
            ),
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: recorder.call(
                        "rerank", client.chat.completions.create, **kw
                    )
                )
            ),
        )
        for module in (embeddings, reranking):
            stack.enter_context(patch.object(module, "get_client", return_value=proxy))
        for stage in ("vector_search", "keyword_search", "reciprocal_rank_fusion", "rerank"):
            original = getattr(search_module, stage)
            stack.enter_context(patch.object(search_module, stage, recorder.stage(stage, original)))
        stack.enter_context(
            patch.object(reranking, "_parse_scores", recorder.scores(reranking._parse_scores))
        )
        try:
            init_db()
            with SessionLocal() as db:
                # No interpolated SQL: these diagnostics are fixed statements only.
                server = db.connection().exec_driver_sql("SELECT version(), current_schema()").one()
                if server[1] != schema:
                    raise BaselineError("Search path does not match isolated run schema")
                write_json(
                    output / "database.json",
                    {
                        "postgres_version": server[0],
                        "schema": schema,
                        "pgvector_version": db.connection()
                        .exec_driver_sql(
                            "SELECT extversion FROM pg_extension WHERE extname='vector'"
                        )
                        .scalar_one(),
                    },
                )
                indexing_start = time.perf_counter()
                for entry in saved_corpus:
                    path = output / "corpus_sources" / Path(entry["path"]).name
                    document = ingestion.ingest_document(db, path.stem, read_file(path), path.name)
                    for chunk in sorted(document.chunks, key=lambda c: c.chunk_index):
                        recorder.append(
                            "indexed_chunks.jsonl",
                            {
                                "id": chunk.id,
                                "document_id": document.id,
                                "document_title": document.title,
                                "source": document.source,
                                "chunk_index": chunk.chunk_index,
                                "content": chunk.content,
                            },
                        )
                indexing_ms = (time.perf_counter() - indexing_start) * 1000

                def configure(mode: str) -> None:
                    settings.use_hybrid = mode != "dense_only"
                    settings.use_rerank = mode != "dense_only"

                rows = capture_queries(
                    load_cases(output / "cases.json"),
                    lambda q, k: search_module.search(db, q, k),
                    configure,
                    recorder,
                    config["top_k"],
                )
                complete = len(rows) == len(load_cases(output / "cases.json")) * len(MODES) and all(
                    r["status"] == "ok" for r in rows
                )
                state.update(
                    status="complete" if complete else "partial",
                    indexing_ms=indexing_ms,
                    completed_queries=len(rows),
                )
        except Exception as exc:
            state.update(status="partial", error_type=type(exc).__name__)
        finally:
            client.close()
            engine.dispose()
            state.update(
                model_calls=len(recorder.calls),
                known_estimated_api_cost_usd=str(ledger.spent),
                estimated_api_cost_usd=(
                    None if ledger.blocked or ledger.reserved is not None else str(ledger.spent)
                ),
                unpriced_request=ledger.blocked or ledger.reserved is not None,
                stop_reason=ledger.stop_reason,
                pricing_scope="operator-supplied rates/bounds, not provider billing audit",
            )
            write_json(output / "status.json", state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--execute-models", action="store_true")
    parser.add_argument("--max-cost-usd", type=Decimal)
    args = parser.parse_args(argv)
    artifact_owned = False
    try:
        config = load_config(args.config, args.execute_models)
        validate_database_url(args.database_url)
        if args.execute_models:
            if args.max_cost_usd is None:
                raise BaselineError("An explicit --max-cost-usd <= 1 is required")
            # Validate all cost controls before connecting to PG or creating an output directory.
            CostLedger(args.max_cost_usd, config["pricing"])
        state = plan(args.output, config)
        artifact_owned = True
        if args.execute_models:
            state = execute(args.output, config, args.database_url, args.max_cost_usd)
        print(f"Baseline artifact status: {state['status']} ({args.output})")
        return 0 if state["status"] in {"planned", "complete"} else 1
    except (BaselineError, OSError, ValueError, KeyError, TypeError, InvalidOperation) as exc:
        # Do not print provider/config error bodies: they may contain credentials.
        if artifact_owned:
            previous = read_json(args.output / "status.json")
            previous.update(status="stopped", error_type=type(exc).__name__)
            write_json(args.output / "status.json", previous)
        print(f"Baseline stopped: {type(exc).__name__}; inspect configuration/guard/budget")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
