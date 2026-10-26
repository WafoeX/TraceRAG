"""Opt-in one-document service-layer RAG smoke in a fresh guarded PG schema.

No API router/streaming/baseline benchmark is claimed. Production pricing
profiles remain unverified. A fixture-only, process-local guard permits exactly
2 embeddings, 1 original LLM rerank and 1 generation, with durable CNY records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch
from uuid import uuid4

from openai import OpenAI
from psycopg import sql
from sqlalchemy import create_engine, text

from app.config import get_settings
from app.services import siliconflow
from scripts.baseline_support import (
    DEFAULT_DATABASE_URL,
    GUARD_MARKER,
    prepare_schema,
    validate_database_url,
)
from scripts.prepare_test_cache import CACHE_DIR, verify_cache
from scripts.smoke_siliconflow import CHAT_RESERVATION, save, settle

T = TypeVar("T")
ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = (
    "BudgetRAG public smoke policy. Customers can request a refund within 30 days "
    "of purchase. Support is available by email."
)
QUESTION = "Can customers request a refund within 30 days?"
ROLES = ("embedding", "embedding", "llm_rerank", "generation")


def database_snapshot(exclude_schema: str | None = None) -> dict[str, Any]:
    """Read-only old-table fingerprints. Values/document text never leave PG."""
    validate_database_url(DEFAULT_DATABASE_URL)
    engine = create_engine(DEFAULT_DATABASE_URL, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            marker = connection.execute(
                text("SELECT marker FROM public.budgetrag_baseline_guard WHERE singleton=true")
            ).scalar_one()
            if marker != GUARD_MARKER:
                raise ValueError("Isolated database marker mismatch")
            tables = connection.execute(
                text(
                    "SELECT n.nspname,c.relname FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE c.relkind='r' AND n.nspname NOT LIKE 'pg_%' "
                    "AND n.nspname<>'information_schema' ORDER BY 1,2"
                )
            ).all()
            result = {}
            for schema, table in tables:
                if schema == exclude_schema:
                    continue
                # Identifier quoting is performed by psycopg, not string interpolation.
                statement = sql.SQL(
                    "SELECT count(*),md5(coalesce(string_agg(md5(row_to_json(t)::text),"
                    "'' ORDER BY md5(row_to_json(t)::text)),'')) FROM {} AS t"
                ).format(sql.Identifier(schema, table))
                count, fingerprint = connection.exec_driver_sql(statement.as_string()).one()
                result[f"{schema}.{table}"] = {"rows": count, "rowset_md5": fingerprint}
            return result
    finally:
        engine.dispose()


class FixtureCalls:
    """Explicitly authorized fixture guard, not an enablement of general models."""

    def __init__(
        self,
        state: dict[str, Any],
        path: Path,
        limit: Decimal,
        key: str,
        *,
        query_only: bool = False,
    ):
        self.state, self.path, self.limit, self.key = state, path, limit, key
        self.query_only = query_only
        self.roles = ROLES[1:] if query_only else ROLES
        self.spent = Decimal(0)
        self.blocked = False

    def checkpoint(self) -> None:
        save(self.path, self.state)

    def safe(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False).replace(self.key, "[REDACTED]"))
        except (TypeError, ValueError):
            raise ValueError("Cannot safely serialize fixture record") from None

    def __call__(self, role: str, model: str, invoke: Callable[[], T], payload: dict) -> T:
        index = len(self.state["calls"])
        if self.blocked or index >= len(self.roles) or role != self.roles[index]:
            raise ValueError("Extra/out-of-order model call denied")
        expected_model = "BAAI/bge-m3" if role == "embedding" else "deepseek-ai/DeepSeek-V4.1-Flash"
        if model != expected_model or payload.get("model") != model:
            raise ValueError("Unexpected fixture model")
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 4096:
            raise ValueError("Fixture request byte ceiling exceeded")
        if role == "embedding":
            expected_input = [DOCUMENT] if index == 0 and not self.query_only else [QUESTION]
            if payload.get("input") != expected_input:
                raise ValueError("Unexpected embedding fixture input")
        elif payload.get("max_tokens") != 64 or "messages" not in payload:
            raise ValueError("Unexpected fixture chat request")
        reservation = Decimal(0) if role == "embedding" else CHAT_RESERVATION
        if self.spent + reservation > self.limit:
            raise ValueError("Remaining CNY budget insufficient")
        row: dict[str, Any] = {
            "role": role,
            "model": model,
            "status": "pending",
            "stage": "request",
            "reserved_cost_cny": str(reservation),
            "request": self.safe(payload),
        }
        self.state["calls"].append(row)
        self.state["estimated_cost_cny"] = None
        self.checkpoint()
        try:
            response = invoke()
            raw: Any = response
            row.update(
                stage="usage_validation",
                raw_usage=self.safe(raw.usage.model_dump() if raw.usage is not None else None),
            )
            if role != "embedding":
                row["response"] = self.safe(raw.model_dump(mode="json"))
            self.checkpoint()
            usage, cost = settle(raw.usage, role)
            self.spent += cost
            row.update(usage=usage, estimated_cost_cny=str(cost), stage="response_validation")
            self.state.update(
                known_estimated_cost_cny=str(self.spent), estimated_cost_cny=str(self.spent)
            )
            self.checkpoint()
            if role == "embedding":
                vectors = siliconflow.ordered_embeddings(raw.data, 1)
                row["dimensions"] = [len(v) for v in vectors]
            else:
                content = siliconflow.chat_content(raw)
                if role == "llm_rerank":
                    siliconflow.validate_llm_scores(content, 1)
            row["status"] = "ok"
            self.checkpoint()
            return response
        except Exception as exc:
            self.blocked = True
            row.update(status="error", error_type=type(exc).__name__)
            self.checkpoint()
            raise ValueError("Fixture request/protocol failed; no further calls") from None


def run(output: Path, execute: bool = False, prior_cost: Decimal = Decimal("0")) -> dict:
    if execute and "app.database" in sys.modules:
        raise ValueError("Execute in a fresh process; database module was already loaded")
    settings = get_settings()
    if (
        settings.model_provider != "siliconflow"
        or settings.embedding_dim != 1024
        or settings.openai_base_url != "https://api.siliconflow.cn/v1"
        or settings.openai_embedding_model != "BAAI/bge-m3"
        or settings.openai_chat_model != "deepseek-ai/DeepSeek-V4.1-Flash"
        or (settings.llm_rerank_model or settings.openai_chat_model) != settings.openai_chat_model
    ):
        raise ValueError("Unexpected authorized provider/model configuration")
    try:
        prices = json.loads((ROOT / settings.siliconflow_pricing_file).read_text())
    except (OSError, ValueError):
        raise ValueError("Cannot read CNY screenshot prices") from None
    for role, inp, out in (
        ("embedding", "0", "0"),
        ("llm_rerank", "2", "8"),
        ("generation", "2", "8"),
    ):
        profile = prices["roles"][role]
        expected_model = (
            settings.openai_embedding_model if role == "embedding" else settings.openai_chat_model
        )
        if (
            profile["model"] != expected_model
            or prices.get("ledger_currency") != "CNY"
            or prices.get("schema_version") != 2
            or profile.get("price_verified_from_screenshot") is not True
            or profile["currency"] != "CNY"
            or profile["unit"] != "per_million_tokens"
            or profile["input_rate"] != inp
            or profile["output_rate"] != out
        ):
            raise ValueError("Screenshot pricing changed; review before execution")
    if not prior_cost.is_finite() or prior_cost < 0:
        raise ValueError("Invalid prior cost")
    available = settings.siliconflow_max_cost_cny - prior_cost
    if execute and (available <= 0 or available < 2 * CHAT_RESERVATION):
        raise ValueError("Insufficient CNY budget; no DB/model access")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    state: dict[str, Any] = {
        "status": "planned",
        "scope": "one-document service-layer RAG, not HTTP/baseline metrics",
        "currency": "CNY",
        "configured_budget_cny": str(settings.siliconflow_max_cost_cny),
        "prior_reported_plus_estimated_cny": str(prior_cost),
        "available_budget_cny": str(available),
        "pricing_policy": "higher uncached rate; all reasoning included; not a billing audit",
        "fixture_sha256": hashlib.sha256(DOCUMENT.encode()).hexdigest(),
        "question": QUESTION,
        "schema": None,
        "expected_call_roles": list(ROLES),
        "calls": [],
        "stages": [],
        "known_estimated_cost_cny": "0",
        "estimated_cost_cny": "0",
        "production_profiles_modified": False,
        "native_rerank_called": False,
    }
    save(path, state)
    (output / "fixture.txt").write_text(DOCUMENT + "\n")
    if not execute:
        return state
    import os

    if not verify_cache():
        raise ValueError("Verified tokenizer cache required")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    before = database_snapshot()
    state["old_tables_before"] = before
    schema = "baseline_" + uuid4().hex
    scoped_url = prepare_schema(DEFAULT_DATABASE_URL, schema)
    state.update(schema=schema, status="running")
    save(path, state)
    scoped = settings.model_copy(
        update={
            "database_url": scoped_url,
            "run_db_init": False,
            "use_hybrid": True,
            "use_rerank": True,
            "top_k": 1,
            "retrieval_candidates": 4,
            "chunk_size": 800,
            "chunk_overlap": 120,
            "siliconflow_llm_max_tokens": 64,
            "siliconflow_generation_max_tokens": 64,
        }
    )
    recorder = FixtureCalls(state, path, available, settings.openai_api_key)
    client = None
    engine = None
    try:
        with patch("app.config.get_settings", return_value=scoped):
            from app import database
            from app.services import embeddings, generation, ingestion, reranking
            from app.services import search as search_module

            engine = database.engine
            # Explicitly qualify ORM table writes even if public contains old app tables.
            engine.update_execution_options(schema_translate_map={None: schema})
            database.init_db()
            with engine.connect() as connection:
                if connection.execute(text("SELECT current_schema()")).scalar_one() != schema:
                    raise ValueError("Schema mismatch")
                dimension = connection.execute(
                    text(
                        "SELECT format_type(a.atttypid,a.atttypmod) FROM pg_attribute a "
                        "JOIN pg_class c ON c.oid=a.attrelid "
                        "JOIN pg_namespace n ON n.oid=c.relnamespace "
                        "WHERE n.nspname=:schema AND c.relname='chunks' AND a.attname='embedding'"
                    ),
                    {"schema": schema},
                ).scalar_one()
                if dimension != "vector(1024)":
                    raise ValueError("New schema embedding column is not vector(1024)")
                state["vector_column_type"] = dimension
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                max_retries=0,
                timeout=60,
            )
            from contextlib import ExitStack

            with ExitStack() as stack:
                stack.enter_context(patch.object(siliconflow, "guarded_call", recorder))
                for module in (embeddings, reranking, generation):
                    stack.enter_context(patch.object(module, "get_client", return_value=client))

                def observed(name, original):
                    def traced(*args, **kwargs):
                        result = original(*args, **kwargs)
                        state["stages"].append(
                            {"stage": name, "candidates": [asdict(c) for c in result]}
                        )
                        save(path, state)
                        return result

                    return traced

                for stage in (
                    "vector_search",
                    "keyword_search",
                    "reciprocal_rank_fusion",
                    "rerank",
                ):
                    original = getattr(search_module, stage)
                    stack.enter_context(
                        patch.object(search_module, stage, observed(stage, original))
                    )
                with database.SessionLocal() as db:
                    document = ingestion.ingest_document(
                        db, "Public refund policy", DOCUMENT, "public-e2e-fixture.txt"
                    )
                    if len(document.chunks) != 1:
                        raise ValueError("Expected exactly one fixture chunk")
                    state.update(document_id=document.id, chunk_count=len(document.chunks))
                    result = search_module.search(db, QUESTION, 1)
                    if recorder.blocked or len(state["calls"]) != 3:
                        raise ValueError("Reranker swallowed an error or expected calls missing")
                    if (
                        not result.chunks
                        or result.chunks[0].document_id != document.id
                        or result.confidence < scoped.min_relevance
                    ):
                        raise ValueError("Unexpected retrieval result/confidence")
                    state["retrieval_confidence"] = result.confidence
                    state["answer"] = generation.generate_answer(QUESTION, result.chunks)
                    if (
                        len(state["calls"]) != 4
                        or "30" not in state["answer"]
                        or "[1]" not in state["answer"]
                    ):
                        raise ValueError(
                            "Expected four calls and answer with 30-day source citation"
                        )
                    state["new_table_counts"] = {
                        table: db.connection()
                        .exec_driver_sql(
                            sql.SQL("SELECT count(*) FROM {}")
                            .format(sql.Identifier(schema, table))
                            .as_string()
                        )
                        .scalar_one()
                        for table in ("documents", "chunks")
                    }
                    state["status"] = "complete"
    except Exception as exc:
        state.update(status="partial", error_type=type(exc).__name__)
    finally:
        if client is not None:
            client.close()
        if engine is not None:
            engine.dispose()
        try:
            after = database_snapshot(exclude_schema=schema)
            state["old_tables_after"] = after
            state["old_data_unchanged"] = before == after
            if before != after:
                state["status"] = "partial"
        except Exception as exc:
            state.update(status="partial", old_data_check_error=type(exc).__name__)
        save(path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prior-cost-cny", type=Decimal, default=Decimal("0"))
    args = parser.parse_args()
    try:
        state = run(args.output, args.execute, args.prior_cost_cny)
        print(
            json.dumps(
                {
                    k: state.get(k)
                    for k in (
                        "status",
                        "schema",
                        "estimated_cost_cny",
                        "old_data_unchanged",
                        "answer",
                    )
                }
            )
        )
        return 0 if state["status"] in {"planned", "complete"} else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "stopped",
                    "error_type": type(exc).__name__,
                    "note": "Inspect report; no request-count inference",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
