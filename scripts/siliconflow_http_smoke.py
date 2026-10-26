"""Opt-in in-process ASGI HTTP smoke: one real /query, existing public fixture.

No listener, DDL, ingestion or rate-limit writes. Plan is no-call.
--stream checks the existing buffered text + citation-frame protocol, not token
latency. The DB is read-only; the fixture guard permits exactly 3 model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai import OpenAI
from psycopg import sql
from sqlalchemy.engine import make_url

from app.config import get_settings
from app.services import siliconflow
from scripts.baseline_support import DEFAULT_DATABASE_URL, read_json, validate_database_url
from scripts.prepare_test_cache import CACHE_DIR, verify_cache
from scripts.siliconflow_e2e import DOCUMENT, QUESTION, ROOT, FixtureCalls, database_snapshot
from scripts.smoke_siliconflow import CHAT_RESERVATION, save


def readonly_url(schema: str) -> str:
    validate_database_url(DEFAULT_DATABASE_URL)
    if not re.fullmatch(r"baseline_[0-9a-f]{32}", schema):
        raise ValueError("Only an allowlisted existing fixture schema is accepted")
    return (
        make_url(DEFAULT_DATABASE_URL)
        .update_query_dict(
            {"options": f"-csearch_path={schema},public -cdefault_transaction_read_only=on"}
        )
        .render_as_string(hide_password=False)
    )


def validate_answer(body: dict[str, Any], document_id: int) -> None:
    if set(body) != {"answer", "citations"}:
        raise ValueError("Unexpected HTTP response fields")
    answer, citations = body["answer"], body["citations"]
    if not isinstance(answer, str) or "30" not in answer or "[1]" not in answer:
        raise ValueError("Expected fixture answer with source citation")
    if not isinstance(citations, list) or len(citations) != 1:
        raise ValueError("Expected one citation")
    citation = citations[0]
    if (
        citation.get("document_id") != document_id
        or citation.get("chunk_index") != 0
        or citation.get("document_title") != "Public refund policy"
        or citation.get("snippet") != DOCUMENT[:240].strip()
    ):
        raise ValueError("HTTP citation does not identify the stored fixture")


def decode_stream(wire: bytes) -> dict[str, Any]:
    """Reassemble bytes before strict UTF-8 decoding; reject truncated/extra frames."""
    content = wire.decode("utf-8")
    if content.count("\x1f") != 1:
        raise ValueError("Expected exactly one final citation frame")
    answer, frame = content.split("\x1f")
    if not answer.strip():
        raise ValueError("Empty stream answer")
    try:
        parsed = json.loads(frame)
    except ValueError:
        raise ValueError("Invalid final citation JSON") from None
    if not isinstance(parsed, dict) or set(parsed) != {"citations"}:
        raise ValueError("Unexpected final citation fields")
    if not isinstance(parsed["citations"], list):
        raise ValueError("Citation frame must contain a list")
    return {"answer": answer, "citations": parsed["citations"]}


def run(
    output: Path,
    source_run: Path,
    execute: bool = False,
    *,
    stream: bool = False,
    accounting_run: Path | None = None,
) -> dict:
    if execute and "app.database" in sys.modules:
        raise ValueError("HTTP smoke requires a fresh process")
    source_run = source_run.resolve()
    if not source_run.is_relative_to(ROOT / "runs"):
        raise ValueError("Source evidence must be under local runs/")
    source = read_json(source_run / "report.json")
    source_accounting = read_json(source_run / "accounting.json")
    accounting_run = source_run if accounting_run is None else accounting_run.resolve()
    if not accounting_run.is_relative_to(ROOT / "runs"):
        raise ValueError("Accounting evidence must be under local runs/")
    accounting = read_json(accounting_run / "accounting.json")
    if accounting_run != source_run:
        prior_run = read_json(accounting_run / "report.json")
        if (
            prior_run.get("status") != "complete"
            or prior_run.get("schema") != source.get("schema")
            or prior_run.get("database_data_unchanged") is not True
        ):
            raise ValueError("Latest accounting must identify a completed unchanged-schema run")
    if (
        source.get("status") != "complete"
        or source.get("vector_column_type") != "vector(1024)"
        or source.get("fixture_sha256") != hashlib.sha256(DOCUMENT.encode()).hexdigest()
        or source.get("question") != QUESTION
        or source.get("old_data_unchanged") is not True
        or accounting.get("currency") != "CNY"
    ):
        raise ValueError("Source fixture evidence is not a successful isolated E2E run")
    schema = source["schema"]
    scoped_url = readonly_url(schema)
    settings = get_settings()
    if (
        settings.model_provider != "siliconflow"
        or settings.embedding_dim != 1024
        or settings.openai_base_url != "https://api.siliconflow.cn/v1"
        or settings.openai_embedding_model != "BAAI/bge-m3"
        or settings.openai_chat_model != "deepseek-ai/DeepSeek-V4.1-Flash"
        or (settings.llm_rerank_model or settings.openai_chat_model) != settings.openai_chat_model
    ):
        raise ValueError("Unexpected authorized model/endpoint")
    prices = read_json(ROOT / settings.siliconflow_pricing_file)
    for role, model, inp, out in (
        ("embedding", "BAAI/bge-m3", "0", "0"),
        ("llm_rerank", settings.openai_chat_model, "2", "8"),
        ("generation", settings.openai_chat_model, "2", "8"),
    ):
        profile = prices["roles"][role]
        if (
            prices.get("ledger_currency") != "CNY"
            or prices.get("schema_version") != 2
            or profile["model"] != model
            or profile["currency"] != "CNY"
            or profile["unit"] != "per_million_tokens"
            or profile.get("price_verified_from_screenshot") is not True
            or profile["input_rate"] != inp
            or profile["output_rate"] != out
        ):
            raise ValueError("Screenshot price policy changed")
    prior = Decimal(accounting["reference_cumulative_reported_plus_estimates_cny"])
    if (
        not prior.is_finite()
        or prior < 0
        or prior < Decimal(source_accounting["reference_cumulative_reported_plus_estimates_cny"])
    ):
        raise ValueError("Invalid/stale prior cost reference")
    available = settings.siliconflow_max_cost_cny - prior
    if execute and (available <= 0 or available < 2 * CHAT_RESERVATION):
        raise ValueError("Insufficient available budget; no DB/model calls")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    state: dict[str, Any] = {
        "status": "planned",
        "scope": (
            "in-process buffered ASGI /query/stream" if stream else "in-process ASGI JSON /query"
        ),
        "source_run": source_run.relative_to(ROOT).as_posix(),
        "accounting_source_run": accounting_run.relative_to(ROOT).as_posix(),
        "stream_mode": stream,
        "token_latency_verified": False,
        "schema": schema,
        "currency": "CNY",
        "configured_budget_cny": str(settings.siliconflow_max_cost_cny),
        "prior_reported_plus_estimated_cny": str(prior),
        "available_budget_cny": str(available),
        "calls": [],
        "http": [],
        "expected_call_roles": ["embedding", "llm_rerank", "generation"],
        "known_estimated_cost_cny": "0",
        "estimated_cost_cny": "0",
        "billing_audited": False,
        "run_db_init": False,
        "rate_limit_enabled": False,
        "read_only_requested": True,
        "production_profiles_modified": False,
        "public_listener_started": False,
    }
    save(path, state)
    if not execute:
        return state
    if not verify_cache():
        raise ValueError("Hash-verified tokenizer cache required before application import")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    before = database_snapshot()
    state["tables_before"] = before
    scoped = settings.model_copy(
        update={
            "database_url": scoped_url,
            "run_db_init": False,
            "rate_limit_enabled": False,
            "top_k": 1,
            "retrieval_candidates": 4,
            "use_hybrid": True,
            "use_rerank": True,
            "siliconflow_llm_max_tokens": 64,
            "siliconflow_generation_max_tokens": 64,
        }
    )
    recorder = FixtureCalls(state, path, available, settings.openai_api_key, query_only=True)
    engine = None
    model_client = None
    try:
        with patch("app.config.get_settings", return_value=scoped):
            from app import database
            from app.main import app
            from app.services import embeddings, generation, reranking

            engine = database.engine
            engine.update_execution_options(schema_translate_map={None: schema})
            with engine.connect() as connection:
                if (
                    connection.exec_driver_sql("SHOW transaction_read_only", {}).scalar_one()
                    != "on"
                    or connection.exec_driver_sql("SELECT current_schema()", {}).scalar_one()
                    != schema
                ):
                    raise ValueError("Read-only/schema connection check failed")
                contents = (
                    connection.exec_driver_sql(
                        sql.SQL("SELECT content FROM {} ORDER BY id")
                        .format(sql.Identifier(schema, "chunks"))
                        .as_string()
                    )
                    .scalars()
                    .all()
                )
                ids = (
                    connection.exec_driver_sql(
                        sql.SQL("SELECT id FROM {} ORDER BY id")
                        .format(sql.Identifier(schema, "documents"))
                        .as_string()
                    )
                    .scalars()
                    .all()
                )
                if contents != [DOCUMENT] or ids != [source["document_id"]]:
                    raise ValueError("Stored fixture differs; no model calls allowed")
                state["read_only_verified"] = True
            model_client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                max_retries=0,
                timeout=60,
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(siliconflow, "guarded_call", recorder))
                for module in (embeddings, generation, reranking):
                    stack.enter_context(
                        patch.object(module, "get_client", return_value=model_client)
                    )
                with TestClient(app, raise_server_exceptions=False) as client:
                    health = client.get("/health")
                    state["http"].append(
                        {
                            "method": "GET",
                            "path": "/health",
                            "status": health.status_code,
                            "body": health.json(),
                        }
                    )
                    if health.status_code != 200 or health.json() != {"status": "ok"}:
                        raise ValueError("Health response mismatch")
                    endpoint = "/query/stream" if stream else "/query"
                    invalid = client.post(endpoint, json={"question": "", "top_k": 1})
                    state["http"].append(
                        {
                            "method": "POST",
                            "path": endpoint + " (invalid)",
                            "status": invalid.status_code,
                            "body": invalid.json(),
                        }
                    )
                    if invalid.status_code != 422 or state["calls"]:
                        raise ValueError("Validation/health unexpectedly invoked models")
                    save(path, state)
                    if stream:
                        with client.stream(
                            "POST", endpoint, json={"question": QUESTION, "top_k": 1}
                        ) as response:
                            chunks = list(response.iter_bytes())
                            wire = b"".join(chunks)
                            content_type = response.headers.get("content-type", "")
                            state["http"].append(
                                {
                                    "method": "POST",
                                    "path": endpoint,
                                    "status": response.status_code,
                                    "content_type": content_type,
                                    "body_text": recorder.safe(wire.decode("utf-8")),
                                    "transport_chunks_observed": len(chunks),
                                    "wire_sha256": hashlib.sha256(wire).hexdigest(),
                                }
                            )
                            save(path, state)
                            if response.status_code != 200 or not content_type.startswith(
                                "text/plain"
                            ):
                                raise ValueError("Unexpected stream status/content type")
                            body = decode_stream(wire)
                    else:
                        response = client.post(endpoint, json={"question": QUESTION, "top_k": 1})
                        state["http"].append(
                            {
                                "method": "POST",
                                "path": endpoint,
                                "status": response.status_code,
                                "body_text": recorder.safe(response.text),
                            }
                        )
                        save(path, state)
                        if response.status_code != 200:
                            raise ValueError("Unexpected query status")
                        body = response.json()
                    if recorder.blocked or len(state["calls"]) != 3:
                        raise ValueError("Query failed or expected three model calls missing")
                    validate_answer(body, source["document_id"])
                    state["query_response"] = recorder.safe(body)
                    state["status"] = "complete"
    except Exception as exc:
        state.update(status="partial", error_type=type(exc).__name__)
    finally:
        if model_client is not None:
            model_client.close()
        if engine is not None:
            engine.dispose()
        try:
            after = database_snapshot()
            state["tables_after"] = after
            state["database_data_unchanged"] = before == after
            if before != after:
                state["status"] = "partial"
        except Exception as exc:
            state.update(status="partial", database_check_error=type(exc).__name__)
        save(path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--accounting-run", type=Path)
    args = parser.parse_args()
    try:
        state = run(
            args.output,
            args.source_run,
            args.execute,
            stream=args.stream,
            accounting_run=args.accounting_run,
        )
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "model_attempts": len(state["calls"]),
                    "estimated_cost_cny": state["estimated_cost_cny"],
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
                    "note": "Inspect report; do not infer zero attempts",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
