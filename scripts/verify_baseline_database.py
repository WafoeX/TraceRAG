"""Real isolated PostgreSQL checks with fake vectors, NOT a retrieval benchmark.

No model client is constructed. Reopen the saved record after Docker restart with
--existing-report PATH to check persistence without creating tables or new data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from .baseline_support import (
    DEFAULT_DATABASE_URL,
    GUARD_MARKER,
    BaselineError,
    existing_schema_url,
    prepare_schema,
    read_json,
    runtime_manifest,
    validate_database_url,
    write_json,
)
from .prepare_test_cache import CACHE_DIR, verify_cache

TEXT = "Customers may request a full refund within 30 days. Data uses AES-256 encryption."


def verify(database_url: str, output: Path, existing_report: Path | None = None) -> dict:
    validate_database_url(database_url)
    previous = read_json(existing_report) if existing_report is not None else None
    if existing_report is not None:
        if (
            not isinstance(previous, dict)
            or previous.get("status") != "passed"
            or type(previous.get("document_id")) is not int
            or previous["document_id"] <= 0
            or not isinstance(previous.get("content_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", previous["content_sha256"])
            or not isinstance(previous.get("schema"), str)
        ):
            raise BaselineError("Invalid successful infrastructure report; refuse any DDL")
        existing_schema_url(database_url, previous["schema"])
    runtime_manifest()
    if "app.database" in sys.modules:
        raise BaselineError("Run infrastructure verification in a fresh process")
    if not verify_cache():
        raise BaselineError(
            "Prepare the hash-verified tokenizer cache before database verification"
        )
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    schema = previous["schema"] if previous is not None else "infra_" + uuid4().hex
    # Fail on any existing output path rather than destroying an earlier result.
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "schema": schema,
        "model_calls": 0,
        "scope": "database infrastructure with fixed 3-D fake vectors, not retrieval quality",
        "quality_metrics": None,
        "reopen_only": previous is not None,
    }
    write_json(output / "report.json", state)
    try:
        scoped_url = (
            existing_schema_url(database_url, schema)
            if previous is not None
            else prepare_schema(database_url, schema)
        )
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        write_json(output / "report.json", state)
        raise BaselineError("Cannot prepare isolated database verification") from None
    from app.config import Settings

    settings = Settings.model_construct(
        openai_api_key="infra-only-never-sent",
        database_url=scoped_url,
        embedding_dim=3,
        chunk_size=64,
        chunk_overlap=8,
        run_db_init=False,
        rate_limit_enabled=False,
    )
    with ExitStack() as stack:
        stack.enter_context(patch("app.config.get_settings", return_value=settings))
        from app.database import SessionLocal, engine, init_db
        from app.models import Chunk, Document
        from app.services import embeddings, ingestion, openai_client, reranking, retrieval

        def forbidden_model_client():
            state["model_calls"] += 1
            raise BaselineError("Model access is forbidden in infrastructure verification")

        for module in (embeddings, reranking, openai_client):
            stack.enter_context(patch.object(module, "get_client", forbidden_model_client))
        stack.enter_context(
            patch.object(ingestion, "embed_texts", lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
        )
        stack.enter_context(patch.object(retrieval, "embed_query", lambda text: [1.0, 0.0, 0.0]))
        try:
            # Existing-report mode performs only reads, not startup DDL.
            if previous is None:
                init_db()
                init_db()
            with SessionLocal() as db:
                connection = db.connection()
                marker = connection.exec_driver_sql(
                    "SELECT marker FROM public.budgetrag_baseline_guard WHERE singleton = true"
                ).scalar_one()
                if marker != GUARD_MARKER:
                    raise BaselineError("Guard mismatch")
                actual_schema = connection.exec_driver_sql("SELECT current_schema()").scalar_one()
                if actual_schema != schema:
                    raise BaselineError("Schema mismatch")
                checks: dict[str, object]
                if previous is not None:
                    document = db.get(Document, previous["document_id"])
                    if document is None:
                        raise BaselineError("Persisted document missing")
                    checks = {"persisted_document": True}
                else:
                    if connection.exec_driver_sql("SELECT count(*) FROM documents").scalar_one():
                        raise BaselineError("New run schema was not empty")
                    document = ingestion.ingest_document(db, "Infrastructure probe", TEXT, "infra")
                    if len(document.chunks) != 1:
                        raise BaselineError("Unexpected chunk count")
                    clone = ingestion.ingest_document(db, "Infrastructure probe", TEXT, "infra")
                    if clone.id == document.id:
                        raise BaselineError("Unexpected legacy duplicate behavior")
                    checks = {
                        "schema_init_idempotent": True,
                        "original_ingestion": True,
                        "legacy_duplicate_creates_separate_document": True,
                    }
                    clone_chunk_id = clone.chunks[0].id
                    db.delete(clone)
                    db.commit()
                    if db.get(Chunk, clone_chunk_id) is not None:
                        raise BaselineError("ORM cascade delete failed")
                    checks["orm_cascade_delete"] = True
                    chunk = document.chunks[0]
                    chunk.content = TEXT + " Updated infrastructure probe."
                    db.commit()
                    updated = retrieval.keyword_search(db, "updated", 5)
                    if not updated or updated[0].document_id != document.id:
                        raise BaselineError("Persisted update/FTS failed")
                    chunk.content = TEXT
                    db.commit()
                    checks["orm_update_and_fts"] = True
                    rejected = False
                    try:
                        with db.begin_nested():
                            db.add(
                                Chunk(
                                    document_id=document.id,
                                    chunk_index=9,
                                    content="wrong dimension",
                                    embedding=[1.0, 0.0],
                                )
                            )
                            db.flush()
                    except Exception as exc:
                        rejected = True
                        checks["dimension_rejection_type"] = type(exc).__name__
                    if not rejected:
                        raise BaselineError("Dimension mismatch was not rejected")
                    checks["dimension_mismatch_rejected_without_losing_original"] = True
                content = document.chunks[0].content
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                expected_hash = (
                    previous["content_sha256"]
                    if previous is not None
                    else hashlib.sha256(TEXT.encode()).hexdigest()
                )
                if content_hash != expected_hash:
                    raise BaselineError("Persisted content hash mismatch")
                vector = retrieval.vector_search(db, "refund", 5)
                keyword = retrieval.keyword_search(db, "refund", 5)
                fused = retrieval.reciprocal_rank_fusion([vector, keyword])
                for result in (vector, keyword, fused):
                    if not result or result[0].document_id != document.id:
                        raise BaselineError("Real database retrieval infrastructure failed")
                connection = db.connection()
                doc_count = connection.exec_driver_sql(
                    "SELECT count(*) FROM documents"
                ).scalar_one()
                chunk_count = connection.exec_driver_sql("SELECT count(*) FROM chunks").scalar_one()
                if (doc_count, chunk_count) != (1, 1):
                    raise BaselineError("Unexpected persisted counts")
                indexes = [
                    row[0]
                    for row in connection.exec_driver_sql(
                        "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
                    )
                ]
                if not {"chunks_embedding_hnsw", "chunks_content_fts"}.issubset(indexes):
                    raise BaselineError("Expected HNSW/GIN indexes missing")
                checks.update(
                    real_pgvector_query=True,
                    real_english_fts=True,
                    rrf=True,
                    content_hash=True,
                    expected_indexes=True,
                )
                state.update(
                    status="passed",
                    checks=checks,
                    document_id=document.id,
                    chunk_id=document.chunks[0].id,
                    content_sha256=content_hash,
                    documents=doc_count,
                    chunks=chunk_count,
                    indexes=sorted(indexes),
                    postgres_version=connection.exec_driver_sql("SELECT version()").scalar_one(),
                    pgvector_version=connection.exec_driver_sql(
                        "SELECT extversion FROM pg_extension WHERE extname='vector'"
                    ).scalar_one(),
                )
        except Exception as exc:
            state.update(status="failed", error_type=type(exc).__name__)
        finally:
            engine.dispose()
            write_json(output / "report.json", state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-report", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.database_url, args.output, args.existing_report)
        print(
            f"Infrastructure verification: {result['status']}; model calls={result['model_calls']}"
        )
        return 0 if result["status"] == "passed" else 1
    except (BaselineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Infrastructure verification stopped: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
