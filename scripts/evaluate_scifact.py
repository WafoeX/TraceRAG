"""Offline SciFact document evaluation and stable chunk manifest (no retrieval).

Without --predictions, write an index-input plan only. With predictions, score
all queries in the chosen split. --fixture marks synthetic plumbing checks;
test scoring requires --acknowledge-test-scoring. No model or DB clients exist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from scripts.prepare_scifact import ROOT, hash_file, write_json, write_rows
from scripts.retrieval_metrics import (
    EvaluationError,
    aggregate_chunks,
    evaluate_rankings,
    identifier,
)

PROTOCOL = ROOT / "configs/evaluation/scifact.protocol.json"
DEFAULT_PREPARED = ROOT / "runs/scifact-prepared-v1"


def parse_rows(data: bytes) -> list[dict]:
    try:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    except (UnicodeError, ValueError):
        raise EvaluationError("Cannot read JSONL input") from None
    if any(not isinstance(r, dict) for r in rows):
        raise EvaluationError("JSONL must contain objects")
    return rows


def read_rows(path: Path) -> list[dict]:
    try:
        return parse_rows(path.read_bytes())
    except OSError:
        raise EvaluationError("Cannot read JSONL file") from None


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise EvaluationError("Cannot read JSON object input") from None
    if not isinstance(value, dict):
        raise EvaluationError("Expected a JSON object")
    return value


def load_prepared(folder: Path, split: str) -> dict[str, Any]:
    if split not in {"development", "test"}:
        raise EvaluationError("Unknown evaluation split")
    folder = folder.resolve()
    report = read_object(folder / "report.json")
    if report.get("status") != "prepared" or report.get("dataset") != "beir-scifact":
        raise EvaluationError("Prepared SciFact artifact required")
    required = {
        "dataset_config.json",
        "corpus.jsonl",
        f"splits/{split}/queries.jsonl",
        f"splits/{split}/qrels.jsonl",
    }
    if not required <= set(report["files"]):
        raise EvaluationError("Missing source manifest entries")
    for name, info in report["files"].items():
        candidate = folder / name
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(folder):
            raise EvaluationError("Unsafe source manifest path")
        if hash_file(candidate) != info["sha256"] or candidate.stat().st_size != info["bytes"]:
            raise EvaluationError("Prepared file hash/size mismatch")
    corpus = {}
    for row in read_rows(folder / "corpus.jsonl"):
        doc_id = identifier(row.get("document_id"))
        if (
            doc_id in corpus
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
            or not isinstance(row.get("title"), str)
        ):
            raise EvaluationError("Duplicate or invalid corpus row")
        corpus[doc_id] = row
    queries = {}
    for row in read_rows(folder / f"splits/{split}/queries.jsonl"):
        query_id = identifier(row.get("query_id"))
        if query_id in queries or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise EvaluationError("Duplicate or invalid query row")
        queries[query_id] = row["text"]
    qrels: dict[str, dict[str, int]] = {q: {} for q in queries}
    for row in read_rows(folder / f"splits/{split}/qrels.jsonl"):
        query_id, doc_id = identifier(row.get("query_id")), identifier(row.get("document_id"))
        grade = row.get("relevance")
        if (
            query_id not in queries
            or doc_id not in corpus
            or doc_id in qrels[query_id]
            or type(grade) is not int
            or grade <= 0
        ):
            raise EvaluationError("Invalid, duplicate or dangling qrel")
        qrels[query_id][doc_id] = grade
    expected = (
        report["development_queries"]
        if split == "development"
        else report["counts"]["test_queries"]
    )
    if (
        not queries
        or len(queries) != expected
        or any(not rel for rel in qrels.values())
        or len(corpus) != report["counts"]["corpus"]
    ):
        raise EvaluationError("Prepared counts/qrels coverage mismatch")
    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "report": report,
        "report_sha256": hash_file(folder / "report.json"),
        "folder": folder,
    }


def load_chunker():
    from scripts.prepare_test_cache import CACHE_DIR, verify_cache

    if not verify_cache():
        raise EvaluationError("Hash-verified tokenizer cache required; no implicit downloads")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    from app.services.chunking import chunk_text

    return chunk_text


def build_chunks(corpus, protocol, corpus_sha256, chunker) -> list[dict]:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol") != "scifact-document-evaluation-v1"
        or protocol.get("chunk_size") != 800
        or protocol.get("chunk_overlap") != 120
    ):
        raise EvaluationError("Unknown frozen chunk protocol")
    namespace = hashlib.sha256(
        json.dumps([corpus_sha256, protocol], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    chunks = []
    for doc_id, doc in corpus.items():
        text = (doc["title"] + "\n\n" + doc["text"]).strip()
        pieces = chunker(text, protocol["chunk_size"], protocol["chunk_overlap"])
        if not pieces:
            raise EvaluationError("Document produced no chunks")
        for index, piece in enumerate(pieces):
            if not isinstance(piece, str) or not piece.strip():
                raise EvaluationError("Invalid chunk content")
            content_hash = hashlib.sha256(piece.encode()).hexdigest()
            stable_id = hashlib.sha256(
                json.dumps([namespace, doc_id, index, content_hash], ensure_ascii=False).encode()
            ).hexdigest()
            chunks.append(
                {
                    "chunk_id": "scifact:" + stable_id,
                    "document_id": doc_id,
                    "chunk_index": index,
                    "text": piece,
                    "text_sha256": content_hash,
                }
            )
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise EvaluationError("Stable chunk ID collision")
    return chunks


def score_predictions(data: bytes, queries, qrels, chunks) -> dict:
    chunk_to_document = {c["chunk_id"]: c["document_id"] for c in chunks}
    rankings = {}
    for row in parse_rows(data):
        if set(row) != {"query_id", "chunks"}:
            raise EvaluationError("Prediction requires exactly query_id and chunks")
        query_id = identifier(row["query_id"])
        if query_id in rankings or query_id not in queries or not isinstance(row["chunks"], list):
            raise EvaluationError("Duplicate/unknown query or invalid chunks list")
        rankings[query_id] = aggregate_chunks(row["chunks"], chunk_to_document)
    result = evaluate_rankings(rankings, qrels)
    result["document_rankings"] = rankings
    return result


def run(
    prepared: Path,
    output: Path,
    *,
    split="development",
    predictions: Path | None = None,
    fixture=False,
    acknowledge_test_scoring=False,
) -> dict:
    if output.exists():
        raise EvaluationError("Output already exists; never overwrite an evaluation artifact")
    if output.resolve().is_relative_to(prepared.resolve()):
        raise EvaluationError("Output cannot be inside the immutable prepared dataset")
    if fixture and predictions is None:
        raise EvaluationError("--fixture requires --predictions")
    if split == "test" and predictions is not None and not acknowledge_test_scoring:
        raise EvaluationError("Explicit test-scoring acknowledgement required")
    data = load_prepared(prepared, split)
    protocol = read_object(PROTOCOL)
    chunks = build_chunks(
        data["corpus"], protocol, data["report"]["files"]["corpus.jsonl"]["sha256"], load_chunker()
    )
    prediction_bytes = None if predictions is None else predictions.read_bytes()
    result = (
        None
        if prediction_bytes is None
        else score_predictions(prediction_bytes, data["queries"], data["qrels"], chunks)
    )
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "chunks.jsonl", chunks)
    write_json(output / "protocol.json", protocol)
    write_json(output / "source_preparation.json", data["report"])
    (output / "evaluator_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "metrics_source.py").write_bytes((ROOT / "scripts/retrieval_metrics.py").read_bytes())
    (output / "chunker_source.py").write_bytes((ROOT / "app/services/chunking.py").read_bytes())
    from scripts.prepare_test_cache import TOKENIZER_SHA256

    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "tiktoken": importlib.metadata.version("tiktoken"),
            "required_tokenizer_sha256": TOKENIZER_SHA256,
        },
    )
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        (output / name).write_bytes((ROOT / name).read_bytes())
    if prediction_bytes is not None:
        (output / "predictions.jsonl").write_bytes(prediction_bytes)
        write_json(output / "scores.json", result)
    state = {
        "status": "planned" if predictions is None else "offline_scored",
        "dataset": "beir-scifact",
        "split": split,
        "documents": len(data["corpus"]),
        "chunks": len(chunks),
        "queries": len(data["queries"]),
        "metrics": None if result is None else result["metrics"],
        "prediction_kind": "none"
        if predictions is None
        else ("fixture" if fixture else "external"),
        "quality_claim": "none"
        if predictions is None or fixture
        else "offline scoring of supplied rankings; retrieval provenance unverified",
        "backend_status": protocol["backend_status"],
        "model_calls_in_this_step": 0,
        "database_operations_in_this_step": 0,
        "model_api_cost_cny_in_this_step": "0",
        "upstream_prediction_generation_cost_cny": None,
        "source_preparation_report_sha256": data["report_sha256"],
        "manifest": {
            p.name: {"sha256": hash_file(p), "bytes": p.stat().st_size}
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    }
    write_json(output / "status.json", state)
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--acknowledge-test-scoring", action="store_true")
    args = parser.parse_args(argv)
    try:
        state = run(
            args.prepared,
            args.output,
            split=args.split,
            predictions=args.predictions,
            fixture=args.fixture,
            acknowledge_test_scoring=args.acknowledge_test_scoring,
        )
        print(
            json.dumps(
                {
                    k: state[k]
                    for k in (
                        "status",
                        "split",
                        "documents",
                        "chunks",
                        "queries",
                        "prediction_kind",
                        "metrics",
                    )
                }
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Offline SciFact evaluation stopped: {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
