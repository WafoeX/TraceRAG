"""Download/normalize pinned BEIR SciFact; no app, database or model imports.

Default is offline. --download explicitly permits the official HTTPS ZIP fetch.
Preserves full corpus and official train/test IDs; train is local development.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/scifact.json"
MEMBERS = {"corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"}
MAX_BYTES = 100 * 1024 * 1024


class DatasetError(ValueError):
    """Dataset integrity/shape mismatch. Do not silently repair or resample."""


def hash_file(path: Path, algorithm="sha256") -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DatasetError("Dataset redirects require separate source verification")


def download(path: Path, config: dict) -> None:
    """One bounded HTTPS attempt, default TLS verification, no redirect/retry."""
    if path.exists():
        raise DatasetError("Archive exists; verify/reuse explicitly rather than overwrite")
    url = config["download_url"]
    if url != "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip":
        raise DatasetError("Only the pinned official artifact URL is accepted")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    # Exclusive creation; on failure leave owned bytes for inspection, never retry.
    with partial.open("xb") as handle:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(url, timeout=60) as response:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > MAX_BYTES:
                    raise DatasetError("Archive exceeds download limit")
                handle.write(block)
    if hash_file(partial, "md5") != config["publisher_md5"]:
        raise DatasetError("Publisher archive checksum mismatch")
    # Preserve the verified bytes; exclusive target creation prevents replacement.
    with path.open("xb") as target, partial.open("rb") as source:
        shutil.copyfileobj(source, target)
    partial.unlink()


def read_archive(archive: Path, expected_md5: str) -> dict[str, bytes]:
    if archive.stat().st_size > MAX_BYTES or hash_file(archive, "md5") != expected_md5:
        raise DatasetError("Publisher archive checksum/size mismatch")
    result = {}
    try:
        with zipfile.ZipFile(archive) as package:
            total = 0
            for item in package.infolist():
                name = item.filename
                parts = name.split("/")
                if (
                    name.startswith("/")
                    or ".." in parts
                    or "\\" in name
                    or (item.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise DatasetError("Unsafe archive member")
                if item.is_dir():
                    continue
                if not name.startswith("scifact/") or name[8:] not in MEMBERS:
                    raise DatasetError("Unexpected archive member")
                key = name[8:]
                total += item.file_size
                if key in result or total > MAX_BYTES:
                    raise DatasetError("Duplicate member or excessive expanded size")
                result[key] = package.read(item)  # CRC verified by zipfile.
    except (zipfile.BadZipFile, RuntimeError):
        raise DatasetError("Invalid ZIP/CRC") from None
    if set(result) != MEMBERS:
        raise DatasetError("Missing expected corpus/query/qrels files")
    return result


def jsonl(raw: bytes, *, corpus: bool) -> dict[str, dict]:
    result = {}
    try:
        for line in raw.decode("utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DatasetError("JSONL row must be an object")
            ident, text = row.get("_id"), row.get("text")
            if (
                not isinstance(ident, str)
                or not ident.strip()
                or ident in result
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise DatasetError("Duplicate/empty/invalid ID or text")
            record = {"document_id" if corpus else "query_id": ident, "text": text}
            if corpus:
                if not isinstance(row.get("title"), str):
                    raise DatasetError("Corpus title must be a string")
                record["title"] = row["title"]
            result[ident] = record
    except (UnicodeError, json.JSONDecodeError):
        raise DatasetError("Invalid UTF-8/JSONL") from None
    return result


def qrels(raw: bytes, queries: dict, corpus: dict) -> list[dict]:
    result, seen = [], set()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise DatasetError("Unexpected qrels header")
        for row in reader:
            qid, did = row["query-id"], row["corpus-id"]
            if qid not in queries or did not in corpus or (qid, did) in seen:
                raise DatasetError("Dangling or duplicate qrel")
            score = row["score"]
            if score is None or not score.isdecimal() or int(score) <= 0:
                raise DatasetError("SciFact requires positive integer relevance")
            seen.add((qid, did))
            result.append({"query_id": qid, "document_id": did, "relevance": int(score)})
    except UnicodeError:
        raise DatasetError("Invalid qrels UTF-8") from None
    if not result:
        raise DatasetError("Empty qrels split")
    return result


def prepare(archive: Path, output: Path, config: dict) -> dict:
    if output.exists():
        raise DatasetError("Output exists; artifacts are immutable")
    raw = read_archive(archive, config["publisher_md5"])
    if config.get("archive_sha256") and hash_file(archive) != config["archive_sha256"]:
        raise DatasetError("Pinned local archive SHA256 mismatch")
    corpus = jsonl(raw["corpus.jsonl"], corpus=True)
    queries = jsonl(raw["queries.jsonl"], corpus=False)
    train = qrels(raw["qrels/train.tsv"], queries, corpus)
    test = qrels(raw["qrels/test.tsv"], queries, corpus)
    train_ids, test_ids = {r["query_id"] for r in train}, {r["query_id"] for r in test}
    if train_ids & test_ids or train_ids | test_ids != set(queries):
        raise DatasetError("Query split overlap or unassigned queries")

    def normalize(text):
        return " ".join(text.casefold().split())

    test_texts = {normalize(queries[q]["text"]) for q in test_ids}
    excluded = {q for q in train_ids if normalize(queries[q]["text"]) in test_texts}
    if config["split_policy"].get("development_overlap_policy") != (
        "exclude_train_exact_normalized_test_text"
    ):
        raise DatasetError("Explicit development overlap policy required")
    development_ids = train_ids - excluded
    development = [r for r in train if r["query_id"] in development_ids]
    if not development_ids:
        raise DatasetError("No development queries remain after overlap quarantine")
    quarantine = [
        {
            "train_query_id": q,
            "matching_test_query_ids": sorted(
                t
                for t in test_ids
                if normalize(queries[t]["text"]) == normalize(queries[q]["text"])
            ),
        }
        for q in sorted(excluded)
    ]
    counts = {"corpus": len(corpus), "train_queries": len(train_ids), "test_queries": len(test_ids)}
    if counts != config["expected_counts"]:
        raise DatasetError("Counts differ from frozen dataset configuration")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    shutil.copyfile(archive, output / "source/scifact.zip")
    for name, data in raw.items():
        target = output / "raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_rows(output / "corpus.jsonl", list(corpus.values()))
    for name, ids, labels in (
        ("development", development_ids, development),
        ("test", test_ids, test),
    ):
        split = output / "splits" / name
        split.mkdir(parents=True)
        write_rows(split / "queries.jsonl", [v for q, v in queries.items() if q in ids])
        write_rows(split / "qrels.jsonl", labels)
    write_json(output / "quarantined_train_queries.json", quarantine)
    write_json(output / "dataset_config.json", config)
    shutil.copyfile(Path(__file__), output / "preparation_source.py")
    # Test labels are validated structurally, not consumed to choose models/parameters.
    report = {
        "status": "prepared",
        "dataset": config["dataset"],
        "counts": counts,
        "development_queries": len(development_ids),
        "qrels": {"original_train": len(train), "development": len(development), "test": len(test)},
        "quarantined_train_queries": quarantine,
        "split_policy": config["split_policy"],
        "evaluation_unit": "document_id",
        "checks": {
            "publisher_md5_match": True,
            "archive_crc": True,
            "all_ids_unique": True,
            "all_qrels_resolve": True,
            "all_queries_have_positive_qrels": True,
            "query_id_overlap": 0,
            "original_train_test_text_overlap_train_queries": len(excluded),
            "development_test_normalized_text_overlap": 0,
            "full_corpus_no_sampling": True,
        },
        "archive_sha256": hash_file(archive),
        "archive_md5": hash_file(archive, "md5"),
        "files": {
            p.relative_to(output).as_posix(): {"sha256": hash_file(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "licenses": config["licenses"],
        "real_model_calls": 0,
        "database_writes": 0,
        "model_api_cost_cny": "0",
        "cost_scope": "No embedding/rerank/generation requests",
        "retrieval_metrics": None,
    }
    write_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / ".test-cache/datasets/scifact.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise DatasetError("Output exists")
        config = json.loads(CONFIG.read_text())
        if not args.archive.exists():
            if not args.download:
                raise DatasetError("Missing archive; explicit --download required")
            download(args.archive, config)
        report = prepare(args.archive, args.output, config)
        print(
            json.dumps(
                {k: report[k] for k in ("status", "counts", "qrels", "archive_sha256")},
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"SciFact preparation stopped: {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
