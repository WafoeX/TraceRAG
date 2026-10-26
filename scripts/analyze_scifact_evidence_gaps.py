"""Offline attribution of frozen Hit@4-but-insufficient development contexts.

No application imports, clients, embeddings, database access, or semantic scoring.
Primary categories are explicit AI annotations; structural checks do not prove
scientific correctness, corpus-wide absence, or candidate recoverability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LABELS = ROOT / "configs/evaluation/scifact.evidence_gaps.v1.json"
PACK = RUNS / "scifact-six-metrics-v1"
CONTEXT = RUNS / "scifact-deepeval-context100-v1"
ASSESSMENT = RUNS / "scifact-answer-assessment200-v1"
PREPARED = RUNS / "scifact-prepared-v1"
CHUNKS = RUNS / "scifact-evaluation-plan-v1/chunks.jsonl"
RANKINGS = {
    "A": RUNS / "scifact-nonthinking-development807-v1/query_results.jsonl",
    "B": RUNS / "scifact-fallback-development807-v1/query_results.jsonl",
}
BLOCKED: list[str] = []


def audit(event, values):
    if event in {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.bind",
        "sqlite3.connect",
    }:
        BLOCKED.append(event)
        raise RuntimeError("Offline evidence analysis: network/database forbidden")
    if event == "open" and isinstance(values[0], str | bytes):
        path = Path(values[0].decode() if isinstance(values[0], bytes) else values[0])
        if path.name.startswith(".env") or "test" in path.parts:
            BLOCKED.append("secret_or_test_open")
            raise RuntimeError("Offline evidence analysis: secrets/test data forbidden")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read saved JSON: {path}") from exc


def rows(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read saved JSONL: {path}") from exc


def unique(records, key):
    result = {key(row): row for row in records}
    if len(result) != len(records):
        raise ValueError("Duplicate source key")
    return result


def target(arm):
    return (
        arm["document_qrels_metrics"]["Hit@4"] == 1 and arm["context_sufficiency"] == "insufficient"
    )


def covered_chars(length, spans):
    """Union length, not sum: overlapping chunks are not double counted."""
    end = total = 0
    for start, stop in sorted(spans):
        if not 0 <= start < stop <= length:
            raise ValueError("Invalid source interval")
        total += max(0, stop - max(end, start))
        end = max(end, stop)
    return total


def document_coverage(document, raw, pieces, sent):
    full = document["title"] + "\n\n" + document["text"]
    if (document["title"], document["text"]) != (raw["title"], raw["text"]):
        raise ValueError("Prepared/raw corpus text mismatch")
    spans = []
    for piece in pieces:
        text = piece["text"]
        if text_sha(text) != piece["text_sha256"]:
            raise ValueError("Chunk hash mismatch")
        start = full.find(text)
        if not text or start < 0 or full.find(text, start + 1) >= 0:
            raise ValueError("Chunk cannot be uniquely anchored to local document")
        if piece["chunk_id"] in sent:
            spans.append((start, start + len(text)))
    chars = covered_chars(len(full), spans)
    return {
        "document_id": document["document_id"],
        "raw_prepared_text_equal": True,
        "local_document_sha256": text_sha(full),
        "local_document_chars_including_title": len(full),
        "manifest_chunk_count": len(pieces),
        "sent_chunk_count": len(spans),
        "sent_document_intervals": spans,
        "covered_characters": chars,
        "full_local_document_sent": chars == len(full),
        "whole_corpus_answer_absence": "not_established",
    }


def validate_labels(specs, query_ids):
    if specs["schema_version"] != 1:
        raise ValueError("Unknown annotation schema")
    labels = unique(specs["cases"], lambda r: r["query_id"])
    if set(labels) != set(query_ids):
        raise ValueError("Annotations must exactly cover the selected query union")
    for label in labels.values():
        if label["category"] not in specs["categories"] or not label["missing_conditions"]:
            raise ValueError("Invalid category or empty necessary-condition gap")
    return labels


def write_new(path, value, lines=False):
    with path.open("x") as handle:
        if lines:
            for row in value:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def analyze(output):
    output = output.resolve()
    if output.exists() or output.parent != RUNS.resolve():
        raise ValueError("Output must be a new immediate child of runs/; no overwrites")
    paths = [
        Path(__file__),
        LABELS,
        PACK / "answers.pending.jsonl",
        CONTEXT / "contexts.reviewed.jsonl",
        CONTEXT / "metrics.jsonl",
        ASSESSMENT / "pipeline_per_query.jsonl",
        PREPARED / "corpus.jsonl",
        PREPARED / "raw/corpus.jsonl",
        PREPARED / "splits/development/qrels.jsonl",
        CHUNKS,
        *RANKINGS.values(),
    ]
    # Verify the original frozen source manifests before consuming their evidence.
    for directory, filename in [
        (PACK, "manifest.json"),
        (CONTEXT, "verification.json"),
        (ASSESSMENT, "pipeline_verification.json"),
    ]:
        manifest = directory / filename
        paths.append(manifest)
        for name, expected in load(manifest)["outputs_sha256"].items():
            path = directory / name
            if sha(path) != expected:
                raise ValueError(f"Frozen output hash mismatch: {path.name}")
            paths.append(path)
    # The collector binds rankings/corpus/chunks to the original experiment too.
    for name, expected in load(PACK / "manifest.json")["inputs_sha256"].items():
        path = ROOT / name
        if path in paths and sha(path) != expected:
            raise ValueError(f"Frozen collector input hash mismatch: {path.name}")
    preparation = RUNS / "scifact-evaluation-plan-v1/source_preparation.json"
    expected_raw = load(preparation)["files"]["raw/corpus.jsonl"]["sha256"]
    if sha(PREPARED / "raw/corpus.jsonl") != expected_raw:
        raise ValueError("Raw corpus does not match original preparation")
    paths.append(preparation)
    paths = list(dict.fromkeys(paths))
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    pairs = rows(ASSESSMENT / "pipeline_per_query.jsonl")
    selected = [p for p in pairs if any(target(p[a]) for a in ("A", "B"))]
    specs = load(LABELS)
    labels = validate_labels(specs, [p["query_id"] for p in selected])
    if len(pairs) != 100 or len(selected) != 26:
        raise ValueError("Frozen 100-query cohort or selected 26-query union changed")
    pending = unique(rows(PACK / "answers.pending.jsonl"), lambda r: (r["query_id"], r["arm"]))
    metrics = unique(rows(CONTEXT / "metrics.jsonl"), lambda r: (r["query_id"], r["arm"]))
    reviewed = unique(
        rows(CONTEXT / "contexts.reviewed.jsonl"), lambda r: (r["query_id"], r["chunk_id"])
    )
    chunks = unique(rows(CHUNKS), lambda r: r["chunk_id"])
    corpus = unique(rows(PREPARED / "corpus.jsonl"), lambda r: r["document_id"])
    raw = unique(rows(PREPARED / "raw/corpus.jsonl"), lambda r: r["_id"])
    qrels = defaultdict(set)
    for rel in rows(PREPARED / "splits/development/qrels.jsonl"):
        if rel["relevance"] > 0:
            qrels[rel["query_id"]].add(rel["document_id"])
    by_document = defaultdict(list)
    for chunk in chunks.values():
        by_document[chunk["document_id"]].append(chunk)
    rankings = {arm: unique(rows(path), lambda r: r["query_id"]) for arm, path in RANKINGS.items()}
    cases = []
    arm_rows = {"A": [], "B": []}
    for pair in selected:
        q = pair["query_id"]
        case = {"query_id": q, "query_position": pair["query_position"], **labels[q], "arms": {}}
        for arm in ("A", "B"):
            answer, metric = pending[q, arm], metrics[q, arm]
            sent = [chunks[c["chunk_id"]] for c in answer["contexts"]]
            ranking = rankings[arm][q]
            if [c["chunk_id"] for c in sent] != [c["chunk_id"] for c in ranking["chunks"][:4]]:
                raise ValueError("Actual sent context is not the saved arm's top4")
            if metric["context_sufficiency"] != pair[arm]["context_sufficiency"]:
                raise ValueError("Frozen sufficiency join mismatch")
            case["query"] = answer["query"]
            sent_ids = {c["chunk_id"] for c in sent}
            related_documents = []
            for did in sorted(qrels[q]):
                related_documents.append(
                    document_coverage(corpus[did], raw[did], by_document[did], sent_ids)
                )
            evidence = []
            for c in sent:
                annotation = reviewed[q, c["chunk_id"]]
                if annotation["text"] != c["text"] or annotation["text_sha256"] != c["text_sha256"]:
                    raise ValueError("Reviewed text binding mismatch")
                for span in annotation["evidence_spans"]:
                    if c["text"][span["start"] : span["end"]] != span["quote"]:
                        raise ValueError("Frozen evidence span mismatch")
                evidence.append(
                    {
                        k: annotation[k]
                        for k in (
                            "chunk_id",
                            "document_id",
                            "text_sha256",
                            "grade",
                            "reason",
                            "evidence_spans",
                        )
                    }
                )
            if [v["grade"] for v in evidence] != pair[arm]["actual_top4_chunk_grades"]:
                raise ValueError("Frozen grade join mismatch")
            record = {
                "in_target_stratum": target(pair[arm]),
                "frozen_sufficiency": metric["context_sufficiency"],
                "frozen_sufficiency_reason": metric["context_sufficiency_reason"],
                "qrels_hit4": pair[arm]["document_qrels_metrics"]["Hit@4"],
                "qrel_document_coverage": related_documents,
                "identical_text_duplicate_slots": 4 - len({c["text_sha256"] for c in sent}),
                "same_document_extra_slots": 4 - len({c["document_id"] for c in sent}),
                "sent_evidence": evidence,
                "candidate_recoverability": "not_systematically_semantically_reviewed",
            }
            case["arms"][arm] = record
            if record["in_target_stratum"]:
                arm_rows[arm].append({**record, "category": case["category"]})
        cases.append(case)
    if [len(arm_rows[a]) for a in ("A", "B")] != [25, 24]:
        raise ValueError("Frozen target counts changed")
    summaries = {}
    for arm, group in arm_rows.items():
        summaries[arm] = {
            "target_query_arm_count": len(group),
            "primary_gap_categories": dict(Counter(r["category"] for r in group)),
            "at_least_one_full_qrel_document_sent": sum(
                any(d["full_local_document_sent"] for d in r["qrel_document_coverage"])
                for r in group
            ),
            "all_positive_qrel_documents_fully_sent": sum(
                all(d["full_local_document_sent"] for d in r["qrel_document_coverage"])
                for r in group
            ),
            "identical_text_duplicate_slots": sum(
                r["identical_text_duplicate_slots"] for r in group
            ),
            "cases_with_same_document_extra_slots": sum(
                r["same_document_extra_slots"] > 0 for r in group
            ),
        }
    probe = specs["candidate_probe"]
    ranked = rankings["B"][probe["query_id"]]
    matches = []
    for position, item in enumerate(ranked["chunks"], 1):
        c = chunks[item["chunk_id"]]
        if probe["literal"].casefold() in c["text"].casefold():
            matches.append(
                {"rank": position, "chunk_id": c["chunk_id"], "document_id": c["document_id"]}
            )
    focal = next(c for c in chunks.values() if c["document_id"] == probe["reviewed_document_id"])
    start = focal["text"].find(probe["reviewed_quote"])
    if start < 0 or not any(m["chunk_id"] == focal["chunk_id"] for m in matches):
        raise ValueError("Candidate probe quote is not bound to a matching saved candidate")
    request_text = ranked["call"]["request"]["messages"][1]["content"]
    probe_result = {
        **probe,
        "saved_B_fallback": ranked["fallback"],
        "saved_rerank_preview_count": len(
            re.findall(r"^\[\d+\] \(from ", request_text, re.MULTILINE)
        ),
        "matching_saved_candidate_ranks": matches,
        "quote_binding": {
            "chunk_id": focal["chunk_id"],
            "text_sha256": focal["text_sha256"],
            "start": start,
            "end": start + len(probe["reviewed_quote"]),
        },
        "literal_in_saved_rerank_passages": probe["literal"].casefold()
        in request_text.split("Passages:\n", 1)[1].casefold(),
        "complete_claim_sufficiency": "not_established",
        "is_systematic_full_candidate_review": False,
    }
    summary = {
        "status": "frozen_context_gap_attribution_complete_candidate_recoverability_pending",
        "selection": "Hit@4 == 1 and frozen context_sufficiency == insufficient in A or B",
        "reviewer": specs["reviewer"],
        "unique_query_ids": len(cases),
        "query_arm_cases": sum(len(group) for group in arm_rows.values()),
        "primary_categories_unique_queries": dict(Counter(c["category"] for c in cases)),
        "category_definitions": specs["categories"],
        "summaries": summaries,
        "exploratory_candidate_probe": probe_result,
        "new_model_API_calls": 0,
        "database_requests": 0,
        "main_session_AI_cost_cny": None,
        "limits": [
            "Nonblind main-assistant AI taxonomy over frozen development reviews; not gold.",
            "49 arm-cases share 26 query IDs and related claims/studies; not independent trials.",
            "Full local document coverage does not imply sufficiency or full-paper availability.",
            "No confirmed chunk-truncation or identical-text-duplication cause in this stratum.",
            "Unsent candidates/other documents have no systematic semantic review in this run.",
            "Literal matches do not establish sufficiency or whole-corpus absence.",
            "No new generation, quality gain, default change or test evaluation.",
        ],
    }
    if BLOCKED or any(sha(ROOT / name) != value for name, value in before.items()):
        raise ValueError("Forbidden event or frozen input mutation")
    output.mkdir()
    write_new(output / "cases.jsonl", cases, lines=True)
    write_new(output / "summary.json", summary)
    write_new(
        output / "verification.json",
        {
            "status": "passed",
            "inputs_sha256": before,
            "outputs_sha256": {
                name: sha(output / name) for name in ("cases.jsonl", "summary.json")
            },
            "frozen_input_hashes_unchanged": True,
            "forbidden_runtime_events": BLOCKED,
            "semantic_judgments_independently_validated": False,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "queries": len(cases),
                "query_arm_cases": summary["query_arm_cases"],
                "summaries": summaries,
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.addaudithook(audit)
    analyze(args.output)


if __name__ == "__main__":
    main()
