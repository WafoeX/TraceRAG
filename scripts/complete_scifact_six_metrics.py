"""Offline saved-cohort metrics and pending review pack; stdlib + pure IR functions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.retrieval_metrics import aggregate_chunks, query_metrics

ROOT = Path(__file__).resolve().parents[1]
# Frozen scientific-claim wrapper used by the original paired generation, not a new prompt.
QUESTION_PREFIX = (
    "Assess this scientific claim using only the provided context. State whether it is "
    "supported, refuted, or there is insufficient evidence; explain the essential evidence "
    "and cite sources inline. Claim: "
)
SOURCES = {
    "protocol": "docs/refactor/SCIFACT_SIX_METRICS_PROTOCOL.md",
    "queries": "runs/scifact-prepared-v1/splits/development/queries.jsonl",
    "qrels": "runs/scifact-prepared-v1/splits/development/qrels.jsonl",
    "chunks": "runs/scifact-evaluation-plan-v1/chunks.jsonl",
    "plan": "runs/scifact-answers-paired-v1/plan.json",
    "answers": "runs/scifact-answers-paired-v1/answers.jsonl",
    "baseline": "runs/scifact-nonthinking-development807-v1/query_results.jsonl",
    "candidate": "runs/scifact-fallback-development807-v1/query_results.jsonl",
    "old_analysis": "runs/scifact-fallback-development807-v1/analysis.json",
    "tool": "scripts/complete_scifact_six_metrics.py",
    "metrics_source": "scripts/retrieval_metrics.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(row: dict[str, Any]) -> str:
    """This pack's row binding, not an alias for historical answer_sha256."""
    return digest(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())


def parse_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON input") from None
    if not isinstance(value, dict):
        raise ValueError("Object row required")
    return value


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            require(bool(line.strip()), "Blank JSONL row")
            yield parse_object(line)


def document_metrics(ranking: list[str], relevance: dict[str, int]) -> dict[str, float]:
    result = query_metrics(ranking, relevance)
    positive = {doc for doc, grade in relevance.items() if grade > 0}
    for k in (4, 10):
        hits = len(set(ranking[:k]) & positive)
        result[f"Hit@{k}"] = 1.0 if hits > 0 else 0.0
        result[f"Precision@{k}"] = hits / k
    result["Recall@4"] = len(set(ranking[:4]) & positive) / len(positive)
    return result


def build() -> tuple[dict[str, Any], list[dict], list[dict], list[dict]]:
    paths = {key: ROOT / value for key, value in SOURCES.items()}
    require(all(p.is_file() and not p.is_symlink() for p in paths.values()), "Missing/linked input")
    inputs = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths.values()}
    query_rows = list(rows(paths["queries"]))
    queries = {row["query_id"]: row["text"] for row in query_rows}
    require(len(queries) == len(query_rows) == 807, "Development queries changed")
    cohort = list(queries)[:100]
    selected = set(cohort)
    qrels: dict[str, dict[str, int]] = {q: {} for q in queries}
    for row in rows(paths["qrels"]):
        q, d, grade = row["query_id"], row["document_id"], row["relevance"]
        require(q in qrels and d not in qrels[q], "Invalid/duplicate qrel")
        require(type(grade) is int and grade > 0, "Positive qrel required")
        qrels[q][d] = grade
    require(all(qrels.values()), "Missing query qrels")
    chunks = {}
    for row in rows(paths["chunks"]):
        cid = row["chunk_id"]
        require(cid not in chunks, "Duplicate chunk")
        require(digest(row["text"].encode()) == row["text_sha256"], "Chunk hash mismatch")
        chunks[cid] = row
    mapping = {cid: c["document_id"] for cid, c in chunks.items()}
    plan = parse_object(paths["plan"].read_text())
    require(plan["planned_calls"] == 200, "Unexpected generation plan")
    # The observed answers must cover exactly this frozen cohort, both arms.
    expected = {(q, arm) for q in cohort for arm in ("A", "B")}
    require(
        len(plan["order"]) == 200
        and {(row["query_id"], row["arm"]) for row in plan["order"]} == expected,
        "Generation plan cohort mismatch",
    )
    require(
        list(dict.fromkeys(row["query_id"] for row in plan["order"])) == cohort,
        "Generation plan query order mismatch",
    )
    answers = {}
    for row in rows(paths["answers"]):
        key = (row["query_id"], row["arm"])
        require(key in expected and key not in answers, "Unexpected/duplicate answer")
        answers[key] = row
    require(set(answers) == expected, "Incomplete paired answer cohort")
    rankings = {}
    for arm, source in (("A", "baseline"), ("B", "candidate")):
        found = {}
        seen = set()
        for row in rows(paths[source]):
            q = row["query_id"]
            require(q in queries and q not in seen, "Unexpected/duplicate ranking")
            seen.add(q)
            if q in selected:
                found[q] = row["chunks"]
        require(seen == set(queries) and set(found) == selected, "Incomplete saved rankings")
        rankings[arm] = found
    per_query = []
    pending_contexts: dict[tuple[str, str], dict] = {}
    pending_answers = []
    for position, q in enumerate(cohort):
        for arm in ("A", "B"):
            answer = answers[q, arm]
            contexts = answer["contexts"]
            require(len(contexts) == 4, "Expected actual top4; investigate, do not drop query")
            require(
                [c["chunk_id"] for c in contexts] == [c["chunk_id"] for c in rankings[arm][q][:4]],
                "Answer contexts differ from saved ranking head",
            )
            blocks = []
            keys = []
            for source_number, c in enumerate(contexts, 1):
                saved = chunks[c["chunk_id"]]
                require(
                    all(
                        c[k] == saved[k]
                        for k in ("document_id", "text", "text_sha256", "chunk_index")
                    ),
                    "Context content/identity mismatch",
                )
                blocks.append(f'[{source_number}] (from "{c["title"]}")\n{c["text"]}')
                key = (q, c["chunk_id"])
                if key not in pending_contexts:
                    pending_contexts[key] = {
                        "query_id": q,
                        "query_position": position,
                        "query": queries[q],
                        "chunk_id": c["chunk_id"],
                        "document_id": c["document_id"],
                        "title": c["title"],
                        "text": c["text"],
                        "text_sha256": c["text_sha256"],
                        "positions": [],
                        "review_status": "pending",
                        "grade": None,
                        "reason": None,
                        "evidence_spans": [],
                        "reviewer": None,
                        "human_verified": False,
                        "independent_blind_review": False,
                    }
                pending_contexts[key]["positions"].append({"arm": arm, "source": source_number})
                keys.append({"query_id": q, "chunk_id": c["chunk_id"]})
            actual_user = answer["request"]["messages"][-1]
            require(
                actual_user
                == {
                    "role": "user",
                    "content": (
                        "Context:\n"
                        + "\n\n".join(blocks)
                        + f"\n\nQuestion: {QUESTION_PREFIX}{queries[q]}"
                    ),
                },
                "Actual request/query/context mismatch",
            )
            require(
                all(
                    answer["request"][k] == plan["config"][k]
                    for k in ("model", "temperature", "max_tokens", "stream")
                ),
                "Generation configuration mismatch",
            )
            doc_ranking = aggregate_chunks(rankings[arm][q], mapping)
            per_query.append(
                {
                    "query_id": q,
                    "query_position": position,
                    "arm": arm,
                    "ranked_document_ids": doc_ranking,
                    "metrics": document_metrics(doc_ranking, qrels[q]),
                    "positive_qrels": len(qrels[q]),
                    "actual_context_count": len(contexts),
                }
            )
            pending_answers.append(
                {
                    "query_id": q,
                    "query_position": position,
                    "arm": arm,
                    "query": queries[q],
                    "source_answer_row_sha256": canonical_hash(answer),
                    "answer": answer["answer"],
                    "contexts": keys,
                    "status": answer["status"],
                    "finish_reason": answer["finish_reason"],
                    "review_status": "pending",
                    "context_sufficiency": None,
                    "context_sufficiency_reason": None,
                    "answer_relevance": None,
                    "answer_relevance_reason": None,
                    "conclusion_support": None,
                    "assertions": [],
                    "citation_support": [],
                    "refusal": None,
                    "reviewer": None,
                    "human_verified": False,
                    "independent_blind_review": False,
                }
            )
    summaries = {}
    old = parse_object(paths["old_analysis"].read_text())["screen100"]
    for arm, old_key in (("A", "baseline"), ("B", "candidate")):
        group = [row for row in per_query if row["arm"] == arm]
        metric_names = group[0]["metrics"]
        means = {m: math.fsum(row["metrics"][m] for row in group) / 100 for m in metric_names}
        for m in ("nDCG@10", "MRR@10", "Recall@10", "Recall@100"):
            require(
                math.isclose(means[m], old[old_key]["metrics"][m], abs_tol=1e-12),
                "Historical screen100 metric mismatch",
            )
        summaries[arm] = {"queries": 100, "metrics": means}
    # A second read confirms inputs were not changed during collection; no historical writes.
    require(
        all(digest((ROOT / name).read_bytes()) == value for name, value in inputs.items()),
        "Input changed during collection",
    )
    result = {
        "schema_version": 1,
        "status": "retrieval_complete_semantic_review_pending",
        "cohort": "development_original_first100",
        "query_ids": cohort,
        "arms": {"A": "historical_nonthinking_rerank", "B": "relevance_v1_rerank"},
        "summaries": summaries,
        "deltas_B_minus_A": {
            m: summaries["B"]["metrics"][m] - summaries["A"]["metrics"][m]
            for m in summaries["A"]["metrics"]
        },
        "generation_question_prefix": QUESTION_PREFIX,
        "document_level_only": True,
        "precision_denominator": "fixed K; unjudged is nonhit, not proven irrelevant",
        "aggregation": "first document occurrence; macro all 100 queries; linear nDCG gain",
        "context_positions": sum(len(row["positions"]) for row in pending_contexts.values()),
        "unique_query_chunk_reviews": len(pending_contexts),
        "answer_reviews_pending": 200,
        "semantic_metrics": {
            "context_relevance": None,
            "faithfulness": None,
            "answer_relevance": None,
        },
        "inputs_sha256": inputs,
        "historical_screen100_metrics_match": True,
        "new_model_requests": 0,
        "database_requests": 0,
        "test_split_read": False,
        "historical_inputs_unchanged": True,
        "new_API_cost_cny": "0",
        "cost_scope": "No new API calls; historical generation and main-session AI costs unknown",
    }
    return result, per_query, list(pending_contexts.values()), pending_answers


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(
        output.is_relative_to((ROOT / "runs").resolve()) and output != ROOT / "runs",
        "Output must be a new directory under repository runs",
    )
    require(not output.exists(), "Output exists; never overwrite frozen artifacts")
    result, per_query, contexts, answers = build()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    for filename, data in (
        ("per_query.jsonl", per_query),
        ("contexts.pending.jsonl", contexts),
        ("answers.pending.jsonl", answers),
    ):
        (output / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in data)
        )
    (output / "protocol.md").write_bytes((ROOT / SOURCES["protocol"]).read_bytes())
    (output / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    lines = [
        "# 六指标补齐：第一步离线结果",
        "",
        "同100题；A旧重排，B relevance_v1。全部为文档级qrels口径。",
        "",
        "| 指标 | A | B | B−A |",
        "|---|---:|---:|---:|",
    ]
    for m, a in result["summaries"]["A"]["metrics"].items():
        b = result["summaries"]["B"]["metrics"][m]
        lines.append(f"| {m} | {a:.6f} | {b:.6f} | {b - a:+.6f} |")
    lines.extend(
        [
            "",
            f"实际上下文800个位置，去重后{len(contexts)}个query–chunk待评；回答200份待评。",
            "",
            "Context Relevance、Faithfulness和Answer Relevance本轮尚未评分，报告为null，不是0。",
            "",
            "Precision未命中不代表人工已证明无关；前4唯一文档不等于实际送入模型的4个chunks。",
            "",
            "旧screen100指标复算一致；输入hash前后未变；无新API/DB调用，不读test或.env。",
            "",
            "这是已经用于选参的development补充报告，不是独立验证或新增质量提升。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    outputs = {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    write_json(
        output / "manifest.json",
        {
            "status": "complete_stage1_only",
            "outputs_sha256": outputs,
            "inputs_sha256": result["inputs_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "status": result["status"],
                "context_positions": result["context_positions"],
                "unique_contexts": len(contexts),
            }
        )
    )


if __name__ == "__main__":
    main()
