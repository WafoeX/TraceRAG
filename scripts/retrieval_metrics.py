"""Pure document-level IR metrics; no model, DB or tokenizer dependencies.

Ranked input order is authoritative (including score ties). Linear nDCG gain
matches trec_eval ndcg_cut; unjudged documents have gain 0, not a new annotation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class EvaluationError(ValueError):
    """Malformed evaluation inputs must not silently change the denominator."""


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError("IDs must be nonempty strings; no implicit numeric ID conversion")
    return value


def aggregate_chunks(chunks: Sequence[dict], chunk_to_document: Mapping[str, str]) -> list[str]:
    """Keep first/highest-ranked chunk per document; deduplicate before cutoff.

    Provider/DB integer document IDs are NOT SciFact IDs. Resolve via the explicit
    stable chunk mapping, never by stringifying a database ID or consulting qrels.
    """
    documents, seen_chunks, seen_docs = [], set(), set()
    for row in chunks:
        if not isinstance(row, dict) or set(row) != {"chunk_id", "score"}:
            raise EvaluationError("Chunk result requires exactly chunk_id and score")
        chunk_id, score = identifier(row["chunk_id"]), row["score"]
        if chunk_id not in chunk_to_document or chunk_id in seen_chunks:
            raise EvaluationError("Unknown or duplicate ranked chunk ID")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise EvaluationError("Score must be finite and numeric")
        seen_chunks.add(chunk_id)
        doc_id = identifier(chunk_to_document[chunk_id])
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            documents.append(doc_id)
    return documents


def query_metrics(
    ranked_documents: Sequence[str], relevance: Mapping[str, int]
) -> dict[str, float]:
    if not relevance or any(type(v) is not int or v < 0 for v in relevance.values()):
        raise EvaluationError("Qrels must contain nonnegative integer relevance grades")
    for doc_id in relevance:
        identifier(doc_id)
    positive = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not positive:
        raise EvaluationError("No positive qrel; cannot silently drop this query from the mean")
    ranked = [identifier(d) for d in ranked_documents]
    if len(ranked) != len(set(ranked)):
        raise EvaluationError("Document rankings must already be deduplicated")
    dcg = sum(relevance.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:10], 1))
    ideal = sum(
        grade / math.log2(i + 1)
        for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:10], 1)
    )
    first = next((i for i, d in enumerate(ranked[:10], 1) if d in positive), None)
    return {
        "nDCG@10": dcg / ideal,
        "Recall@10": len(set(ranked[:10]) & positive) / len(positive),
        "Recall@100": len(set(ranked[:100]) & positive) / len(positive),
        "MRR@10": 0.0 if first is None else 1 / first,
    }


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    if not qrels or set(rankings) != set(qrels):
        raise EvaluationError(
            "Exactly one ranking per labeled query required; no omitted hard cases"
        )
    per_query = {identifier(q): query_metrics(rankings[q], rel) for q, rel in qrels.items()}
    return {
        "queries": len(qrels),
        "metrics": {
            name: math.fsum(row[name] for row in per_query.values()) / len(qrels)
            for name in ("nDCG@10", "Recall@10", "Recall@100", "MRR@10")
        },
        "per_query": per_query,
        "empty_rankings": sum(not ranking for ranking in rankings.values()),
        "fewer_than_100_unique_documents": sum(len(r) < 100 for r in rankings.values()),
        "metric_policy": {
            "aggregation": "macro mean over all selected labeled queries",
            "ndcg_gain": "linear relevance grade (trec_eval ndcg_cut)",
            "discount": "log2(rank+1)",
            "positive_relevance": ">0",
            "unjudged_documents": "gain 0; not asserted human-judged irrelevant",
            "short_rankings": "no padding; original relevant-doc denominator retained",
            "ties": "supplied ranking order retained; scorer does not re-sort scores",
        },
    }
