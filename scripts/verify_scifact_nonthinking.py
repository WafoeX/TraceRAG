"""Offline raw-response, cost, cohort and frozen-evidence verification. No API calls."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from openai.types.chat import ChatCompletion

import scripts.validate_scifact_nonthinking as full
from app.services import reranking, siliconflow
from scripts.evaluate_scifact import read_object, read_rows
from scripts.prepare_scifact import hash_file, write_json
from scripts.retrieval_metrics import aggregate_chunks, evaluate_rankings


def verify(folder: Path):
    report = read_object(folder / "report.json")
    data, chunks, cases, reused = full.full_inputs(
        full.experiment.SOURCE, full.experiment.CHUNKS, full.RETRIEVAL, full.REUSE
    )
    rows = read_rows(folder / "query_results.jsonl")
    by_id = {c["query_id"]: c for c in cases}
    assert len(rows) == len(by_id) == 807
    assert {r["query_id"] for r in rows} == set(by_id)
    assert [c["query_id"] for c in cases] == report["cohort_query_ids"]
    events = read_rows(folder / "cost_events.jsonl")
    starts = {e["request_id"]: e for e in events if e["event"] == "started"}
    ends = {
        e["request_id"]: {k: v for k, v in e.items() if k != "event"}
        for e in events
        if e["event"] == "finished"
    }
    assert len(events) == 1414
    assert len(starts) == len(ends) == 707 and starts.keys() == ends.keys()
    reused_ids = {r["call"]["request_id"] for r in reused}
    assert {r["call"]["request_id"] for r in rows} == set(ends) | reused_ids
    reasons = Counter()
    known_cost = Decimal(0)
    for row in rows:
        case = by_id[row["query_id"]]
        call = row["call"]
        assert call["request"] == {**case["payload"], "enable_thinking": False}
        if call["request_id"] in ends:
            assert call == ends[call["request_id"]]
            assert starts[call["request_id"]]["request"] == call["request"]
        expected = case["fused"]
        try:
            response = ChatCompletion.model_validate(call["response"])
            content = siliconflow.chat_content(response)
            siliconflow.validate_llm_scores(content, 10)
            scores = reranking._parse_scores(content, 10)
            if not scores or not any(s > 0 for s in scores):
                raise ValueError("nonpositive_scores")
            expected = full.experiment.reorder(case["fused"], scores)
            assert row["fallback"] is None
        except (siliconflow.ProviderResponseError, ValueError, KeyError) as exc:
            assert row["fallback"] is not None
            reason = "all_zero_scores" if "nonpositive" in str(exc) else type(exc).__name__
            reasons[reason] += 1
        assert row["chunks"] == expected
        assert row["chunks"][10:] == case["fused"][10:]
        assert {r["chunk_id"] for r in row["chunks"][:10]} == {
            r["chunk_id"] for r in case["fused"][:10]
        }
        usage = call.get("usage")
        if usage is None:
            assert call["estimated_cost_cny"] is None
        else:
            cost = (
                Decimal(usage["prompt_tokens"]) * Decimal(call["price_input_per_million_cny"])
                + Decimal(usage["completion_tokens"])
                * Decimal(call["price_output_per_million_cny"])
            ) / Decimal(1000000)
            assert cost == Decimal(call["estimated_cost_cny"])
            known_cost += cost
    score = full.summary(rows, cases, data, chunks, folder / "pricing.json")
    assert score == report["full_nonthinking"]
    assert known_cost == Decimal(score["costs"]["known_estimated_cost_cny"])
    mapping = {k: c["document_id"] for k, c in chunks.items()}
    baseline = evaluate_rankings(
        {c["query_id"]: aggregate_chunks(c["fused"], mapping) for c in cases}, data["qrels"]
    )
    assert baseline == report["full_fts_rrf"]
    assert score["fewer_than_100_unique_documents"] == 0
    first_ids = {c["query_id"] for c in cases[:100]}
    assert (
        full.experiment.summarize(
            [r for r in rows if r["query_id"] in first_ids],
            cases[:100],
            data,
            chunks,
            folder / "pricing.json",
        )
        == report["first100"]
    )
    assert sum(reasons.values()) == score["fallbacks"]
    for p, digest in report["frozen_hashes"].items():
        assert hash_file(full.ROOT / p) == digest
    result = {
        "status": "passed",
        "queries": 807,
        "new_started_finished": 707,
        "reused_calls": 100,
        "raw_response_rankings_verified": True,
        "metrics_recomputed": True,
        "costs_recomputed": True,
        "frozen_hashes_unchanged": True,
        "fallback_reasons": dict(reasons),
        "http_sql_blocked": True,
    }
    write_json(folder / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    with (
        patch("httpx.Client.send", side_effect=AssertionError("HTTP forbidden")),
        patch("sqlalchemy.engine.Engine.connect", side_effect=AssertionError("SQL forbidden")),
    ):
        print(verify(args.folder))


if __name__ == "__main__":
    main()
