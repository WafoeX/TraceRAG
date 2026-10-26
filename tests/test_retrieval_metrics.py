"""Hand-calculated document metrics and aggregation negative controls."""

import math

import pytest

from scripts.retrieval_metrics import (
    EvaluationError,
    aggregate_chunks,
    evaluate_rankings,
    query_metrics,
)


def test_rank_two_single_relevant_hand_calculation():
    metrics = query_metrics(["unjudged", "relevant"], {"relevant": 1})
    assert metrics == {"nDCG@10": 1 / math.log2(3), "Recall@10": 1, "Recall@100": 1, "MRR@10": 0.5}


def test_graded_linear_gain_not_exponential():
    metrics = query_metrics(["low", "high"], {"high": 3, "low": 1})
    assert metrics["nDCG@10"] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert metrics["Recall@10"] == 1


def test_multiple_relevant_and_unjudged():
    result = query_metrics(["u", "a"], {"a": 1, "b": 1, "zero": 0})
    assert result["Recall@10"] == result["Recall@100"] == 0.5
    assert result["MRR@10"] == 0.5
    assert result["nDCG@10"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_cutoffs_are_document_rank_and_mrr_is_truncated():
    ranked = [f"u{i}" for i in range(10)] + ["relevant"]
    assert query_metrics(ranked, {"relevant": 1}) == {
        "nDCG@10": 0,
        "Recall@10": 0,
        "Recall@100": 1,
        "MRR@10": 0,
    }
    ranked = [f"u{i}" for i in range(100)] + ["relevant"]
    assert query_metrics(ranked, {"relevant": 1})["Recall@100"] == 0


def test_empty_ranking_is_zero_and_remains_in_macro_denominator():
    result = evaluate_rankings({"q1": ["a"], "q2": []}, {"q1": {"a": 1}, "q2": {"b": 1}})
    assert all(value == 0.5 for value in result["metrics"].values())
    assert result["queries"] == 2 and result["empty_rankings"] == 1
    assert result["fewer_than_100_unique_documents"] == 2


@pytest.mark.parametrize("bad", [{}, {"a": 0}, {"a": -1}, {"a": True}, {"a": 0.5}])
def test_invalid_or_no_positive_qrels_not_omitted(bad):
    with pytest.raises(EvaluationError):
        query_metrics([], bad)


@pytest.mark.parametrize("rankings", [{}, {"extra": []}, {"q": [], "extra": []}])
def test_exact_query_coverage_required(rankings):
    with pytest.raises(EvaluationError):
        evaluate_rankings(rankings, {"q": {"a": 1}})


def test_duplicate_document_ranking_not_double_counted():
    with pytest.raises(EvaluationError):
        query_metrics(["a", "a"], {"a": 1})


def test_chunk_dedup_before_document_cutoff_and_preserve_rank_order():
    mapping = {f"c{i}": "same" for i in range(20)} | {"last": "target"}
    chunks = [{"chunk_id": f"c{i}", "score": 0} for i in range(20)]
    # Scores deliberately do not determine order; different stages use different scales.
    chunks.append({"chunk_id": "last", "score": 999})
    docs = aggregate_chunks(chunks, mapping)
    assert docs == ["same", "target"]
    assert query_metrics(docs, {"target": 1})["MRR@10"] == 0.5
    assert query_metrics(docs, {"target": 1})["Recall@10"] == 1


@pytest.mark.parametrize(
    "row",
    [
        {"chunk_id": "unknown", "score": 1},
        {"chunk_id": 1, "score": 1},
        {"chunk_id": "c", "score": True},
        {"chunk_id": "c", "score": float("nan")},
        {"chunk_id": "c", "score": float("inf")},
        {"chunk_id": "c", "score": None},
        {"chunk_id": "c", "score": 1, "document_id": "forged"},
    ],
)
def test_chunk_input_negative_controls(row):
    with pytest.raises(EvaluationError):
        aggregate_chunks([row], {"c": "dataset-id"})


def test_duplicate_chunk_and_bad_mapping_rejected():
    row = {"chunk_id": "c", "score": 1}
    with pytest.raises(EvaluationError):
        aggregate_chunks([row, row], {"c": "d"})
    with pytest.raises(EvaluationError):
        aggregate_chunks([row], {"c": ""})
    assert aggregate_chunks([], {}) == []
