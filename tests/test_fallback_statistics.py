"""Small synthetic checks of paired query uncertainty, not benchmark-dependent tests."""

import pytest

from scripts.analyze_scifact_fallbacks import paired_stats, percentile


def score(values, fallbacks):
    return {"per_query": {q: {"nDCG@10": v} for q, v in values.items()}, "fallback_ids": fallbacks}


def test_constant_paired_gain_and_recovery():
    before = score({"b": 0, "a": 0}, ["b", "a"])
    after = score({"b": 1, "a": 1}, [])
    result = paired_stats(before, after, ["b", "a"], iterations=20)
    assert result["nDCG_delta_95pct"] == [1, 1]
    assert result["fallback_reduction_pp_95pct"] == [100, 100]
    assert result["fallback_recovered"] == 2
    assert result["fallback_new"] == 0


def test_point_gate_is_not_interval_noninferiority():
    before = score({"b": 0.5, "a": 0.5}, ["b"])
    after = score({"b": 0.51, "a": 0.48}, [])
    result = paired_stats(before, after, ["b", "a"], iterations=1000)
    assert result["quality_point_guard_passed"]
    assert not result["quality_interval_lower_above_minus_001"]
    assert result == paired_stats(before, after, ["b", "a"], iterations=1000)
    assert "not held-out" in result["scope"]


def test_new_fallbacks_not_only_recovered_count():
    before = score({"a": 0.5, "b": 0.5}, ["a"])
    after = score({"a": 0.5, "b": 0.5}, ["b"])
    result = paired_stats(before, after, ["a", "b"], iterations=30)
    assert result["fallback_reduction_pp"] == 0
    assert result["fallback_recovered"] == result["fallback_new"] == 1
    assert result["ndcg_tied"] == 2


def test_percentile_interpolation():
    assert percentile([3, 1, 2, 0], 0.25) == pytest.approx(0.75)
