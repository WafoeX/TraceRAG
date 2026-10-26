"""Unit tests for Reciprocal Rank Fusion — pure, no network or DB."""
from app.services.retrieval import RetrievedChunk, reciprocal_rank_fusion


def _chunk(cid: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=1,
        document_title="doc",
        chunk_index=cid,
        content=f"content {cid}",
        score=0.0,
    )


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_items_ranked_high_in_both_lists_win():
    vector = [_chunk(1), _chunk(2), _chunk(3)]
    keyword = [_chunk(2), _chunk(1), _chunk(4)]
    fused = reciprocal_rank_fusion([vector, keyword])
    top_two = {c.chunk_id for c in fused[:2]}
    # 1 and 2 appear near the top of BOTH lists → they should fuse to the top.
    assert top_two == {1, 2}


def test_fusion_deduplicates_and_unions():
    vector = [_chunk(1), _chunk(2)]
    keyword = [_chunk(2), _chunk(3)]
    fused = reciprocal_rank_fusion([vector, keyword])
    assert sorted(c.chunk_id for c in fused) == [1, 2, 3]  # union, no duplicates


def test_scores_are_descending():
    fused = reciprocal_rank_fusion([[_chunk(1), _chunk(2), _chunk(3)]])
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)
