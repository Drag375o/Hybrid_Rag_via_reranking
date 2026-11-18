import pytest

from conftest import make_chunk
from hybrid_rag.indexing import IndexStore
from hybrid_rag.retrieval import HybridRetriever, reciprocal_rank_fusion


# --- Reciprocal Rank Fusion -------------------------------------------------

def test_rrf_matches_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_over_single_top_rank():
    # "both" is 2nd in each list; "solo" is 1st in only one list
    fused = reciprocal_rank_fusion([["solo", "both"], ["other", "both"]], k=60)
    assert fused[0][0] == "both"


def test_rrf_weights_shift_the_balance():
    rankings = [["dense_pick"], ["sparse_pick"]]
    assert reciprocal_rank_fusion(rankings, weights=[2.0, 1.0])[0][0] == "dense_pick"
    assert reciprocal_rank_fusion(rankings, weights=[1.0, 2.0])[0][0] == "sparse_pick"


def test_rrf_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([["x"], ["y"]])
    assert [item for item, _ in fused] == ["x", "y"]


def test_rrf_edge_cases():
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 1.0])


# --- HybridRetriever --------------------------------------------------------

CHUNKS = [
    make_chunk("cats.txt::c0", "Cats are small domesticated mammals that purr and sleep a lot."),
    make_chunk("dogs.txt::c0", "Dogs are loyal domesticated mammals that bark."),
    make_chunk("errors.pdf::p2::c0", "Windows error 0x80070005 means access is denied.", page=2),
    make_chunk("bikes.md::c0", "Repairing a punctured bicycle tire needs a patch kit and pump."),
]


@pytest.fixture
def retriever(settings, fake_embedder):
    store = IndexStore("retrieval_docs", settings=settings, embedder=fake_embedder)
    store.build(CHUNKS)
    return HybridRetriever(store)


def test_sparse_mode_uses_bm25_order(retriever):
    results = retriever.retrieve("error 0x80070005", mode="sparse")
    assert results[0].chunk.id == "errors.pdf::p2::c0"
    assert results[0].sparse_rank == 1 and results[0].dense_rank is None
    assert results[0].score == results[0].sparse_score


def test_dense_mode_returns_cosine_scores(retriever):
    results = retriever.retrieve("bicycle tire", mode="dense")
    assert results[0].chunk.id == "bikes.md::c0"
    assert results[0].sparse_rank is None
    assert results[0].score == results[0].dense_score
    assert len(results) == len(CHUNKS)  # dense search always returns neighbours


def test_hybrid_mode_records_both_ranks(retriever):
    results = retriever.retrieve("domesticated mammals that purr", mode="hybrid")
    top = results[0]
    assert top.chunk.id == "cats.txt::c0"
    assert top.dense_rank == 1 and top.sparse_rank == 1
    assert top.score == pytest.approx(2 / (60 + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_hybrid_includes_chunks_found_by_only_one_retriever(retriever):
    results = retriever.retrieve("mammals that bark", mode="hybrid")
    ids = {r.chunk.id for r in results}
    dense_only = [r for r in results if r.sparse_rank is None]
    assert {"dogs.txt::c0", "cats.txt::c0"} <= ids
    assert dense_only, "chunks with no keyword overlap should still come from dense search"


def test_top_k_limits_results(retriever):
    assert len(retriever.retrieve("mammals", mode="hybrid", top_k=2)) == 2


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError, match="mode"):
        retriever.retrieve("cats", mode="magic")


def test_empty_collection_returns_nothing(settings, fake_embedder):
    store = IndexStore("empty_docs", settings=settings, embedder=fake_embedder)
    assert HybridRetriever(store).retrieve("anything") == []
