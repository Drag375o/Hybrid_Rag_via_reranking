import pytest

from conftest import make_chunk
from hybrid_rag.indexing import IndexStore
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.reranking import rerank
from hybrid_rag.schema import ScoredChunk


def candidate(chunk_id: str, text: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id, text), score=1 / rank, retrieval_rank=rank)


CANDIDATES = [
    candidate("a::c0", "general overview of retrieval systems", 1),
    candidate("b::c0", "bm25 keyword scoring explained", 2),
    candidate("c::c0", "how reciprocal rank fusion merges bm25 and dense rankings", 3),
]


# --- rerank() ---------------------------------------------------------------

def test_rerank_reorders_by_relevance(fake_reranker):
    results = rerank(fake_reranker, "how does rank fusion merge bm25 rankings", CANDIDATES, top_k=3)
    assert [r.chunk.id for r in results] == ["c::c0", "b::c0", "a::c0"]
    assert [r.retrieval_rank for r in results] == [3, 2, 1]  # original positions kept
    assert results[0].rerank_score > results[1].rerank_score > results[2].rerank_score


def test_rerank_keeps_top_k(fake_reranker):
    assert len(rerank(fake_reranker, "bm25", CANDIDATES, top_k=1)) == 1


def test_rerank_does_not_modify_inputs(fake_reranker):
    rerank(fake_reranker, "bm25", CANDIDATES, top_k=3)
    assert all(c.rerank_score is None for c in CANDIDATES)


def test_rerank_skips_model_when_nothing_to_rank(fake_reranker):
    assert rerank(fake_reranker, "anything", [], top_k=5) == []
    assert fake_reranker.calls == 0


# --- RAGPipeline.search() ---------------------------------------------------

CHUNKS = [
    make_chunk("cats.txt::c0", "Cats are small domesticated mammals that purr."),
    make_chunk("dogs.txt::c0", "Dogs are loyal domesticated mammals that bark loudly."),
    make_chunk("errors.pdf::p2::c0", "Windows error 0x80070005 means access is denied.", page=2),
    make_chunk("bikes.md::c0", "Repairing a punctured bicycle tire needs a patch kit."),
]


@pytest.fixture
def pipeline(settings, fake_embedder, fake_reranker):
    store = IndexStore("pipeline_docs", settings=settings, embedder=fake_embedder)
    store.build(CHUNKS)
    return RAGPipeline(settings=settings, store=store, reranker=fake_reranker)


def test_search_with_reranker_returns_rerank_top_k(pipeline):
    results = pipeline.search("which mammals bark loudly")
    assert len(results) <= pipeline.settings.rerank_top_k
    assert results[0].chunk.id == "dogs.txt::c0"
    assert all(r.rerank_score is not None and r.retrieval_rank is not None for r in results)


def test_search_without_reranker_skips_model(pipeline, fake_reranker):
    results = pipeline.search("mammals", use_reranker=False)
    assert fake_reranker.calls == 0
    assert all(r.rerank_score is None for r in results)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_supports_every_mode(pipeline, mode):
    results = pipeline.search("error 0x80070005 access denied", mode=mode, top_k=2)
    assert results[0].chunk.id == "errors.pdf::p2::c0"


def test_reranker_is_loaded_lazily(settings, fake_embedder):
    store = IndexStore("lazy_docs", settings=settings, embedder=fake_embedder)
    pipeline = RAGPipeline(settings=settings, store=store)
    assert pipeline._reranker is None
    pipeline.search("anything", use_reranker=False)
    assert pipeline._reranker is None
