import json
import math
import zipfile

import pytest

from conftest import FakeReranker, make_chunk
from eval.datasets import read_beir_zip
from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from eval.run_eval import corpus_documents, doc_ranking, evaluate, results_table
from hybrid_rag.indexing import IndexStore
from hybrid_rag.ingestion import chunk_documents
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.schema import ScoredChunk


# --- metrics ---------------------------------------------------------------------

RELEVANT = {"a": 1, "b": 1}


def test_recall_at_k():
    assert recall_at_k(["x", "a", "y", "b"], RELEVANT, 2) == 0.5
    assert recall_at_k(["x", "a", "y", "b"], RELEVANT, 4) == 1.0
    assert recall_at_k(["x"], {}, 5) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["x", "y", "a"], RELEVANT, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "a"], RELEVANT, 2) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b", "x"], RELEVANT, 10) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))  # one hit at rank 2 of 2 relevant
    assert ndcg_at_k(["x", "a"], RELEVANT, 10) == pytest.approx(expected)
    assert ndcg_at_k(["x"], RELEVANT, 10) == 0.0


# --- dataset reading -----------------------------------------------------------------

CORPUS = {
    "d1": ("Coffee and sleep", "Caffeine taken late in the day delays sleep onset."),
    "d2": ("Vitamin D", "Vitamin D supplements reduce fracture risk in older adults."),
    "d3": ("Exercise", "Regular aerobic exercise lowers resting blood pressure."),
}
QUERIES = {"q1": "Does caffeine delay sleep?", "q2": "Does exercise lower blood pressure?",
           "q3": "A question with no judgements"}
QRELS = [("q1", "d1", 1), ("q2", "d3", 1), ("q2", "d2", 0)]


@pytest.fixture
def beir_zip(tmp_path):
    path = tmp_path / "mini.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mini/corpus.jsonl", "\n".join(
            json.dumps({"_id": i, "title": t, "text": x}) for i, (t, x) in CORPUS.items()))
        zf.writestr("mini/queries.jsonl", "\n".join(
            json.dumps({"_id": i, "text": t}) for i, t in QUERIES.items()))
        zf.writestr("mini/qrels/test.tsv", "query-id\tcorpus-id\tscore\n" + "\n".join(
            f"{q}\t{d}\t{s}" for q, d, s in QRELS))
    return path


def test_read_beir_zip(beir_zip):
    data = read_beir_zip(beir_zip, "mini")
    assert data.corpus["d2"]["title"] == "Vitamin D"
    assert set(data.queries) == {"q1", "q2"}  # q3 has no judgements
    assert data.qrels == {"q1": {"d1": 1}, "q2": {"d3": 1}}  # score 0 = not relevant


# --- evaluation loop -------------------------------------------------------------------

def test_doc_ranking_keeps_best_chunk_per_document():
    results = [ScoredChunk(chunk=make_chunk(f"{doc}::c{i}", "t"), score=0)
               for doc, i in [("d2", 0), ("d1", 1), ("d2", 1), ("d3", 0)]]
    assert doc_ranking(results) == ["d2", "d1", "d3"]


def test_evaluate_end_to_end_on_tiny_benchmark(beir_zip, settings, fake_embedder):
    data = read_beir_zip(beir_zip, "mini")
    store = IndexStore("beir_mini", settings=settings, embedder=fake_embedder)
    store.build(chunk_documents(corpus_documents(data), 800, 100))
    pipeline = RAGPipeline(settings=settings, store=store, reranker=FakeReranker())

    results = evaluate(pipeline, data, ["dense", "bm25", "hybrid", "hybrid+rerank"],
                       sorted(data.queries))

    assert [r.setup for r in results] == ["dense", "bm25", "hybrid", "hybrid+rerank"]
    for r in results:  # each question shares obvious keywords with its one relevant doc
        assert r.recall_at_10 == 1.0 and r.mrr_at_10 == 1.0 and r.ndcg_at_10 == pytest.approx(1.0)
    assert "| **dense** | 1.000 |" in results_table(results)
