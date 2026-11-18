"""Compare dense, BM25, hybrid and hybrid + rerank retrieval on a BEIR benchmark.

Usage (from the project root):
    python -m eval.run_eval                       # SciFact, all 300 test questions
    python -m eval.run_eval --limit 50            # quick run on the first 50 questions
    python -m eval.run_eval --rebuild             # re-index after changing chunking/models

Documents are chunked and indexed exactly like your own files, then each chunk
list is collapsed to a document ranking (a document's position is that of its
best chunk) and scored against the benchmark's answer key.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from eval.datasets import BeirDataset, load_beir
from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from hybrid_rag.config import get_settings
from hybrid_rag.indexing import IndexStore
from hybrid_rag.ingestion import chunk_documents
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.reranking import rerank
from hybrid_rag.schema import Document, ScoredChunk

RESULTS_DIR = Path(__file__).parent / "results"

# setup name -> (retrieval mode, rerank?)
SETUPS: dict[str, tuple[str, bool]] = {
    "dense": ("dense", False),
    "bm25": ("sparse", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}


@dataclass
class SetupResult:
    setup: str
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_5: float
    recall_at_10: float
    ms_per_query: float


def corpus_documents(dataset: BeirDataset) -> list[Document]:
    return [
        Document(
            text=f"{doc['title']}\n\n{doc['text']}".strip(),
            source=doc_id,
            metadata={"title": doc["title"]},
        )
        for doc_id, doc in dataset.corpus.items()
    ]


def doc_ranking(results: list[ScoredChunk]) -> list[str]:
    """Collapse a chunk ranking to a document ranking (first = best chunk wins)."""
    return list(dict.fromkeys(r.chunk.source for r in results))


def evaluate(
    pipeline: RAGPipeline,
    dataset: BeirDataset,
    setups: list[str],
    query_ids: list[str],
    progress: bool = False,
) -> list[SetupResult]:
    # Retrieval for each mode is shared, e.g. "hybrid" and "hybrid+rerank" reuse the same candidates
    candidates: dict[str, dict[str, list[ScoredChunk]]] = {}
    retrieval_seconds: dict[str, float] = {}
    for mode in dict.fromkeys(SETUPS[s][0] for s in setups):
        started = time.perf_counter()
        candidates[mode] = {
            qid: pipeline.retriever.retrieve(dataset.queries[qid], mode=mode) for qid in query_ids
        }
        retrieval_seconds[mode] = time.perf_counter() - started
        if progress:
            print(f"  retrieved with {mode:<6} in {retrieval_seconds[mode]:.1f}s", flush=True)

    results = []
    for setup in setups:
        mode, use_reranker = SETUPS[setup]
        seconds = retrieval_seconds[mode]
        scores = {"ndcg": 0.0, "mrr": 0.0, "r5": 0.0, "r10": 0.0}
        started = time.perf_counter()
        for qid in query_ids:
            ranked = candidates[mode][qid]
            if use_reranker:
                ranked = rerank(pipeline.reranker, dataset.queries[qid], ranked, top_k=None)
            docs, relevant = doc_ranking(ranked), dataset.qrels[qid]
            scores["ndcg"] += ndcg_at_k(docs, relevant, 10)
            scores["mrr"] += mrr_at_k(docs, relevant, 10)
            scores["r5"] += recall_at_k(docs, relevant, 5)
            scores["r10"] += recall_at_k(docs, relevant, 10)
        if use_reranker:
            seconds += time.perf_counter() - started
            if progress:
                print(f"  reranked in {time.perf_counter() - started:.1f}s", flush=True)

        n = len(query_ids)
        results.append(
            SetupResult(
                setup=setup,
                ndcg_at_10=scores["ndcg"] / n,
                mrr_at_10=scores["mrr"] / n,
                recall_at_5=scores["r5"] / n,
                recall_at_10=scores["r10"] / n,
                ms_per_query=1000 * seconds / n,
            )
        )
    return results


def results_table(results: list[SetupResult]) -> str:
    lines = [
        "| Setup | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | ms / query |",
        "|---|---|---|---|---|---|",
    ]
    best = max(r.ndcg_at_10 for r in results)
    for r in results:
        name = f"**{r.setup}**" if r.ndcg_at_10 == best else r.setup
        lines.append(
            f"| {name} | {r.ndcg_at_10:.3f} | {r.mrr_at_10:.3f} | {r.recall_at_5:.3f} "
            f"| {r.recall_at_10:.3f} | {r.ms_per_query:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Benchmark the retrieval setups on BEIR.")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="only use the first N questions")
    parser.add_argument("--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    parser.add_argument("--rebuild", action="store_true", help="re-index the corpus")
    parser.add_argument("--no-save", action="store_true", help="don't write eval/results/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(errors="replace")

    print(f"Loading BEIR/{args.dataset} ({args.split} split) ...")
    dataset = load_beir(args.dataset, settings.benchmarks_dir, args.split)
    query_ids = sorted(dataset.queries)[: args.limit]
    print(f"  {len(dataset.corpus)} documents, {len(query_ids)} questions")

    store = IndexStore(f"beir_{args.dataset}", settings=settings)
    if args.rebuild or store.count() == 0:
        chunks = chunk_documents(
            corpus_documents(dataset), settings.chunk_size, settings.chunk_overlap
        )
        print(f"Indexing {len(chunks)} chunks (one-time; takes a few minutes on CPU) ...")
        started = time.perf_counter()
        store.build(chunks)
        print(f"  done in {time.perf_counter() - started:.0f}s")
    else:
        print(f"Using existing index ({store.count()} chunks); pass --rebuild to re-index")

    pipeline = RAGPipeline(settings=settings, store=store)
    print("Evaluating ...")
    results = evaluate(pipeline, dataset, args.setups, query_ids, progress=True)

    table = results_table(results)
    print(f"\n{table}\n")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        config = {
            "dataset": args.dataset,
            "split": args.split,
            "questions": len(query_ids),
            "documents": len(dataset.corpus),
            "chunks": store.count(),
            "date": date.today().isoformat(),
            **{key: getattr(settings, key) for key in (
                "embedding_model", "reranker_model", "chunk_size", "chunk_overlap", "chunk_title_prefix",
                "dense_top_k", "sparse_top_k", "rrf_k", "fusion_top_k",
            )},
        }
        stem = RESULTS_DIR / f"{args.dataset}_{args.split}"
        stem.with_suffix(".json").write_text(
            json.dumps({"config": config, "results": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )
        settings_lines = "\n".join(f"- {key}: `{value}`" for key, value in config.items())
        stem.with_suffix(".md").write_text(
            f"# BEIR/{args.dataset} ({args.split})\n\n{table}\n\n## Configuration\n\n{settings_lines}\n",
            encoding="utf-8",
        )
        print(f"Saved {stem.with_suffix('.md')} and .json")


if __name__ == "__main__":
    main()
