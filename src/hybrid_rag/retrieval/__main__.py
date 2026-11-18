"""Try retrieval (and reranking) from the command line and see where each result came from.

Usage (after running ``python -m scripts.ingest``):
    python -m hybrid_rag.retrieval "your question"
    python -m hybrid_rag.retrieval "your question" --mode dense --top-k 5
    python -m hybrid_rag.retrieval "your question" --rerank
"""

import argparse
import logging
import sys

from hybrid_rag.config import get_settings
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retrieval import RETRIEVAL_MODES


def _fmt(value: int | float | None, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid")
    parser.add_argument("--rerank", action="store_true", help="rerank with the cross-encoder")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(errors="replace")

    pipeline = RAGPipeline(args.collection)
    if pipeline.store.count() == 0:
        sys.exit(f"Collection '{pipeline.store.name}' is empty. Run: python -m scripts.ingest")

    top_k = args.top_k or (settings.rerank_top_k if args.rerank else 10)
    results = pipeline.search(args.query, mode=args.mode, use_reranker=args.rerank, top_k=top_k)

    label = f"{args.mode} + rerank" if args.rerank else args.mode
    print(f"\n{label} results for: {args.query!r}\n")
    print(f"{'#':>3}  {'rerank':>6}  {'was #':>5}  {'dense':>5}  {'bm25':>5}  chunk")
    for position, r in enumerate(results, start=1):
        snippet = " ".join(r.chunk.text.split())[:70]
        print(
            f"{position:>3}  {_fmt(r.rerank_score, '.3f'):>6}  {_fmt(r.retrieval_rank):>5}  "
            f"{_fmt(r.dense_rank):>5}  {_fmt(r.sparse_rank):>5}  {r.chunk.id}\n{'':>34}{snippet}"
        )
    print(
        "\nrerank = cross-encoder relevance (0-1); 'was #' = position before reranking;"
        "\ndense / bm25 = rank in each retriever's own list ('-' = not found)"
    )


if __name__ == "__main__":
    main()
