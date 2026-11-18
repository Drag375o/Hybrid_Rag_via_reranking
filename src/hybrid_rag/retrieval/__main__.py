"""Try retrieval from the command line and see where each result came from.

Usage (after running ``python -m scripts.ingest``):
    python -m hybrid_rag.retrieval "your question"
    python -m hybrid_rag.retrieval "your question" --mode dense --top-k 5
"""

import argparse
import logging
import sys

from hybrid_rag.config import get_settings
from hybrid_rag.indexing import IndexStore
from hybrid_rag.retrieval import RETRIEVAL_MODES, HybridRetriever


def _rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(errors="replace")

    store = IndexStore(args.collection)
    if store.count() == 0:
        sys.exit(f"Collection '{store.name}' is empty. Run: python -m scripts.ingest")

    results = HybridRetriever(store).retrieve(args.query, mode=args.mode, top_k=args.top_k)

    print(f"\n{args.mode} results for: {args.query!r}\n")
    print(f"{'#':>3}  {'score':>8}  {'dense':>5}  {'bm25':>5}  chunk")
    for position, result in enumerate(results, start=1):
        snippet = " ".join(result.chunk.text.split())[:70]
        print(
            f"{position:>3}  {result.score:>8.4f}  {_rank(result.dense_rank):>5}  "
            f"{_rank(result.sparse_rank):>5}  {result.chunk.id}\n{'':>29}{snippet}"
        )
    print("\n'dense' / 'bm25' = the chunk's rank in each retriever's own list ('-' = not found)")


if __name__ == "__main__":
    main()
