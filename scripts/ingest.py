"""Load, chunk and index a folder of documents (rebuilds the collection from scratch).

Usage (from the project root):
    python -m scripts.ingest                      # indexes data/docs
    python -m scripts.ingest --path my/folder --collection my_docs
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.indexing import IndexStore
from hybrid_rag.ingestion import SUPPORTED_EXTENSIONS


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the dense + BM25 indexes for a folder.")
    parser.add_argument("--path", type=Path, default=settings.docs_dir)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    # Progress is printed below; logging only surfaces problems (e.g. a file that failed to load)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    started = time.perf_counter()
    if not any(p.suffix.lower() in SUPPORTED_EXTENSIONS for p in args.path.rglob("*")):
        sys.exit(f"No supported documents (.pdf, .txt, .md, .docx) found in {args.path}")

    print(f"Loading embedding model {settings.embedding_model} ...")
    store = IndexStore(args.collection)
    print("Loading, chunking, embedding and indexing ...")
    stats = store.build_from_folder(args.path)

    elapsed = time.perf_counter() - started
    print(f"Loaded {stats.files} files ({stats.documents} documents) -> {stats.chunks} chunks")
    print(
        f"Done in {elapsed:.1f}s: collection '{store.name}' has {store.dense.count()} dense "
        f"and {store.sparse.count()} BM25 entries in {settings.index_dir / store.name}"
    )


if __name__ == "__main__":
    main()
