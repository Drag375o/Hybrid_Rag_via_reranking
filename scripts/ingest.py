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
from hybrid_rag.ingestion import chunk_documents, load_directory


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the dense + BM25 indexes for a folder.")
    parser.add_argument("--path", type=Path, default=settings.docs_dir)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    started = time.perf_counter()
    documents = load_directory(args.path)
    if not documents:
        sys.exit(f"No supported documents (.pdf, .txt, .md, .docx) found in {args.path}")
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")

    print(f"Loading embedding model {settings.embedding_model} ...")
    store = IndexStore(args.collection)
    print("Embedding and indexing ...")
    store.build(chunks)

    elapsed = time.perf_counter() - started
    print(
        f"Done in {elapsed:.1f}s: collection '{store.name}' has {store.dense.count()} dense "
        f"and {store.sparse.count()} BM25 entries in {settings.index_dir / store.name}"
    )


if __name__ == "__main__":
    main()
