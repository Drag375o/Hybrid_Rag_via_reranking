"""Preview how a folder of documents gets loaded and chunked.

Usage:
    python -m hybrid_rag.ingestion                # uses data/docs
    python -m hybrid_rag.ingestion path/to/folder --show 5
"""

import argparse
import logging
import sys
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.ingestion import chunk_documents, load_directory


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Preview document loading and chunking.")
    parser.add_argument("path", nargs="?", type=Path, default=settings.docs_dir)
    parser.add_argument("--show", type=int, default=3, help="number of chunks to print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Windows consoles often can't encode every Unicode character; don't crash on them.
    sys.stdout.reconfigure(errors="replace")

    documents = load_directory(args.path)
    chunks = chunk_documents(documents)
    sources = {doc.source for doc in documents}

    print(f"Folder:    {args.path}")
    print(f"Files:     {len(sources)}")
    print(f"Documents: {len(documents)} (PDFs count one per page)")
    print(f"Chunks:    {len(chunks)} (size {settings.chunk_size}, overlap {settings.chunk_overlap})")
    if chunks:
        avg = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"Avg chunk: {avg:.0f} characters")

    for chunk in chunks[: args.show]:
        print(f"\n--- {chunk.id} ({len(chunk.text)} chars) ---")
        print(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))


if __name__ == "__main__":
    main()
