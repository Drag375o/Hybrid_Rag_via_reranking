"""Ask a question about your indexed documents from the terminal.

Usage (after running ``python -m scripts.ingest``):
    python -m hybrid_rag "your question"
    python -m hybrid_rag "your question" --mode dense --no-rerank
"""

import argparse
import logging
import sys

from hybrid_rag.config import get_settings
from hybrid_rag.generation import GenerationError, check_citations, make_answer, source_label
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retrieval import RETRIEVAL_MODES


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask a question about your documents.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(errors="replace")

    pipeline = RAGPipeline(args.collection)
    if pipeline.store.count() == 0:
        sys.exit(f"Collection '{pipeline.store.name}' is empty. Run: python -m scripts.ingest")

    sources = pipeline.retrieve_context(
        args.question, mode=args.mode, use_reranker=not args.no_rerank
    )
    if not sources:
        print(f"\nNo passage scored above the relevance threshold ({settings.min_relevance}).")

    print(f"\n[{settings.llm_model}] ", end="", flush=True)
    pieces = []
    try:
        for piece in pipeline.generator.stream(args.question, sources):
            print(piece, end="", flush=True)
            pieces.append(piece)
    except GenerationError as exc:
        sys.exit(f"\n\nError: {exc}")
    answer = make_answer(args.question, "".join(pieces).strip(), sources, settings.llm_model)

    if not answer.sources:
        print()
        return

    print("\n\nSources:")
    for number, source in enumerate(answer.sources, start=1):
        marker = "*" if number in answer.cited else " "
        print(f" {marker}[{number}] {source_label(source)}  ({source.chunk.id})")
    print("\n* = cited in the answer")

    checks = check_citations(answer, settings.citation_min_support)
    problems = [c for c in checks if c.status != "supported"]
    if checks:
        print(f"\nCitation check: {len(checks) - len(problems)}/{len(checks)} sentences supported")
    for check in problems:
        if check.status == "uncited":
            print(f'  ! No citation: "{check.sentence}" (best match: [{check.best_source}])')
        else:
            cited = ", ".join(f"[{n}]" for n in check.cited)
            print(
                f'  ! "{check.sentence}" cites {cited}, which doesn\'t clearly support it '
                f"(best match: [{check.best_source}])"
            )


if __name__ == "__main__":
    main()
