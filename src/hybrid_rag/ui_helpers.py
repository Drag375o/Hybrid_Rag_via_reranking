"""Pure helpers for the Streamlit app, kept separate so they can be unit-tested."""

import re
from pathlib import Path
from typing import Any

from hybrid_rag.ingestion import SUPPORTED_EXTENSIONS
from hybrid_rag.pipeline import RetrievalTrace

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_answer_markdown(text: str) -> str:
    """Bold the [n] citations and escape '$' so Streamlit doesn't render it as maths."""
    return _CITATION.sub(r"**[\1]**", text.replace("$", r"\$"))


def safe_upload_name(name: str) -> str | None:
    """File name to save an upload under, or None if its type isn't supported.

    Strips any folder parts so an upload can't be written outside the docs folder.
    """
    clean = Path(name.replace("\\", "/")).name.strip()
    if not clean or Path(clean).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    return clean


def trace_rows(trace: RetrievalTrace, limit: int = 20) -> list[dict[str, Any]]:
    """Table rows showing each candidate's journey through the pipeline.

    Includes the top ``limit`` retrieval candidates plus any chunk the reranker
    promoted into the final list from further down. Final picks come first.
    """
    final_position = {r.chunk.id: i for i, r in enumerate(trace.ranked, start=1)}
    shown = trace.candidates[:limit] + [
        c for c in trace.candidates[limit:] if c.chunk.id in final_position
    ]
    rows = [
        {
            "final #": final_position.get(c.chunk.id),
            "retrieval #": c.retrieval_rank,
            "dense #": c.dense_rank,
            "BM25 #": c.sparse_rank,
            "retrieval score": c.score,
            "rerank score": trace.rerank_scores.get(c.chunk.id),
            "chunk": c.chunk.id,
            "text": " ".join(c.chunk.text.split())[:160],
        }
        for c in shown
    ]
    rows.sort(key=lambda r: (r["final #"] is None, r["final #"] or 0, r["retrieval #"] or 0))
    return rows
