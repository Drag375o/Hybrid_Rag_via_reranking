"""Core data types shared by every stage of the pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of loaded text: one page of a PDF, or one whole TXT/MD/DOCX file."""

    text: str
    source: str  # file path relative to the ingested folder, or a benchmark doc id
    page: int | None = None  # 1-based page number (PDFs only)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a Document small enough to embed, index and retrieve."""

    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int  # position of this chunk within its Document
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A retrieved Chunk plus the evidence for why it was retrieved.

    ``score`` is the retrieval score (RRF score in hybrid mode, cosine similarity
    in dense mode, BM25 score in sparse mode). Ranks are 1-based and None when
    that stage did not see the chunk. After reranking, ``rerank_score`` (0-1
    relevance) decides the order and ``retrieval_rank`` keeps the position the
    chunk had before reranking.
    """

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    retrieval_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class Answer:
    """A generated answer and the passages it was grounded in.

    ``sources`` are numbered 1..n in the order they were shown to the LLM, and
    ``cited`` lists the numbers the answer actually referenced, e.g. ``[1, 3]``.
    """

    question: str
    text: str
    sources: list[ScoredChunk]
    cited: list[int]
    model: str

    @property
    def cited_sources(self) -> list[ScoredChunk]:
        return [self.sources[n - 1] for n in self.cited]
