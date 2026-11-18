"""Rerank retrieved chunks with a cross-encoder.

Unlike the embedding model, which encodes the query and each chunk separately,
a cross-encoder reads the query and a chunk *together* in one pass, so it can
judge whether the chunk actually answers the question. That is far more
accurate but too slow to run over a whole collection, so it is only applied to
the shortlist produced by hybrid retrieval.
"""

from dataclasses import replace
from typing import Protocol

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.indexing.embeddings import resolve_device
from hybrid_rag.schema import ScoredChunk


class Reranker(Protocol):
    """Anything that scores (query, passage) relevance (lets tests swap in a fake)."""

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores relevance in [0, 1] with a local cross-encoder model."""

    def __init__(self, settings: Settings | None = None):
        import torch  # heavy imports, load lazily
        from sentence_transformers import CrossEncoder

        self.settings = settings or get_settings()
        self.model = CrossEncoder(
            self.settings.reranker_model,
            device=resolve_device(self.settings.device),
            max_length=self.settings.reranker_max_length,
            activation_fn=torch.nn.Sigmoid(),  # raw logits -> 0..1 relevance
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = self.model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def rerank(
    reranker: Reranker,
    query: str,
    candidates: list[ScoredChunk],
    top_k: int,
) -> list[ScoredChunk]:
    """Re-order candidates by cross-encoder relevance and keep the best ``top_k``.

    Returns copies with ``rerank_score`` filled in; the inputs are left untouched.
    """
    if not candidates:
        return []
    scores = reranker.score(query, [c.chunk.text for c in candidates])
    rescored = [replace(c, rerank_score=s) for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda c: c.rerank_score, reverse=True)
    return rescored[:top_k]
