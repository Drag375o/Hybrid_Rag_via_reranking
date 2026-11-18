"""Retrieve chunks with dense search, BM25, or both fused with RRF."""

from typing import Literal, get_args

from hybrid_rag.config import Settings
from hybrid_rag.indexing import IndexStore
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.schema import ScoredChunk

RetrievalMode = Literal["dense", "sparse", "hybrid"]
RETRIEVAL_MODES: tuple[str, ...] = get_args(RetrievalMode)


class HybridRetriever:
    def __init__(self, store: IndexStore, settings: Settings | None = None):
        self.store = store
        self.settings = settings or store.settings

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks (default: ``fusion_top_k``), best first."""
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        top_k = top_k or self.settings.fusion_top_k

        dense = self.store.dense.search(query, self.settings.dense_top_k) if mode != "sparse" else []
        sparse = self.store.sparse.search(query, self.settings.sparse_top_k) if mode != "dense" else []

        if mode == "dense":
            ranked = dense[:top_k]
        elif mode == "sparse":
            ranked = sparse[:top_k]
        else:
            ranked = reciprocal_rank_fusion(
                [[chunk_id for chunk_id, _ in dense], [chunk_id for chunk_id, _ in sparse]],
                k=self.settings.rrf_k,
                weights=[self.settings.dense_weight, self.settings.sparse_weight],
            )[:top_k]

        dense_hits = {cid: (rank, score) for rank, (cid, score) in enumerate(dense, start=1)}
        sparse_hits = {cid: (rank, score) for rank, (cid, score) in enumerate(sparse, start=1)}
        chunks = {c.id: c for c in self.store.get_chunks([cid for cid, _ in ranked])}

        results = []
        for chunk_id, score in ranked:
            if chunk_id not in chunks:  # indexes out of sync; skip rather than crash
                continue
            dense_rank, dense_score = dense_hits.get(chunk_id, (None, None))
            sparse_rank, sparse_score = sparse_hits.get(chunk_id, (None, None))
            results.append(
                ScoredChunk(
                    chunk=chunks[chunk_id],
                    score=score,
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                    sparse_rank=sparse_rank,
                    sparse_score=sparse_score,
                )
            )
        return results
