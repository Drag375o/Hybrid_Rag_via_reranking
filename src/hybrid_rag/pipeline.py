"""End-to-end pipeline: hybrid retrieval -> cross-encoder reranking (-> generation)."""

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.indexing import IndexStore
from hybrid_rag.reranking import CrossEncoderReranker, Reranker, rerank
from hybrid_rag.retrieval import HybridRetriever, RetrievalMode
from hybrid_rag.schema import ScoredChunk


class RAGPipeline:
    """One entry point used by the CLI, the UI and the evaluation harness."""

    def __init__(
        self,
        collection: str | None = None,
        settings: Settings | None = None,
        store: IndexStore | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or IndexStore(collection, settings=self.settings)
        self.retriever = HybridRetriever(self.store, self.settings)
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        # The cross-encoder is large, so only load it the first time it is needed
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings)
        return self._reranker

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve candidates and (optionally) rerank them.

        ``top_k`` defaults to ``rerank_top_k`` when reranking, else ``fusion_top_k``.
        """
        candidates = self.retriever.retrieve(query, mode=mode)
        if not use_reranker:
            return candidates[: top_k or self.settings.fusion_top_k]
        return rerank(self.reranker, query, candidates, top_k or self.settings.rerank_top_k)
