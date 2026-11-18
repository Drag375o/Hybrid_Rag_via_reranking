"""End-to-end pipeline: hybrid retrieval -> cross-encoder reranking -> generation."""

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.generation import OllamaGenerator
from hybrid_rag.indexing import IndexStore
from hybrid_rag.reranking import CrossEncoderReranker, Reranker, rerank
from hybrid_rag.retrieval import HybridRetriever, RetrievalMode
from hybrid_rag.schema import Answer, ScoredChunk


class RAGPipeline:
    """One entry point used by the CLI, the UI and the evaluation harness."""

    def __init__(
        self,
        collection: str | None = None,
        settings: Settings | None = None,
        store: IndexStore | None = None,
        reranker: Reranker | None = None,
        generator: OllamaGenerator | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or IndexStore(collection, settings=self.settings)
        self.retriever = HybridRetriever(self.store, self.settings)
        self._reranker = reranker
        self.generator = generator or OllamaGenerator(self.settings)

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

    def retrieve_context(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        use_reranker: bool = True,
    ) -> list[ScoredChunk]:
        """The passages to give the LLM, or [] if none is relevant enough to answer from.

        The relevance gate needs reranker scores, so it only applies when reranking.
        """
        # Without reranking, still only pass a short list of passages to the LLM
        sources = self.search(
            question, mode=mode, use_reranker=use_reranker, top_k=self.settings.rerank_top_k
        )
        if use_reranker and sources and sources[0].rerank_score < self.settings.min_relevance:
            return []
        return sources

    def ask(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        use_reranker: bool = True,
    ) -> Answer:
        """Answer a question from the indexed documents, with citations."""
        sources = self.retrieve_context(question, mode=mode, use_reranker=use_reranker)
        return self.generator.generate(question, sources)
