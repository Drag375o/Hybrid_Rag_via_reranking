"""End-to-end pipeline: hybrid retrieval -> cross-encoder reranking -> generation."""

from dataclasses import dataclass

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.generation import OllamaGenerator
from hybrid_rag.indexing import IndexStore
from hybrid_rag.reranking import CrossEncoderReranker, Reranker, rerank
from hybrid_rag.retrieval import HybridRetriever, RetrievalMode
from hybrid_rag.schema import Answer, ScoredChunk


@dataclass
class RetrievalTrace:
    """Every stage of one search, so the UI can show how the context was chosen."""

    candidates: list[ScoredChunk]  # retriever output before reranking (up to fusion_top_k)
    ranked: list[ScoredChunk]  # final top-k (reranked when the reranker is on)
    context: list[ScoredChunk]  # what the LLM is given ([] when the relevance gate blocks)
    gated: bool  # True when no passage scored above min_relevance
    rerank_scores: dict[str, float]  # chunk id -> reranker score for every candidate


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

    def trace(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> RetrievalTrace:
        """Run retrieval and reranking, keeping every intermediate list.

        The relevance gate needs reranker scores, so it only applies when reranking.
        """
        top_k = top_k or self.settings.rerank_top_k
        candidates = self.retriever.retrieve(question, mode=mode)
        if use_reranker:
            all_reranked = rerank(self.reranker, question, candidates, top_k=None)
            ranked = all_reranked[:top_k]
            rerank_scores = {r.chunk.id: r.rerank_score for r in all_reranked}
        else:
            ranked = candidates[:top_k]  # still only pass a short list to the LLM
            rerank_scores = {}
        gated = use_reranker and bool(ranked) and ranked[0].rerank_score < self.settings.min_relevance
        return RetrievalTrace(
            candidates=candidates,
            ranked=ranked,
            context=[] if gated else ranked,
            gated=gated,
            rerank_scores=rerank_scores,
        )

    def retrieve_context(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """The passages to give the LLM, or [] if none is relevant enough to answer from."""
        return self.trace(question, mode=mode, use_reranker=use_reranker, top_k=top_k).context

    def ask(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> Answer:
        """Answer a question from the indexed documents, with citations."""
        sources = self.retrieve_context(question, mode=mode, use_reranker=use_reranker, top_k=top_k)
        return self.generator.generate(question, sources)
