"""Turn text into embedding vectors with a sentence-transformers model."""

from typing import Protocol

from hybrid_rag.config import Settings, get_settings


class Embedder(Protocol):
    """Anything that can embed passages and queries (lets tests swap in a fake)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class SentenceTransformerEmbedder:
    """Embeds text with a local model; vectors are L2-normalised for cosine similarity."""

    def __init__(self, settings: Settings | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.settings = settings or get_settings()
        self.model = SentenceTransformer(
            self.settings.embedding_model, device=resolve_device(self.settings.device)
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            self.settings.embedding_query_prefix + text, normalize_embeddings=True
        )
        return vector.tolist()
