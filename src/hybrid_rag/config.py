"""Central configuration for the hybrid RAG pipeline.

Every setting can be overridden with an environment variable prefixed with
``RAG_`` (for example ``RAG_LLM_MODEL=llama3.1:8b``) or in a ``.env`` file at
the project root. See ``.env.example`` for the full list.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # --- Models ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models retrieve better when short queries carry this instruction (passages don't)
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=32, gt=0)
    # Small and fast on CPU (~90 MB). For higher accuracy at ~10x the size and
    # CPU time, use "BAAI/bge-reranker-base".
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_max_length: int = Field(default=512, gt=0)  # tokens per (query, chunk) pair
    # Small enough (~1 GB) to download and run quickly on CPU. Larger models such as
    # "qwen2.5:3b" or "qwen2.5:7b" write better answers if you can download/run them.
    llm_model: str = "qwen2.5:1.5b"
    ollama_host: str = "http://localhost:11434"
    llm_temperature: float = Field(default=0.1, ge=0)  # low = stick closely to the sources
    llm_num_ctx: int = Field(default=8192, gt=0)  # context window in tokens
    # A citation counts as supported if the cited passage contains at least this
    # share of the sentence's content words
    citation_min_support: float = Field(default=0.5, ge=0, le=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Chunking (sizes in characters) ---
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    # Repeat a document's title (when known) at the top of every chunk, so later
    # chunks keep their context. Re-index after changing this.
    chunk_title_prefix: bool = True

    # --- Retrieval ---
    # Reduce words to their root in BM25 ("vitamins" -> "vitamin"); existing BM25
    # indexes are re-tokenized automatically when this changes
    bm25_stemming: bool = True
    dense_top_k: int = Field(default=50, gt=0)
    sparse_top_k: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    # Relative trust in each retriever when fusing (1.0 / 1.0 = classic, unweighted RRF)
    dense_weight: float = Field(default=1.0, ge=0)
    sparse_weight: float = Field(default=1.0, ge=0)
    fusion_top_k: int = Field(default=50, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    # If even the best reranked passage scores below this, answer "not found" without
    # calling the LLM (stops small models answering from their own knowledge).
    # Calibrated for the MiniLM reranker, where off-topic questions score ~0.0000 and
    # answerable ones >= ~0.03. Set to 0 to disable; re-check it if you change models.
    min_relevance: float = Field(default=0.01, ge=0, le=1)

    # --- Storage ---
    data_dir: Path = PROJECT_ROOT / "data"
    collection_name: str = "documents"

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.rerank_top_k > self.fusion_top_k:
            raise ValueError("rerank_top_k cannot exceed fusion_top_k")
        if not self.data_dir.is_absolute():
            self.data_dir = PROJECT_ROOT / self.data_dir
        return self

    @property
    def docs_dir(self) -> Path:
        """Where users drop their own documents for ingestion."""
        return self.data_dir / "docs"

    @property
    def index_dir(self) -> Path:
        """Where the ChromaDB and BM25 indexes are persisted."""
        return self.data_dir / "index"

    @property
    def benchmarks_dir(self) -> Path:
        """Where downloaded evaluation datasets are cached."""
        return self.data_dir / "benchmarks"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
