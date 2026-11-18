"""Keep the dense and sparse indexes for one collection in sync."""

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.indexing.dense import DenseIndex
from hybrid_rag.indexing.embeddings import Embedder, SentenceTransformerEmbedder
from hybrid_rag.indexing.sparse import BM25Index
from hybrid_rag.schema import Chunk


class IndexStore:
    """Both indexes for one named collection, stored under ``<index_dir>/<name>/``.

    Separate names keep your own documents apart from benchmark datasets.
    """

    def __init__(
        self,
        name: str | None = None,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ):
        self.settings = settings or get_settings()
        self.name = name or self.settings.collection_name
        root = self.settings.index_dir / self.name
        self.dense = DenseIndex(
            root / "chroma", self.name, embedder or SentenceTransformerEmbedder(self.settings)
        )
        self.sparse = BM25Index(root / "bm25.json")

    def count(self) -> int:
        return self.dense.count()

    def reset(self) -> None:
        self.dense.reset()
        self.sparse.reset()

    def add(self, chunks: list[Chunk]) -> None:
        existing = set(c.id for c in self.dense.get([c.id for c in chunks]))
        if existing:
            raise ValueError(
                f"{len(existing)} chunk ids are already indexed (e.g. {next(iter(existing))}); "
                "rebuild the collection instead of adding the same files twice"
            )
        self.dense.add(chunks)
        self.sparse.add(chunks)

    def build(self, chunks: list[Chunk]) -> None:
        """Replace everything in the collection with these chunks."""
        self.reset()
        self.add(chunks)

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        return self.dense.get(ids)
