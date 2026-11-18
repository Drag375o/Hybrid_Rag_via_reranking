"""Dense (semantic) index: chunk embeddings stored and searched in ChromaDB.

ChromaDB also stores each chunk's text and metadata, so it doubles as the
chunk store that later stages use to look chunks up by id.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from hybrid_rag.indexing.embeddings import Embedder
from hybrid_rag.schema import Chunk

_RESERVED_KEYS = {"source", "page", "chunk_index"}


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    # ChromaDB metadata values must be str/int/float/bool, so drop None and complex values
    metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if isinstance(value, (str, int, float, bool)) and key not in _RESERVED_KEYS
    }
    metadata["source"] = chunk.source
    metadata["chunk_index"] = chunk.chunk_index
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


def _from_record(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    metadata = dict(metadata)
    return Chunk(
        id=chunk_id,
        text=text,
        source=metadata.pop("source"),
        page=metadata.pop("page", None),
        chunk_index=metadata.pop("chunk_index"),
        metadata=metadata,
    )


class DenseIndex:
    def __init__(self, path: Path, collection_name: str, embedder: Embedder):
        self.embedder = embedder
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._open_collection()

    def _open_collection(self):
        return self._client.get_or_create_collection(
            self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we always pass our own embeddings
        )

    def reset(self) -> None:
        """Delete every chunk in this collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk]) -> None:
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
            )

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine similarity) pairs, most similar first."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.count()),
            include=["distances"],
        )
        return [
            (chunk_id, 1.0 - distance)
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0])
        ]

    def all_chunks(self, batch_size: int = 5000) -> list[Chunk]:
        """Every stored chunk (text and metadata, no embeddings), in pages."""
        chunks: list[Chunk] = []
        for offset in range(0, self.count(), batch_size):
            page = self._collection.get(
                include=["documents", "metadatas"], limit=batch_size, offset=offset
            )
            chunks.extend(
                _from_record(chunk_id, text, metadata)
                for chunk_id, text, metadata in zip(page["ids"], page["documents"], page["metadatas"])
            )
        return chunks

    def get(self, ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id, in the order requested (unknown ids are skipped)."""
        if not ids:
            return []
        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {
            chunk_id: _from_record(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }
        return [by_id[i] for i in ids if i in by_id]
