import math
import zlib

import pytest

from hybrid_rag.config import Settings
from hybrid_rag.indexing import tokenize
from hybrid_rag.schema import Chunk

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words get similar embeddings.

    Lets index and retrieval tests run offline without downloading a model.
    """

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for token in tokenize(text):
            vector[zlib.crc32(token.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def make_chunk(chunk_id: str, text: str, page: int | None = None, **metadata) -> Chunk:
    source = chunk_id.split("::")[0]
    return Chunk(id=chunk_id, text=text, source=source, page=page, chunk_index=0, metadata=metadata)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
