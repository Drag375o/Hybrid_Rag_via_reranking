"""Sparse (keyword) index: BM25 over the same chunks as the dense index.

Only the chunk ids and their tokens are saved to disk; the BM25 statistics are
rebuilt on load, which takes well under a second even for large collections.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from hybrid_rag.schema import Chunk

# Very common English words carry almost no signal for keyword matching
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its "
    "me my no not of on or our she so than that the their them then there these they "
    "this to was we were what when where which who why will with you your".split()
)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    def __init__(self, path: Path):
        self.path = path
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._ids, self._tokens = data["ids"], data["tokens"]
            self._rebuild()

    def _rebuild(self) -> None:
        # BM25Okapi cannot be built over an empty corpus
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._tokens, self._bm25 = [], [], None
        self.path.unlink(missing_ok=True)

    def add(self, chunks: list[Chunk]) -> None:
        self._ids.extend(c.id for c in chunks)
        self._tokens.extend(tokenize(c.text) for c in chunks)
        self._rebuild()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"ids": self._ids, "tokens": self._tokens}), encoding="utf-8")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, BM25 score) pairs, best first; chunks sharing no terms are dropped."""
        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self._ids[i], float(scores[i])) for i in ranked[:top_k] if scores[i] > 0]
