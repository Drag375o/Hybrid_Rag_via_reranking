"""Sparse (keyword) index: BM25 over the same chunks as the dense index.

Only the chunk ids and their tokens are saved to disk; the BM25 statistics are
rebuilt on load, which takes well under a second even for large collections.
The file also records which tokenizer produced the tokens, so an index built
with different settings is detected instead of silently mismatching queries.
"""

import json
import re
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

from hybrid_rag.schema import Chunk

# Very common English words carry almost no signal for keyword matching
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its "
    "me my no not of on or our she so than that the their them then there these they "
    "this to was we were what when where which who why will with you your".split()
)

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STEMMER = snowballstemmer.stemmer("english")


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Lowercase words minus stopwords; with ``stem``, reduce each to its root
    ("vitamins" -> "vitamin", "diabetic"/"diabetes" -> "diabet")."""
    tokens = [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]
    return _STEMMER.stemWords(tokens) if stem else tokens


def tokenizer_name(stem: bool) -> str:
    return "words+stopwords+snowball-en" if stem else "words+stopwords"


class BM25Index:
    def __init__(self, path: Path, stem: bool = False):
        self.path = path
        self.stem = stem
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        # True when the file on disk was tokenized differently; the caller should re-add chunks
        self.needs_rebuild = False
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("tokenizer", tokenizer_name(False)) == tokenizer_name(stem):
                self._ids, self._tokens = data["ids"], data["tokens"]
                self._rebuild()
            else:
                self.needs_rebuild = True

    def _rebuild(self) -> None:
        # BM25Okapi cannot be built over an empty corpus
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._tokens, self._bm25 = [], [], None
        self.needs_rebuild = False
        self.path.unlink(missing_ok=True)

    def add(self, chunks: list[Chunk]) -> None:
        self._ids.extend(c.id for c in chunks)
        self._tokens.extend(tokenize(c.text, self.stem) for c in chunks)
        self._rebuild()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tokenizer": tokenizer_name(self.stem), "ids": self._ids, "tokens": self._tokens}
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, BM25 score) pairs, best first; chunks sharing no terms are dropped."""
        query_tokens = tokenize(query, self.stem)
        if self._bm25 is None or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self._ids[i], float(scores[i])) for i in ranked[:top_k] if scores[i] > 0]
