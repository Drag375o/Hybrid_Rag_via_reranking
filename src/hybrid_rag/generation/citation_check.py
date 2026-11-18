"""Check whether each cited sentence is actually supported by the passage it cites.

Small LLMs sometimes attach the wrong passage number, or none at all. This is a
cheap word-overlap check, not proof of correctness: it flags citations whose
passage shares too few of the sentence's content words, and points to the
passage that matches best so a reader can verify it.
"""

import re
from dataclasses import dataclass

from hybrid_rag.generation.prompts import NO_ANSWER, parse_citations
from hybrid_rag.indexing.sparse import tokenize
from hybrid_rag.schema import Answer

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATIONS = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_LEADING_CITATIONS = re.compile(r"^(?:\[\d+(?:\s*,\s*\d+)*\][\s.]*)+")


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]  # passage numbers the sentence cites
    support: dict[int, float]  # passage number -> share of the sentence's words it contains
    best_source: int | None  # passage with the highest support
    min_support: float

    @property
    def status(self) -> str:
        """'supported', 'unsupported' (cites a passage that doesn't match) or 'uncited'."""
        if not self.cited:
            return "uncited"
        if all(self.support.get(n, 0.0) >= self.min_support for n in self.cited):
            return "supported"
        return "unsupported"


def split_sentences(text: str) -> list[str]:
    """Split into sentences, re-attaching citations written after the full stop.

    "It is red. [2] It is big [1]." -> ["It is red. [2]", "It is big [1]."]
    """
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text.strip()):
        leading = _LEADING_CITATIONS.match(piece)
        if sentences and leading:
            sentences[-1] += " " + leading.group().strip()
            piece = piece[leading.end():]
        if piece.strip():
            sentences.append(piece.strip())
    return sentences


def check_citations(answer: Answer, min_support: float = 0.5) -> list[SentenceCheck]:
    """One check per answer sentence that makes a claim (refusals are skipped)."""
    if not answer.sources or answer.text.strip() == NO_ANSWER:
        return []

    passage_tokens = [set(tokenize(s.chunk.text)) for s in answer.sources]
    checks = []
    for sentence in split_sentences(answer.text):
        words = set(tokenize(_CITATIONS.sub(" ", sentence)))
        if not words:
            continue
        support = {
            number: len(words & tokens) / len(words)
            for number, tokens in enumerate(passage_tokens, start=1)
        }
        checks.append(
            SentenceCheck(
                sentence=sentence,
                cited=parse_citations(sentence, len(answer.sources)),
                support=support,
                best_source=max(support, key=support.get),
                min_support=min_support,
            )
        )
    return checks
