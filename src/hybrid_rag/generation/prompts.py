"""Prompt templates and citation parsing for grounded answers.

Small local models follow *demonstrated* behaviour far more reliably than
written rules, so every prompt starts with a short made-up example exchange
(one cited answer, one refusal) on a topic unrelated to any real documents.
The rules are also repeated right before the question, where small models
pay the most attention.
"""

import re

from hybrid_rag.schema import ScoredChunk

NO_ANSWER = "I couldn't find the answer in the provided documents."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages you are given, "
    "never your own knowledge, and you cite the passages you use. "
    f'If the passages do not answer the question, you reply exactly: "{NO_ANSWER}"'
)

RULES = f"""Instructions:
- Answer using ONLY the context passages above. Do not use your own knowledge.
- End every sentence with the number of the passage it came from, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "{NO_ANSWER}\""""

# Demonstration exchange: deliberately about a fictional product, so the model
# learns the format without being able to copy facts from it.
_DEMO_CONTEXT = """Context passages:

[1] (source: zephyr_manual.pdf, page 2)
The Zephyr 300 kettle has a 1.7 litre capacity and switches off automatically when the water boils.

[2] (source: zephyr_warranty.txt)
Every Zephyr kettle comes with a two-year warranty. Repairs are free during the warranty period."""

FEW_SHOT = [
    {
        "role": "user",
        "content": f"{_DEMO_CONTEXT}\n\n{RULES}\n\nQuestion: How long is the warranty and what does it cover?\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "The kettle has a two-year warranty [2]. Repairs are free during that period [2].",
    },
    {
        "role": "user",
        "content": f"{_DEMO_CONTEXT}\n\n{RULES}\n\nQuestion: What colour options are available?\nAnswer:",
    },
    {"role": "assistant", "content": NO_ANSWER},
]


def source_label(chunk: ScoredChunk) -> str:
    """Human-readable origin of a chunk, e.g. 'paper.pdf, page 3'."""
    c = chunk.chunk
    return f"{c.source}, page {c.page}" if c.page is not None else c.source


def build_user_prompt(question: str, sources: list[ScoredChunk]) -> str:
    passages = "\n\n".join(
        f"[{number}] (source: {source_label(s)})\n{s.chunk.text}"
        for number, s in enumerate(sources, start=1)
    )
    return f"Context passages:\n\n{passages}\n\n{RULES}\n\nQuestion: {question}\nAnswer:"


def build_messages(question: str, sources: list[ScoredChunk]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT,
        {"role": "user", "content": build_user_prompt(question, sources)},
    ]


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(text: str, num_sources: int) -> list[int]:
    """Source numbers cited in the text (``[2]``, ``[1, 3]``), in first-cited order.

    Numbers outside 1..num_sources (hallucinated citations) are ignored.
    """
    cited: list[int] = []
    for group in _CITATION.findall(text):
        for number in (int(n) for n in group.split(",")):
            if 1 <= number <= num_sources and number not in cited:
                cited.append(number)
    return cited
