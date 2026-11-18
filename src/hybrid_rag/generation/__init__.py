"""Answer generation with a local LLM via Ollama."""

from hybrid_rag.generation.citation_check import SentenceCheck, check_citations, split_sentences
from hybrid_rag.generation.ollama_llm import GenerationError, OllamaGenerator, make_answer
from hybrid_rag.generation.prompts import (
    NO_ANSWER,
    build_messages,
    parse_citations,
    source_label,
)

__all__ = [
    "NO_ANSWER",
    "GenerationError",
    "OllamaGenerator",
    "SentenceCheck",
    "build_messages",
    "check_citations",
    "make_answer",
    "parse_citations",
    "source_label",
    "split_sentences",
]
