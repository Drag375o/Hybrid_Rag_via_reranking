"""Generate grounded answers with a local LLM served by Ollama."""

from collections.abc import Iterator

import ollama

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.generation.prompts import NO_ANSWER, build_messages, parse_citations
from hybrid_rag.schema import Answer, ScoredChunk


class GenerationError(RuntimeError):
    """Raised with a user-friendly fix when Ollama can't produce an answer."""


class OllamaGenerator:
    def __init__(self, settings: Settings | None = None, client: ollama.Client | None = None):
        self.settings = settings or get_settings()
        self.client = client or ollama.Client(host=self.settings.ollama_host)

    @property
    def model(self) -> str:
        return self.settings.llm_model

    def stream(self, question: str, sources: list[ScoredChunk]) -> Iterator[str]:
        """Yield the answer piece by piece as the model writes it."""
        if not sources:
            yield NO_ANSWER
            return
        try:
            for part in self.client.chat(
                model=self.model,
                messages=build_messages(question, sources),
                options={
                    "temperature": self.settings.llm_temperature,
                    "num_ctx": self.settings.llm_num_ctx,
                },
                stream=True,
            ):
                yield part["message"]["content"]
        except ConnectionError as exc:
            raise GenerationError(
                f"Can't reach Ollama at {self.settings.ollama_host}. "
                "Start the Ollama app (or run `ollama serve`) and try again."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise GenerationError(
                    f"The model '{self.model}' isn't downloaded. Run: ollama pull {self.model}"
                ) from exc
            raise GenerationError(f"Ollama error: {exc.error}") from exc

    def generate(self, question: str, sources: list[ScoredChunk]) -> Answer:
        text = "".join(self.stream(question, sources)).strip()
        return make_answer(question, text, sources, self.model)


def make_answer(question: str, text: str, sources: list[ScoredChunk], model: str) -> Answer:
    """Package generated text with its sources (used after streaming, too)."""
    return Answer(
        question=question,
        text=text,
        sources=sources,
        cited=parse_citations(text, len(sources)),
        model=model,
    )
