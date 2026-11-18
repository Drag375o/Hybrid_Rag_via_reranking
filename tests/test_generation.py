import ollama
import pytest

from conftest import make_chunk
from hybrid_rag.generation import (
    NO_ANSWER,
    GenerationError,
    OllamaGenerator,
    build_messages,
    parse_citations,
)
from hybrid_rag.indexing import IndexStore
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.schema import ScoredChunk


class FakeOllamaClient:
    """Streams a canned reply word by word, or raises a configured error."""

    def __init__(self, reply="", error=None):
        self.reply, self.error = reply, error
        self.calls = []

    def chat(self, model, messages, options, stream):
        self.calls.append({"model": model, "messages": messages, "options": options})
        if self.error:
            raise self.error
        return self._stream()

    def _stream(self):
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield {"message": {"content": word if i == len(words) - 1 else word + " "}}


def source(chunk_id: str, text: str, page: int | None = None) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id, text, page=page), score=1.0)


SOURCES = [
    source("guide.pdf::p4::c0", "Python 3.11 or newer is required.", page=4),
    source("notes.md::c2", "Ollama must be running before you ask questions."),
]


# --- prompts & citations ----------------------------------------------------

def test_prompt_numbers_sources_with_labels():
    messages = build_messages("Which Python?", SOURCES)
    system, user = messages[0], messages[-1]
    assert system["role"] == "system" and NO_ANSWER in system["content"]
    assert user["role"] == "user"
    assert "[1] (source: guide.pdf, page 4)\nPython 3.11 or newer is required." in user["content"]
    assert "[2] (source: notes.md)" in user["content"]
    assert user["content"].endswith("Question: Which Python?\nAnswer:")


def test_prompt_demonstrates_citing_and_refusing():
    demo = build_messages("q", SOURCES)[1:-1]
    assert [m["role"] for m in demo] == ["user", "assistant", "user", "assistant"]
    assert parse_citations(demo[1]["content"], num_sources=2)  # demo answer cites
    assert demo[3]["content"] == NO_ANSWER  # demo refusal
    # the demo must not leak into the real context passages
    assert "Zephyr" not in build_messages("q", SOURCES)[-1]["content"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Use Python 3.11 [1].", [1]),
        ("Both apply [2][1].", [2, 1]),
        ("Grouped [1, 2] and repeated [1].", [1, 2]),
        ("Out of range [7] and [0] are ignored [2].", [2]),
        ("No citations here.", []),
    ],
)
def test_parse_citations(text, expected):
    assert parse_citations(text, num_sources=2) == expected


# --- OllamaGenerator --------------------------------------------------------

def test_generate_streams_and_parses_citations(settings):
    client = FakeOllamaClient("You need Python 3.11 or newer [1].")
    answer = OllamaGenerator(settings, client=client).generate("Which Python?", SOURCES)

    assert answer.text == "You need Python 3.11 or newer [1]."
    assert answer.cited == [1]
    assert answer.cited_sources[0].chunk.id == "guide.pdf::p4::c0"
    assert answer.model == settings.llm_model
    assert client.calls[0]["options"] == {"temperature": settings.llm_temperature,
                                          "num_ctx": settings.llm_num_ctx}


def test_no_sources_skips_the_llm(settings):
    client = FakeOllamaClient("should not be used")
    answer = OllamaGenerator(settings, client=client).generate("Anything?", [])
    assert answer.text == NO_ANSWER and answer.cited == []
    assert client.calls == []


def test_ollama_not_running_gives_clear_error(settings):
    client = FakeOllamaClient(error=ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(GenerationError, match="Start the Ollama app"):
        OllamaGenerator(settings, client=client).generate("q", SOURCES)


def test_missing_model_tells_user_how_to_pull_it(settings):
    client = FakeOllamaClient(error=ollama.ResponseError("model not found", status_code=404))
    with pytest.raises(GenerationError, match=f"ollama pull {settings.llm_model}"):
        OllamaGenerator(settings, client=client).generate("q", SOURCES)


# --- RAGPipeline.ask() ------------------------------------------------------

def make_pipeline(settings, fake_embedder, fake_reranker, reply="Install Python 3.11 first [1]."):
    store = IndexStore("ask_docs", settings=settings, embedder=fake_embedder)
    store.build([
        make_chunk("setup.md::c0", "Install Python 3.11 before anything else."),
        make_chunk("cats.txt::c0", "Cats purr when they are content."),
    ])
    client = FakeOllamaClient(reply)
    pipeline = RAGPipeline(
        settings=settings,
        store=store,
        reranker=fake_reranker,
        generator=OllamaGenerator(settings, client=client),
    )
    return pipeline, client


def test_pipeline_ask_passes_reranked_sources_to_llm(settings, fake_embedder, fake_reranker):
    pipeline, client = make_pipeline(settings, fake_embedder, fake_reranker)

    answer = pipeline.ask("Which Python should I install?")

    assert answer.sources[0].chunk.id == "setup.md::c0"
    assert answer.cited_sources[0].chunk.id == "setup.md::c0"
    assert len(answer.sources) <= settings.rerank_top_k
    assert "[1] (source: setup.md)" in client.calls[0]["messages"][-1]["content"]


def test_irrelevant_question_is_refused_without_calling_llm(settings, fake_embedder, fake_reranker):
    pipeline, client = make_pipeline(settings, fake_embedder, fake_reranker)

    # FakeReranker scores 0 when no query word appears in any passage
    answer = pipeline.ask("Who won the football world cup?")

    assert answer.text == NO_ANSWER and answer.sources == []
    assert client.calls == []


def test_relevance_gate_can_be_disabled(fake_embedder, fake_reranker, tmp_path):
    from hybrid_rag.config import Settings

    settings = Settings(_env_file=None, data_dir=tmp_path, min_relevance=0)
    pipeline, client = make_pipeline(settings, fake_embedder, fake_reranker)

    pipeline.ask("Who won the football world cup?")

    assert len(client.calls) == 1


def test_relevance_gate_needs_reranker_scores(settings, fake_embedder, fake_reranker):
    pipeline, client = make_pipeline(settings, fake_embedder, fake_reranker)

    pipeline.ask("Who won the football world cup?", use_reranker=False)

    assert len(client.calls) == 1  # no rerank scores -> no gate, the LLM decides
