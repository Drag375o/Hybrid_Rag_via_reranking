"""Tests for the UI helpers, the pipeline trace, and the Streamlit app itself (with fake models)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import hybrid_rag.config
import hybrid_rag.pipeline
from conftest import FakeEmbedder, FakeOllamaClient, FakeReranker, make_chunk
from hybrid_rag.config import Settings
from hybrid_rag.generation import OllamaGenerator
from hybrid_rag.indexing import IndexStore
from hybrid_rag.pipeline import RAGPipeline, RetrievalTrace
from hybrid_rag.schema import ScoredChunk
from hybrid_rag.ui_helpers import format_answer_markdown, safe_upload_name, trace_rows

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

CHUNKS = [
    make_chunk("setup.md::c0", "Install Python 3.11 before anything else."),
    make_chunk("cats.txt::c0", "Cats purr when they are content."),
    make_chunk("dogs.txt::c0", "Dogs bark at the mail carrier."),
]


# --- helpers -------------------------------------------------------------------

def test_format_answer_markdown_bolds_citations_and_escapes_dollars():
    assert format_answer_markdown("Costs $5 [1] and [2, 3].") == r"Costs \$5 **[1]** and **[2, 3]**."


@pytest.mark.parametrize(
    "name, expected",
    [
        ("report.PDF", "report.PDF"),
        ("../../secrets/notes.md", "notes.md"),
        ("C:\\Users\\x\\paper.docx", "paper.docx"),
        ("image.png", None),
        ("", None),
    ],
)
def test_safe_upload_name(name, expected):
    assert safe_upload_name(name) == expected


def scored(chunk_id: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id, f"text {chunk_id}"), score=1 / rank, retrieval_rank=rank)


def test_trace_rows_show_final_picks_first_including_promoted_ones():
    candidates = [scored(f"c{i}::c0", i) for i in range(1, 31)]
    ranked = [candidates[29], candidates[0]]  # reranker promoted #30 to the top
    trace = RetrievalTrace(
        candidates=candidates, ranked=ranked, context=ranked, gated=False,
        rerank_scores={c.chunk.id: 0.0 for c in candidates} | {"c30::c0": 0.9, "c1::c0": 0.5},
    )

    rows = trace_rows(trace, limit=5)

    assert [(r["chunk"], r["final #"]) for r in rows[:2]] == [("c30::c0", 1), ("c1::c0", 2)]
    assert len(rows) == 6  # top 5 candidates + the promoted #30
    assert rows[0]["rerank score"] == 0.9 and rows[2]["final #"] is None


# --- pipeline.trace() ------------------------------------------------------------

@pytest.fixture
def pipeline(settings, fake_embedder, fake_reranker):
    store = IndexStore("ui_docs", settings=settings, embedder=fake_embedder)
    store.build(CHUNKS)
    client = FakeOllamaClient("Install Python 3.11 first [1].")
    return RAGPipeline(settings=settings, store=store, reranker=fake_reranker,
                       generator=OllamaGenerator(settings, client=client))


def test_trace_keeps_every_stage(pipeline):
    trace = pipeline.trace("which python to install", top_k=2)
    assert len(trace.candidates) == 3 and len(trace.ranked) == 2
    assert trace.context == trace.ranked and not trace.gated
    assert set(trace.rerank_scores) == {c.id for c in CHUNKS}  # scores for all candidates


def test_trace_gates_irrelevant_questions(pipeline):
    trace = pipeline.trace("football world cup winner")
    assert trace.gated and trace.context == [] and trace.ranked


def test_trace_without_reranker_has_no_scores_or_gate(pipeline):
    trace = pipeline.trace("football world cup winner", use_reranker=False, top_k=2)
    assert not trace.gated and trace.rerank_scores == {} and len(trace.context) == 2


# --- Streamlit app (AppTest runs the real script with fake models) -----------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    """Run the app against a temporary data folder, a fake LLM reply and fake models."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    hybrid_rag.config.get_settings.cache_clear()
    st.cache_resource.clear()
    state = {"reply": "Install Python 3.11 first [1].", "error": None, "chunks": CHUNKS}

    def fake_pipeline(*args, **kwargs):
        settings = Settings(_env_file=None, data_dir=tmp_path)
        store = IndexStore(settings=settings, embedder=FakeEmbedder())
        if state["chunks"]:
            store.build(state["chunks"])
        client = FakeOllamaClient(state["reply"], error=state["error"])
        return RAGPipeline(settings=settings, store=store, reranker=FakeReranker(),
                           generator=OllamaGenerator(settings, client=client))

    monkeypatch.setattr(hybrid_rag.pipeline, "RAGPipeline", fake_pipeline)
    yield state
    st.cache_resource.clear()
    hybrid_rag.config.get_settings.cache_clear()


def run_app(question: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    if question:
        at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    return at


def test_app_answers_with_citations_and_details(app):
    at = run_app("Which Python should I install?")

    assert at.chat_message[1].markdown[0].value == "Install Python 3.11 first **[1]**."
    assert any("all 1 sentences are supported" in c.value for c in at.caption)
    labels = [e.label for e in at.expander]
    assert "Sources (3 passages given to the LLM)" in labels
    assert "How these passages were chosen" in labels


def test_app_explains_relevance_gate(app):
    at = run_app("Who won the football world cup?")
    assert any("wasn't asked" in i.value for i in at.info)


def test_app_flags_wrong_citation(app):
    app["reply"] = "Install Python 3.11 first [2]."  # [2] is not the setup passage
    at = run_app("Which Python should I install?")
    assert any("doesn't clearly support it" in w.value for w in at.warning)


def test_app_shows_ollama_errors(app):
    app["error"] = ConnectionError("Failed to connect to Ollama")
    at = run_app("Which Python should I install?")
    assert any("Start the Ollama app" in e.value for e in at.error)


def test_app_with_empty_index_disables_chat(app):
    app["chunks"] = []
    at = run_app()
    assert at.chat_input[0].disabled
    assert any("No documents are indexed yet" in i.value for i in at.info)
