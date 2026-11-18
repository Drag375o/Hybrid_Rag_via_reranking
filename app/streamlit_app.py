"""Streamlit chat UI for the hybrid RAG pipeline.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import logging

import streamlit as st

from hybrid_rag.config import get_settings
from hybrid_rag.generation import GenerationError, check_citations, make_answer, source_label
from hybrid_rag.pipeline import RAGPipeline, RetrievalTrace
from hybrid_rag.retrieval import RETRIEVAL_MODES
from hybrid_rag.schema import Answer
from hybrid_rag.ui_helpers import format_answer_markdown, safe_upload_name, trace_rows

st.set_page_config(page_title="Hybrid RAG", page_icon="🔎", layout="wide")
logging.basicConfig(level=logging.WARNING)

settings = get_settings()
MODE_LABELS = {"hybrid": "Hybrid (dense + BM25)", "dense": "Dense only", "sparse": "BM25 only"}


@st.cache_resource(show_spinner="Loading models (the first time can take a minute)...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = load_pipeline()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("upload_key", 0)


# --- Rendering -----------------------------------------------------------------

def render_details(answer: Answer, trace: RetrievalTrace) -> None:
    """Relevance gate notice, citation check, sources and the pipeline table."""
    if trace.gated:
        best = trace.ranked[0].rerank_score
        st.info(
            f"No passage was relevant enough to answer from (best reranker score "
            f"{best:.4f} < threshold {settings.min_relevance}), so the LLM wasn't asked."
        )

    checks = check_citations(answer, settings.citation_min_support)
    problems = [c for c in checks if c.status != "supported"]
    for check in problems:
        if check.status == "uncited":
            st.warning(f"No citation for: *{check.sentence}* (best match: [{check.best_source}])")
        else:
            cited = ", ".join(f"[{n}]" for n in check.cited)
            st.warning(
                f"Citation check: *{check.sentence}* cites {cited}, which doesn't clearly "
                f"support it (best match: [{check.best_source}])"
            )
    if checks and not problems:
        st.caption(f"✅ Citation check: all {len(checks)} sentences are supported by what they cite")

    if answer.sources:
        with st.expander(f"Sources ({len(answer.sources)} passages given to the LLM)"):
            for number, source in enumerate(answer.sources, start=1):
                details = [source_label(source)]
                if source.rerank_score is not None:
                    details.append(f"relevance {source.rerank_score:.3f}")
                if number in answer.cited:
                    details.append("**cited**")
                st.markdown(f"**[{number}]** " + " · ".join(details))
                st.code(source.chunk.text, language=None, wrap_lines=True)

    if trace.candidates:
        with st.expander("How these passages were chosen"):
            st.caption(
                "Each row is a candidate passage. **dense #** / **BM25 #** = its rank in each "
                "search (empty = not found), **retrieval #** = rank after fusion, "
                "**rerank score** = cross-encoder relevance (0-1), **final #** = position in "
                "the list given to the LLM."
            )
            st.dataframe(
                trace_rows(trace),
                hide_index=True,
                width="stretch",
                column_config={
                    "retrieval score": st.column_config.NumberColumn(format="%.4f"),
                    "rerank score": st.column_config.NumberColumn(format="%.4f"),
                },
            )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        if message.get("error"):
            st.error(message["error"])
            return
        st.markdown(format_answer_markdown(message["answer"].text))
        st.caption(message["caption"])
        render_details(message["answer"], message["trace"])


# --- Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.header("Search settings")
    mode = st.radio(
        "Retrieval", RETRIEVAL_MODES, index=RETRIEVAL_MODES.index("hybrid"),
        format_func=MODE_LABELS.get,
    )
    use_reranker = st.toggle("Cross-encoder reranking", value=True)
    top_k = st.slider("Passages given to the LLM", 1, 10, settings.rerank_top_k)
    st.caption(
        f"LLM `{settings.llm_model}` · embeddings `{settings.embedding_model.split('/')[-1]}` · "
        f"reranker `{settings.reranker_model.split('/')[-1]}`"
    )

    st.header("Documents")
    st.metric("Indexed chunks", pipeline.store.count())
    uploads = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"uploads_{st.session_state.upload_key}",
    )
    if st.button("Save & rebuild index", type="primary", width="stretch"):
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads or []:
            name = safe_upload_name(upload.name)
            if name:
                (settings.docs_dir / name).write_bytes(upload.getvalue())
        with st.spinner("Loading, chunking and embedding documents..."):
            stats = pipeline.store.build_from_folder(settings.docs_dir)
        st.session_state.upload_key += 1  # clears the uploader
        st.session_state.last_build = (
            f"Indexed {stats.files} files ({stats.documents} documents) into {stats.chunks} chunks."
        )
        st.rerun()
    if "last_build" in st.session_state:
        st.success(st.session_state.last_build)
    st.caption(f"Documents folder: `{settings.docs_dir}`")

    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --- Chat ------------------------------------------------------------------------

st.title("🔎 Hybrid RAG")
st.caption(
    "Ask questions about your documents. Answers come only from the indexed files, "
    "found with dense + BM25 search, reranked by a cross-encoder, and cited."
)

for message in st.session_state.messages:
    render_message(message)

indexed = pipeline.store.count() > 0
if not indexed:
    st.info("No documents are indexed yet. Add files in the sidebar and click **Save & rebuild index**.")

question = st.chat_input("Ask a question about your documents...", disabled=not indexed)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            trace = pipeline.trace(question, mode=mode, use_reranker=use_reranker, top_k=top_k)

        placeholder, text = st.empty(), ""
        try:
            for piece in pipeline.generator.stream(question, trace.context):
                text += piece
                placeholder.markdown(format_answer_markdown(text) + " ▌")
        except GenerationError as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.append({"role": "assistant", "error": str(exc)})
        else:
            answer = make_answer(question, text.strip(), trace.context, settings.llm_model)
            placeholder.markdown(format_answer_markdown(answer.text))
            caption = f"{MODE_LABELS[mode]} · reranking {'on' if use_reranker else 'off'} · {settings.llm_model}"
            st.caption(caption)
            render_details(answer, trace)
            st.session_state.messages.append(
                {"role": "assistant", "answer": answer, "trace": trace, "caption": caption}
            )
