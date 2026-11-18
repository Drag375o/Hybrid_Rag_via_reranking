import re

import pytest

from hybrid_rag.ingestion import chunk_documents, split_text
from hybrid_rag.schema import Document

PROSE = " ".join(
    f"Sentence number {i} explains one small idea about hybrid retrieval." for i in range(60)
)


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def test_short_text_is_a_single_chunk():
    assert split_text("A short note.", chunk_size=100, chunk_overlap=10) == ["A short note."]


def test_chunks_respect_size_limit():
    chunks = split_text(PROSE, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_no_text_is_lost():
    chunks = split_text(PROSE, chunk_size=200, chunk_overlap=40)
    covered = set(" ".join(chunks).split())
    assert set(words(PROSE)) <= covered


def test_neighbouring_chunks_overlap():
    chunks = split_text(PROSE, chunk_size=200, chunk_overlap=60)
    for previous, current in zip(chunks, chunks[1:]):
        tail = words(previous)[-3:]
        assert " ".join(tail) in current


def test_prefers_paragraph_boundaries():
    text = "Paragraph one is here.\n\nParagraph two is here.\n\nParagraph three is here."
    chunks = split_text(text, chunk_size=30, chunk_overlap=0)
    assert chunks == ["Paragraph one is here.", "Paragraph two is here.", "Paragraph three is here."]


def test_unbroken_text_is_hard_split():
    chunks = split_text("x" * 250, chunk_size=100, chunk_overlap=10)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text(PROSE, chunk_size=100, chunk_overlap=100)


def test_title_is_repeated_on_every_chunk_within_size():
    doc = Document(text=f"Hybrid Retrieval\n\n{PROSE}", source="paper", metadata={"title": "Hybrid Retrieval"})

    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50, title_prefix=True)

    assert len(chunks) > 1
    assert all(c.text.startswith("Hybrid Retrieval\n\n") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[0].text.count("Hybrid Retrieval") == 1  # title in the text isn't doubled
    assert set(words(PROSE)) <= set(" ".join(c.text for c in chunks).split())


def test_title_prefix_is_skipped_without_title_or_when_disabled():
    untitled = Document(text=PROSE, source="notes.txt")
    titled = Document(text=PROSE, source="paper", metadata={"title": "Hybrid Retrieval"})

    assert not chunk_documents([untitled], 300, 50, title_prefix=True)[1].text.startswith("Hybrid")
    assert not chunk_documents([titled], 300, 50, title_prefix=False)[1].text.startswith("Hybrid")


def test_overlong_title_falls_back_to_normal_splitting():
    doc = Document(text=PROSE, source="paper", metadata={"title": "T" * 290})
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50, title_prefix=True)
    assert all(len(c.text) <= 300 for c in chunks) and not chunks[1].text.startswith("TTT")


def test_chunk_documents_ids_and_metadata():
    docs = [
        Document(text=PROSE, source="paper.pdf", page=3, metadata={"title": "RAG"}),
        Document(text="Short file.", source="notes/a.txt"),
    ]

    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)

    pdf_chunks = [c for c in chunks if c.source == "paper.pdf"]
    assert pdf_chunks[0].id == "paper.pdf::p3::c0"
    assert [c.chunk_index for c in pdf_chunks] == list(range(len(pdf_chunks)))
    assert all(c.page == 3 and c.metadata == {"title": "RAG"} for c in pdf_chunks)

    txt_chunk = next(c for c in chunks if c.source == "notes/a.txt")
    assert txt_chunk.id == "notes/a.txt::c0" and txt_chunk.page is None

    assert len({c.id for c in chunks}) == len(chunks)
