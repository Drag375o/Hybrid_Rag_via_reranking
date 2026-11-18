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
