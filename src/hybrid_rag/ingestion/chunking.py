"""Split Documents into overlapping Chunks.

Uses recursive splitting: try to break text at paragraph boundaries first, then
lines, then sentences, then words, and only cut mid-word as a last resort. The
small pieces are then packed back together into chunks of up to ``chunk_size``
characters, with roughly ``chunk_overlap`` characters repeated between
neighbouring chunks so that a sentence cut at a boundary still appears whole
in one of them.
"""

from collections.abc import Iterable

from hybrid_rag.config import get_settings
from hybrid_rag.schema import Chunk, Document

SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [piece for piece in pieces if piece]


def _split_recursive(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Break text into pieces no longer than chunk_size, preferring coarse separators."""
    if len(text) <= chunk_size:
        return [text]

    for i, separator in enumerate(separators):
        if separator == "" or separator in text:
            remaining = separators[i + 1:]
            break

    if separator == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    for piece in _split_keeping_separator(text, separator):
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, remaining, chunk_size))
    return pieces


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack pieces into chunks, carrying trailing pieces over as overlap."""
    chunks: list[str] = []
    window: list[str] = []
    length = 0
    for piece in pieces:
        if window and length + len(piece) > chunk_size:
            chunks.append("".join(window))
            while window and (length > chunk_overlap or length + len(piece) > chunk_size):
                length -= len(window.pop(0))
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most ``chunk_size`` characters."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return _merge(_split_recursive(text, SEPARATORS, chunk_size), chunk_size, chunk_overlap)


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split Documents into Chunks with stable, human-readable ids."""
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    chunks: list[Chunk] = []
    for doc in documents:
        prefix = f"{doc.source}::p{doc.page}" if doc.page is not None else doc.source
        for index, text in enumerate(split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    id=f"{prefix}::c{index}",
                    text=text,
                    source=doc.source,
                    page=doc.page,
                    chunk_index=index,
                    metadata=dict(doc.metadata),
                )
            )
    return chunks
