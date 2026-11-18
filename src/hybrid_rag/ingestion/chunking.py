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


def _split_with_title(doc: Document, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a document so every chunk starts with its title (when it has one).

    Without this, only the first chunk of a document says what it is about, so
    the embedder and reranker judge later chunks without that context. The title
    counts towards ``chunk_size``; if it is too long to leave room for text, the
    document is split normally instead.
    """
    title = str(doc.metadata.get("title") or "").strip()
    header = f"{title}\n\n"
    body_size = chunk_size - len(header)
    if not title or body_size <= chunk_overlap:
        return split_text(doc.text, chunk_size, chunk_overlap)

    body = doc.text.strip()
    if body.startswith(title):  # don't repeat a title the text already starts with
        body = body[len(title):].lstrip()
    return [header + part for part in split_text(body, body_size, chunk_overlap)] if body else [title]


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    title_prefix: bool | None = None,
) -> list[Chunk]:
    """Split Documents into Chunks with stable, human-readable ids.

    With ``title_prefix`` (default: the ``chunk_title_prefix`` setting), documents
    that have a ``title`` in their metadata get it at the top of every chunk.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    title_prefix = settings.chunk_title_prefix if title_prefix is None else title_prefix

    chunks: list[Chunk] = []
    for doc in documents:
        prefix = f"{doc.source}::p{doc.page}" if doc.page is not None else doc.source
        texts = (
            _split_with_title(doc, chunk_size, chunk_overlap)
            if title_prefix
            else split_text(doc.text, chunk_size, chunk_overlap)
        )
        for index, text in enumerate(texts):
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
