"""Core data types shared by every stage of the pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of loaded text: one page of a PDF, or one whole TXT/MD/DOCX file."""

    text: str
    source: str  # file path relative to the ingested folder, or a benchmark doc id
    page: int | None = None  # 1-based page number (PDFs only)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a Document small enough to embed, index and retrieve."""

    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int  # position of this chunk within its Document
    metadata: dict[str, Any] = field(default_factory=dict)
