"""Load PDF, TXT, Markdown and DOCX files into Documents."""

import logging
import re
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from hybrid_rag.schema import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def clean_text(text: str) -> str:
    """Normalise whitespace and undo common PDF extraction artefacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path, source: str) -> list[Document]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt("")  # many "encrypted" PDFs only have an empty user password
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            documents.append(Document(text=text, source=source, page=page_number))
    return documents


def _load_text(path: Path, source: str) -> list[Document]:
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [Document(text=text, source=source)] if text else []


def _load_docx(path: Path, source: str) -> list[Document]:
    docx = DocxDocument(str(path))
    blocks = [p.text for p in docx.paragraphs]
    for table in docx.tables:
        for row in table.rows:
            blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
    text = clean_text("\n\n".join(b for b in blocks if b.strip()))
    return [Document(text=text, source=source)] if text else []


_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
    ".docx": _load_docx,
}


def load_file(path: str | Path, base_dir: str | Path | None = None) -> list[Document]:
    """Load one file. ``source`` is recorded relative to ``base_dir`` when given."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path})")
    source = path.relative_to(base_dir).as_posix() if base_dir else path.name
    return loader(path, source)


def load_directory(directory: str | Path, recursive: bool = True) -> list[Document]:
    """Load every supported file in a folder, skipping (and logging) files that fail."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents: list[Document] = []
    for path in files:
        try:
            documents.extend(load_file(path, base_dir=directory))
        except Exception as exc:  # one bad file should not stop the whole ingestion
            logger.warning("Skipping %s: %s", path, exc)
    return documents
