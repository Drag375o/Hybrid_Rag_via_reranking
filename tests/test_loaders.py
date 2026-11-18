from pathlib import Path

import pytest
from docx import Document as DocxDocument

from hybrid_rag.ingestion import load_directory, load_file
from hybrid_rag.ingestion.loaders import clean_text


def write_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal valid PDF with one line of Helvetica text per page."""
    font_id = 3 + 2 * len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages))), len(pages)
        ),
    ]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    path.write_bytes(out)


def test_clean_text_normalises_whitespace_and_hyphenation():
    raw = "Retrie-\nval   augmented\r\n\n\n\ngeneration \t works"
    assert clean_text(raw) == "Retrieval augmented\n\ngeneration works"


def test_load_txt_and_md(tmp_path):
    (tmp_path / "notes.txt").write_text("Plain text file.", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Title\n\nSome markdown.", encoding="utf-8")

    [txt] = load_file(tmp_path / "notes.txt")
    [md] = load_file(tmp_path / "readme.md")

    assert txt.text == "Plain text file." and txt.source == "notes.txt" and txt.page is None
    assert md.text == "# Title\n\nSome markdown."


def test_load_docx_includes_paragraphs_and_tables(tmp_path):
    docx = DocxDocument()
    docx.add_paragraph("First paragraph.")
    docx.add_paragraph("Second paragraph.")
    table = docx.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Model"
    table.rows[0].cells[1].text = "bge-reranker"
    docx.save(tmp_path / "report.docx")

    [doc] = load_file(tmp_path / "report.docx")

    assert "First paragraph.\n\nSecond paragraph." in doc.text
    assert "Model | bge-reranker" in doc.text


def test_load_pdf_creates_one_document_per_page(tmp_path):
    write_pdf(tmp_path / "paper.pdf", ["Hello page one", "Hello page two"])

    docs = load_file(tmp_path / "paper.pdf")

    assert [d.page for d in docs] == [1, 2]
    assert docs[0].text == "Hello page one"
    assert docs[1].text == "Hello page two"


def test_load_file_rejects_unsupported_type(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(tmp_path / "image.png")


def test_load_directory_is_recursive_and_skips_bad_files(tmp_path, caplog):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Top level.", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("Nested.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("x,y", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a real pdf")

    docs = load_directory(tmp_path)

    assert sorted(d.source for d in docs) == ["a.txt", "sub/b.md"]
    assert "broken.pdf" in caplog.text
