from pathlib import Path

import pytest

from rag_ingestion.pipeline import collect_pdf_paths


def test_collect_pdf_paths_file(tmp_path: Path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert collect_pdf_paths(pdf) == [pdf.resolve()]


def test_collect_pdf_paths_directory_recursive(tmp_path: Path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    one = tmp_path / "one.pdf"
    two = nested / "two.PDF"
    (tmp_path / "notes.txt").write_text("nope")
    one.write_bytes(b"%PDF")
    two.write_bytes(b"%PDF")
    found = collect_pdf_paths(tmp_path)
    names = {p.name.lower() for p in found}
    assert names == {"one.pdf", "two.pdf"}


def test_collect_pdf_paths_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        collect_pdf_paths(tmp_path)
