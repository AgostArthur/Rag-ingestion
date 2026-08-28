from rag_ingestion.chunk import chunk_markdown
from rag_ingestion.models import PageSpan
from rag_ingestion.parse import page_for_span


def test_offsets_are_slices_of_source():
    text = "# Titre\n\n" + ("Paragraphe. " * 40 + "\n\n") * 8
    chunks = chunk_markdown(text, document_id="abc", max_chars=200, overlap_chars=20)
    assert chunks
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert chunk.document_id == "abc"
        assert chunk.chunk_id == f"abc:{chunk.chunk_index}"


def test_heading_path():
    text = "# Rapport\n\nIntro.\n\n## Méthode\n\n" + ("détail " * 30)
    chunks = chunk_markdown(text, document_id="d", max_chars=80, overlap_chars=10)
    later = [c for c in chunks if "détail" in c.text]
    assert later
    assert "Rapport" in later[-1].heading_path
    assert "Méthode" in later[-1].heading_path


def test_empty_markdown():
    assert chunk_markdown("   \n", document_id="x") == []
    assert chunk_markdown("", document_id="x") == []


def test_page_for_span():
    spans = [
        PageSpan(page_num=1, start=0, end=50),
        PageSpan(page_num=2, start=50, end=120),
    ]
    assert page_for_span(spans, 0, 20) == 1
    assert page_for_span(spans, 60, 80) == 2
    assert page_for_span(spans, 40, 70) == 2
