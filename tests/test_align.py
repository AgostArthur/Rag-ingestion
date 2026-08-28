from rag_ingestion.align import align_extractions
from rag_ingestion.models import Chunk, GroundedExtraction


def _chunk(index: int, start: int, end: int, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=f"doc:{index}",
        document_id="doc",
        chunk_index=index,
        text=text,
        start=start,
        end=end,
        heading_path="A",
        page=1,
    )


def test_overlap_attaches_entity_only_to_matching_chunk():
    chunks = [
        _chunk(0, 0, 50, "alpha"),
        _chunk(1, 50, 100, "beta"),
    ]
    extractions = [
        GroundedExtraction(
            extraction_id="e1",
            extraction_class="entity",
            extraction_text="ACME",
            start=10,
            end=14,
        ),
        GroundedExtraction(
            extraction_id="d1",
            extraction_class="doc_type",
            extraction_text="rapport",
            start=None,
            end=None,
            attributes={"normalized": "rapport"},
        ),
        GroundedExtraction(
            extraction_id="date1",
            extraction_class="date",
            extraction_text="2024-01-01",
            start=70,
            end=80,
        ),
    ]
    payloads = align_extractions(
        chunks,
        extractions,
        source_path="/tmp/a.pdf",
        parse_quality="ok",
    )
    assert payloads[0].entities == ["ACME"]
    assert payloads[0].dates == []
    assert payloads[1].entities == []
    assert payloads[1].dates == ["2024-01-01"]
    assert payloads[0].doc_type == "rapport"
    assert payloads[1].doc_type == "rapport"


def test_ungrounded_entity_is_not_copied_to_chunks():
    chunks = [_chunk(0, 0, 10)]
    extractions = [
        GroundedExtraction(
            extraction_id="e",
            extraction_class="entity",
            extraction_text="Ghost",
            start=None,
            end=None,
        )
    ]
    payloads = align_extractions(
        chunks, extractions, source_path="x", parse_quality="ok"
    )
    assert payloads[0].entities == []


def test_project_id_is_copied_to_every_chunk_even_if_ungrounded():
    chunks = [
        _chunk(0, 0, 50, "page titre"),
        _chunk(1, 50, 100, "conclusions"),
    ]
    extractions = [
        GroundedExtraction(
            extraction_id="p",
            extraction_class="entity",
            extraction_text="Projet nº : 4405",
            start=None,
            end=None,
            attributes={"type": "project_id", "normalized": "4405"},
        ),
        GroundedExtraction(
            extraction_id="e",
            extraction_class="entity",
            extraction_text="ACME",
            start=10,
            end=14,
        ),
    ]
    payloads = align_extractions(
        chunks, extractions, source_path="x", parse_quality="ok"
    )
    assert payloads[0].entities == ["4405", "ACME"]
    assert payloads[1].entities == ["4405"]


def test_numeric_entity_is_treated_as_project_id_but_year_is_not():
    chunks = [_chunk(0, 0, 20), _chunk(1, 20, 40)]
    extractions = [
        GroundedExtraction(
            extraction_id="p",
            extraction_class="entity",
            extraction_text="2259",
            start=2,
            end=6,
        ),
        GroundedExtraction(
            extraction_id="y",
            extraction_class="entity",
            extraction_text="2019",
            start=2,
            end=6,
        ),
    ]
    payloads = align_extractions(
        chunks, extractions, source_path="x", parse_quality="ok"
    )
    assert payloads[0].entities == ["2259", "2019"]
    assert payloads[1].entities == ["2259"]
