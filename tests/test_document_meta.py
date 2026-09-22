from rag_ingestion.document_meta import build_document_meta, stamp_payloads
from rag_ingestion.models import Chunk, ChunkPayload, GroundedExtraction


def test_build_document_meta_lot_and_typed_dates():
    extractions = [
        GroundedExtraction(
            extraction_id="t",
            extraction_class="title",
            extraction_text="ÉES phase II",
            start=0,
            end=10,
        ),
        GroundedExtraction(
            extraction_id="d",
            extraction_class="doc_type",
            extraction_text="phase II",
            start=None,
            end=None,
            attributes={"normalized": "ees_phase_2"},
        ),
        GroundedExtraction(
            extraction_id="p",
            extraction_class="entity",
            extraction_text="4405",
            start=None,
            end=None,
            attributes={"type": "project_id", "normalized": "4405"},
        ),
        GroundedExtraction(
            extraction_id="f",
            extraction_class="entity",
            extraction_text="Enviro-Experts",
            start=0,
            end=8,
            attributes={"type": "firm"},
        ),
        GroundedExtraction(
            extraction_id="a",
            extraction_class="location",
            extraction_text="619, route 341, L'Épiphanie",
            start=20,
            end=40,
            attributes={"type": "address"},
        ),
        GroundedExtraction(
            extraction_id="l",
            extraction_class="location",
            extraction_text="lot 2 363 352",
            start=40,
            end=55,
            attributes={"type": "lot", "normalized": "2363352"},
        ),
        GroundedExtraction(
            extraction_id="r",
            extraction_class="date",
            extraction_text="3 janvier 2023",
            start=60,
            end=74,
            attributes={"normalized": "2023-01-03", "role": "report"},
        ),
        GroundedExtraction(
            extraction_id="w",
            extraction_class="date",
            extraction_text="17 août 2019",
            start=80,
            end=92,
            attributes={"normalized": "2019-08-17", "role": "fieldwork"},
        ),
        GroundedExtraction(
            extraction_id="k",
            extraction_class="date",
            extraction_text="5 juin 2019",
            start=93,
            end=104,
            attributes={"normalized": "2019-06-05", "role": "contract"},
        ),
        GroundedExtraction(
            extraction_id="o",
            extraction_class="date",
            extraction_text="30 avril 2021",
            start=105,
            end=118,
            attributes={"normalized": "2021-04-30", "role": "other"},
        ),
        GroundedExtraction(
            extraction_id="c",
            extraction_class="topic",
            extraction_text="HAM",
            start=120,
            end=123,
            attributes={"kind": "contaminant"},
        ),
    ]
    meta = build_document_meta(
        extractions,
        document_id="doc",
        source_path="/tmp/a.pdf",
        parse_quality="ok",
    )
    assert meta.project_id == "4405"
    assert meta.doc_type == "ees_phase_2"
    assert meta.title == "ÉES phase II"
    assert meta.firm == "Enviro-Experts"
    assert meta.lot_cadastral == "2363352"
    assert meta.address == "619, route 341, L'Épiphanie"
    assert meta.site_id == "lot:2363352"
    assert meta.report_date == "2023-01-03"
    assert meta.contaminants == ["HAM"]
    roles = {e.role: e.iso_date for e in meta.events}
    assert roles["report"] == "2023-01-03"
    assert roles["fieldwork"] == "2019-08-17"
    assert roles["contract"] == "2019-06-05"
    assert "other" not in roles


def test_stamp_payloads_copies_site_and_project():
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="x",
        start=0,
        end=1,
        heading_path="",
        page=1,
    )
    payloads = [
        ChunkPayload(
            chunk=chunk,
            doc_type="rapport",
            entities=["4405"],
            dates=[],
            topics=[],
            extraction_ids=[],
            parse_quality="ok",
            source_path="/tmp/a.pdf",
        )
    ]
    meta = build_document_meta(
        [
            GroundedExtraction(
                extraction_id="p",
                extraction_class="entity",
                extraction_text="4405",
                start=None,
                end=None,
                attributes={"type": "project_id"},
            ),
            GroundedExtraction(
                extraction_id="l",
                extraction_class="location",
                extraction_text="2363352",
                start=None,
                end=None,
                attributes={"type": "lot"},
            ),
        ],
        document_id="doc",
        source_path="/tmp/a.pdf",
        parse_quality="ok",
    )
    stamp_payloads(payloads, meta)
    assert payloads[0].site_id == "lot:2363352"
    assert payloads[0].project_id == "4405"
    assert payloads[0].contaminants == []


def test_lab_certificate_date_becomes_event():
    meta = build_document_meta(
        [
            GroundedExtraction(
                extraction_id="c",
                extraction_class="date",
                extraction_text="2025-09-03",
                start=None,
                end=None,
                attributes={"normalized": "2025-09-03", "role": "lab_certificate"},
            ),
        ],
        document_id="doc",
        source_path="/tmp/a.pdf",
        parse_quality="ok",
    )
    assert [(e.role, e.iso_date) for e in meta.events] == [
        ("lab_certificate", "2025-09-03")
    ]


def test_analysis_date_alias_becomes_lab_analysis_event():
    meta = build_document_meta(
        [
            GroundedExtraction(
                extraction_id="a",
                extraction_class="date",
                extraction_text="17 août 2019",
                start=None,
                end=None,
                attributes={"normalized": "2019-08-17", "role": "analysis"},
            ),
        ],
        document_id="doc",
        source_path="/tmp/a.pdf",
        parse_quality="ok",
    )
    assert [(e.role, e.iso_date) for e in meta.events] == [
        ("lab_analysis", "2019-08-17")
    ]

