from chatbot.focus import (
    citations_from_hits,
    envelope_from_result,
    parse_tool_hits,
    unique_keep_order,
)
from chatbot.tools import format_hits, is_table_heavy


def test_parse_tool_hits_roundtrip_format_hits():
    markdown = format_hits(
        [
            {
                "score": 0.91,
                "text": "ligne 1\nligne 2",
                "source_path": "/docs/a.pdf",
                "document_id": "abc",
                "project_id": "4405",
                "site_id": "lot:2363352",
                "chunk_id": "abc:3",
                "page": 4,
                "doc_type": "ees_phase_2",
                "heading_path": "Conclusions",
                "entities": ["4405"],
            }
        ],
        max_chars=600,
    )
    hits = parse_tool_hits(markdown)
    assert len(hits) == 1
    assert hits[0]["document_id"] == "abc"
    assert hits[0]["project_id"] == "4405"
    assert hits[0]["site_id"] == "lot:2363352"
    assert "ligne 1" in hits[0]["text"]
    assert "ligne 2" in hits[0]["text"]
    citations = citations_from_hits(hits)
    assert citations[0]["page"] == 4
    assert citations[0]["document_id"] == "abc"


def test_parse_tool_hits_ignores_catalog_fiches():
    from chatbot.tools import format_catalog_fiches

    fiche = format_catalog_fiches(
        [{"document_id": "abc", "client": "9342-9967 Québec Inc.", "source_path": "a.pdf"}]
    )
    chunks = format_hits(
        [
            {
                "score": 0.5,
                "text": "labo",
                "document_id": "abc",
                "page": 2,
                "source_path": "a.pdf",
            }
        ]
    )
    hits = parse_tool_hits(f"{fiche}\n\n{chunks}")
    assert len(hits) == 1
    assert hits[0]["text"] == "labo"
    assert hits[0]["document_id"] == "abc"


def test_is_table_heavy_and_preserves_pipes():
    table = "| Paramètre | F-1 |\n| --- | --- |\n| HAM | 120 |\n| HAP | 3 |"
    assert is_table_heavy(table)
    out = format_hits(
        [{"score": 0.5, "text": table, "document_id": "x", "source_path": "a.pdf"}],
        max_chars=20,
        table_max_chars=500,
    )
    assert "| HAM | 120 |" in out
    assert "\n" in out.split("### Hit", 1)[-1]


def test_unique_keep_order():
    assert unique_keep_order(["4405", None, "4405", "2259", ""]) == ["4405", "2259"]


class _Human:
    type = "human"


class _Tool:
    type = "tool"

    def __init__(self, content: str):
        self.content = content


def test_envelope_from_result_uses_tool_hits_not_reply(monkeypatch):
    monkeypatch.setattr(
        "rag_ingestion.catalog.get_documents",
        lambda ids, settings=None: [
            {
                "document_id": "abc",
                "project_id": "4405",
                "site_id": "lot:2363352",
                "title": "ÉES",
            }
        ],
    )
    tool_md = format_hits(
        [
            {
                "score": 0.7,
                "text": "extrait",
                "document_id": "abc",
                "project_id": "4405",
                "site_id": "lot:2363352",
                "source_path": "a.pdf",
                "page": 2,
            }
        ]
    )
    result = {
        "messages": [
            _Human(),
            _Tool(tool_md),
            type("AI", (), {"type": "ai", "content": "Le projet 9999 est contaminé"})(),
        ]
    }
    envelope = envelope_from_result(result)
    assert envelope["focus"]["document_ids"] == ["abc"]
    assert envelope["focus"]["project_ids"] == ["4405"]
    assert envelope["focus"]["site_ids"] == ["lot:2363352"]
    assert "9999" not in envelope["focus"]["project_ids"]
    assert envelope["documents"][0]["title"] == "ÉES"
