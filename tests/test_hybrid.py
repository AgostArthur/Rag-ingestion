from rag_ingestion.qdrant_store import build_filter, build_keyword_filter
from rag_ingestion.retrieve import merge_hits, project_tokens_for_search, search


def test_project_tokens_for_search_merges_query_and_filters():
    tokens = project_tokens_for_search(
        "contamination 4405 en 2019",
        {"entities": "2259", "doc_type": "ees_phase_2"},
    )
    assert tokens == ["4405", "2259"]


def test_merge_hits_prefers_higher_score_and_respects_limit():
    merged = merge_hits(
        [{"chunk_id": "a", "score": 0.2, "text": "old"}],
        [
            {"chunk_id": "a", "score": 0.9, "text": "new"},
            {"chunk_id": "b", "score": 0.5, "text": "other"},
            {"chunk_id": "c", "score": 0.1, "text": "low"},
        ],
        limit=2,
    )
    assert [h["chunk_id"] for h in merged] == ["a", "b"]
    assert merged[0]["text"] == "new"


def test_search_hybrid_unions_keyword_when_query_has_project_id(monkeypatch):
    calls: list[dict] = []

    def fake_embed(text, settings=None):
        return [0.1, 0.2]

    def fake_similar(collection, vector, *, limit=5, filters=None, query_filter=None, settings=None):
        calls.append({"filters": filters, "query_filter": query_filter, "limit": limit})
        if query_filter is not None:
            return [
                {
                    "score": 0.91,
                    "text": "Projet nº : 4405 conclusions",
                    "document_id": "sha-4405",
                    "chunk_id": "sha-4405:8",
                    "chunk_index": 8,
                    "project_id": None,
                    "entities": [],
                }
            ]
        return [
            {
                "score": 0.88,
                "text": "même site, autre dossier 2259",
                "document_id": "sha-2259",
                "chunk_id": "sha-2259:3",
                "chunk_index": 3,
                "project_id": "2259",
                "entities": ["2259"],
            }
        ]

    monkeypatch.setattr("rag_ingestion.retrieve.embed_query", fake_embed)
    monkeypatch.setattr("rag_ingestion.retrieve.search_similar", fake_similar)

    from rag_ingestion.config import Settings
    from pathlib import Path

    settings = Settings(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks_test",
        embed_model="x",
        embed_dim=8,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=Path("config/langextract/prompt.base.txt"),
        langextract_few_shots_file=Path("config/langextract/profiles/default/few_shots.json"),
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=Path("data"),
        project_root=Path("."),
    )
    hits = search("contamination 4405", limit=5, settings=settings)
    assert len(calls) == 2
    assert calls[0]["filters"] is None
    assert calls[1]["query_filter"] is not None
    ids = {h["document_id"] for h in hits}
    assert "sha-4405" in ids
    assert "sha-2259" in ids


def test_search_hybrid_text_runs_keyword_without_project_id(monkeypatch):
    calls: list[dict] = []

    def fake_embed(text, settings=None):
        return [0.1, 0.2]

    def fake_similar(collection, vector, *, limit=5, filters=None, query_filter=None, settings=None):
        calls.append({"query_filter": query_filter, "limit": limit})
        if query_filter is not None:
            return [
                {
                    "score": 0.7,
                    "text": "Client : 9342-9967 Québec Inc.",
                    "document_id": "sha-g25",
                    "chunk_id": "sha-g25:0",
                    "chunk_index": 0,
                }
            ]
        return [
            {
                "score": 0.4,
                "text": "tableau d'analyses",
                "document_id": "sha-g25",
                "chunk_id": "sha-g25:9",
                "chunk_index": 9,
            }
        ]

    monkeypatch.setattr("rag_ingestion.retrieve.embed_query", fake_embed)
    monkeypatch.setattr("rag_ingestion.retrieve.search_similar", fake_similar)

    from pathlib import Path

    from rag_ingestion.config import Settings

    settings = Settings(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks_test",
        embed_model="x",
        embed_dim=8,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=Path("config/langextract/prompt.txt"),
        langextract_few_shots_file=Path("config/langextract/few_shots.json"),
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=Path("data"),
        project_root=Path("."),
    )
    hits = search("quel est le nom du client", limit=5, hybrid_text=True, settings=settings)
    assert len(calls) == 2
    assert calls[1]["query_filter"] is not None
    assert hits[0]["text"].startswith("Client")


def test_keyword_filter_drops_strict_entities():
    built = build_keyword_filter(["4405"], {"entities": "4405", "doc_type": "ees_phase_2"})
    assert built is not None
    assert built.must is not None
    assert len(built.must) == 2
    assert built.must[0].key == "doc_type"
    nested = built.must[1]
    assert nested.should


def test_build_filter_site_and_project():
    f = build_filter({"site_id": "lot:2363352", "project_id": "4405"})
    assert f is not None
    assert len(f.must) == 2
