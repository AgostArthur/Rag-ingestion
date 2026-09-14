from pathlib import Path
from unittest.mock import MagicMock

from rag_ingestion.config import Settings
from rag_ingestion.retrieve import lexical_tokens_for_search, search


def test_lexical_tokens_keep_client_and_drop_stopwords():
    tokens = lexical_tokens_for_search("Quel est le nom du client")
    assert "client" in [t.lower() for t in tokens]
    assert "nom" not in [t.lower() for t in tokens]
    assert "quel" not in [t.lower() for t in tokens]


def _settings(**overrides: object) -> Settings:
    base = dict(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks_test",
        embed_model="BAAI/bge-m3",
        embed_dim=512,
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
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_search_embeds_then_queries_qdrant(monkeypatch):
    captured: dict = {}

    def fake_embed(text, settings=None):
        captured["query"] = text
        captured["embed_settings"] = settings
        return [0.1, 0.2, 0.3]

    def fake_similar(collection, vector, *, limit=5, filters=None, settings=None, **kwargs):
        captured["collection"] = collection
        captured["vector"] = vector
        captured["limit"] = limit
        captured["filters"] = filters
        captured["search_settings"] = settings
        return [{"score": 0.91, "text": "clause de responsabilité", "page": 3}]

    monkeypatch.setattr("rag_ingestion.retrieve.embed_query", fake_embed)
    monkeypatch.setattr("rag_ingestion.retrieve.search_similar", fake_similar)

    settings = _settings()
    hits = search(
        "clause de responsabilité",
        limit=3,
        filters={"doc_type": "rapport"},
        settings=settings,
    )

    assert hits == [{"score": 0.91, "text": "clause de responsabilité", "page": 3}]
    assert captured["query"] == "clause de responsabilité"
    assert captured["collection"] == "chunks_test"
    assert captured["vector"] == [0.1, 0.2, 0.3]
    assert captured["limit"] == 3
    assert captured["filters"] == {"doc_type": "rapport"}
    assert captured["embed_settings"] is settings
    assert captured["search_settings"] is settings


def test_search_default_limit_and_no_filters(monkeypatch):
    fake_similar = MagicMock(return_value=[])
    monkeypatch.setattr("rag_ingestion.retrieve.embed_query", lambda *a, **k: [1.0])
    monkeypatch.setattr("rag_ingestion.retrieve.search_similar", fake_similar)

    search("q", settings=_settings())

    _, kwargs = fake_similar.call_args
    assert kwargs["limit"] == 5
    assert kwargs["filters"] is None


def test_search_prefetch_queries_more_then_returns_limit(monkeypatch):
    captured: dict = {}

    def fake_similar(collection, vector, *, limit=5, filters=None, settings=None, **kwargs):
        captured["limit"] = limit
        return [
            {"chunk_id": f"c{i}", "score": 1.0 - i * 0.01, "text": str(i)}
            for i in range(12)
        ]

    monkeypatch.setattr("rag_ingestion.retrieve.embed_query", lambda *a, **k: [1.0])
    monkeypatch.setattr("rag_ingestion.retrieve.search_similar", fake_similar)

    hits = search("q", limit=3, prefetch=12, settings=_settings())
    assert captured["limit"] == 12
    assert len(hits) == 3


def test_search_score_threshold_drops_weak_hits(monkeypatch):
    def fake_similar(collection, vector, *, limit=5, filters=None, settings=None, **kwargs):
        return [
            {"chunk_id": "hi", "score": 0.91, "text": "ok"},
            {"chunk_id": "lo", "score": 0.12, "text": "noise"},
        ]

    monkeypatch.setattr("rag_ingestion.retrieve.embed_query", lambda *a, **k: [1.0])
    monkeypatch.setattr("rag_ingestion.retrieve.search_similar", fake_similar)

    hits = search("q", limit=5, score_threshold=0.5, settings=_settings())
    assert [h["chunk_id"] for h in hits] == ["hi"]
