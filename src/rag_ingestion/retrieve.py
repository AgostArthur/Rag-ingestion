"""Façade publique de recherche RAG (embeddings + Qdrant).

Le chatbot et la CLI `query` passent par ici : pas d'accès direct à
`embed` / `qdrant_store` depuis l'extérieur du package d'ingestion.
"""

from __future__ import annotations

from typing import Any

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.embed import embed_query
from rag_ingestion.qdrant_store import search_similar


def search(
    query: str,
    *,
    limit: int = 5,
    filters: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Embed la question puis retourne les chunks les plus proches.

    Args:
        query: Question ou phrase de recherche.
        limit: Nombre max de hits.
        filters: Filtres payload (`doc_type`, `entities`, `document_id`, …).
        settings: Config ; `.env` si omis.

    Returns:
        Hits (score, texte, source, étiquettes), du plus proche au plus loin.
    """
    s = settings or load_settings()
    vector = embed_query(query, settings=s)
    return search_similar(
        s.qdrant_collection,
        vector,
        limit=limit,
        filters=filters,
        settings=s,
    )
