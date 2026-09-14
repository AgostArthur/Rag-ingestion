"""Façade publique de recherche RAG (embeddings + Qdrant).

Le chatbot et la CLI `query` passent par ici : pas d'accès direct à
`embed` / `qdrant_store` depuis l'extérieur du package d'ingestion.

Si la question (ou le filtre) contient un n° de projet, une branche mot-clé
relâche le filtre `entities` exact et cherche aussi dans le texte du chunk —
cas LangExtract manqué, ou LLM qui oublie `entities=`.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.embed import embed_query
from rag_ingestion.normalize import is_project_token, project_tokens_in_text
from rag_ingestion.qdrant_store import build_keyword_filter, search_similar

logger = logging.getLogger(__name__)


def _csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def project_tokens_for_search(query: str, filters: dict[str, str] | None) -> list[str]:
    """N° de projet issus de la question et des filtres payload."""
    tokens = project_tokens_in_text(query)
    if filters:
        for key in ("entities", "project_id"):
            for value in _csv_values(filters.get(key)):
                if is_project_token(value):
                    tokens.append(value)
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def merge_hits(*groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Déduplique par chunk_id ; garde le meilleur score ; trie décroissant."""
    best: dict[str, dict[str, Any]] = {}
    for group in groups:
        for hit in group:
            key = str(
                hit.get("chunk_id")
                or f"{hit.get('document_id')}:{hit.get('chunk_index')}"
            )
            previous = best.get(key)
            if previous is None or float(hit.get("score") or 0) > float(
                previous.get("score") or 0
            ):
                best[key] = hit
    ranked = sorted(
        best.values(),
        key=lambda hit: float(hit.get("score") or 0),
        reverse=True,
    )
    return ranked[:limit]


def search(
    query: str,
    *,
    limit: int = 5,
    filters: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Embed la question puis retourne les chunks les plus proches.

    Branche hybride : si un n° de projet est présent, interroge aussi Qdrant
    avec un filtre OR (entities / project_id / texte).

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
    dense = search_similar(
        s.qdrant_collection,
        vector,
        limit=limit,
        filters=filters,
        settings=s,
    )
    tokens = project_tokens_for_search(query, filters)
    if not tokens:
        return dense
    keyword_filter = build_keyword_filter(tokens, filters)
    try:
        keyword = search_similar(
            s.qdrant_collection,
            vector,
            limit=limit,
            query_filter=keyword_filter,
            settings=s,
        )
    except Exception:
        logger.exception("Keyword / text match search failed — dense hits only.")
        return dense
    return merge_hits(keyword, dense, limit=limit)
