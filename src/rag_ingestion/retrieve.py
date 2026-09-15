"""Façade publique de recherche RAG (embeddings + Qdrant).

Le chatbot et la CLI `query` passent par ici : pas d'accès direct à
`embed` / `qdrant_store` depuis l'extérieur du package d'ingestion.

Si la question (ou le filtre) contient un n° de projet, une branche mot-clé
relâche le filtre `entities` exact et cherche aussi dans le texte du chunk —
cas LangExtract manqué, ou LLM qui oublie `entities=`.

`hybrid_text=True` (chatbot) étend cette branche aux termes métier
(« client », « firme », …) pour les questions d'identité.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.embed import embed_query
from rag_ingestion.normalize import is_project_token, project_tokens_in_text
from rag_ingestion.qdrant_store import build_keyword_filter, search_similar

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]{4,}")

# Mots trop fréquents : un MatchText dessus noie la recherche.
_STOPWORDS = frozenset(
    {
        "alors",
        "avec",
        "cette",
        "dans",
        "dont",
        "est",
        "les",
        "name",
        "nom",
        "pour",
        "quel",
        "quelle",
        "quelles",
        "quels",
        "sont",
        "that",
        "the",
        "this",
        "une",
        "what",
    }
)

# Termes d'identité / ÉES : même 4 lettres, on les garde.
_DOMAIN_TERMS = frozenset(
    {
        "adresse",
        "address",
        "cadastral",
        "client",
        "contaminant",
        "contamination",
        "firm",
        "firme",
        "forage",
        "lot",
        "project",
        "projet",
        "rapport",
    }
)


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


def lexical_tokens_for_search(query: str) -> list[str]:
    """Termes métier + mots significatifs (hors stopwords) pour MatchText."""
    if not query or not query.strip():
        return []
    lowered = query.lower()
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        key = raw.lower()
        if key in seen:
            return
        seen.add(key)
        tokens.append(raw)

    for term in _DOMAIN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered, flags=re.IGNORECASE):
            _add(term)
    for raw in _TOKEN_RE.findall(query):
        key = raw.lower()
        if key in _STOPWORDS or key in _DOMAIN_TERMS:
            continue
        if len(key) >= 5 or any(ch.isdigit() for ch in key):
            _add(raw)
    return tokens


def _apply_score_threshold(
    hits: list[dict[str, Any]], *, threshold: float, limit: int
) -> list[dict[str, Any]]:
    if threshold > 0:
        hits = [h for h in hits if float(h.get("score") or 0) >= threshold]
    return hits[:limit]


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
    prefetch: int = 0,
    score_threshold: float = 0.0,
    hybrid_text: bool = False,
) -> list[dict[str, Any]]:
    """Embed la question puis retourne les chunks les plus proches.

    Branche hybride : si un n° de projet est présent (ou `hybrid_text`),
    interroge aussi Qdrant avec un filtre OR (entities / project_id / texte).

    Args:
        query: Question ou phrase de recherche.
        limit: Nombre max de hits renvoyés.
        filters: Filtres payload (`doc_type`, `entities`, `document_id`, …).
        settings: Config ; `.env` si omis.
        prefetch: Si > `limit`, Qdrant ramène plus de candidats avant coupe.
        score_threshold: Score cosine minimum (0 = pas de filtre).
        hybrid_text: Ajoute les termes métier / mots longs à la branche mot-clé.

    Returns:
        Hits (score, texte, source, étiquettes), du plus proche au plus loin.
    """
    s = settings or load_settings()
    fetch = max(limit, int(prefetch) if prefetch else 0)
    threshold = max(0.0, float(score_threshold or 0))
    vector = embed_query(query, settings=s)
    dense = search_similar(
        s.qdrant_collection,
        vector,
        limit=fetch,
        filters=filters,
        settings=s,
    )
    tokens = project_tokens_for_search(query, filters)
    if hybrid_text:
        for token in lexical_tokens_for_search(query):
            if token not in tokens and token.lower() not in {t.lower() for t in tokens}:
                tokens.append(token)
    if not tokens:
        return _apply_score_threshold(dense, threshold=threshold, limit=limit)
    keyword_filter = build_keyword_filter(tokens, filters)
    try:
        keyword = search_similar(
            s.qdrant_collection,
            vector,
            limit=fetch,
            query_filter=keyword_filter,
            settings=s,
        )
    except Exception:
        logger.exception("Keyword / text match search failed — dense hits only.")
        return _apply_score_threshold(dense, threshold=threshold, limit=limit)
    merged = merge_hits(keyword, dense, limit=fetch)
    return _apply_score_threshold(merged, threshold=threshold, limit=limit)
