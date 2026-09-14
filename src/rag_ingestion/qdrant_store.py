"""Collection Qdrant `chunks` : indexes payload, upsert, recherche filtrée."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.models import ChunkPayload

logger = logging.getLogger(__name__)

_INDEXED_FIELDS: tuple[tuple[str, PayloadSchemaType], ...] = (
    ("document_id", PayloadSchemaType.KEYWORD),
    ("source_path", PayloadSchemaType.KEYWORD),
    ("page", PayloadSchemaType.INTEGER),
    ("heading_path", PayloadSchemaType.KEYWORD),
    ("doc_type", PayloadSchemaType.KEYWORD),
    ("entities", PayloadSchemaType.KEYWORD),
    ("dates", PayloadSchemaType.KEYWORD),
    ("parse_quality", PayloadSchemaType.KEYWORD),
    ("site_id", PayloadSchemaType.KEYWORD),
    ("project_id", PayloadSchemaType.KEYWORD),
    ("text", PayloadSchemaType.TEXT),
)

_FILTER_KEYS = frozenset(
    {
        "document_id",
        "source_path",
        "doc_type",
        "entities",
        "dates",
        "heading_path",
        "parse_quality",
        "page",
        "site_id",
        "project_id",
    }
)

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def ping_qdrant(settings: Settings | None = None) -> tuple[bool, str]:
    """Health-check HTTP Qdrant (liste des collections)."""
    s = settings or load_settings()
    try:
        client = _client(s)
        names = [c.name for c in client.get_collections().collections]
        return True, ",".join(names) if names else "ok"
    except Exception as exc:
        return False, str(exc)


def _client(settings: Settings) -> QdrantClient:
    """Client HTTP vers `settings.qdrant_url`."""
    return QdrantClient(url=settings.qdrant_url)


def _point_id(document_id: str, chunk_index: int) -> str:
    """UUID v5 déterministe pour ré-upsert le même chunk sans doublon d'id."""
    return str(uuid.uuid5(_NAMESPACE, f"{document_id}:{chunk_index}"))


def ensure_collection(
    collection_name: str,
    vector_size: int,
    settings: Settings | None = None,
) -> None:
    """Crée la collection si besoin et pose les indexes payload filtrables.

    Vérifie que la dimension existante correspond à `vector_size`.

    Args:
        collection_name: Nom Qdrant (`QDRANT_COLLECTION`).
        vector_size: Dimension dense (ex. 1024).
        settings: Config ; `.env` si omis.

    Raises:
        ValueError: Collection déjà là avec une autre dimension.
    """
    s = settings or load_settings()
    client = _client(s)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        info = client.get_collection(collection_name)
        params = info.config.params
        vectors = params.vectors
        size = vectors.size if isinstance(vectors, VectorParams) else None
        if isinstance(vectors, dict):
            if len(vectors) == 1:
                size = next(iter(vectors.values())).size
        if size is not None and size != vector_size:
            raise ValueError(
                f"Collection « {collection_name} » has dimension {size}, "
                f"expected {vector_size}."
            )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection « %s » created (dimension %s).", collection_name, vector_size)

    for field_name, schema in _INDEXED_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema,
            wait=True,
        )


def delete_by_document_id(
    collection_name: str,
    document_id: str,
    settings: Settings | None = None,
) -> None:
    """Supprime tous les points d'un document (ré-ingest idempotent).

    Args:
        collection_name: Collection cible.
        document_id: SHA-256 du fichier.
        settings: Config ; `.env` si omis.
    """
    s = settings or load_settings()
    client = _client(s)
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
        wait=True,
    )
    logger.info("  Previous points for this document deleted (if re-ingest).")


def upsert_payloads(
    collection_name: str,
    payloads: list[ChunkPayload],
    vectors: list[list[float]],
    settings: Settings | None = None,
) -> int:
    """Écrit un point par chunk : vecteur + payload plat (pas le JSON complet).

    Args:
        collection_name: Collection cible.
        payloads: Chunks étiquetés (même ordre que `vectors`).
        vectors: Embeddings des textes de chunks.
        settings: Config ; `.env` si omis.

    Returns:
        Nombre de points upsertés.

    Raises:
        ValueError: Longueurs `payloads` / `vectors` différentes.
    """
    if len(payloads) != len(vectors):
        raise ValueError("payloads and vectors must have the same length")
    s = settings or load_settings()
    client = _client(s)
    points: list[PointStruct] = []
    for item, vec in zip(payloads, vectors, strict=True):
        chunk = item.chunk
        payload: dict[str, Any] = {
            "document_id": chunk.document_id,
            "source_path": item.source_path,
            "page": chunk.page,
            "heading_path": chunk.heading_path,
            "doc_type": item.doc_type,
            "entities": item.entities,
            "dates": item.dates,
            "topics": item.topics,
            "extraction_ids": item.extraction_ids,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "chunk_id": chunk.chunk_id,
            "parse_quality": item.parse_quality,
            "char_start": chunk.start,
            "char_end": chunk.end,
            "site_id": item.site_id,
            "project_id": item.project_id,
        }
        points.append(
            PointStruct(
                id=_point_id(chunk.document_id, chunk.chunk_index),
                vector=vec,
                payload=payload,
            )
        )
    batch = 64
    for i in range(0, len(points), batch):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch],
            wait=True,
        )
    return len(points)


def build_filter(filters: dict[str, str]) -> Filter | None:
    """Construit un filtre Qdrant à partir de `clé=valeur` (virgules = MatchAny).

    Clés autorisées : document_id, source_path, doc_type, entities, dates,
    heading_path, parse_quality, page, site_id, project_id.

    Args:
        filters: Mapping clé → valeur brute (page en entier, listes en CSV).

    Returns:
        `Filter` Qdrant, ou None si rien à appliquer.

    Raises:
        ValueError: Clé inconnue.
    """
    must: list[FieldCondition] = []
    for key, raw in filters.items():
        if not raw:
            continue
        if key not in _FILTER_KEYS:
            raise ValueError(f"Unknown filter: {key}")
        if key == "page":
            must.append(
                FieldCondition(key="page", match=MatchValue(value=int(raw)))
            )
            continue
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if not values:
            continue
        if len(values) == 1:
            must.append(FieldCondition(key=key, match=MatchValue(value=values[0])))
        else:
            must.append(FieldCondition(key=key, match=MatchAny(any=values)))
    if not must:
        return None
    return Filter(must=must)


def search_similar(
    collection_name: str,
    query_vector: list[float],
    *,
    limit: int = 5,
    filters: dict[str, str] | None = None,
    query_filter: Filter | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Recherche dense, éventuellement restreinte par le payload.

    Args:
        collection_name: Collection à interroger.
        query_vector: Embedding de la question.
        limit: Nombre max de hits.
        filters: Filtres optionnels (`doc_type`, `entities`, …).
        query_filter: Filtre Qdrant déjà construit (prioritaire sur `filters`).
        settings: Config ; `.env` si omis.

    Returns:
        Hits (score, texte, source, étiquettes), du plus proche au plus loin.
    """
    s = settings or load_settings()
    client = _client(s)
    built = query_filter if query_filter is not None else build_filter(filters or {})
    resp = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=built,
        limit=limit,
        with_payload=True,
    )
    results: list[dict[str, Any]] = []
    for hit in resp.points:
        pl = dict(hit.payload) if isinstance(hit.payload, dict) else {}
        results.append(
            {
                "score": hit.score,
                "text": pl.get("text", ""),
                "source_path": pl.get("source_path", ""),
                "document_id": pl.get("document_id"),
                "chunk_index": pl.get("chunk_index"),
                "page": pl.get("page"),
                "heading_path": pl.get("heading_path"),
                "doc_type": pl.get("doc_type"),
                "entities": pl.get("entities") or [],
                "dates": pl.get("dates") or [],
                "site_id": pl.get("site_id"),
                "project_id": pl.get("project_id"),
                "chunk_id": pl.get("chunk_id"),
            }
        )
    return results


def keyword_should_conditions(tokens: list[str]) -> list[FieldCondition]:
    """Conditions OR : n° dans `entities` / `project_id` / texte du chunk."""
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    should: list[FieldCondition] = [
        FieldCondition(key="entities", match=MatchAny(any=tokens)),
        FieldCondition(key="project_id", match=MatchAny(any=tokens)),
    ]
    for token in tokens:
        should.append(FieldCondition(key="text", match=MatchText(text=token)))
    return should


def build_keyword_filter(
    tokens: list[str],
    filters: dict[str, str] | None = None,
) -> Filter | None:
    """Filtre hybride : conserve doc_type / site / etc., relâche entities.

    Le n° de projet est cherché dans le payload ET dans le Markdown (MatchText),
    pour le cas où LangExtract a manqué `project_id`.
    """
    base = {
        k: v
        for k, v in (filters or {}).items()
        if k not in {"entities", "project_id"}
    }
    must_filter = build_filter(base)
    should = keyword_should_conditions(tokens)
    must: list[Any] = list(must_filter.must) if must_filter is not None else []
    if should:
        # Nested should is required (top-level `should` would only boost scores).
        must.append(Filter(should=should))
    if not must:
        return None
    return Filter(must=must)
