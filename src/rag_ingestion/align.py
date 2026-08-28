"""Aligne les extractions LangExtract (char_interval) sur les chunks."""

from __future__ import annotations

import re

from rag_ingestion.models import Chunk, ChunkPayload, GroundedExtraction

_DOC_LEVEL = frozenset({"doc_type", "title"})
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_DIGITS_RE = re.compile(r"\d{3,8}")


def _overlaps(ext: GroundedExtraction, chunk: Chunk) -> bool:
    """True si l'intervalle de l'extraction recouvre celui du chunk."""
    if not ext.grounded:
        return False
    assert ext.start is not None and ext.end is not None
    return min(chunk.end, ext.end) > max(chunk.start, ext.start)


def _label(ext: GroundedExtraction) -> str:
    """Valeur d'étiquette : `attributes.normalized` si présent, sinon le texte."""
    normalized = (ext.attributes or {}).get("normalized")
    if isinstance(normalized, str) and normalized.strip():
        return normalized.strip()
    return ext.extraction_text.strip()


def _entity_type(ext: GroundedExtraction) -> str:
    raw = (ext.attributes or {}).get("type")
    return raw.strip().lower() if isinstance(raw, str) else ""


def project_id_label(ext: GroundedExtraction) -> str | None:
    """N° de projet indexable (`4405`), ou None si ce n'est pas un project_id."""
    if ext.extraction_class != "entity":
        return None
    if _entity_type(ext) == "project_id":
        normalized = (ext.attributes or {}).get("normalized")
        if isinstance(normalized, str) and normalized.strip().isdigit():
            return normalized.strip()
        text = ext.extraction_text.strip()
        if text.isdigit():
            return text
        found = _DIGITS_RE.findall(text)
        if len(found) == 1 and not _YEAR_RE.match(found[0]):
            return found[0]
        return None
    # Repli : entité purement numérique (pas une année), même sans type=.
    text = _label(ext)
    if text.isdigit() and 3 <= len(text) <= 8 and not _YEAR_RE.match(text):
        return text
    return None


def _doc_project_ids(extractions: list[GroundedExtraction]) -> list[str]:
    """N° de projet au niveau document (recopiés sur tous les chunks)."""
    out: list[str] = []
    seen: set[str] = set()
    for ext in extractions:
        label = project_id_label(ext)
        if not label:
            continue
        key = label.lower()
        if key not in seen:
            seen.add(key)
            out.append(label)
    return out


def align_extractions(
    chunks: list[Chunk],
    extractions: list[GroundedExtraction],
    *,
    source_path: str,
    parse_quality: str,
) -> list[ChunkPayload]:
    """Rattache à chaque chunk uniquement les extractions dont le span recouvre.

    `doc_type` et les n° de projet (`entity` type=project_id) sont recopiés sur
    tous les chunks (niveau document), y compris s'ils ne sont pas ancrés.
    Les autres entités non ancrées ne sont pas copiées.

    Args:
        chunks: Morceaux avec offsets dans le Markdown.
        extractions: Sortie LangExtract (avec `char_interval` si possible).
        source_path: Chemin du PDF, recopié dans le payload.
        parse_quality: Qualité LiteParse (`ok`, `ocr_partial`, `ocr_heavy`).

    Returns:
        Un `ChunkPayload` par chunk, prêt pour l'upsert Qdrant.
    """
    doc_types = [
        _label(e).lower()
        for e in extractions
        if e.extraction_class == "doc_type" and _label(e)
    ]
    doc_type = doc_types[0] if doc_types else None
    project_ids = _doc_project_ids(extractions)

    payloads: list[ChunkPayload] = []
    for chunk in chunks:
        entities: list[str] = []
        dates: list[str] = []
        topics: list[str] = []
        ids: list[str] = []
        seen_ent: set[str] = {p.lower() for p in project_ids}
        seen_date: set[str] = set()
        seen_topic: set[str] = set()
        entities.extend(project_ids)

        for ext in extractions:
            if ext.extraction_class in _DOC_LEVEL:
                if ext.extraction_class == "doc_type":
                    ids.append(ext.extraction_id)
                continue
            if project_id_label(ext):
                ids.append(ext.extraction_id)
                continue
            if ext.grounded and not _overlaps(ext, chunk):
                continue
            if not ext.grounded:
                continue
            ids.append(ext.extraction_id)
            label = _label(ext)
            if not label:
                continue
            if ext.extraction_class == "entity":
                key = label.lower()
                if key not in seen_ent:
                    seen_ent.add(key)
                    entities.append(label)
            elif ext.extraction_class == "date":
                key = label.lower()
                if key not in seen_date:
                    seen_date.add(key)
                    dates.append(label)
            elif ext.extraction_class == "topic":
                key = label.lower()
                if key not in seen_topic:
                    seen_topic.add(key)
                    topics.append(label)

        payloads.append(
            ChunkPayload(
                chunk=chunk,
                doc_type=doc_type,
                entities=entities,
                dates=dates,
                topics=topics,
                extraction_ids=ids,
                parse_quality=parse_quality,
                source_path=source_path,
            )
        )
    return payloads
