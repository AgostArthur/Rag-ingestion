"""Agrège les extractions LangExtract au niveau document (fiche métier)."""

from __future__ import annotations

from rag_ingestion.align import _doc_project_ids, _label
from rag_ingestion.models import (
    ChunkPayload,
    DocumentMeta,
    GroundedExtraction,
    TypedEvent,
)
from rag_ingestion.normalize import (
    infer_date_role,
    iso_date,
    normalize_address,
    normalize_lot,
    site_id_for,
)


def _entity_type(ext: GroundedExtraction) -> str:
    raw = (ext.attributes or {}).get("type")
    return raw.strip().lower() if isinstance(raw, str) else ""


def _location_type(ext: GroundedExtraction) -> str:
    if ext.extraction_class != "location":
        return ""
    return _entity_type(ext)


def _first_label(
    extractions: list[GroundedExtraction],
    *,
    extraction_class: str,
    entity_type: str | None = None,
) -> str | None:
    for ext in extractions:
        if ext.extraction_class != extraction_class:
            continue
        if entity_type is not None and _entity_type(ext) != entity_type:
            continue
        label = _label(ext)
        if label:
            return label
    return None


def _collect_events(extractions: list[GroundedExtraction]) -> list[TypedEvent]:
    events: list[TypedEvent] = []
    seen: set[tuple[str, str]] = set()
    for ext in extractions:
        if ext.extraction_class != "date":
            continue
        label = _label(ext)
        attrs = ext.attributes or {}
        normalized = attrs.get("normalized") if isinstance(attrs.get("normalized"), str) else None
        parsed = iso_date(normalized) or iso_date(label)
        if not parsed:
            continue
        role_raw = attrs.get("role")
        role = infer_date_role(
            role_raw if isinstance(role_raw, str) else None,
            ext.extraction_text or label,
        )
        key = (parsed, role)
        if key in seen:
            continue
        seen.add(key)
        events.append(TypedEvent(iso_date=parsed, role=role, label=label or parsed))
    return events


def build_document_meta(
    extractions: list[GroundedExtraction],
    *,
    document_id: str,
    source_path: str,
    parse_quality: str,
) -> DocumentMeta:
    """Fiche document : projet, titre, firme, client, lot/adresse, dates typées.

    `location` et `title` sont des propriétés du document (comme `project_id`),
    pas des étiquettes de chunk. Le `site_id` privilégie le lot cadastral.
    """
    project_ids = _doc_project_ids(extractions)
    doc_types = [
        _label(e).lower()
        for e in extractions
        if e.extraction_class == "doc_type" and _label(e)
    ]
    title = _first_label(extractions, extraction_class="title")
    firm = _first_label(extractions, extraction_class="entity", entity_type="firm")
    client = _first_label(extractions, extraction_class="entity", entity_type="client")

    address: str | None = None
    lot: str | None = None
    city: str | None = None
    for ext in extractions:
        if ext.extraction_class != "location":
            continue
        kind = _location_type(ext)
        label = _label(ext)
        if not label:
            continue
        if kind == "lot" and lot is None:
            lot = normalize_lot(label) or normalize_lot(ext.extraction_text)
        elif kind in {"city", "town", "village"} and city is None:
            city = normalize_address(label)
        elif kind in {"address", "", "other"} and address is None:
            address = normalize_address(label)
        elif kind == "region" and city is None:
            city = normalize_address(label)

    if lot is None:
        for ext in extractions:
            if ext.extraction_class != "location":
                continue
            guessed = normalize_lot(_label(ext) or ext.extraction_text)
            if guessed:
                lot = guessed
                break

    events = _collect_events(extractions)
    report_date = next((e.iso_date for e in events if e.role == "report"), None)
    if report_date is None and events:
        # Repli : première date ISO si aucun rôle `report` (few-shots incomplets).
        report_date = events[0].iso_date

    site_id = site_id_for(lot=lot, address=address)
    return DocumentMeta(
        document_id=document_id,
        source_path=source_path,
        parse_quality=parse_quality,
        project_ids=project_ids,
        title=title,
        doc_type=doc_types[0] if doc_types else None,
        firm=firm,
        client=client,
        address=address,
        lot_cadastral=lot,
        city=city,
        site_id=site_id,
        report_date=report_date,
        events=events,
    )


def stamp_payloads(payloads: list[ChunkPayload], meta: DocumentMeta) -> list[ChunkPayload]:
    """Recopie `site_id` / `project_id` sur chaque chunk (filtre Qdrant)."""
    for item in payloads:
        item.site_id = meta.site_id
        item.project_id = meta.project_id
    return payloads
