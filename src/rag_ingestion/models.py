"""Structures de données du pipeline (parse, chunks, extractions, ingest)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageSpan:
    """Intervalle de caractères d'une page dans le Markdown concaténé."""

    page_num: int
    start: int
    end: int
    needs_ocr: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class InspectReport:
    """Verdict LiteParse `is_complex` : pages, besoin d'OCR, qualité globale."""

    n_pages: int
    n_needs_ocr: int
    pages: tuple[PageSpan, ...]
    parse_quality: str

    @property
    def ocr_ratio(self) -> float:
        """Fraction de pages marquées `needs_ocr` (0.0 si aucune page)."""
        if self.n_pages == 0:
            return 0.0
        return self.n_needs_ocr / self.n_pages


@dataclass
class ParsedDocument:
    """PDF mis au propre : Markdown, offsets de pages, rapport d'inspect."""

    document_id: str
    source_path: str
    content_sha256: str
    markdown: str
    page_spans: list[PageSpan]
    inspect: InspectReport
    markdown_path: str | None = None


@dataclass(frozen=True)
class Chunk:
    """Morceau de Markdown : `text == markdown[start:end]`."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start: int
    end: int
    heading_path: str
    page: int | None = None


@dataclass(frozen=True)
class GroundedExtraction:
    """Entité LangExtract, éventuellement ancrée (`start`/`end`) dans le Markdown."""

    extraction_id: str
    extraction_class: str
    extraction_text: str
    start: int | None
    end: int | None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """True si l'extraction a un intervalle caractère dans le texte source."""
        return self.start is not None and self.end is not None


@dataclass
class ChunkPayload:
    """Chunk + étiquettes plates à indexer dans Qdrant (pas le JSON LangExtract brut)."""

    chunk: Chunk
    doc_type: str | None
    entities: list[str]
    dates: list[str]
    topics: list[str]
    extraction_ids: list[str]
    parse_quality: str
    source_path: str
    site_id: str | None = None
    project_id: str | None = None
    contaminants: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TypedEvent:
    """Date typée pour la timeline (`events`), jamais générée par le chatbot."""

    iso_date: str
    role: str
    label: str


@dataclass
class DocumentMeta:
    """Fiche métier d'un PDF (catalog SQL + éventuellement JSON parent)."""

    document_id: str
    source_path: str
    parse_quality: str
    project_ids: list[str] = field(default_factory=list)
    title: str | None = None
    doc_type: str | None = None
    firm: str | None = None
    client: str | None = None
    address: str | None = None
    lot_cadastral: str | None = None
    city: str | None = None
    site_id: str | None = None
    report_date: str | None = None
    events: list[TypedEvent] = field(default_factory=list)
    contaminants: list[str] = field(default_factory=list)

    @property
    def project_id(self) -> str | None:
        """N° de projet principal : le premier extrait, ou None."""
        return self.project_ids[0] if self.project_ids else None


@dataclass
class IngestResult:
    """Compte-rendu d'une ingestion (succès, skip, avertissements)."""

    document_id: str
    n_chunks: int
    n_extractions: int
    parse_quality: str
    warnings: list[str]
    skipped: bool = False
    skip_reason: str | None = None
    step_seconds: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    site_id: str | None = None
    project_id: str | None = None
    error: str | None = None
    # Profil LangExtract retenu (toujours renseigné, même si skip_extract=True)
    profile_id: str | None = None
    profile_match_source: str | None = None
    # Chemin source du PDF (avant déplacement par l'inbox)
    source_path: str | None = None
    # Métadonnées ingest supplémentaires pour le journal
    n_pages: int | None = None
    n_needs_ocr: int | None = None
    doc_type: str | None = None
