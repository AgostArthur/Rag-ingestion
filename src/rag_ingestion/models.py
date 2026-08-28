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
