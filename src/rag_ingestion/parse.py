"""LiteParse : inspect (is_complex) puis parse Markdown. OCR = plugin, pas une étape amont."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.headers import strip_running_headers
from rag_ingestion.models import InspectReport, PageSpan, ParsedDocument

logger = logging.getLogger(__name__)


def document_id_from_bytes(data: bytes) -> str:
    """Identifiant stable : SHA-256 du contenu binaire (pas du chemin).

    Args:
        data: Octets du fichier source.

    Returns:
        Hex digest SHA-256, utilisé comme `document_id`.
    """
    return hashlib.sha256(data).hexdigest()


def _quality_from_ratio(ratio: float, heavy: float) -> str:
    """Classe la qualité de parse selon la part de pages `needs_ocr`."""
    if ratio <= 0:
        return "ok"
    if ratio >= heavy:
        return "ocr_heavy"
    return "ocr_partial"


def inspect_pdf(path: Path, settings: Settings | None = None) -> InspectReport:
    """Contrôle LiteParse (`is_complex`) : quelles pages ont besoin d'OCR.

    N'extrait pas le texte. Sert de quality gate avant le parse Markdown.

    Args:
        path: Chemin du PDF.
        settings: Config ; `.env` si omis.

    Returns:
        Rapport page par page (`needs_ocr`, raisons) et `parse_quality`.
    """
    from liteparse import LiteParse

    s = settings or load_settings()
    parser = LiteParse(
        ocr_language=s.ocr_language,
        ocr_server_url=s.ocr_server_url,
        quiet=True,
    )
    stats = parser.is_complex(str(path))
    pages: list[PageSpan] = []
    n_ocr = 0
    ocr_details: list[str] = []
    for st in stats:
        needs = bool(st.needs_ocr)
        reasons = tuple(st.reasons or ())
        if needs:
            n_ocr += 1
            ocr_details.append(
                f"page {st.page_number} ({', '.join(reasons) or 'scanned'})"
            )
        pages.append(
            PageSpan(
                page_num=int(st.page_number),
                start=0,
                end=0,
                needs_ocr=needs,
                reasons=reasons,
            )
        )
    n_pages = len(pages)
    ratio = n_ocr / n_pages if n_pages else 0.0
    quality = _quality_from_ratio(ratio, s.ocr_heavy_ratio)
    preview = ocr_details[:12]
    for line in preview:
        logger.info("  OCR needed: %s", line)
    if len(ocr_details) > 12:
        logger.info("  … and %s more page(s) needing OCR", len(ocr_details) - 12)
    logger.info(
        "  %s page(s) · %s need OCR (%.0f%%) · quality=%s",
        n_pages,
        n_ocr,
        ratio * 100,
        quality,
    )
    return InspectReport(
        n_pages=n_pages,
        n_needs_ocr=n_ocr,
        pages=tuple(pages),
        parse_quality=quality,
    )


def parse_pdf(path: Path, settings: Settings | None = None) -> ParsedDocument:
    """Transforme un PDF en Markdown (OCR LiteParse si une page en a besoin).

    Inspecte d'abord, parse ensuite, écrit `data/parsed/{document_id}.md`.

    Args:
        path: Chemin du PDF.
        settings: Config ; `.env` si omis.

    Returns:
        Document parsé (Markdown, spans de pages, inspect, chemin du `.md`).
    """
    from liteparse import LiteParse

    s = settings or load_settings()
    path = path.resolve()
    data = path.read_bytes()
    doc_id = document_id_from_bytes(data)
    logger.info("  OCR check for « %s »…", path.name)
    inspect = inspect_pdf(path, settings=s)

    if inspect.n_pages == 0:
        logger.warning("  PDF has no detected pages.")
    if inspect.parse_quality == "ocr_heavy":
        logger.warning(
            "  Many scanned pages (%s/%s): OCR will slow down parsing.",
            inspect.n_needs_ocr,
            inspect.n_pages,
        )

    ocr_note = "built-in Tesseract" if not s.ocr_server_url else s.ocr_server_url
    logger.info("  Markdown conversion (OCR %s, language=%s)…", ocr_note, s.ocr_language)

    kwargs: dict = {
        "output_format": "markdown",
        "ocr_enabled": True,
        "ocr_language": s.ocr_language,
        "image_mode": "placeholder",
        "extract_links": True,
        "include_complexity": True,
        # On retire le bandeau nous-mêmes après la page 1 (identité du rapport).
        "keep_headers_footers": True,
        "quiet": True,
    }
    if s.ocr_server_url:
        kwargs["ocr_server_url"] = s.ocr_server_url
    parser = LiteParse(**kwargs)
    result = parser.parse(str(path))

    parts: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    inspect_by_page = {p.page_num: p for p in inspect.pages}
    raw_pages = [(page.markdown or page.text or "").strip() for page in result.pages]
    page_texts, header_stats = strip_running_headers(raw_pages)
    if header_stats.lines_removed:
        logger.info(
            "  Running headers dropped after page 1: %s page(s), %s line(s)",
            header_stats.pages_stripped,
            header_stats.lines_removed,
        )

    for i, page in enumerate(result.pages):
        md = page_texts[i] if i < len(page_texts) else ""
        start = offset
        parts.append(md)
        offset += len(md)
        if i < len(result.pages) - 1:
            parts.append("\n\n")
            offset += 2
        end = offset
        meta = inspect_by_page.get(page.page_num)
        needs = bool(meta.needs_ocr) if meta else False
        reasons = meta.reasons if meta else ()
        if page.complexity is not None:
            needs = bool(page.complexity.needs_ocr)
            reasons = tuple(page.complexity.reasons or ())
        spans.append(
            PageSpan(
                page_num=int(page.page_num),
                start=start,
                end=end,
                needs_ocr=needs,
                reasons=reasons,
            )
        )

    markdown = "".join(parts)

    s.parsed_dir.mkdir(parents=True, exist_ok=True)
    md_path = s.parsed_dir / f"{doc_id}.md"
    md_path.write_text(markdown, encoding="utf-8")
    logger.info(
        "  Markdown: %s character(s), %s page(s) → %s",
        len(markdown),
        len(spans),
        md_path,
    )
    return ParsedDocument(
        document_id=doc_id,
        source_path=str(path),
        content_sha256=doc_id,
        markdown=markdown,
        page_spans=spans,
        inspect=inspect,
        markdown_path=str(md_path),
    )


def page_for_span(page_spans: list[PageSpan], start: int, end: int) -> int | None:
    """Page principale d'un intervalle `[start, end)` dans le Markdown concaténé.

    Args:
        page_spans: Offsets caractère par numéro de page.
        start: Début (inclus) dans le Markdown.
        end: Fin (exclue) dans le Markdown.

    Returns:
        Numéro de page qui recouvre le plus le span, ou None si liste vide.
    """
    best_page: int | None = None
    best_overlap = 0
    for span in page_spans:
        overlap = min(end, span.end) - max(start, span.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = span.page_num
    if best_page is not None:
        return best_page
    for span in page_spans:
        if span.start <= start < span.end:
            return span.page_num
    return page_spans[0].page_num if page_spans else None
