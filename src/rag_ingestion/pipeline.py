"""Orchestration ingest : parse → chunk → LangExtract → embed → Qdrant."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from rag_ingestion.align import align_extractions
from rag_ingestion.chunk import chunk_markdown
from rag_ingestion.config import Settings, load_settings
from rag_ingestion.embed import embed_texts, embedding_dimension
from rag_ingestion.extract import extract_structured
from rag_ingestion.logging_setup import StepTimer
from rag_ingestion.models import GroundedExtraction, IngestResult
from rag_ingestion.parse import parse_pdf
from rag_ingestion.qdrant_store import (
    delete_by_document_id,
    ensure_collection,
    upsert_payloads,
)

logger = logging.getLogger(__name__)


def _write_parent(
    settings: Settings,
    *,
    document_id: str,
    source_path: str,
    inspect: dict,
    n_chunks: int,
    n_extractions: int,
    markdown_path: str | None,
    warnings: list[str],
    step_seconds: dict[str, float],
    total_seconds: float,
) -> None:
    """Écrit la fiche parent JSON (JSON LangExtract brut reste dans extractions/)."""
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document_id,
        "source_path": source_path,
        "content_sha256": document_id,
        "n_pages": inspect.get("n_pages"),
        "n_pages_needs_ocr": inspect.get("n_needs_ocr"),
        "parse_quality": inspect.get("parse_quality"),
        "n_chunks": n_chunks,
        "n_extractions": n_extractions,
        "markdown_path": markdown_path,
        "extraction_jsonl": str(settings.extractions_dir / f"{document_id}.jsonl"),
        "extraction_html": str(settings.extractions_dir / f"{document_id}.html"),
        "warnings": warnings,
        "step_seconds": {k: round(v, 4) for k, v in step_seconds.items()},
        "total_seconds": round(total_seconds, 4),
    }
    path = settings.documents_dir / f"{document_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Document record written: %s", path)


def _finish(
    *,
    document_id: str,
    n_chunks: int,
    n_extractions: int,
    parse_quality: str,
    warnings: list[str],
    timer: StepTimer,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> IngestResult:
    total = timer.total_seconds()
    for line in timer.summary_lines():
        logger.info("%s", line)
    return IngestResult(
        document_id=document_id,
        n_chunks=n_chunks,
        n_extractions=n_extractions,
        parse_quality=parse_quality,
        warnings=warnings,
        skipped=skipped,
        skip_reason=skip_reason,
        step_seconds=dict(timer.steps),
        total_seconds=total,
    )


def ingest_path(
    path: Path,
    *,
    settings: Settings | None = None,
    skip_extract: bool = False,
) -> IngestResult:
    """Ingest un PDF : parse, chunk, extrait, embed, upsert (un point par chunk).

    Quality gates : Markdown vide ou aucun chunk → skip. LangExtract en échec
    → avertissement et poursuite sans métadonnées. Ré-ingest : delete puis upsert.

    Args:
        path: Chemin du PDF.
        settings: Config ; `.env` si omis.
        skip_extract: Si True, n'appelle pas LangExtract.

    Returns:
        Compte-rendu (`n_chunks`, extractions, `parse_quality`, timings).

    Raises:
        FileNotFoundError: Le chemin n'est pas un fichier.
    """
    s = settings or load_settings()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    timer = StepTimer(logger)
    size_mo = path.stat().st_size / (1024 * 1024)
    logger.info("Ingesting « %s » (%.2f MB)", path.name, size_mo)
    logger.info(
        "Qdrant collection « %s » · LangExtract %s · chunks %s chars / overlap %s",
        s.qdrant_collection,
        "disabled" if skip_extract else s.langextract_model,
        s.chunk_size_chars,
        s.chunk_overlap_chars,
    )

    warnings: list[str] = []
    with timer.step(
        "parse",
        "Step 1/6 — Inspect PDF then convert to Markdown…",
    ):
        parsed = parse_pdf(path, settings=s)
    logger.info("  document_id=%s (%s)", parsed.document_id, timer.took("parse"))

    if parsed.inspect.parse_quality == "ocr_heavy":
        warnings.append(
            f"Heavy OCR: {parsed.inspect.n_needs_ocr}/{parsed.inspect.n_pages} pages"
        )

    if not parsed.markdown.strip():
        msg = "Empty markdown after parse — ingest skipped"
        logger.warning("%s", msg)
        return _finish(
            document_id=parsed.document_id,
            n_chunks=0,
            n_extractions=0,
            parse_quality=parsed.inspect.parse_quality,
            warnings=warnings + [msg],
            timer=timer,
            skipped=True,
            skip_reason=msg,
        )

    with timer.step(
        "chunk",
        f"Step 2/6 — Chunking text ({s.chunk_size_chars} chars, "
        f"overlap {s.chunk_overlap_chars})…",
    ):
        chunks = chunk_markdown(
            parsed.markdown,
            document_id=parsed.document_id,
            max_chars=s.chunk_size_chars,
            overlap_chars=s.chunk_overlap_chars,
            page_spans=parsed.page_spans,
        )
        chunks = [c for c in chunks if c.text.strip()]
    logger.info("  %s chunk(s) ready to index (%s).", len(chunks), timer.took("chunk"))

    if not chunks:
        msg = "No non-empty chunks — ingest skipped"
        logger.warning("%s", msg)
        return _finish(
            document_id=parsed.document_id,
            n_chunks=0,
            n_extractions=0,
            parse_quality=parsed.inspect.parse_quality,
            warnings=warnings + [msg],
            timer=timer,
            skipped=True,
            skip_reason=msg,
        )

    extractions: list[GroundedExtraction] = []
    if skip_extract:
        logger.info("Step 3/6 — LangExtract skipped (--skip-extract).")
        timer.steps["langextract"] = 0.0
    else:
        with timer.step(
            "langextract",
            f"Step 3/6 — Structured extraction (Ollama {s.langextract_model})…",
        ):
            try:
                extractions = extract_structured(
                    parsed.markdown,
                    document_id=parsed.document_id,
                    settings=s,
                )
            except Exception as exc:
                warnings.append(f"LangExtract failed: {exc}")
                logger.exception(
                    "LangExtract failed — continuing without metadata (embeddings still run)."
                )
        counts = Counter(e.extraction_class for e in extractions)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        logger.info(
            "  %s extraction(s) (%s) (%s).",
            len(extractions),
            detail,
            timer.took("langextract"),
        )
        if not extractions and not skip_extract:
            warnings.append("0 entities extracted by LangExtract")

    with timer.step(
        "align",
        "Step 4/6 — Attach labels to chunks…",
    ):
        payloads = align_extractions(
            chunks,
            extractions,
            source_path=parsed.source_path,
            parse_quality=parsed.inspect.parse_quality,
        )
    n_labeled = sum(1 for p in payloads if p.entities or p.dates or p.topics)
    logger.info(
        "  %s chunk(s) have at least one local label (%s).",
        n_labeled,
        timer.took("align"),
    )

    dim = embedding_dimension(s)
    with timer.step(
        "embed",
        f"Step 5/6 — Computing embeddings ({len(payloads)} texts, dim {dim})…",
    ):
        vectors = embed_texts([p.chunk.text for p in payloads], settings=s)
    logger.info("  %s vector(s) computed (%s).", len(vectors), timer.took("embed"))

    with timer.step(
        "qdrant",
        f"Step 6/6 — Writing to Qdrant « {s.qdrant_collection} »…",
    ):
        ensure_collection(s.qdrant_collection, vector_size=dim, settings=s)
        delete_by_document_id(s.qdrant_collection, parsed.document_id, settings=s)
        n = upsert_payloads(s.qdrant_collection, payloads, vectors, settings=s)
    logger.info("  %s point(s) upserted (%s).", n, timer.took("qdrant"))

    total = timer.total_seconds()
    _write_parent(
        s,
        document_id=parsed.document_id,
        source_path=parsed.source_path,
        inspect={
            "n_pages": parsed.inspect.n_pages,
            "n_needs_ocr": parsed.inspect.n_needs_ocr,
            "parse_quality": parsed.inspect.parse_quality,
        },
        n_chunks=n,
        n_extractions=len(extractions),
        markdown_path=parsed.markdown_path,
        warnings=warnings,
        step_seconds=dict(timer.steps),
        total_seconds=total,
    )
    logger.info(
        "Ingest complete — document_id=%s · %s chunk(s) · %s extraction(s) · quality=%s",
        parsed.document_id,
        n,
        len(extractions),
        parsed.inspect.parse_quality,
    )
    return _finish(
        document_id=parsed.document_id,
        n_chunks=n,
        n_extractions=len(extractions),
        parse_quality=parsed.inspect.parse_quality,
        warnings=warnings,
        timer=timer,
    )
