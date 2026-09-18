"""Orchestration ingest : parse → chunk → LangExtract → catalog → embed → Qdrant."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from rag_ingestion.align import align_extractions
from rag_ingestion.catalog import upsert_document_meta
from rag_ingestion.chunk import chunk_markdown
from rag_ingestion.config import Settings, load_settings
from rag_ingestion.document_meta import build_document_meta, stamp_payloads
from rag_ingestion.embed import embed_texts, embedding_dimension
from rag_ingestion.extract import extract_structured
from rag_ingestion.extract_profile import resolve_extract_schema
from rag_ingestion.ingest_log import append_ingest_log, build_pipeline_record
from rag_ingestion.llm_providers import resolve_extract_llm
from rag_ingestion.logging_setup import StepTimer
from rag_ingestion.models import DocumentMeta, GroundedExtraction, IngestResult
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
    meta: DocumentMeta | None = None,
) -> None:
    """Écrit la fiche parent JSON (journal d'ingest + identité catalog)."""
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
        "catalog_path": str(settings.catalog_path),
        "warnings": warnings,
        "step_seconds": {k: round(v, 4) for k, v in step_seconds.items()},
        "total_seconds": round(total_seconds, 4),
    }
    if meta is not None:
        payload.update(
            {
                "site_id": meta.site_id,
                "project_id": meta.project_id,
                "project_ids": meta.project_ids,
                "title": meta.title,
                "doc_type": meta.doc_type,
                "firm": meta.firm,
                "client": meta.client,
                "address": meta.address,
                "lot_cadastral": meta.lot_cadastral,
                "city": meta.city,
                "report_date": meta.report_date,
                "contaminants": meta.contaminants,
                "events": [
                    {"iso_date": e.iso_date, "role": e.role, "label": e.label}
                    for e in meta.events
                ],
            }
        )
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
    site_id: str | None = None,
    project_id: str | None = None,
    error: str | None = None,
    profile_id: str | None = None,
    profile_match_source: str | None = None,
    source_path: str | None = None,
    n_pages: int | None = None,
    n_needs_ocr: int | None = None,
    doc_type: str | None = None,
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
        site_id=site_id,
        project_id=project_id,
        error=error,
        profile_id=profile_id,
        profile_match_source=profile_match_source,
        source_path=source_path,
        n_pages=n_pages,
        n_needs_ocr=n_needs_ocr,
        doc_type=doc_type,
    )


def ingest_path(
    path: Path,
    *,
    settings: Settings | None = None,
    skip_extract: bool = False,
    extract_profile: str | None = None,
    trigger: str = "cli",
    dest_path: Path | None = None,
) -> IngestResult:
    """Ingest un PDF : parse, chunk, extrait, catalog, embed, upsert (un point par chunk).

    Quality gates : Markdown vide ou aucun chunk → skip. LangExtract en échec
    → avertissement et poursuite sans métadonnées. Ré-ingest : delete puis upsert.
    Le catalog SQLite (sites / documents / events) est écrit après l'alignement.
    Un enregistrement JSONL est toujours ajouté au journal d'ingest (best-effort).

    Args:
        path: Chemin du PDF (après déplacement archive pour l'inbox).
        settings: Config ; `.env` si omis.
        skip_extract: Si True, n'appelle pas LangExtract (profil déduit quand même).
        extract_profile: Profil LangExtract (`--profile`) ; sinon déduit du nom.
        trigger: Déclencheur (``"cli"`` ou ``"inbox"``).
        dest_path: Chemin de destination finale (archive ou failed) — fourni par inbox.

    Returns:
        Compte-rendu (`n_chunks`, extractions, `parse_quality`, timings, profil).

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

    # Profil LangExtract résolu avant tout (sans appel LLM) pour le journal.
    _schema = None
    try:
        _schema = resolve_extract_schema(
            path,
            markdown=None,
            override=extract_profile,
            settings=s,
        )
    except Exception:
        pass  # Résolution via le Markdown uniquement (heading) arrive plus loin.

    # Vrai modèle LangExtract selon le provider actif (gemini, openai, ollama…).
    _extract_llm = resolve_extract_llm(
        langextract_model=s.langextract_model,
        ollama_base_url=s.ollama_base_url,
    )
    _effective_langextract_model = (
        f"{_extract_llm.provider}/{_extract_llm.model}" if not skip_extract else "disabled"
    )

    warnings: list[str] = []

    # --- Contexte de log partagé : rempli au fur et à mesure ---
    _log_ctx: dict = {
        "status": "failed",
        "source_path": str(path),
        "dest_path": str(dest_path) if dest_path else None,
        "document_id": "",
        "filename": path.name,
        "profile_id": _schema.profile_id if _schema else None,
        "profile_match_source": _schema.match_source if _schema else None,
        "skip_extract": skip_extract,
        "langextract_model": _effective_langextract_model,
        "embed_model": s.embed_model,
        "qdrant_collection": s.qdrant_collection,
        "n_pages": None,
        "n_needs_ocr": None,
        "parse_quality": None,
        "n_chunks": 0,
        "n_extractions": 0,
        "doc_type": None,
        "site_id": None,
        "project_id": None,
        "step_seconds": {},
        "total_seconds": 0.0,
        "warnings": warnings,
        "error": None,
    }

    try:
        with timer.step(
            "parse",
            "Step 1/7 — Inspect PDF then convert to Markdown…",
        ):
            parsed = parse_pdf(path, settings=s)
        logger.info("  document_id=%s (%s)", parsed.document_id, timer.took("parse"))

        _log_ctx["document_id"] = parsed.document_id
        _log_ctx["n_pages"] = parsed.inspect.n_pages
        _log_ctx["n_needs_ocr"] = parsed.inspect.n_needs_ocr
        _log_ctx["parse_quality"] = parsed.inspect.parse_quality

        if parsed.inspect.parse_quality == "ocr_heavy":
            warnings.append(
                f"Heavy OCR: {parsed.inspect.n_needs_ocr}/{parsed.inspect.n_pages} pages"
            )

        if not parsed.markdown.strip():
            msg = "Empty markdown after parse — ingest skipped"
            logger.warning("%s", msg)
            _log_ctx["status"] = "skipped"
            _log_ctx["error"] = msg
            return _finish(
                document_id=parsed.document_id,
                n_chunks=0,
                n_extractions=0,
                parse_quality=parsed.inspect.parse_quality,
                warnings=warnings + [msg],
                timer=timer,
                skipped=True,
                skip_reason=msg,
                profile_id=_log_ctx["profile_id"],
                profile_match_source=_log_ctx["profile_match_source"],
                source_path=str(path),
                n_pages=parsed.inspect.n_pages,
                n_needs_ocr=parsed.inspect.n_needs_ocr,
            )

        with timer.step(
            "chunk",
            f"Step 2/7 — Chunking text ({s.chunk_size_chars} chars, "
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
            _log_ctx["status"] = "skipped"
            _log_ctx["error"] = msg
            return _finish(
                document_id=parsed.document_id,
                n_chunks=0,
                n_extractions=0,
                parse_quality=parsed.inspect.parse_quality,
                warnings=warnings + [msg],
                timer=timer,
                skipped=True,
                skip_reason=msg,
                profile_id=_log_ctx["profile_id"],
                profile_match_source=_log_ctx["profile_match_source"],
                source_path=str(path),
                n_pages=parsed.inspect.n_pages,
                n_needs_ocr=parsed.inspect.n_needs_ocr,
            )

        extractions: list[GroundedExtraction] = []
        schema = _schema  # Peut être affiné avec le Markdown si heading match.
        if skip_extract:
            logger.info("Step 3/7 — LangExtract skipped (--skip-extract).")
            timer.steps["langextract"] = 0.0
            # Affiner la résolution avec le Markdown (heading) si besoin.
            if schema is None or schema.match_source in {"default"}:
                try:
                    schema = resolve_extract_schema(
                        path,
                        markdown=parsed.markdown,
                        override=extract_profile,
                        settings=s,
                    )
                    _log_ctx["profile_id"] = schema.profile_id
                    _log_ctx["profile_match_source"] = schema.match_source
                except Exception:
                    pass
        else:
            with timer.step(
                "langextract",
                f"Step 3/7 — Structured extraction ({s.langextract_model})…",
            ):
                schema = resolve_extract_schema(
                    path,
                    markdown=parsed.markdown,
                    override=extract_profile,
                    settings=s,
                )
                _log_ctx["profile_id"] = schema.profile_id
                _log_ctx["profile_match_source"] = schema.match_source
                if schema.match_source == "default":
                    warnings.append(
                        f"LangExtract type={schema.profile_id}  fichier=« {path.name} »  "
                        f"source=default — aucun motif de profil n'a matché"
                    )
                try:
                    extractions = extract_structured(
                        parsed.markdown,
                        document_id=parsed.document_id,
                        source_path=path,
                        settings=s,
                        schema=schema,
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
            "Step 4/7 — Attach labels to chunks…",
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

        meta: DocumentMeta | None = None
        with timer.step(
            "catalog",
            "Step 5/7 — Document catalog (SQLite sites / documents / events)…",
        ):
            meta = build_document_meta(
                extractions,
                document_id=parsed.document_id,
                source_path=parsed.source_path,
                parse_quality=parsed.inspect.parse_quality,
            )
            try:
                meta = upsert_document_meta(meta, settings=s)
            except Exception as exc:
                warnings.append(f"Catalog upsert failed: {exc}")
                logger.exception("Catalog upsert failed — continuing with Qdrant.")
            stamp_payloads(payloads, meta)
        logger.info(
            "  site_id=%s project_id=%s events=%s (%s).",
            meta.site_id,
            meta.project_id,
            len(meta.events),
            timer.took("catalog"),
        )

        _log_ctx["doc_type"] = meta.doc_type if meta else None
        _log_ctx["site_id"] = meta.site_id if meta else None
        _log_ctx["project_id"] = meta.project_id if meta else None

        dim = embedding_dimension(s)
        with timer.step(
            "embed",
            f"Step 6/7 — Computing embeddings ({len(payloads)} texts, dim {dim})…",
        ):
            vectors = embed_texts([p.chunk.text for p in payloads], settings=s)
        logger.info("  %s vector(s) computed (%s).", len(vectors), timer.took("embed"))

        with timer.step(
            "qdrant",
            f"Step 7/7 — Writing to Qdrant « {s.qdrant_collection} »…",
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
            meta=meta,
        )
        logger.info(
            "Ingest complete — document_id=%s · %s chunk(s) · %s extraction(s) · "
            "quality=%s · site_id=%s · project_id=%s",
            parsed.document_id,
            n,
            len(extractions),
            parsed.inspect.parse_quality,
            meta.site_id if meta else None,
            meta.project_id if meta else None,
        )

        _log_ctx.update(
            status="ingested",
            n_chunks=n,
            n_extractions=len(extractions),
        )
        result = _finish(
            document_id=parsed.document_id,
            n_chunks=n,
            n_extractions=len(extractions),
            parse_quality=parsed.inspect.parse_quality,
            warnings=warnings,
            timer=timer,
            site_id=meta.site_id if meta else None,
            project_id=meta.project_id if meta else None,
            profile_id=_log_ctx["profile_id"],
            profile_match_source=_log_ctx["profile_match_source"],
            source_path=str(path),
            n_pages=parsed.inspect.n_pages,
            n_needs_ocr=parsed.inspect.n_needs_ocr,
            doc_type=meta.doc_type if meta else None,
        )
        return result

    except Exception as exc:
        _log_ctx["status"] = "failed"
        _log_ctx["error"] = str(exc)
        raise

    finally:
        _log_ctx["step_seconds"] = {k: round(v, 4) for k, v in timer.steps.items()}
        _log_ctx["total_seconds"] = round(timer.total_seconds(), 4)
        _log_ctx["warnings"] = list(warnings)
        record = build_pipeline_record(
            status=_log_ctx["status"],
            trigger=trigger,
            source_path=_log_ctx["source_path"],
            dest_path=_log_ctx["dest_path"],
            document_id=_log_ctx["document_id"],
            filename=_log_ctx["filename"],
            profile_id=_log_ctx["profile_id"],
            profile_match_source=_log_ctx["profile_match_source"],
            skip_extract=skip_extract,
            langextract_model=_log_ctx["langextract_model"],
            embed_model=_log_ctx["embed_model"],
            qdrant_collection=_log_ctx["qdrant_collection"],
            n_pages=_log_ctx["n_pages"],
            n_needs_ocr=_log_ctx["n_needs_ocr"],
            parse_quality=_log_ctx["parse_quality"],
            n_chunks=_log_ctx["n_chunks"],
            n_extractions=_log_ctx["n_extractions"],
            doc_type=_log_ctx["doc_type"],
            site_id=_log_ctx["site_id"],
            project_id=_log_ctx["project_id"],
            step_seconds=_log_ctx["step_seconds"],
            total_seconds=_log_ctx["total_seconds"],
            warnings=_log_ctx["warnings"],
            error=_log_ctx["error"],
        )
        append_ingest_log(record, s.resolved_ingest_log_path())


def collect_pdf_paths(path: Path) -> list[Path]:
    """Un PDF, ou tous les PDF d'un dossier (récursif, dédupliqués)."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {path}")
        return [path]
    if path.is_dir():
        found: list[Path] = []
        seen: set[Path] = set()
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
        if not found:
            raise FileNotFoundError(f"No PDF found in {path}")
        return found
    raise FileNotFoundError(f"File not found: {path}")


def ingest_many(
    paths: list[Path],
    *,
    settings: Settings | None = None,
    skip_extract: bool = False,
    extract_profile: str | None = None,
) -> list[IngestResult]:
    """Ingest une liste de PDF ; une exception n'interrompt pas le lot."""
    s = settings or load_settings()
    results: list[IngestResult] = []
    for index, path in enumerate(paths, start=1):
        logger.info("Batch ingest %s/%s — %s", index, len(paths), path.name)
        try:
            results.append(
                ingest_path(
                    path,
                    settings=s,
                    skip_extract=skip_extract,
                    extract_profile=extract_profile,
                )
            )
        except Exception as exc:
            logger.exception("Batch item failed: %s", path)
            results.append(
                IngestResult(
                    document_id="",
                    n_chunks=0,
                    n_extractions=0,
                    parse_quality="error",
                    warnings=[],
                    skipped=True,
                    skip_reason=str(exc),
                    error=str(exc),
                    total_seconds=0.0,
                )
            )
    n_ok = sum(1 for item in results if not item.skipped and not item.error)
    n_fail = sum(1 for item in results if item.error)
    logger.info(
        "Batch done — %s file(s), %s ok, %s failed.",
        len(paths),
        n_ok,
        n_fail,
    )
    return results
