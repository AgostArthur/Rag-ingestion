"""Journal JSONL append-only d'ingestion.

Une ligne JSON par tentative (succès, skip, doublon, échec, instable).
Chemin par défaut : ``{DATA_DIR}/log_ingest.jsonl`` (surcharge via ``INGEST_LOG_FILE``).

Le fichier est écrit en best-effort : une erreur d'I/O logue un warning
stdout et n'interrompt jamais le pipeline principal.

Sur les systèmes POSIX (Linux, macOS) un verrou ``fcntl.flock`` est posé avant
l'écriture pour que le CLI et le worker watch puissent tourner simultanément
sans corrompre le fichier.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import conditionnel : fcntl n'existe pas sur Windows.
try:
    import fcntl as _fcntl  # noqa: PLC0415
    _HAS_FLOCK = True
except ImportError:
    _fcntl = None  # type: ignore[assignment]
    _HAS_FLOCK = False


def now_utc_iso() -> str:
    """Horodatage ISO-8601 UTC (secondes, suffixe Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_ingest_log(record: dict[str, Any], log_path: Path) -> None:
    """Ajoute une ligne JSON au journal d'ingest (best-effort, verrou POSIX).

    Args:
        record:   Dictionnaire plat sérialisable en JSON.
        log_path: Chemin du fichier ``.jsonl``.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with log_path.open("a", encoding="utf-8") as fh:
            if _HAS_FLOCK:
                _fcntl.flock(fh, _fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                if _HAS_FLOCK:
                    _fcntl.flock(fh, _fcntl.LOCK_UN)
    except Exception:
        logger.warning(
            "ingest_log: impossible d'écrire dans %s (ingest non affecté).",
            log_path,
            exc_info=True,
        )


def build_pipeline_record(
    *,
    status: str,
    trigger: str,
    source_path: str,
    dest_path: str | None,
    document_id: str,
    filename: str,
    profile_id: str | None,
    profile_match_source: str | None,
    skip_extract: bool,
    langextract_model: str,
    embed_model: str,
    qdrant_collection: str,
    n_pages: int | None,
    n_needs_ocr: int | None,
    parse_quality: str | None,
    n_chunks: int,
    n_extractions: int,
    doc_type: str | None,
    site_id: str | None,
    project_id: str | None,
    step_seconds: dict[str, float],
    total_seconds: float,
    warnings: list[str],
    error: str | None,
) -> dict[str, Any]:
    """Construit le dictionnaire d'un enregistrement issu du pipeline principal."""
    return {
        "ts": now_utc_iso(),
        "status": status,
        "trigger": trigger,
        "filename": filename,
        "source_path": source_path,
        "dest_path": dest_path,
        "document_id": document_id,
        "profile_id": profile_id,
        "profile_match_source": profile_match_source,
        "skip_extract": skip_extract,
        "langextract_model": langextract_model,
        "embed_model": embed_model,
        "qdrant_collection": qdrant_collection,
        "n_pages": n_pages,
        "n_needs_ocr": n_needs_ocr,
        "parse_quality": parse_quality,
        "n_chunks": n_chunks,
        "n_extractions": n_extractions,
        "doc_type": doc_type,
        "site_id": site_id,
        "project_id": project_id,
        "step_seconds": {k: round(v, 4) for k, v in step_seconds.items()},
        "total_seconds": round(total_seconds, 4),
        "warnings": warnings,
        "error": error,
    }


def build_inbox_record(
    *,
    status: str,
    source_path: str,
    dest_path: str | None,
    document_id: str,
    filename: str,
    detail: str,
) -> dict[str, Any]:
    """Construit le dictionnaire d'un enregistrement inbox-only (doublon, instable)."""
    return {
        "ts": now_utc_iso(),
        "status": status,
        "trigger": "inbox",
        "filename": filename,
        "source_path": source_path,
        "dest_path": dest_path,
        "document_id": document_id,
        "profile_id": None,
        "profile_match_source": None,
        "skip_extract": None,
        "langextract_model": None,
        "embed_model": None,
        "qdrant_collection": None,
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
        "warnings": [],
        "error": detail or None,
    }
