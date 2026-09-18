"""Drop folder : scan `incoming/`, ingest, archiver (ou `failed/`).

Le catalog SQLite décide si un SHA-256 a déjà été traité. Rien n'est ingéré
tant qu'un PDF n'est pas stable dans le dossier d'arrivée.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag_ingestion.catalog import document_exists
from rag_ingestion.config import Settings, load_settings
from rag_ingestion.ingest_log import append_ingest_log, build_inbox_record
from rag_ingestion.parse import document_id_from_bytes
from rag_ingestion.pipeline import ingest_path

logger = logging.getLogger(__name__)


@dataclass
class InboxItem:
    """Résultat du traitement d'un PDF du drop folder."""

    path: Path
    document_id: str
    action: str
    dest: Path | None = None
    detail: str = ""


def list_inbox_pdfs(incoming: Path) -> list[Path]:
    """PDF du drop folder (récursif). Ignore fichiers cachés et non-.pdf."""
    incoming.mkdir(parents=True, exist_ok=True)
    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in sorted(incoming.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name.startswith(".") or candidate.name.startswith("~$"):
            continue
        if candidate.suffix.lower() != ".pdf":
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def file_is_stable(path: Path, *, wait_seconds: float = 1.5) -> bool:
    """True si taille et mtime n'ont pas changé pendant `wait_seconds` (copie en cours)."""
    try:
        first = path.stat()
    except OSError:
        return False
    if first.st_size <= 0:
        return False
    time.sleep(wait_seconds)
    try:
        second = path.stat()
    except OSError:
        return False
    return first.st_size == second.st_size and first.st_mtime == second.st_mtime


def _unique_path(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / name
    if not dest.exists():
        return dest
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 1
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _move(path: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return dest


def existing_archive(archive_root: Path, document_id: str) -> Path | None:
    """Premier PDF déjà présent sous ``archive/{sha16}/``, ou ``None``.

    Le ``document_id`` est un SHA-256 du contenu : un second drop du même
    fichier n'a pas besoin d'une autre copie dans l'archive.
    """
    folder = archive_root / document_id[:16]
    if not folder.is_dir():
        return None
    pdfs = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    return pdfs[0] if pdfs else None


def archive_destination(archive_root: Path, document_id: str, original_name: str) -> Path:
    """``archive/{sha16}/{filename}`` — sans suffixe ``_N`` (un seul fichier par SHA)."""
    folder = archive_root / document_id[:16]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / original_name


def _archive_or_reuse(path: Path, archive_root: Path, document_id: str, original_name: str) -> Path:
    """Réutilise l'archive existante pour ce SHA, sinon déplace ``path`` vers l'archive."""
    existing = existing_archive(archive_root, document_id)
    if existing is not None:
        if path.resolve() != existing.resolve():
            path.unlink(missing_ok=True)
        return existing
    return _move(path, archive_destination(archive_root, document_id, original_name))


def process_inbox_file(
    path: Path,
    *,
    settings: Settings,
    skip_extract: bool | None = None,
    force: bool = False,
    stable_wait: float = 1.5,
) -> InboxItem:
    """Déplace le PDF vers l'archive, ingère depuis ce chemin, ou vers `failed/`."""
    if not file_is_stable(path, wait_seconds=stable_wait):
        logger.info("  Unstable (copy in progress?): %s", path.name)
        item = InboxItem(path=path, document_id="", action="unstable", detail="file still changing")
        append_ingest_log(
            build_inbox_record(
                status="unstable",
                source_path=str(path),
                dest_path=None,
                document_id="",
                filename=path.name,
                detail="file still changing",
            ),
            settings.resolved_ingest_log_path(),
        )
        return item

    data = path.read_bytes()
    document_id = document_id_from_bytes(data)
    skip_extract = settings.ingest_skip_extract if skip_extract is None else skip_extract
    skip_existing = settings.ingest_skip_existing and not force

    original_name = path.name
    archive_root = settings.resolved_archive_dir()
    if skip_existing and document_exists(document_id, settings=settings):
        dest = _archive_or_reuse(path, archive_root, document_id, original_name)
        logger.info("  Duplicate SHA-256 %s — archived without re-ingest → %s", document_id[:12], dest)
        item = InboxItem(
            path=path,
            document_id=document_id,
            action="skipped_duplicate",
            dest=dest,
            detail="already in catalog",
        )
        append_ingest_log(
            build_inbox_record(
                status="skipped_duplicate",
                source_path=str(path),
                dest_path=str(dest),
                document_id=document_id,
                filename=original_name,
                detail="already in catalog",
            ),
            settings.resolved_ingest_log_path(),
        )
        return item

    dest = _archive_or_reuse(path, archive_root, document_id, original_name)

    def _to_failed(detail: str, doc_id: str) -> InboxItem:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failed = _unique_path(settings.resolved_failed_dir(), f"{stamp}_{original_name}")
        if dest.exists():
            failed = _move(dest, failed)
        return InboxItem(
            path=path,
            document_id=doc_id,
            action="failed",
            dest=failed,
            detail=detail,
        )

    try:
        result = ingest_path(
            dest,
            settings=settings,
            skip_extract=skip_extract,
            trigger="inbox",
            dest_path=dest,
        )
    except Exception as exc:
        logger.exception("Inbox ingest failed: %s", original_name)
        return _to_failed(str(exc), document_id)

    if result.skipped or result.error:
        return _to_failed(
            result.skip_reason or result.error or "ingest skipped",
            result.document_id or document_id,
        )

    logger.info("  Ingested %s → archive %s", document_id[:12], dest)
    return InboxItem(
        path=path,
        document_id=result.document_id or document_id,
        action="ingested",
        dest=dest,
    )


def process_inbox(
    *,
    settings: Settings | None = None,
    skip_extract: bool | None = None,
    force: bool = False,
    stable_wait: float = 1.5,
) -> list[InboxItem]:
    """Traite tous les PDF stables du drop folder (une passe)."""
    s = settings or load_settings()
    incoming = s.resolved_incoming_dir()
    incoming.mkdir(parents=True, exist_ok=True)
    s.resolved_archive_dir().mkdir(parents=True, exist_ok=True)
    s.resolved_failed_dir().mkdir(parents=True, exist_ok=True)
    items: list[InboxItem] = []
    for path in list_inbox_pdfs(incoming):
        items.append(
            process_inbox_file(
                path,
                settings=s,
                skip_extract=skip_extract,
                force=force,
                stable_wait=stable_wait,
            )
        )
    return items


def watch_inbox(
    *,
    settings: Settings | None = None,
    skip_extract: bool | None = None,
    force: bool = False,
    poll_seconds: int | None = None,
    once: bool = False,
    stable_wait: float = 1.5,
) -> None:
    """Boucle : scan périodique du drop folder jusqu'à interruption."""
    s = settings or load_settings()
    interval = poll_seconds if poll_seconds is not None else s.ingest_poll_seconds
    incoming = s.resolved_incoming_dir()
    logger.info(
        "Watching incoming PDF folder %s (poll %ss, archive %s)",
        incoming,
        interval,
        s.resolved_archive_dir(),
    )
    while True:
        items = process_inbox(
            settings=s,
            skip_extract=skip_extract,
            force=force,
            stable_wait=stable_wait,
        )
        if items:
            counts: dict[str, int] = {}
            for item in items:
                counts[item.action] = counts.get(item.action, 0) + 1
            logger.info("  Inbox pass: %s", counts)
        if once:
            return
        time.sleep(interval)
