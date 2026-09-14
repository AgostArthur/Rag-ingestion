"""Catalog SQLite : sites, documents, events. Écrit à l'ingest uniquement."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.models import DocumentMeta
from rag_ingestion.normalize import TIMELINE_ROLES
from rag_ingestion.normalize import address_key

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    site_id TEXT PRIMARY KEY,
    lot_cadastral TEXT,
    address TEXT,
    address_key TEXT,
    city TEXT,
    lat REAL,
    lon REAL,
    geocode_status TEXT NOT NULL DEFAULT 'skipped',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    project_id TEXT,
    project_ids TEXT,
    title TEXT,
    doc_type TEXT,
    firm TEXT,
    client TEXT,
    site_id TEXT,
    source_path TEXT,
    parse_quality TEXT,
    report_date TEXT,
    contaminants TEXT,
    ingested_at TEXT NOT NULL,
    FOREIGN KEY (site_id) REFERENCES sites(site_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    site_id TEXT,
    iso_date TEXT NOT NULL,
    role TEXT NOT NULL,
    label TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_sites_address_key ON sites(address_key);
CREATE INDEX IF NOT EXISTS idx_documents_site ON documents(site_id);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_events_site ON events(site_id);
CREATE INDEX IF NOT EXISTS idx_events_document ON events(document_id);
"""


def catalog_path(settings: Settings | None = None) -> Path:
    """Chemin du fichier SQLite (`DATA_DIR/catalog.sqlite`)."""
    s = settings or load_settings()
    return s.catalog_path


def connect(path: Path) -> sqlite3.Connection:
    """Ouvre SQLite avec clés étrangères et WAL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    """Ajoute une colonne absente (bases SQLite déjà créées)."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Crée les tables si besoin et complète les colonnes ajoutées plus tard."""
    conn.executescript(_SCHEMA)
    _ensure_column(conn, "documents", "contaminants", "TEXT")
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fill_if_empty(current: str | None, incoming: str | None) -> str | None:
    if current:
        return current
    return incoming


def _resolve_site_id(conn: sqlite3.Connection, meta: DocumentMeta) -> str | None:
    """Réutilise un site existant (lot, puis adresse normalisée)."""
    if meta.site_id and meta.site_id.startswith("lot:"):
        return meta.site_id
    key = address_key(meta.address)
    if key:
        row = conn.execute(
            "SELECT site_id FROM sites WHERE address_key = ? LIMIT 1",
            (key,),
        ).fetchone()
        if row:
            return str(row["site_id"])
    return meta.site_id


def _upsert_site(conn: sqlite3.Connection, meta: DocumentMeta, site_id: str) -> None:
    now = _now()
    existing = conn.execute(
        "SELECT lot_cadastral, address, address_key, city, lat, lon, geocode_status "
        "FROM sites WHERE site_id = ?",
        (site_id,),
    ).fetchone()
    incoming_key = address_key(meta.address)
    if existing is None:
        conn.execute(
            "INSERT INTO sites (site_id, lot_cadastral, address, address_key, city, "
            "lat, lon, geocode_status, updated_at) VALUES (?, ?, ?, ?, ?, NULL, NULL, 'skipped', ?)",
            (site_id, meta.lot_cadastral, meta.address, incoming_key, meta.city, now),
        )
        return
        conn.execute(
            "UPDATE sites SET lot_cadastral = ?, address = ?, address_key = ?, city = ?, "
            "updated_at = ? WHERE site_id = ?",
            (
                _fill_if_empty(existing["lot_cadastral"], meta.lot_cadastral),
                _fill_if_empty(existing["address"], meta.address),
                _fill_if_empty(existing["address_key"], incoming_key),
                _fill_if_empty(existing["city"], meta.city),
                now,
                site_id,
            ),
        )


def _merge_address_sites_into(
    conn: sqlite3.Connection,
    canonical_id: str,
    key: str | None,
) -> None:
    """Si un site `addr:` partage la même adresse, le fusionne dans `canonical_id`."""
    if not key:
        return
    rows = conn.execute(
        "SELECT site_id FROM sites WHERE address_key = ? AND site_id != ?",
        (key, canonical_id),
    ).fetchall()
    for row in rows:
        old_id = str(row["site_id"])
        conn.execute(
            "UPDATE documents SET site_id = ? WHERE site_id = ?",
            (canonical_id, old_id),
        )
        conn.execute(
            "UPDATE events SET site_id = ? WHERE site_id = ?",
            (canonical_id, old_id),
        )
        conn.execute("DELETE FROM sites WHERE site_id = ?", (old_id,))
        logger.info("  Merged catalog site %s into %s", old_id, canonical_id)


def upsert_document_meta(meta: DocumentMeta, settings: Settings | None = None) -> DocumentMeta:
    """Écrit / met à jour sites, documents, events. Remplace les events du document.

    Args:
        meta: Fiche agrégée à l'ingest.
        settings: Config ; `.env` si omis.

    Returns:
        La même fiche, avec `site_id` éventuellement réconcilié (adresse → lot).
    """
    s = settings or load_settings()
    path = catalog_path(s)
    conn = connect(path)
    try:
        ensure_schema(conn)
        site_id = _resolve_site_id(conn, meta)
        if site_id:
            _upsert_site(conn, meta, site_id)
            _merge_address_sites_into(conn, site_id, address_key(meta.address))
            meta.site_id = site_id
        conn.execute(
            """
            INSERT INTO documents (
                document_id, project_id, project_ids, title, doc_type, firm, client,
                site_id, source_path, parse_quality, report_date, contaminants, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                project_id = excluded.project_id,
                project_ids = excluded.project_ids,
                title = excluded.title,
                doc_type = excluded.doc_type,
                firm = excluded.firm,
                client = excluded.client,
                site_id = excluded.site_id,
                source_path = excluded.source_path,
                parse_quality = excluded.parse_quality,
                report_date = excluded.report_date,
                contaminants = excluded.contaminants,
                ingested_at = excluded.ingested_at
            """,
            (
                meta.document_id,
                meta.project_id,
                json.dumps(meta.project_ids, ensure_ascii=False),
                meta.title,
                meta.doc_type,
                meta.firm,
                meta.client,
                meta.site_id,
                meta.source_path,
                meta.parse_quality,
                meta.report_date,
                json.dumps(meta.contaminants, ensure_ascii=False),
                _now(),
            ),
        )
        conn.execute("DELETE FROM events WHERE document_id = ?", (meta.document_id,))
        for event in meta.events:
            if event.role not in TIMELINE_ROLES:
                continue
            conn.execute(
                "INSERT INTO events (document_id, site_id, iso_date, role, label) "
                "VALUES (?, ?, ?, ?, ?)",
                (meta.document_id, meta.site_id, event.iso_date, event.role, event.label),
            )
        conn.commit()
        logger.info(
            "  Catalog upsert document_id=%s site_id=%s project_id=%s events=%s",
            meta.document_id,
            meta.site_id,
            meta.project_id,
            len(meta.events),
        )
        return meta
    finally:
        conn.close()


def _parse_project_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [raw] if raw else []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _document_dict(row: sqlite3.Row, site: sqlite3.Row | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "document_id": row["document_id"],
        "project_id": row["project_id"],
        "project_ids": _parse_project_ids(row["project_ids"]),
        "title": row["title"],
        "doc_type": row["doc_type"],
        "firm": row["firm"],
        "client": row["client"],
        "site_id": row["site_id"],
        "source_path": row["source_path"],
        "parse_quality": row["parse_quality"],
        "report_date": row["report_date"],
        "contaminants": _parse_project_ids(
            row["contaminants"] if "contaminants" in row.keys() else None
        ),
        "ingested_at": row["ingested_at"],
    }
    if site is not None:
        out["lot_cadastral"] = site["lot_cadastral"]
        out["address"] = site["address"]
        out["city"] = site["city"]
        out["lat"] = site["lat"]
        out["lon"] = site["lon"]
        out["geocode_status"] = site["geocode_status"]
    else:
        out["lot_cadastral"] = None
        out["address"] = None
        out["city"] = None
        out["lat"] = None
        out["lon"] = None
        out["geocode_status"] = None
    return out


def _site_dict(row: sqlite3.Row, document_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "site_id": row["site_id"],
        "lot_cadastral": row["lot_cadastral"],
        "address": row["address"],
        "city": row["city"],
        "lat": row["lat"],
        "lon": row["lon"],
        "geocode_status": row["geocode_status"],
        "updated_at": row["updated_at"],
        "document_ids": document_ids if document_ids is not None else [],
    }


def get_document(document_id: str, settings: Settings | None = None) -> dict[str, Any] | None:
    """Fiche document + champs du site lié, ou None."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        site = None
        if row["site_id"]:
            site = conn.execute(
                "SELECT * FROM sites WHERE site_id = ?",
                (row["site_id"],),
            ).fetchone()
        return _document_dict(row, site)
    finally:
        conn.close()


def document_exists(document_id: str, settings: Settings | None = None) -> bool:
    """True si `document_id` (SHA-256) est déjà dans le catalog."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return False
    conn = connect(path)
    try:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
            (document_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_documents(document_ids: list[str], settings: Settings | None = None) -> list[dict[str, Any]]:
    """Fiches dans l'ordre demandé ; les ids inconnus sont omis."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc_id in document_ids:
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        row = get_document(doc_id, settings=settings)
        if row is not None:
            out.append(row)
    return out


def get_site(site_id: str, settings: Settings | None = None) -> dict[str, Any] | None:
    """Site + liste des document_id liés."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        ensure_schema(conn)
        row = conn.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,)).fetchone()
        if row is None:
            return None
        docs = conn.execute(
            "SELECT document_id FROM documents WHERE site_id = ? ORDER BY ingested_at",
            (site_id,),
        ).fetchall()
        return _site_dict(row, [str(d["document_id"]) for d in docs])
    finally:
        conn.close()


def get_site_timeline(site_id: str, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Events du site, triés par date."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        ensure_schema(conn)
        roles = tuple(sorted(TIMELINE_ROLES))
        placeholders = ",".join("?" * len(roles))
        rows = conn.execute(
            "SELECT id, document_id, site_id, iso_date, role, label "
            f"FROM events WHERE site_id = ? AND role IN ({placeholders}) "
            "ORDER BY iso_date, id",
            (site_id, *roles),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "site_id": row["site_id"],
                "iso_date": row["iso_date"],
                "role": row["role"],
                "label": row["label"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def ping_catalog(settings: Settings | None = None) -> tuple[bool, str]:
    """Vérifie que le fichier SQLite s'ouvre (health check)."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return True, "not created yet"
    try:
        conn = connect(path)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        return True, str(path)
    except Exception as exc:
        return False, str(exc)
