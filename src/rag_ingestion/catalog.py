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
    lot_geojson TEXT,
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
    _ensure_column(conn, "sites", "lot_geojson", "TEXT")
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


def _geocode_site(conn: sqlite3.Connection, site_id: str, *, settings: Settings) -> None:
    """Remplit lat/lon (+ polygone de lot) si GEOCODE_ENABLED.

    Priorité : polygone Cadastre QC si `lot_cadastral`, sinon Nominatim sur l'adresse.
    """
    if not settings.geocode_enabled:
        return
    row = conn.execute(
        "SELECT lot_cadastral, address, city, lat, lon, geocode_status, lot_geojson "
        "FROM sites WHERE site_id = ?",
        (site_id,),
    ).fetchone()
    if row is None:
        return
    lot = row["lot_cadastral"]
    has_lot_geom = bool(row["lot_geojson"])
    has_point = row["lat"] is not None and row["lon"] is not None and row["geocode_status"] == "ok"
    if has_lot_geom and has_point:
        return
    now = _now()
    if lot and not has_lot_geom:
        from rag_ingestion.cadastre import lookup_lot_geometry

        found = lookup_lot_geometry(lot, user_agent=settings.geocode_user_agent)
        if found:
            conn.execute(
                "UPDATE sites SET lat = ?, lon = ?, geocode_status = ?, lot_geojson = ?, "
                "updated_at = ? WHERE site_id = ?",
                (
                    found["lat"],
                    found["lon"],
                    "ok",
                    json.dumps(found["geometry"], separators=(",", ":")),
                    now,
                    site_id,
                ),
            )
            logger.info(
                "  Cadastre lot %s → %s, %s",
                lot,
                found["lat"],
                found["lon"],
            )
            return
        logger.warning("  Cadastre polygon missing for lot %s", lot)
    if has_point:
        return
    from rag_ingestion.geocode import geocode_address

    coords = geocode_address(
        row["address"],
        city=row["city"],
        user_agent=settings.geocode_user_agent,
    )
    if coords is None:
        conn.execute(
            "UPDATE sites SET geocode_status = ?, updated_at = ? WHERE site_id = ?",
            ("failed", now, site_id),
        )
        logger.warning("  Geocode failed for site %s", site_id)
        return
    lat, lon = coords
    conn.execute(
        "UPDATE sites SET lat = ?, lon = ?, geocode_status = ?, updated_at = ? WHERE site_id = ?",
        (lat, lon, "ok", now, site_id),
    )
    logger.info("  Geocoded site %s (address) → %s, %s", site_id, lat, lon)


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
            _geocode_site(conn, site_id, settings=s)
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


def _parse_lot_geometry(row: sqlite3.Row) -> dict[str, Any] | None:
    if "lot_geojson" not in row.keys():
        return None
    raw = row["lot_geojson"]
    if not raw:
        return None
    try:
        geom = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(geom, dict) or not geom.get("type"):
        return None
    return geom


def _site_dict(row: sqlite3.Row, document_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "site_id": row["site_id"],
        "lot_cadastral": row["lot_cadastral"],
        "address": row["address"],
        "city": row["city"],
        "lat": row["lat"],
        "lon": row["lon"],
        "geocode_status": row["geocode_status"],
        "lot_geometry": _parse_lot_geometry(row),
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


def list_site_documents(
    site_id: str, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Fiches `documents` du site (catalog SQLite), triées par date de rapport."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        ensure_schema(conn)
        site = conn.execute(
            "SELECT * FROM sites WHERE site_id = ?",
            (site_id,),
        ).fetchone()
        rows = conn.execute(
            "SELECT * FROM documents WHERE site_id = ? "
            "ORDER BY report_date IS NULL, report_date, ingested_at",
            (site_id,),
        ).fetchall()
        return [_document_dict(row, site) for row in rows]
    finally:
        conn.close()


_TIMELINE_SELECT = """
SELECT
  e.id AS id,
  e.document_id AS document_id,
  e.site_id AS site_id,
  e.iso_date AS iso_date,
  e.role AS role,
  e.label AS label,
  d.title AS title,
  d.firm AS firm,
  d.doc_type AS doc_type,
  d.project_id AS project_id,
  d.source_path AS source_path,
  d.client AS client,
  d.contaminants AS contaminants,
  s.address AS address,
  s.lot_cadastral AS lot_cadastral,
  s.city AS city
FROM events e
LEFT JOIN documents d ON d.document_id = e.document_id
LEFT JOIN sites s ON s.site_id = e.site_id
"""


def get_site_timeline(site_id: str, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Events du site, triés par date, avec métadonnées document + site.

    S'il n'y a pas d'events LangExtract, repli sur `report_date` des documents.
    """
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
            _TIMELINE_SELECT
            + f"WHERE e.site_id = ? AND e.role IN ({placeholders}) "
            "ORDER BY e.iso_date, e.id",
            (site_id, *roles),
        ).fetchall()
        events = [_event_dict(row) for row in rows]
        if events:
            return events
        docs = conn.execute(
            """
            SELECT
              d.document_id AS document_id,
              d.title AS title,
              d.firm AS firm,
              d.doc_type AS doc_type,
              d.project_id AS project_id,
              d.source_path AS source_path,
              d.report_date AS report_date,
              d.client AS client,
              d.contaminants AS contaminants,
              s.address AS address,
              s.lot_cadastral AS lot_cadastral,
              s.city AS city
            FROM documents d
            LEFT JOIN sites s ON s.site_id = d.site_id
            WHERE d.site_id = ?
            ORDER BY d.report_date, d.ingested_at
            """,
            (site_id,),
        ).fetchall()
        fallback: list[dict[str, Any]] = []
        for row in docs:
            iso = row["report_date"]
            if not iso:
                continue
            fallback.append(
                _timeline_event(
                    event_id=None,
                    document_id=row["document_id"],
                    site_id=site_id,
                    iso_date=iso,
                    role="report",
                    label=row["title"] or row["firm"] or "rapport",
                    title=row["title"],
                    firm=row["firm"],
                    doc_type=row["doc_type"],
                    project_id=row["project_id"],
                    source_path=row["source_path"],
                    client=row["client"],
                    contaminants=row["contaminants"],
                    address=row["address"],
                    lot_cadastral=row["lot_cadastral"],
                    city=row["city"],
                )
            )
        return fallback
    finally:
        conn.close()


def _timeline_event(
    *,
    event_id: int | None,
    document_id: str,
    site_id: str,
    iso_date: str,
    role: str,
    label: str | None,
    title: str | None,
    firm: str | None,
    doc_type: str | None,
    project_id: str | None,
    source_path: str | None,
    client: str | None = None,
    contaminants: Any = None,
    address: str | None = None,
    lot_cadastral: str | None = None,
    city: str | None = None,
) -> dict[str, Any]:
    parsed = contaminants
    if isinstance(contaminants, str) or contaminants is None:
        parsed = _parse_project_ids(contaminants if isinstance(contaminants, str) else None)
    elif not isinstance(contaminants, list):
        parsed = []
    return {
        "id": event_id,
        "document_id": document_id,
        "site_id": site_id,
        "iso_date": iso_date,
        "role": role,
        "label": label,
        "title": title,
        "firm": firm,
        "doc_type": doc_type,
        "project_id": project_id,
        "source_path": source_path,
        "client": client,
        "contaminants": parsed,
        "address": address,
        "lot_cadastral": lot_cadastral,
        "city": city,
    }


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()

    def col(name: str) -> Any:
        return row[name] if name in keys else None

    return _timeline_event(
        event_id=col("id"),
        document_id=col("document_id"),
        site_id=col("site_id"),
        iso_date=col("iso_date"),
        role=col("role"),
        label=col("label"),
        title=col("title"),
        firm=col("firm"),
        doc_type=col("doc_type"),
        project_id=col("project_id"),
        source_path=col("source_path"),
        client=col("client"),
        contaminants=col("contaminants"),
        address=col("address"),
        lot_cadastral=col("lot_cadastral"),
        city=col("city"),
    )


def parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    """`min_lon,min_lat,max_lon,max_lat` ou None si vide.

    Raises:
        ValueError: Format invalide.
    """
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be min_lon,min_lat,max_lon,max_lat")
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError("bbox values must be numbers") from exc
    return min_lon, min_lat, max_lon, max_lat


def list_sites(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Tous les sites du catalog, optionnellement filtrés par bbox."""
    s = settings or load_settings()
    path = catalog_path(s)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        ensure_schema(conn)
        sql = "SELECT * FROM sites"
        args: list[Any] = []
        if bbox is not None:
            sql += (
                " WHERE lat IS NOT NULL AND lon IS NOT NULL "
                "AND lon >= ? AND lat >= ? AND lon <= ? AND lat <= ?"
            )
            args.extend(bbox)
        sql += " ORDER BY site_id"
        rows = conn.execute(sql, args).fetchall()
        docs = conn.execute(
            "SELECT site_id, document_id FROM documents ORDER BY ingested_at"
        ).fetchall()
        by_site: dict[str, list[str]] = {}
        for doc in docs:
            sid = doc["site_id"]
            if not sid:
                continue
            by_site.setdefault(str(sid), []).append(str(doc["document_id"]))
        return [_site_dict(row, by_site.get(str(row["site_id"]), [])) for row in rows]
    finally:
        conn.close()


def sites_geojson(sites: list[dict[str, Any]]) -> dict[str, Any]:
    """FeatureCollection : polygone de lot si présent, sinon point lat/lon."""
    features: list[dict[str, Any]] = []
    for site in sites:
        lot_geom = site.get("lot_geometry")
        lat = site.get("lat")
        lon = site.get("lon")
        if isinstance(lot_geom, dict) and lot_geom.get("type"):
            geometry = lot_geom
        elif lat is not None and lon is not None:
            geometry = {"type": "Point", "coordinates": [float(lon), float(lat)]}
        else:
            continue
        props = {
            k: v
            for k, v in site.items()
            if k not in {"lat", "lon", "lot_geometry"}
        }
        features.append(
            {
                "type": "Feature",
                "id": site.get("site_id"),
                "geometry": geometry,
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


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
