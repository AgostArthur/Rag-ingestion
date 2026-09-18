"""Polygones de lots — Cadastre du Québec (MELCCFP / MRNF), à l'ingest.

Le pin carte doit viser le **lot**, pas l'adresse Nominatim. Le service public
`Cadastre_allege` expose `NO_LOT` avec espaces (ex. `2 363 352`).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

CADASTRE_QUERY_URL = (
    "https://geo.environnement.gouv.qc.ca/donnees/rest/services/"
    "Reference/Cadastre_allege/MapServer/0/query"
)
DEFAULT_USER_AGENT = "rag-ingestion/0.1 (enviro-rag)"


def format_cadastre_no_lot(lot: str | None) -> str | None:
    """Chiffres du lot → `NO_LOT` Cadastre QC (groupes de 3 depuis la droite)."""
    if not lot:
        return None
    digits = "".join(ch for ch in str(lot) if ch.isdigit())
    if len(digits) < 5:
        return None
    groups: list[str] = []
    rest = digits
    while rest:
        groups.append(rest[-3:])
        rest = rest[:-3]
    return " ".join(reversed(groups))


def _rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon" and coords:
        return [coords[0]]
    if kind == "MultiPolygon":
        return [poly[0] for poly in coords if poly]
    return []


def _bbox_area(ring: list[list[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def geometry_centroid(geometry: dict[str, Any]) -> tuple[float, float] | None:
    """Centroïde approximatif `(lat, lon)` de l'anneau extérieur le plus grand."""
    rings = _rings(geometry)
    if not rings:
        return None
    ring = max(rings, key=_bbox_area)
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    if not pts:
        return None
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return lat, lon


def _pick_feature(features: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_area = -1.0
    for feature in features:
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        rings = _rings(geom)
        area = max((_bbox_area(r) for r in rings), default=0.0)
        if area > best_area:
            best_area = area
            best = feature
    return best


def lookup_lot_geometry(
    lot: str | None,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 15.0,
    opener: Any = None,
) -> dict[str, Any] | None:
    """Retourne `{geometry, lat, lon, no_lot}` ou None."""
    no_lot = format_cadastre_no_lot(lot)
    if not no_lot:
        return None
    params = urllib.parse.urlencode(
        {
            "where": f"NO_LOT='{no_lot}'",
            "outFields": "NO_LOT",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    url = f"{CADASTRE_QUERY_URL}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent or DEFAULT_USER_AGENT,
            "Accept": "application/geo+json, application/json",
        },
    )
    try:
        if opener is not None:
            raw = opener(url, timeout)
        else:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception:
        logger.exception("Cadastre lookup failed for lot %s", no_lot)
        return None
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list) or not features:
        logger.warning("Cadastre: no polygon for NO_LOT=%s", no_lot)
        return None
    feature = _pick_feature(features)
    if feature is None:
        return None
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or not geometry.get("type"):
        return None
    centroid = geometry_centroid(geometry)
    if centroid is None:
        return None
    lat, lon = centroid
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    return {
        "geometry": geometry,
        "lat": lat,
        "lon": lon,
        "no_lot": props.get("NO_LOT") or no_lot,
    }
