"""Géocodage d'une adresse de site (Nominatim) — à l'ingest, pas au chat."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "rag-ingestion/0.1 (enviro-rag)"


def compose_query(address: str | None, city: str | None = None) -> str | None:
    """Adresse + ville + Québec, pour Nominatim."""
    parts: list[str] = []
    if address and address.strip():
        parts.append(address.strip())
    if city and city.strip() and (not address or city.strip().lower() not in address.lower()):
        parts.append(city.strip())
    if not parts:
        return None
    text = ", ".join(parts)
    if "québec" not in text.lower() and "quebec" not in text.lower() and "canada" not in text.lower():
        text = f"{text}, Québec, Canada"
    return text


def geocode_address(
    address: str | None,
    *,
    city: str | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 10.0,
    opener: Any = None,
) -> tuple[float, float] | None:
    """Retourne `(lat, lon)` ou None. `opener` = callable(url, timeout) pour les tests."""
    query = compose_query(address, city)
    if not query:
        return None
    params = urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": "1", "countrycodes": "ca"}
    )
    url = f"{NOMINATIM_URL}?{params}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent or DEFAULT_USER_AGENT, "Accept": "application/json"},
    )
    try:
        if opener is not None:
            raw = opener(url, timeout)
        else:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception:
        logger.exception("Geocode failed for %s", query)
        return None
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    try:
        lat = float(first["lat"])
        lon = float(first["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    return lat, lon
