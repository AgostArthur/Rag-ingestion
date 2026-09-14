"""Normalisation lot / adresse / dates pour le catalog (pas pour les embeddings)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_PROJECT_TOKEN_RE = re.compile(r"\b(\d{3,8})\b")
_DIGITS_RE = re.compile(r"\d+")

DATE_ROLES = frozenset(
    {"report", "fieldwork", "phase1", "sampling", "unknown"}
)

_FIELDWORK_HINTS = (
    "forage",
    "forages",
    "campagne",
    "prélèvement",
    "prelevement",
    "échantillonnage",
    "echantillonnage",
    "fieldwork",
    "sampling",
)
_REPORT_HINTS = (
    "rapport",
    "report",
    "remise",
    "émission",
    "emission",
)


def fold_ascii(text: str) -> str:
    """Minuscules sans accents (clé de fusion d'adresse)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def normalize_lot(text: str | None) -> str | None:
    """Chiffres seuls d'un lot cadastral ; None si trop court pour un lot."""
    if not text:
        return None
    digits = "".join(_DIGITS_RE.findall(text))
    if 5 <= len(digits) <= 12:
        return digits
    return None


def normalize_address(text: str | None) -> str | None:
    """Adresse affichable : espaces collapsés, pas de virgules doubles."""
    if not text:
        return None
    collapsed = re.sub(r"\s+", " ", text).strip(" ,;")
    return collapsed or None


def address_key(text: str | None) -> str | None:
    """Clé de fusion : minuscules, sans accents, ponctuation réduite."""
    raw = normalize_address(text)
    if not raw:
        return None
    folded = fold_ascii(raw)
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    return folded or None


def site_id_for(*, lot: str | None, address: str | None) -> str | None:
    """`lot:{chiffres}` si le lot est connu, sinon hash stable de l'adresse."""
    if lot:
        return f"lot:{lot}"
    key = address_key(address)
    if not key:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"addr:{digest}"


def is_year_token(text: str) -> bool:
    """True si `text` est une année 19xx / 20xx (pas un n° de projet)."""
    return bool(_YEAR_RE.match(text.strip()))


def is_project_token(text: str) -> bool:
    """True si `text` ressemble à un n° de projet (3–8 chiffres, pas une année)."""
    value = text.strip()
    return value.isdigit() and 3 <= len(value) <= 8 and not is_year_token(value)


def project_tokens_in_text(text: str) -> list[str]:
    """N° de projet candidats dans une question (ordre d'apparition, uniques)."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _PROJECT_TOKEN_RE.finditer(text or ""):
        token = match.group(1)
        if not is_project_token(token) or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def iso_date(text: str | None) -> str | None:
    """Retourne YYYY-MM-DD si `text` est une date ISO valide, sinon None."""
    if not text:
        return None
    value = text.strip()
    match = _ISO_DATE_RE.match(value)
    if not match:
        return None
    year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def infer_date_role(role: str | None, label: str) -> str:
    """Rôle de date : attribut LangExtract, sinon indices lexicaux, sinon `unknown`."""
    if isinstance(role, str):
        key = role.strip().lower()
        if key in DATE_ROLES and key != "unknown":
            return key
    hay = fold_ascii(label)
    if any(hint in hay for hint in _FIELDWORK_HINTS):
        return "fieldwork"
    if any(hint in hay for hint in _REPORT_HINTS):
        return "report"
    return "unknown"
