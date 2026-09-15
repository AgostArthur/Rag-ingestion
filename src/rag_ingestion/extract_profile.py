"""Routage du schéma LangExtract : profil selon le nom de fichier (ou un override)."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.extract import load_few_shots, load_prompt

logger = logging.getLogger(__name__)

MatchSource = Literal["override", "filename", "heading", "default"]

_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(.+)$")


@dataclass(frozen=True)
class ExtractSchema:
    """Prompt composé + few-shots prêts pour `langextract.extract`."""

    profile_id: str
    prompt: str
    examples: list[Any]
    few_shots_path: Path
    match_source: MatchSource
    source_name: str


@dataclass(frozen=True)
class _ProfileDef:
    id: str
    addendum: Path
    few_shots: Path
    filename: tuple[re.Pattern[str], ...]
    heading: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class _Catalog:
    default_id: str
    base_prompt: Path
    profiles: tuple[_ProfileDef, ...]
    by_id: dict[str, _ProfileDef]


def _fold(text: str) -> str:
    """Minuscules sans accents, pour matcher « ÉES » comme « ees »."""
    stripped = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in stripped if not unicodedata.combining(ch)).lower()


def _compile_patterns(raw: object, *, field: str, profile_id: str) -> tuple[re.Pattern[str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise ValueError(f"Profile « {profile_id} » : « {field} » must be a list of non-empty strings")
    compiled: list[re.Pattern[str]] = []
    for item in raw:
        try:
            compiled.append(re.compile(item, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(
                f"Profile « {profile_id} » : invalid {field} regex {item!r}: {exc}"
            ) from exc
    return tuple(compiled)


def _resolve_under(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    return path


def _first_heading(markdown: str) -> str:
    match = _HEADING_RE.search(markdown[:8000])
    return match.group(1).strip() if match else ""


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    if not text or not patterns:
        return False
    folded = _fold(text)
    return any(pattern.search(folded) for pattern in patterns)


@lru_cache(maxsize=8)
def _load_catalog(profiles_path: str) -> _Catalog:
    path = Path(profiles_path)
    if not path.is_file():
        raise FileNotFoundError(f"LangExtract profiles file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in LangExtract profiles file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"LangExtract profiles file must be a JSON object: {path}")

    default_id = raw.get("default")
    if not isinstance(default_id, str) or not default_id.strip():
        raise ValueError(f"LangExtract profiles file needs a non-empty « default » id: {path}")
    default_id = default_id.strip()

    root = path.parent
    base_raw = raw.get("base_prompt", "prompt.base.txt")
    if not isinstance(base_raw, str) or not base_raw.strip():
        raise ValueError(f"LangExtract profiles file needs « base_prompt »: {path}")
    base_prompt = _resolve_under(root, base_raw.strip())

    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ValueError(f"LangExtract profiles file needs a non-empty « profiles » list: {path}")

    profiles: list[_ProfileDef] = []
    by_id: dict[str, _ProfileDef] = {}
    for i, item in enumerate(profiles_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Profile #{i} must be an object in {path}")
        profile_id = item.get("id")
        addendum = item.get("addendum")
        few_shots = item.get("few_shots")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError(f"Profile #{i} needs « id » in {path}")
        if not isinstance(addendum, str) or not addendum.strip():
            raise ValueError(f"Profile « {profile_id} » needs « addendum » in {path}")
        if not isinstance(few_shots, str) or not few_shots.strip():
            raise ValueError(f"Profile « {profile_id} » needs « few_shots » in {path}")
        profile_id = profile_id.strip()
        if profile_id in by_id:
            raise ValueError(f"Duplicate LangExtract profile id « {profile_id} » in {path}")
        defined = _ProfileDef(
            id=profile_id,
            addendum=_resolve_under(root, addendum.strip()),
            few_shots=_resolve_under(root, few_shots.strip()),
            filename=_compile_patterns(item.get("filename"), field="filename", profile_id=profile_id),
            heading=_compile_patterns(item.get("heading"), field="heading", profile_id=profile_id),
        )
        profiles.append(defined)
        by_id[profile_id] = defined

    if default_id not in by_id:
        raise ValueError(f"Default profile « {default_id} » is not in {path}")
    return _Catalog(
        default_id=default_id,
        base_prompt=base_prompt,
        profiles=tuple(profiles),
        by_id=by_id,
    )


def _compose_prompt(base: str, addendum: str) -> str:
    parts = [base.strip(), addendum.strip()]
    return "\n\n".join(part for part in parts if part)


def _profiles_path(settings: Settings) -> Path | None:
    if settings.langextract_profiles_file is not None:
        return settings.langextract_profiles_file
    candidate = settings.project_root / "config" / "langextract" / "profiles.json"
    return candidate if candidate.is_file() else None


def _legacy_schema(settings: Settings, *, source_name: str, match_source: MatchSource) -> ExtractSchema:
    """Repli : `LANGEXTRACT_PROMPT_FILE` + `LANGEXTRACT_FEW_SHOTS_FILE`."""
    prompt = load_prompt(settings.langextract_prompt_file)
    examples = load_few_shots(settings.langextract_few_shots_file)
    return ExtractSchema(
        profile_id="legacy",
        prompt=prompt,
        examples=examples,
        few_shots_path=settings.langextract_few_shots_file,
        match_source=match_source,
        source_name=source_name,
    )


def _build_schema(
    catalog: _Catalog,
    profile: _ProfileDef,
    *,
    match_source: MatchSource,
    source_name: str,
) -> ExtractSchema:
    prompt = _compose_prompt(load_prompt(catalog.base_prompt), load_prompt(profile.addendum))
    examples = load_few_shots(profile.few_shots)
    return ExtractSchema(
        profile_id=profile.id,
        prompt=prompt,
        examples=examples,
        few_shots_path=profile.few_shots,
        match_source=match_source,
        source_name=source_name,
    )


def _log_classification(
    profile_id: str,
    source_name: str,
    match_source: MatchSource,
) -> None:
    """Journalise le type retenu et le nom du fichier (contrôle visuel à l'ingest)."""
    if match_source == "default":
        logger.warning(
            "LangExtract type=%s  fichier=« %s »  source=%s "
            "— aucun motif de profil n'a matché ; schéma générique utilisé.",
            profile_id,
            source_name,
            match_source,
        )
        return
    logger.info(
        "LangExtract type=%s  fichier=« %s »  source=%s",
        profile_id,
        source_name,
        match_source,
    )


def resolve_extract_schema(
    source_path: Path,
    *,
    markdown: str | None = None,
    override: str | None = None,
    settings: Settings | None = None,
) -> ExtractSchema:
    """Choisit le profil LangExtract (override > nom de fichier > heading > default).

    Chaque résolution est journalisée (type de profil + nom du fichier) pour
    contrôle visuel à l'ingest. Un nom sans motif connu déclenche un warning
    et le profil `default` (sauf si `override` est fourni).

    Args:
        source_path: Chemin du PDF (seul le nom est inspecté).
        markdown: Markdown déjà parsé ; le premier heading sert de repli.
        override: Identifiant de profil (`--profile` : `ees_phase_1`,
            `ees_phase_2`, `default`) ; gagne toujours. Même id que `doc_type`.
        settings: Config ; `.env` si omis.

    Returns:
        Schéma (prompt composé + exemples) à passer à LangExtract.

    Raises:
        FileNotFoundError: `profiles.json` ou un fichier de profil absent.
        ValueError: JSON invalide, profil inconnu, regex cassée.
    """
    s = settings or load_settings()
    source_name = source_path.name
    profiles_path = _profiles_path(s)

    if profiles_path is None:
        schema = _legacy_schema(s, source_name=source_name, match_source="default")
        _log_classification(schema.profile_id, source_name, schema.match_source)
        return schema

    catalog = _load_catalog(str(profiles_path.resolve()))

    if override is not None and override.strip():
        profile_id = override.strip()
        profile = catalog.by_id.get(profile_id)
        if profile is None:
            known = ", ".join(catalog.by_id)
            raise ValueError(
                f"Unknown LangExtract profile « {profile_id} ». Known: {known}"
            )
        schema = _build_schema(
            catalog, profile, match_source="override", source_name=source_name
        )
        _log_classification(schema.profile_id, source_name, schema.match_source)
        return schema

    for profile in catalog.profiles:
        if _matches(profile.filename, source_name):
            schema = _build_schema(
                catalog, profile, match_source="filename", source_name=source_name
            )
            _log_classification(schema.profile_id, source_name, schema.match_source)
            return schema

    heading = _first_heading(markdown or "")
    if heading:
        for profile in catalog.profiles:
            if _matches(profile.heading, heading):
                schema = _build_schema(
                    catalog, profile, match_source="heading", source_name=source_name
                )
                _log_classification(schema.profile_id, source_name, schema.match_source)
                return schema

    default = catalog.by_id[catalog.default_id]
    schema = _build_schema(
        catalog, default, match_source="default", source_name=source_name
    )
    _log_classification(schema.profile_id, source_name, schema.match_source)
    return schema
