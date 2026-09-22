"""Schéma LangExtract : chargement des fichiers de config et appel LLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_ingestion.config import Settings, load_settings
from rag_ingestion.models import GroundedExtraction

if TYPE_CHECKING:
    from rag_ingestion.extract_profile import ExtractSchema

logger = logging.getLogger(__name__)


def load_prompt(path: Path) -> str:
    """Lit le prompt LangExtract (texte brut UTF-8).

    Args:
        path: Fichier pointé par `LANGEXTRACT_PROMPT_FILE`.

    Returns:
        Description de prompt non vide.

    Raises:
        FileNotFoundError: Fichier absent.
        ValueError: Fichier vide.
    """
    if not path.is_file():
        raise FileNotFoundError(f"LangExtract prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"LangExtract prompt file is empty: {path}")
    return text


def load_few_shots(path: Path) -> list[Any]:
    """Charge les few-shots JSON → objets `langextract.data.ExampleData`.

    Format attendu (liste d'exemples) ::

        [
          {
            "text": "...",
            "extractions": [
              {
                "extraction_class": "title",
                "extraction_text": "...",
                "attributes": { "normalized": "..." }
              }
            ]
          }
        ]

    Args:
        path: Fichier pointé par `LANGEXTRACT_FEW_SHOTS_FILE`
            (alias : `LANGEXTRACT_FEW_SHOTS_EXAMPLE`).

    Returns:
        Liste d'exemples LangExtract.

    Raises:
        FileNotFoundError: Fichier absent.
        ValueError: JSON invalide ou extractions mal formées.
    """
    import langextract as lx

    if not path.is_file():
        raise FileNotFoundError(f"LangExtract few-shots file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in LangExtract few-shots file {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"LangExtract few-shots must be a non-empty JSON list: {path}")

    examples: list[Any] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Few-shot #{i} must be an object in {path}")
        text = item.get("text")
        extractions_raw = item.get("extractions")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Few-shot #{i} needs a non-empty string « text » in {path}")
        if not isinstance(extractions_raw, list) or not extractions_raw:
            raise ValueError(f"Few-shot #{i} needs a non-empty « extractions » list in {path}")
        extractions: list[Any] = []
        for j, ext in enumerate(extractions_raw):
            if not isinstance(ext, dict):
                raise ValueError(f"Few-shot #{i} extraction #{j} must be an object in {path}")
            cls = ext.get("extraction_class")
            etext = ext.get("extraction_text")
            if not isinstance(cls, str) or not cls.strip():
                raise ValueError(
                    f"Few-shot #{i} extraction #{j} needs « extraction_class » in {path}"
                )
            if not isinstance(etext, (str, int, float)):
                raise ValueError(
                    f"Few-shot #{i} extraction #{j} needs string/number "
                    f"« extraction_text » in {path}"
                )
            attrs = ext.get("attributes") or {}
            if attrs is not None and not isinstance(attrs, dict):
                raise ValueError(
                    f"Few-shot #{i} extraction #{j} « attributes » must be an object in {path}"
                )
            extractions.append(
                lx.data.Extraction(
                    extraction_class=cls.strip(),
                    extraction_text=etext if isinstance(etext, str) else str(etext),
                    attributes=dict(attrs) if attrs else None,
                )
            )
        examples.append(lx.data.ExampleData(text=text, extractions=extractions))
    return examples


def _interval(extraction: Any) -> tuple[int | None, int | None]:
    """Lit `char_interval` LangExtract ; `(None, None)` si non ancré."""
    interval = getattr(extraction, "char_interval", None)
    if interval is None:
        return None, None
    start = getattr(interval, "start_pos", None)
    end = getattr(interval, "end_pos", None)
    if start is None or end is None:
        return None, None
    return int(start), int(end)


def _to_grounded(result: Any) -> list[GroundedExtraction]:
    """Convertit le résultat LangExtract en liste d'extractions internes."""
    out: list[GroundedExtraction] = []
    extractions = getattr(result, "extractions", None) or []
    for i, ext in enumerate(extractions):
        start, end = _interval(ext)
        attrs = dict(ext.attributes or {}) if getattr(ext, "attributes", None) else {}
        cls = str(getattr(ext, "extraction_class", "") or "")
        text = str(getattr(ext, "extraction_text", "") or "")
        out.append(
            GroundedExtraction(
                extraction_id=f"{cls}:{i}:{start if start is not None else 'u'}",
                extraction_class=cls,
                extraction_text=text,
                start=start,
                end=end,
                attributes=attrs,
            )
        )
    return out


def extract_structured(
    markdown: str,
    *,
    document_id: str,
    source_path: Path,
    settings: Settings | None = None,
    profile_override: str | None = None,
    schema: ExtractSchema | None = None,
) -> list[GroundedExtraction]:
    """Extrait des métadonnées via LangExtract et sauve JSONL + HTML.

    Le LLM (Ollama, OpenAI ou Gemini) est choisi par `LLM_PROVIDER` /
    `LANGEXTRACT_PROVIDER`. Le prompt et les few-shots viennent du profil
    résolu (nom de fichier, heading Markdown, ou `--profile`).
    Voir `config/langextract/profiles.json`.

    Args:
        markdown: Texte source (même chaîne que celle chunkée ensuite).
        document_id: Nom des artefacts `data/extractions/{id}.jsonl|.html`.
        source_path: PDF d'origine ; son nom sélectionne le profil.
        settings: Config LLM / chemins ; `.env` si omis.
        profile_override: Identifiant de profil (CLI `--profile`).
        schema: Schéma déjà résolu ; sinon calculé ici.

    Returns:
        Extractions (classe, texte, offsets si le modèle a ancré). Liste vide
        si le Markdown est vide.
    """
    s = settings or load_settings()
    if not markdown.strip():
        return []

    import langextract as lx

    from rag_ingestion.extract_profile import resolve_extract_schema
    from rag_ingestion.llm_providers import langextract_extract_kwargs, resolve_extract_llm

    resolved = schema or resolve_extract_schema(
        source_path,
        markdown=markdown,
        override=profile_override,
        settings=s,
    )
    prompt = resolved.prompt
    examples = resolved.examples
    extract_llm = resolve_extract_llm(
        langextract_model=s.langextract_model,
        ollama_base_url=s.ollama_base_url,
    )
    logger.info(
        "  Calling LangExtract provider=%s model=%s (%s characters, "
        "max_char_buffer=%s, timeout %ss)…",
        extract_llm.provider,
        extract_llm.model,
        len(markdown),
        s.langextract_max_char_buffer,
        int(s.langextract_timeout_seconds),
    )
    logger.info(
        "  LangExtract type=%s  fichier=« %s »  source=%s · few-shots=%s (%s example(s))",
        resolved.profile_id,
        resolved.source_name,
        resolved.match_source,
        resolved.few_shots_path,
        len(examples),
    )
    result = lx.extract(
        text_or_documents=markdown,
        prompt_description=prompt,
        examples=examples,
        **langextract_extract_kwargs(
            extract_llm,
            timeout_seconds=s.langextract_timeout_seconds,
            max_char_buffer=s.langextract_max_char_buffer,
        ),
    )

    s.extractions_dir.mkdir(parents=True, exist_ok=True)
    jsonl_name = f"{document_id}.jsonl"
    try:
        lx.io.save_annotated_documents(
            [result],
            output_name=jsonl_name,
            output_dir=str(s.extractions_dir),
        )
        jsonl_path = s.extractions_dir / jsonl_name
        html = lx.visualize(str(jsonl_path))
        html_path = s.extractions_dir / f"{document_id}.html"
        html_path.write_text(
            html if isinstance(html, str) else getattr(html, "data", str(html)),
            encoding="utf-8",
        )
        logger.info("  Artifacts: %s and %s", jsonl_path, html_path)
    except Exception:
        logger.exception("Could not save LangExtract visualization document_id=%s", document_id)

    grounded = _to_grounded(result)
    if not grounded:
        logger.warning("  No entities extracted.")
    return grounded
