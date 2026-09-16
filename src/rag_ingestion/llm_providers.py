"""Presets LLM : `LLM_PROVIDER` = local | ollama | openai | gemini.

Le RAG (FastEmbed + Qdrant) reste local. Chat et LangExtract suivent
`LLM_PROVIDER`, sauf `CHAT_PROVIDER` / `LANGEXTRACT_PROVIDER`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from rag_ingestion.config import rewrite_loopback_url

logger = logging.getLogger(__name__)

LOCAL = "local"
OLLAMA = "ollama"
OPENAI = "openai"
GEMINI = "gemini"

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_LLAMA_SERVER_URL = "http://localhost:8080/v1"

_PLACEHOLDER_KEY = "sk-no-key-required"


def _get(*keys: str) -> str | None:
    for key in keys:
        raw = os.getenv(key)
        if raw and raw.strip():
            return raw.strip()
    return None


def _provider(*keys: str, default: str) -> str:
    name = (_get(*keys) or default).strip().lower()
    if name not in {LOCAL, OLLAMA, OPENAI, GEMINI}:
        raise ValueError(
            f"Unknown LLM provider {name!r}. Use local, ollama, openai, or gemini."
        )
    return name


@dataclass(frozen=True)
class ChatLLM:
    """Paramètres chat : OpenAI-compat (`local`/`ollama`/`openai`) ou Gemini natif."""

    provider: str
    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True)
class ExtractLLM:
    """Paramètres LangExtract (`lx.extract`)."""

    provider: str
    model: str
    api_key: str | None
    model_url: str | None
    openai_base_url: str | None
    fence_output: bool | None
    use_schema_constraints: bool


def resolve_chat_llm(
    *,
    json_provider: str | None = None,
    json_base_url: str | None = None,
    json_model: str | None = None,
    json_api_key: str | None = None,
) -> ChatLLM:
    """URL / modèle / clé du chatbot selon `CHAT_PROVIDER` ou `LLM_PROVIDER`."""
    provider = _provider("CHAT_PROVIDER", "LLM_PROVIDER", default=json_provider or LOCAL)
    ollama_root = rewrite_loopback_url(
        (_get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    )

    if provider == GEMINI:
        url = _get("GEMINI_BASE_URL") or GEMINI_OPENAI_BASE
        model = _get("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        key = _get("GEMINI_API_KEY", "GOOGLE_API_KEY") or _PLACEHOLDER_KEY
    elif provider == OPENAI:
        url = _get("OPENAI_BASE_URL") or OPENAI_API_BASE
        model = _get("OPENAI_MODEL") or "gpt-4o-mini"
        key = _get("OPENAI_API_KEY") or _PLACEHOLDER_KEY
    elif provider == OLLAMA:
        url = _get("LLAMA_SERVER_BASE_URL") or f"{ollama_root}/v1"
        model = _get("OLLAMA_CHAT_MODEL", "LANGEXTRACT_MODEL") or json_model or "llama3.1:8b"
        key = _get("LLAMA_SERVER_API_KEY") or _PLACEHOLDER_KEY
    else:
        url = _get("LLAMA_SERVER_BASE_URL") or json_base_url or DEFAULT_LLAMA_SERVER_URL
        model = _get("LLAMA_SERVER_MODEL") or json_model or "local"
        key = _get("LLAMA_SERVER_API_KEY") or json_api_key or _PLACEHOLDER_KEY

    if provider in {OPENAI, GEMINI} and key == _PLACEHOLDER_KEY:
        logger.warning(
            "Chat provider=%s has no API key. Set %s.",
            provider,
            "GEMINI_API_KEY" if provider == GEMINI else "OPENAI_API_KEY",
        )
    return ChatLLM(
        provider=provider,
        base_url=rewrite_loopback_url(url.rstrip("/")),
        model=model,
        api_key=key,
    )


def resolve_extract_llm(*, langextract_model: str, ollama_base_url: str) -> ExtractLLM:
    """Backend LangExtract. `local` (llama-server côté chat) reste Ollama ici."""
    provider = _provider("LANGEXTRACT_PROVIDER", "LLM_PROVIDER", default=OLLAMA)
    if provider == LOCAL:
        provider = OLLAMA
    ollama_url = rewrite_loopback_url(ollama_base_url.rstrip("/"))

    if provider == GEMINI:
        return ExtractLLM(
            provider=provider,
            model=_get("GEMINI_MODEL") or "gemini-3.5-flash-lite",
            api_key=_get("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            model_url=None,
            openai_base_url=None,
            fence_output=None,
            use_schema_constraints=True,
        )
    if provider == OPENAI:
        base = _get("OPENAI_BASE_URL")
        return ExtractLLM(
            provider=provider,
            model=_get("OPENAI_MODEL") or "gpt-4o-mini",
            api_key=_get("OPENAI_API_KEY"),
            model_url=None,
            openai_base_url=rewrite_loopback_url(base.rstrip("/")) if base else None,
            fence_output=True,
            use_schema_constraints=False,
        )
    return ExtractLLM(
        provider=OLLAMA,
        model=_get("LANGEXTRACT_MODEL") or langextract_model,
        api_key=_get("OLLAMA_API_KEY"),
        model_url=ollama_url,
        openai_base_url=None,
        fence_output=False,
        use_schema_constraints=False,
    )


def langextract_extract_kwargs(extract: ExtractLLM, *, timeout_seconds: float) -> dict:
    """Kwargs `lx.extract` (hors texte / prompt / few-shots)."""
    from langextract.factory import ModelConfig

    if extract.provider == OLLAMA:
        lx_provider = "ollama"
        provider_kwargs = {
            "model_url": extract.model_url,
            "base_url": extract.model_url,
            "timeout": int(timeout_seconds),
        }
    elif extract.provider == OPENAI:
        lx_provider = "openai"
        provider_kwargs = {"api_key": extract.api_key}
        if extract.openai_base_url:
            provider_kwargs["base_url"] = extract.openai_base_url
    else:
        lx_provider = "gemini"
        provider_kwargs = {"api_key": extract.api_key}

    kwargs: dict = {
        "config": ModelConfig(
            model_id=extract.model,
            provider=lx_provider,
            provider_kwargs=provider_kwargs,
        ),
        "use_schema_constraints": extract.use_schema_constraints,
        "show_progress": False,
    }
    if extract.fence_output is not None:
        kwargs["fence_output"] = extract.fence_output
    return kwargs
