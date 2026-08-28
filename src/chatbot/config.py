"""Configuration du chatbot : fichiers sous `config/chatbot/`, comme LangExtract.

- `config/chatbot/prompt.txt` — prompt système de l'agent
- `config/chatbot/settings.json` — llama-server, température, limites, API

`.env` pointe vers ces fichiers (`CHAT_PROMPT_FILE`, `CHAT_SETTINGS_FILE`)
et peut surcharger une clé JSON (même nom d'env que ci-dessous).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CHAT_PROMPT_FILE = "config/chatbot/prompt.txt"
DEFAULT_CHAT_SETTINGS_FILE = "config/chatbot/settings.json"

_JSON_ENV_KEYS: dict[str, str] = {
    "llama_server_base_url": "LLAMA_SERVER_BASE_URL",
    "llama_server_model": "LLAMA_SERVER_MODEL",
    "llama_server_api_key": "LLAMA_SERVER_API_KEY",
    "temperature": "CHAT_TEMPERATURE",
    "rag_limit": "CHAT_RAG_LIMIT",
    "rag_hit_max_chars": "CHAT_RAG_HIT_MAX_CHARS",
    "max_context_tokens": "CHAT_MAX_CONTEXT_TOKENS",
    "max_tool_calls": "CHAT_MAX_TOOL_CALLS",
    "api_host": "CHAT_API_HOST",
    "api_port": "CHAT_API_PORT",
}


def _project_root() -> Path:
    """Racine du dépôt (dossier qui contient `pyproject.toml`), ou le cwd."""
    here = Path(__file__).resolve().parent
    src = here.parent
    if src.name == "src" and (src.parent / "pyproject.toml").exists():
        return src.parent
    if (here.parent / "pyproject.toml").exists():
        return here.parent
    return Path.cwd()


_PROJECT_ROOT = _project_root()
_ENV_PATH = _PROJECT_ROOT / ".env"


def _load_dotenv_file() -> None:
    """Charge `.env` : les valeurs du fichier priment sur l'environnement."""
    load_dotenv(_ENV_PATH, override=True)


def _resolve_path(raw: str, default: str) -> Path:
    """Chemin relatif à la racine du dépôt, ou absolu tel quel."""
    value = (raw or default).strip() or default
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def load_prompt(path: Path) -> str:
    """Lit le prompt système du chatbot (texte brut UTF-8).

    Args:
        path: Fichier pointé par `CHAT_PROMPT_FILE`.

    Returns:
        Prompt non vide.

    Raises:
        FileNotFoundError: Fichier absent.
        ValueError: Fichier vide.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Chatbot prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Chatbot prompt file is empty: {path}")
    return text


def load_settings_file(path: Path) -> dict:
    """Charge `config/chatbot/settings.json`.

    Args:
        path: Fichier pointé par `CHAT_SETTINGS_FILE`.

    Returns:
        Objet JSON (dict).

    Raises:
        FileNotFoundError: Fichier absent.
        ValueError: JSON invalide ou pas un objet.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Chatbot settings file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in chatbot settings file {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Chatbot settings must be a non-empty JSON object: {path}")
    return raw


def _str_from(data: dict, key: str, default: str) -> str:
    env_key = _JSON_ENV_KEYS[key]
    env_val = os.getenv(env_key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    raw = data.get(key, default)
    text = str(raw).strip() if raw is not None else ""
    return text or default


def _int_from(data: dict, key: str, default: int, *, minimum: int) -> int:
    env_key = _JSON_ENV_KEYS[key]
    env_val = os.getenv(env_key)
    source: object = env_val if env_val is not None and env_val.strip() else data.get(key, default)
    try:
        return max(minimum, int(source))
    except (TypeError, ValueError):
        return default


def _float_from(data: dict, key: str, default: float) -> float:
    env_key = _JSON_ENV_KEYS[key]
    env_val = os.getenv(env_key)
    source: object = env_val if env_val is not None and env_val.strip() else data.get(key, default)
    try:
        return float(source)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ChatSettings:
    """Paramètres runtime du chatbot (fichiers `config/chatbot/` + `.env`)."""

    llama_server_base_url: str
    llama_server_model: str
    llama_server_api_key: str
    temperature: float
    rag_limit: int
    rag_hit_max_chars: int
    max_context_tokens: int
    max_tool_calls: int
    api_host: str
    api_port: int
    prompt_file: Path
    settings_file: Path
    project_root: Path


def load_chat_settings() -> ChatSettings:
    """Lit `.env` + `config/chatbot/settings.json` et retourne un `ChatSettings`."""
    _load_dotenv_file()
    prompt_file = _resolve_path(os.getenv("CHAT_PROMPT_FILE", ""), DEFAULT_CHAT_PROMPT_FILE)
    settings_file = _resolve_path(
        os.getenv("CHAT_SETTINGS_FILE", ""), DEFAULT_CHAT_SETTINGS_FILE
    )
    data = load_settings_file(settings_file)
    return ChatSettings(
        llama_server_base_url=_str_from(
            data, "llama_server_base_url", "http://localhost:8080/v1"
        ).rstrip("/")
        or "http://localhost:8080/v1",
        llama_server_model=_str_from(data, "llama_server_model", "local"),
        llama_server_api_key=_str_from(data, "llama_server_api_key", "sk-no-key-required"),
        temperature=_float_from(data, "temperature", 0.2),
        rag_limit=_int_from(data, "rag_limit", 5, minimum=1),
        rag_hit_max_chars=_int_from(data, "rag_hit_max_chars", 600, minimum=100),
        max_context_tokens=_int_from(data, "max_context_tokens", 48000, minimum=2000),
        max_tool_calls=_int_from(data, "max_tool_calls", 2, minimum=1),
        api_host=_str_from(data, "api_host", "127.0.0.1"),
        api_port=_int_from(data, "api_port", 8000, minimum=1),
        prompt_file=prompt_file,
        settings_file=settings_file,
        project_root=_PROJECT_ROOT,
    )
