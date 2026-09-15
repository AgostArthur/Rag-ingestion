"""Chargement de la configuration depuis `.env` à la racine du dépôt.

Tous les défauts vivent ici. Pour changer un paramètre : environnement
(Compose / shell, prioritaire), sinon `.env`, sinon `DEFAULT_*` ci-dessous.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

# --- Défauts : env process > `.env` > ces constantes ---
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "chunks"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_EMBED_FALLBACK_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_EMBED_DIM = 1024
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LANGEXTRACT_MODEL = "nemotron-3-nano:4b"
DEFAULT_LANGEXTRACT_TIMEOUT_SECONDS = 300.0
DEFAULT_LANGEXTRACT_PROMPT_FILE = "config/langextract/prompt.base.txt"
DEFAULT_LANGEXTRACT_FEW_SHOTS_FILE = "config/langextract/profiles/default/few_shots.json"
DEFAULT_LANGEXTRACT_PROFILES_FILE = "config/langextract/profiles.json"
DEFAULT_OCR_LANGUAGE = "fra+eng"
DEFAULT_OCR_HEAVY_RATIO = 0.5
DEFAULT_CHUNK_SIZE_CHARS = 2400
DEFAULT_CHUNK_OVERLAP_CHARS = 300
DEFAULT_DATA_DIR = "data"


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


def running_in_docker() -> bool:
    """True si le process tourne dans un conteneur (fichier `/.dockerenv`)."""
    return Path("/.dockerenv").exists()


def rewrite_loopback_url(url: str, *, in_docker: bool | None = None) -> str:
    """Ajuste l'hôte d'une URL loopback selon le runtime.

    En local, `host.docker.internal` devient `localhost`.
    Dans Docker, `localhost` / `127.0.0.1` deviennent `host.docker.internal`
    (Ollama / llama-server sur la machine hôte). Les autres hôtes
    (`qdrant`, `ollama`, `api.openai.com`, …) restent inchangés.
    """
    text = (url or "").strip()
    if not text:
        return text
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if not host:
        return text
    docker = running_in_docker() if in_docker is None else in_docker
    if docker and host in {"localhost", "127.0.0.1"}:
        new_host = "host.docker.internal"
    elif not docker and host == "host.docker.internal":
        new_host = "localhost"
    else:
        return text
    port = parsed.port
    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    netloc = f"{auth}{new_host}" + (f":{port}" if port else "")
    return urlunparse(parsed._replace(netloc=netloc))


def _load_dotenv_file() -> None:
    """Charge `.env` sans écraser les variables déjà posées (Compose, shell)."""
    load_dotenv(_ENV_PATH, override=False)


def _int_env(key: str, default: int, *, minimum: int) -> int:
    """Entier d'environnement, borné inférieurement ; `default` si invalide."""
    try:
        return max(minimum, int(os.getenv(key, str(default)).strip()))
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    """Flottant d'environnement ; `default` si invalide ou absent."""
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(key: str, default: bool) -> bool:
    """Booléen d'environnement (`1`/`true`/`yes`/`on`)."""
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Paramètres runtime (Qdrant, embeddings, Ollama, OCR, chunking, disque)."""

    qdrant_url: str
    qdrant_collection: str
    embed_model: str
    embed_dim: int
    ollama_base_url: str
    langextract_model: str
    langextract_timeout_seconds: float
    langextract_prompt_file: Path
    langextract_few_shots_file: Path
    ocr_language: str
    ocr_server_url: str | None
    ocr_heavy_ratio: float
    chunk_size_chars: int
    chunk_overlap_chars: int
    data_dir: Path
    project_root: Path
    incoming_dir: Path | None = None
    archive_dir: Path | None = None
    failed_dir: Path | None = None
    ingest_poll_seconds: int = 10
    ingest_skip_existing: bool = True
    ingest_skip_extract: bool = False
    langextract_profiles_file: Path | None = None

    @property
    def parsed_dir(self) -> Path:
        """Dossier des Markdown parsés (`{document_id}.md`)."""
        return self.data_dir / "parsed"

    @property
    def extractions_dir(self) -> Path:
        """Dossier des artefacts LangExtract (JSONL + HTML)."""
        return self.data_dir / "extractions"

    @property
    def documents_dir(self) -> Path:
        """Dossier des fiches parent JSON (métadonnées document)."""
        return self.data_dir / "documents"

    @property
    def catalog_path(self) -> Path:
        """Fichier SQLite du catalog métier (`sites` / `documents` / `events`)."""
        return self.data_dir / "catalog.sqlite"

    def resolved_incoming_dir(self) -> Path:
        """Dossier d'arrivée des PDF (drop folder)."""
        return self.incoming_dir or (self.project_root / "incoming")

    def resolved_archive_dir(self) -> Path:
        """PDF ingérés (ou doublons) après traitement."""
        return self.archive_dir or (self.data_dir / "archive")

    def resolved_failed_dir(self) -> Path:
        """PDF dont l'ingest a échoué."""
        return self.failed_dir or (self.data_dir / "failed")


def load_settings() -> Settings:
    """Lit `.env` et retourne un `Settings` immuable.

    Returns:
        Configuration utilisée par tout le pipeline d'ingestion.
    """
    _load_dotenv_file()
    root = _PROJECT_ROOT
    data_raw = os.getenv("DATA_DIR", DEFAULT_DATA_DIR).strip() or DEFAULT_DATA_DIR
    data_dir = Path(data_raw)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    ocr_server = os.getenv("OCR_SERVER_URL", "").strip() or None

    def _resolve_path(raw: str, default: str) -> Path:
        value = (raw or default).strip() or default
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path

    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip()
        or DEFAULT_QDRANT_COLLECTION,
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL,
        embed_dim=_int_env("EMBED_DIM", DEFAULT_EMBED_DIM, minimum=32),
        ollama_base_url=rewrite_loopback_url(
            os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        ),
        langextract_model=os.getenv("LANGEXTRACT_MODEL", DEFAULT_LANGEXTRACT_MODEL).strip()
        or DEFAULT_LANGEXTRACT_MODEL,
        langextract_timeout_seconds=_float_env(
            "LANGEXTRACT_TIMEOUT_SECONDS", DEFAULT_LANGEXTRACT_TIMEOUT_SECONDS
        ),
        langextract_prompt_file=_resolve_path(
            os.getenv("LANGEXTRACT_PROMPT_FILE", ""),
            DEFAULT_LANGEXTRACT_PROMPT_FILE,
        ),
        langextract_few_shots_file=_resolve_path(
            os.getenv("LANGEXTRACT_FEW_SHOTS_FILE", "")
            or os.getenv("LANGEXTRACT_FEW_SHOTS_EXAMPLE", ""),
            DEFAULT_LANGEXTRACT_FEW_SHOTS_FILE,
        ),
        langextract_profiles_file=_resolve_path(
            os.getenv("LANGEXTRACT_PROFILES_FILE", ""),
            DEFAULT_LANGEXTRACT_PROFILES_FILE,
        ),
        ocr_language=os.getenv("OCR_LANGUAGE", DEFAULT_OCR_LANGUAGE).strip() or DEFAULT_OCR_LANGUAGE,
        ocr_server_url=ocr_server,
        ocr_heavy_ratio=min(1.0, max(0.0, _float_env("OCR_HEAVY_RATIO", DEFAULT_OCR_HEAVY_RATIO))),
        chunk_size_chars=_int_env("CHUNK_SIZE_CHARS", DEFAULT_CHUNK_SIZE_CHARS, minimum=200),
        chunk_overlap_chars=_int_env(
            "CHUNK_OVERLAP_CHARS", DEFAULT_CHUNK_OVERLAP_CHARS, minimum=0
        ),
        data_dir=data_dir,
        project_root=root,
        incoming_dir=_resolve_path(os.getenv("INCOMING_DIR", ""), "incoming"),
        archive_dir=_resolve_path(os.getenv("ARCHIVE_DIR", ""), str(data_dir / "archive")),
        failed_dir=_resolve_path(os.getenv("FAILED_DIR", ""), str(data_dir / "failed")),
        ingest_poll_seconds=_int_env("INGEST_POLL_SECONDS", 10, minimum=1),
        ingest_skip_existing=_bool_env("INGEST_SKIP_EXISTING", True),
        ingest_skip_extract=_bool_env("INGEST_SKIP_EXTRACT", False),
    )
