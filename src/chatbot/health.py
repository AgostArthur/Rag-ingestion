"""Health-checks : Qdrant, catalog SQLite, LLM chat."""

from __future__ import annotations

from urllib.error import URLError
from urllib.request import Request, urlopen

from chatbot.config import ChatSettings, load_chat_settings
from rag_ingestion.catalog import ping_catalog
from rag_ingestion.config import load_settings
from rag_ingestion.qdrant_store import ping_qdrant


def ping_http(
    url: str,
    timeout: float = 2.0,
    api_key: str | None = None,
) -> tuple[bool, str]:
    """GET `url` ; (ok, détail). Envoie `Authorization` si une clé API est fournie."""
    headers: dict[str, str] = {}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        request = Request(url, method="GET", headers=headers)
        with urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            ok = 200 <= status < 400
            return ok, f"http {status}"
    except URLError as exc:
        return False, str(exc.reason if getattr(exc, "reason", None) else exc)
    except Exception as exc:
        return False, str(exc)


def llama_models_url(base_url: str) -> str:
    """`…/v1` → `…/v1/models` (OpenAI-compatible)."""
    return base_url.rstrip("/") + "/models"


def collect_health(chat: ChatSettings | None = None) -> dict[str, object]:
    """Rapport de santé pour `GET /health` (n'écrit rien)."""
    chat_settings = chat or load_chat_settings()
    ingest = load_settings()
    q_ok, q_detail = ping_qdrant(ingest)
    c_ok, c_detail = ping_catalog(ingest)
    l_ok, l_detail = ping_http(
        llama_models_url(chat_settings.llama_server_base_url),
        api_key=chat_settings.llama_server_api_key,
    )
    overall = "ok" if q_ok else "degraded"
    if not q_ok:
        overall = "error"
    elif not l_ok:
        overall = "degraded"
    llm = {
        "ok": l_ok,
        "detail": l_detail,
        "provider": getattr(chat_settings, "llm_provider", "local"),
        "model": chat_settings.llama_server_model,
        "url": chat_settings.llama_server_base_url,
    }
    return {
        "status": overall,
        "qdrant": {"ok": q_ok, "detail": q_detail, "url": ingest.qdrant_url},
        "catalog": {"ok": c_ok, "detail": c_detail},
        "llm": llm,
        "llama_server": llm,
    }
