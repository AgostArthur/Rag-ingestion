"""Erreurs LLM / invoke : messages API et REPL."""

from __future__ import annotations


def _text(exc: BaseException) -> str:
    return str(exc).lower()


def llm_is_unreachable(exc: BaseException) -> bool:
    """True si le client n'a pas atteint le serveur OpenAI-compatible."""
    text = _text(exc)
    needles = (
        "connection refused",
        "connecterror",
        "apiconnectionerror",
        "connection error",
        "nodename nor servname",
        "name or service not known",
        "failed to establish",
        "max retries exceeded",
        "connection reset",
        "timed out",
        "timeout",
        "connect timeout",
        "winerror 10061",
        "errno 111",
    )
    return any(n in text for n in needles)


def context_overflow(exc: BaseException) -> bool:
    """True si le serveur LLM a rejeté la fenêtre de contexte."""
    text = _text(exc)
    return "exceed_context_size" in text or "exceeds the available context size" in text


def http_status_for_llm_error(exc: BaseException) -> int:
    """400 contexte, 503 LLM down, 502 autres erreurs d'invoke."""
    if context_overflow(exc):
        return 400
    if llm_is_unreachable(exc):
        return 503
    return 502


def format_llm_error(
    exc: BaseException, *, llm_url: str = "", model: str = ""
) -> str:
    """Message lisible (UI + REPL) ; n'expose pas de traceback."""
    if context_overflow(exc):
        return (
            "Fenêtre de contexte saturée (conversation + hits RAG). "
            "Recommencez le fil ou augmentez n_ctx / -c du serveur LLM."
        )
    if llm_is_unreachable(exc):
        target = llm_url or "le serveur LLM"
        extra = f" (modèle {model})" if model else ""
        return (
            f"LLM injoignable à {target}{extra}. "
            "Le chat utilise OPENAI_* ou LLAMA_SERVER_* (API OpenAI), "
            "pas OLLAMA_BASE_URL (LangExtract / ingest). "
            "Ollama chat : OPENAI_BASE_URL=http://localhost:11434/v1 et "
            "OPENAI_MODEL=<tag ollama>. "
            "Depuis Docker, localhost du .env est le conteneur : "
            "host.docker.internal est réécrit automatiquement."
        )
    return f"Échec du tour de chat : {exc}"
