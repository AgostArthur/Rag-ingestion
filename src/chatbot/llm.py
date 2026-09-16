"""Client LLM pour le chatbot.

- `local` / `ollama` / `openai` → `ChatOpenAI` (endpoint OpenAI-compatible)
- `gemini` → `ChatGoogleGenerativeAI` (SDK natif)

Gemini 3 exige des `thought_signature` sur les tool calls. L'API
OpenAI-compat de Google + `ChatOpenAI` les perd ; le client natif
(`langchain-google-genai` >= 3.1) les renvoie correctement.
"""

from __future__ import annotations

import logging

from chatbot.config import ChatSettings, load_chat_settings

logger = logging.getLogger(__name__)


def build_chat_model(settings: ChatSettings | None = None):
    """Retourne un chat model LangChain pour le provider configuré.

    Args:
        settings: Config chatbot ; `.env` si omis.

    Returns:
        Instance prête pour `bind_tools` (`ChatOpenAI` ou
        `ChatGoogleGenerativeAI`).
    """
    s = settings or load_chat_settings()
    logger.info(
        "Chat LLM provider=%s model=%s base_url=%s",
        s.llm_provider,
        s.llama_server_model,
        s.llama_server_base_url,
    )

    if s.llm_provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Gemini chat requires langchain-google-genai. "
                "Install with: pip install -e '.[chat]'"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=s.llama_server_model,
            api_key=s.llama_server_api_key,
            temperature=s.temperature,
            top_p=s.top_p,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=s.llama_server_base_url,
        api_key=s.llama_server_api_key,
        model=s.llama_server_model,
        temperature=s.temperature,
        top_p=s.top_p,
    )
