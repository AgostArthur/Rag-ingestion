"""Client LLM : serveur compatible OpenAI (llama-server / llama.cpp).

`ChatOpenAI` est le client HTTP du protocole `/v1/chat/completions`.
L'inférence reste locale (llama-server) ; aucun appel au cloud OpenAI.
"""

from __future__ import annotations

from chatbot.config import ChatSettings, load_chat_settings


def build_chat_model(settings: ChatSettings | None = None):
    """Retourne un chat model LangChain branché sur llama-server.

    Args:
        settings: Config chatbot ; `.env` si omis.

    Returns:
        Instance `ChatOpenAI` (client OpenAI-compatible) prête pour `bind_tools`.
    """
    from langchain_openai import ChatOpenAI

    s = settings or load_chat_settings()
    return ChatOpenAI(
        base_url=s.llama_server_base_url,
        api_key=s.llama_server_api_key,
        model=s.llama_server_model,
        temperature=s.temperature,
    )
