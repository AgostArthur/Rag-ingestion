"""Chatbot LangGraph : RAG en outil, LLM via un serveur OpenAI-compatible (llama-server)."""

__all__ = ["build_graph"]


def __getattr__(name: str):
    """Import paresseux pour ne pas charger LangGraph au `import chatbot`."""
    if name == "build_graph":
        from chatbot.graph import build_graph

        return build_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
