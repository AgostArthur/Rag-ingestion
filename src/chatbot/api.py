"""API HTTP du chatbot : `POST /chat`, `GET /health`."""

from __future__ import annotations

import uuid
from typing import Any

from chatbot.config import ChatSettings, load_chat_settings
from chatbot.graph import build_graph, last_message_text, recursion_limit_for


def create_app(settings: ChatSettings | None = None):
    """Application FastAPI branchée sur le graphe LangGraph.

    Args:
        settings: Config chatbot ; `.env` si omis.
    """
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    s = settings or load_chat_settings()
    graph = build_graph(s)

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1)
        thread_id: str | None = None

    class ChatResponse(BaseModel):
        reply: str
        thread_id: str

    app = FastAPI(title="RAG chat", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> Any:
        thread_id = body.thread_id or str(uuid.uuid4())
        result = graph.invoke(
            {"messages": [{"role": "user", "content": body.message}]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit_for(s.max_tool_calls),
            },
        )
        return ChatResponse(reply=last_message_text(result), thread_id=thread_id)

    return app
