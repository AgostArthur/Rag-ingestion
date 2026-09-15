"""API HTTP du chatbot : `POST /chat`, lecture catalog, `GET /health`."""

from __future__ import annotations

import uuid
from typing import Any

from chatbot.config import ChatSettings, load_chat_settings
from chatbot.focus import envelope_from_result
from chatbot.graph import build_graph, last_message_text, recursion_limit_for
from chatbot.health import collect_health


def create_app(settings: ChatSettings | None = None):
    """Application FastAPI branchée sur le graphe LangGraph.

    Args:
        settings: Config chatbot ; `.env` si omis.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    s = settings or load_chat_settings()
    graph = build_graph(s)

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1)
        thread_id: str | None = None

    class Focus(BaseModel):
        document_ids: list[str]
        project_ids: list[str]
        site_ids: list[str]

    class ChatResponse(BaseModel):
        reply: str
        thread_id: str
        focus: Focus
        documents: list[dict[str, Any]]
        citations: list[dict[str, Any]]

    app = FastAPI(title="RAG chat", version="0.2.0")

    @app.get("/")
    def root() -> dict[str, Any]:
        """Index JSON : le navigateur ouvre `/`, pas `/health`."""
        return {
            "service": "RAG chat",
            "docs": "/docs",
            "health": "/health",
            "chat": "POST /chat",
        }

    @app.get("/favicon.ico")
    def favicon():
        from fastapi.responses import Response

        return Response(status_code=204)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return collect_health(s)

    @app.get("/documents/{document_id}")
    def document(document_id: str) -> dict[str, Any]:
        from rag_ingestion.catalog import get_document

        row = get_document(document_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")
        return row

    @app.get("/sites/{site_id}")
    def site(site_id: str) -> dict[str, Any]:
        from rag_ingestion.catalog import get_site

        row = get_site(site_id)
        if row is None:
            raise HTTPException(status_code=404, detail="site not found")
        return row

    @app.get("/sites/{site_id}/timeline")
    def site_timeline(site_id: str) -> dict[str, Any]:
        from rag_ingestion.catalog import get_site, get_site_timeline

        row = get_site(site_id)
        if row is None:
            raise HTTPException(status_code=404, detail="site not found")
        return {"site_id": site_id, "events": get_site_timeline(site_id)}

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
        envelope = envelope_from_result(result)
        return ChatResponse(
            reply=last_message_text(result),
            thread_id=thread_id,
            focus=Focus(**envelope["focus"]),
            documents=envelope["documents"],
            citations=envelope["citations"],
        )

    return app
