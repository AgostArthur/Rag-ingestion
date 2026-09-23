"""API HTTP du chatbot : `POST /chat`, lecture catalog, `GET /health`."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from chatbot.config import ChatSettings, load_chat_settings
from chatbot.focus import envelope_from_result
from chatbot.graph import build_graph, last_message_text, recursion_limit_for
from chatbot.timing import invoke_turn
from chatbot.health import collect_health
from chatbot.tools import ui_document_id, ui_site_id


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None
    site_id: str | None = None
    document_id: str | None = None
    context: list[str] = Field(default_factory=list)


class Focus(BaseModel):
    document_ids: list[str]
    project_ids: list[str]
    site_ids: list[str]


class TurnTiming(BaseModel):
    total_seconds: float
    steps: dict[str, float] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    focus: Focus
    documents: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    timing: TurnTiming


def message_with_context(message: str, context: list[str] | None) -> str:
    """Préfixe la question avec les chips de la barre de contexte.

    Le texte saisi par l'utilisateur reste inchangé côté UI. Seul le message
    envoyé au graphe porte le fichier et la section sélectionnés.
    """
    lines = [item.strip() for item in (context or []) if isinstance(item, str) and item.strip()]
    if not lines:
        return message
    block = "\n".join(f"- {line}" for line in lines)
    return f"Contexte sélectionné dans l'interface:\n{block}\n\nQuestion: {message}"


def create_app(settings: ChatSettings | None = None):
    """Application FastAPI branchée sur le graphe LangGraph.

    Args:
        settings: Config chatbot ; `.env` si omis.
    """
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    s = settings or load_chat_settings()
    graph = build_graph(s)

    app = FastAPI(title="RAG chat", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.get("/sites")
    def sites(
        bbox: str | None = Query(default=None),
        geojson: bool = Query(default=False),
    ) -> Any:
        from rag_ingestion.catalog import list_sites, parse_bbox, sites_geojson

        try:
            box = parse_bbox(bbox)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rows = list_sites(bbox=box)
        if geojson:
            return sites_geojson(rows)
        return {"sites": rows}

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
        from rag_ingestion.catalog import get_site, get_site_timeline, list_site_documents

        row = get_site(site_id)
        if row is None:
            raise HTTPException(status_code=404, detail="site not found")
        return {
            "site_id": site_id,
            "documents": list_site_documents(site_id),
            "events": get_site_timeline(site_id),
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest = Body()) -> Any:
        thread_id = payload.thread_id or str(uuid.uuid4())
        site_token = ui_site_id.set((payload.site_id or "").strip())
        doc_token = ui_document_id.set((payload.document_id or "").strip())
        try:
            result, timing = invoke_turn(
                graph,
                message_with_context(payload.message, payload.context),
                thread_id=thread_id,
                recursion_limit=recursion_limit_for(s.max_tool_calls),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)[:2000]) from exc
        finally:
            ui_site_id.reset(site_token)
            ui_document_id.reset(doc_token)
        envelope = envelope_from_result(result)
        return ChatResponse(
            reply=last_message_text(result),
            thread_id=thread_id,
            focus=Focus(**envelope["focus"]),
            documents=envelope["documents"],
            citations=envelope["citations"],
            timing=TurnTiming(**timing),
        )

    return app
