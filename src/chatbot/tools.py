"""Outil LangChain : recherche dans la base RAG via `rag_ingestion.retrieve`."""

from __future__ import annotations

import re
from typing import Any

from chatbot.config import ChatSettings, load_chat_settings

# Cap so the model cannot request dozens of chunks in one tool call.
_MAX_RAG_LIMIT = 10
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_DEFAULT_TABLE_MAX_CHARS = 2400


def is_table_heavy(text: str) -> bool:
    """True si le chunk ressemble à un tableau Markdown (analyses labo, etc.)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    table_lines = sum(1 for line in lines if _TABLE_LINE_RE.match(line))
    if table_lines >= 3:
        return True
    return table_lines / len(lines) >= 0.3


def format_hits(
    hits: list[dict[str, Any]],
    *,
    max_chars: int = 600,
    table_max_chars: int = _DEFAULT_TABLE_MAX_CHARS,
) -> str:
    """Formate les hits Qdrant en markdown lisible par le LLM.

    Les retours à la ligne sont conservés (tableaux). Un chunk tabulaire
    utilise `table_max_chars` au lieu de `max_chars`.

    Args:
        hits: Résultats de `rag_ingestion.retrieve.search`.
        max_chars: Troncature du texte de chaque chunk prosaïque.
        table_max_chars: Troncature plus large pour les tableaux Markdown.

    Returns:
        Bloc markdown, ou un message d'absence de résultat.
    """
    if not hits:
        return "No matching chunks found in the knowledge base."
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        raw = (hit.get("text") or "").strip()
        cap = table_max_chars if is_table_heavy(raw) else max_chars
        text = raw if len(raw) <= cap else raw[:cap] + "…"
        heading = hit.get("heading_path") or ""
        entities = hit.get("entities") or []
        entity_s = ", ".join(str(e) for e in entities) if entities else ""
        source = hit.get("source_path") or ""
        if "/" in source:
            source = source.rsplit("/", 1)[-1]
        lines = [
            f"### Hit {i} (score={float(hit.get('score') or 0):.4f})",
            f"- source: {source}",
            f"- page: {hit.get('page')}",
            f"- doc_type: {hit.get('doc_type') or ''}",
            f"- document_id: {hit.get('document_id') or ''}",
            f"- project_id: {hit.get('project_id') or ''}",
            f"- site_id: {hit.get('site_id') or ''}",
            f"- chunk_id: {hit.get('chunk_id') or ''}",
        ]
        if heading:
            lines.append(f"- section: {heading}")
        if entity_s:
            lines.append(f"- entities: {entity_s}")
        lines.append("")
        lines.append(text)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _filters_from_args(
    doc_type: str = "",
    entities: str = "",
    document_id: str = "",
    site_id: str = "",
    project_id: str = "",
) -> dict[str, str]:
    """Construit le dict de filtres Qdrant à partir des arguments d'outil."""
    filters: dict[str, str] = {}
    if doc_type.strip():
        filters["doc_type"] = doc_type.strip()
    if entities.strip():
        filters["entities"] = entities.strip()
    if document_id.strip():
        filters["document_id"] = document_id.strip()
    if site_id.strip():
        filters["site_id"] = site_id.strip()
    if project_id.strip():
        filters["project_id"] = project_id.strip()
    return filters


def build_search_tool(settings: ChatSettings | None = None):
    """Outil `search_knowledge` qui appelle `rag_ingestion.retrieve.search`.

    Args:
        settings: Config chatbot (limite par défaut) ; `.env` si omis.
    """
    from langchain_core.tools import tool

    from rag_ingestion.retrieve import search as retrieve_search

    chat = settings or load_chat_settings()
    default_limit = chat.rag_limit
    hit_max_chars = chat.rag_hit_max_chars
    table_max_chars = chat.rag_table_max_chars

    @tool
    def search_knowledge(
        query: str,
        doc_type: str = "",
        entities: str = "",
        document_id: str = "",
        site_id: str = "",
        project_id: str = "",
        limit: int = 0,
    ) -> str:
        """Search the local RAG knowledge base for relevant document chunks.

        Call this tool for factual questions about ingested documents.
        For a project number (e.g. 2259, 4405), pass it in `entities` or
        `project_id` and include it in the query. Hybrid search also matches
        the number in chunk text if LangExtract missed it.

        Args:
            query: Natural-language search phrase (not keywords only).
            doc_type: Optional payload filter (e.g. rapport, ees_phase_2).
            entities: Optional exact entity filter; comma-separated names
                or project numbers (e.g. 4405).
            document_id: Optional SHA-256 document id to restrict the search.
            site_id: Optional catalog site id (e.g. lot:2363352).
            project_id: Optional project number filter.
            limit: Max chunks to return; 0 uses the configured default.
        """
        n = limit if limit and limit > 0 else default_limit
        n = max(1, min(int(n), _MAX_RAG_LIMIT))
        hits = retrieve_search(
            query,
            limit=n,
            filters=_filters_from_args(
                doc_type, entities, document_id, site_id, project_id
            )
            or None,
        )
        return format_hits(
            hits,
            max_chars=hit_max_chars,
            table_max_chars=table_max_chars,
        )

    return search_knowledge
