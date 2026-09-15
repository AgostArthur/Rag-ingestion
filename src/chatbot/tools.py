"""Outil LangChain : recherche dans la base RAG via `rag_ingestion.retrieve`."""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

from chatbot.config import ChatSettings, load_chat_settings

# Cap so the model cannot request dozens of chunks in one tool call.
_MAX_RAG_LIMIT = 20
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_DEFAULT_TABLE_MAX_CHARS = 2400
_CATALOG_FIELDS = (
    "title",
    "client",
    "firm",
    "project_id",
    "project_ids",
    "doc_type",
    "address",
    "lot_cadastral",
    "city",
    "report_date",
    "site_id",
)

ui_site_id: ContextVar[str] = ContextVar("ui_site_id", default="")
ui_document_id: ContextVar[str] = ContextVar("ui_document_id", default="")


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
        contaminants = hit.get("contaminants") or []
        if contaminants:
            lines.append(
                "- contaminants: " + ", ".join(str(c) for c in contaminants)
            )
        lines.append("")
        lines.append(text)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _short_source(path: str) -> str:
    text = (path or "").strip()
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def format_catalog_fiches(rows: list[dict[str, Any]]) -> str:
    """Formate les fiches SQLite (client, firme, adresse) pour le LLM."""
    if not rows:
        return ""
    parts = [
        "## Fiches catalog (métadonnées d'ingest : client, firme, adresse, lot, projet)"
    ]
    for row in rows:
        lines = [f"### Document {row.get('document_id') or ''}"]
        source = _short_source(str(row.get("source_path") or ""))
        if source:
            lines.append(f"- source: {source}")
        for key in _CATALOG_FIELDS:
            value = row.get(key)
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value if item)
            if value == "":
                continue
            lines.append(f"- {key}: {value}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def catalog_rows_for_search(
    filters: dict[str, str], hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fiches catalog du focus UI, sinon des documents touchés par les hits."""
    from rag_ingestion.catalog import get_documents, get_site

    ids: list[str] = []
    focused_doc = (filters.get("document_id") or "").strip()
    focused_site = (filters.get("site_id") or "").strip()
    if focused_doc:
        ids.append(focused_doc)
    elif focused_site:
        site = get_site(focused_site)
        if site:
            ids.extend(str(doc_id) for doc_id in (site.get("document_ids") or []))
    for hit in hits:
        doc_id = str(hit.get("document_id") or "").strip()
        if doc_id:
            ids.append(doc_id)
    return get_documents(ids)


def _filters_from_args(
    doc_type: str = "",
    entities: str = "",
    document_id: str = "",
    site_id: str = "",
    project_id: str = "",
    contaminants: str = "",
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
    if contaminants.strip():
        filters["contaminants"] = contaminants.strip()
    return filters


def apply_ui_focus_filters(filters: dict[str, str]) -> dict[str, str]:
    """Le focus carte / chronologie prime : document_id, sinon site_id."""
    out = dict(filters)
    focused_doc = (ui_document_id.get() or "").strip()
    focused_site = (ui_site_id.get() or "").strip()
    if focused_doc:
        out["document_id"] = focused_doc
    elif focused_site:
        out["site_id"] = focused_site
    return out


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
    prefetch = chat.rag_prefetch
    score_threshold = chat.rag_score_threshold
    hybrid_text = chat.rag_hybrid_text
    include_catalog = chat.rag_include_catalog

    @tool
    def search_knowledge(
        query: str,
        doc_type: str = "",
        entities: str = "",
        document_id: str = "",
        site_id: str = "",
        project_id: str = "",
        contaminants: str = "",
        limit: int = 0,
    ) -> str:
        """Search the local RAG knowledge base for relevant document chunks.

        Call this tool for factual questions about ingested documents.
        For a project number (e.g. 2259, 4405), pass it in `entities` or
        `project_id` and include it in the query. Hybrid search also matches
        the number in chunk text if LangExtract missed it. Identity questions
        (client, firm, address) should include those words in `query`.

        Args:
            query: Natural-language search phrase (not keywords only).
            doc_type: Optional payload filter (e.g. rapport, ees_phase_1,
                ees_phase_2). Same id as LangExtract `--profile`.
            entities: Optional exact entity filter; comma-separated names
                or project numbers (e.g. 4405).
            document_id: Optional SHA-256 document id to restrict the search.
            site_id: Optional catalog site id (e.g. lot:2363352).
            project_id: Optional project number filter.
            contaminants: Optional detected contaminant filter (e.g. HAM, HAP).
            limit: Max chunks to return; 0 uses the configured default.
        """
        n = limit if limit and limit > 0 else default_limit
        n = max(1, min(int(n), _MAX_RAG_LIMIT))
        filters = apply_ui_focus_filters(
            _filters_from_args(doc_type, entities, document_id, site_id, project_id)
        )
        hits = retrieve_search(
            query,
            limit=n,
            filters=_filters_from_args(
                doc_type,
                entities,
                document_id,
                site_id,
                project_id,
                contaminants,
            )
            or None,
        )
        chunks = format_hits(
            hits,
            max_chars=hit_max_chars,
            table_max_chars=table_max_chars,
        )
        if not include_catalog:
            return chunks
        try:
            fiches = format_catalog_fiches(catalog_rows_for_search(filters, hits))
        except Exception:
            fiches = ""
        if fiches:
            return f"{fiches}\n\n{chunks}"
        return chunks

    return search_knowledge
