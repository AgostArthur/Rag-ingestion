"""Focus et citations : dérivés des ToolMessage, jamais du texte généré."""

from __future__ import annotations

import re
from typing import Any

from chatbot.graph import _is_human, _is_tool

_SCORE_RE = re.compile(r"score=([0-9.]+)")


def parse_tool_hits(content: str) -> list[dict[str, Any]]:
    """Parse le markdown émis par `format_hits` (un bloc `### Hit` par chunk)."""
    hits: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []

    def _flush() -> None:
        nonlocal current, body
        if current is None:
            return
        current["text"] = "\n".join(body).strip()
        hits.append(current)
        current = None
        body = []

    for line in (content or "").splitlines():
        if line.startswith("### Hit"):
            _flush()
            score = None
            match = _SCORE_RE.search(line)
            if match:
                try:
                    score = float(match.group(1))
                except ValueError:
                    score = None
            current = {"score": score}
            continue
        if current is None:
            continue
        if line.startswith("- ") and ":" in line:
            key, _, value = line[2:].partition(":")
            current[key.strip()] = value.strip()
            continue
        body.append(line)
    _flush()
    return hits


def _int_or_none(raw: Any) -> int | None:
    if raw is None or raw == "" or raw == "None":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def citations_from_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Citations stables pour l'API (source, page, document_id)."""
    out: list[dict[str, Any]] = []
    for hit in hits:
        document_id = str(hit.get("document_id") or "").strip()
        if not document_id:
            continue
        out.append(
            {
                "document_id": document_id,
                "source": hit.get("source") or hit.get("source_path") or "",
                "page": _int_or_none(hit.get("page")),
                "heading_path": hit.get("section") or hit.get("heading_path") or "",
                "score": hit.get("score"),
                "project_id": hit.get("project_id") or "",
                "site_id": hit.get("site_id") or "",
                "chunk_id": hit.get("chunk_id") or "",
            }
        )
    return out


def unique_keep_order(values: list[str | None]) -> list[str]:
    """Déduplique en conservant l'ordre, ignore vide / None."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        if not raw:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def tool_contents_this_turn(messages: list[Any]) -> list[str]:
    """Textes des ToolMessage depuis le dernier HumanMessage (ordre chrono)."""
    contents: list[str] = []
    for message in reversed(messages):
        if _is_human(message):
            break
        if not _is_tool(message):
            continue
        content = getattr(message, "content", "") or ""
        if not isinstance(content, str):
            continue
        if content.startswith("[skipped]"):
            continue
        contents.append(content)
    contents.reverse()
    return contents


def envelope_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """`focus` + fiches catalog + citations, à partir des hits de ce tour.

    Ne lit pas la réponse du LLM. Les `site_id` / `project_id` viennent du
    catalog SQLite (jointure sur `document_id`), avec repli sur les champs
    des hits Qdrant si la fiche n'existe pas encore.
    """
    from rag_ingestion.catalog import get_documents

    messages = result.get("messages") or []
    hits: list[dict[str, Any]] = []
    for content in tool_contents_this_turn(messages):
        hits.extend(parse_tool_hits(content))
    citations = citations_from_hits(hits)
    document_ids = unique_keep_order([c.get("document_id") for c in citations])
    documents = get_documents(document_ids)
    project_ids = unique_keep_order(
        [d.get("project_id") for d in documents]
        + [c.get("project_id") for c in citations]
    )
    site_ids = unique_keep_order(
        [d.get("site_id") for d in documents]
        + [c.get("site_id") for c in citations]
    )
    return {
        "focus": {
            "document_ids": document_ids,
            "project_ids": project_ids,
            "site_ids": site_ids,
        },
        "documents": documents,
        "citations": citations,
    }
