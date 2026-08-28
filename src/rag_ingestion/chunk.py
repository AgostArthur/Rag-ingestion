"""Chunker markdown-aware : fenêtres sur le texte original (offsets stables)."""

from __future__ import annotations

import re

from rag_ingestion.models import Chunk, PageSpan
from rag_ingestion.parse import page_for_span

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def _heading_path_at(text: str, pos: int) -> str:
    """Fil d'Ariane des titres Markdown visibles à la position `pos`."""
    stack: dict[int, str] = {}
    for match in _HEADING_RE.finditer(text):
        if match.start() > pos:
            break
        level = len(match.group(1))
        title = match.group(2).strip()
        stack = {k: v for k, v in stack.items() if k < level}
        stack[level] = title
    ordered = [stack[k] for k in sorted(stack)]
    return " > ".join(ordered)


def _table_block_ranges(text: str) -> list[tuple[int, int]]:
    """Intervalles `[start, end)` des tableaux Markdown (lignes `| … |`)."""
    ranges: list[tuple[int, int]] = []
    in_table = False
    start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        is_table = bool(_TABLE_LINE_RE.match(line.rstrip("\n")))
        if is_table and not in_table:
            in_table = True
            start = offset
        elif in_table and not is_table:
            ranges.append((start, offset))
            in_table = False
        offset += len(line)
    if in_table:
        ranges.append((start, len(text)))
    return ranges


def _inside_table(tables: list[tuple[int, int]], pos: int) -> tuple[int, int] | None:
    """Tableau qui contient strictement `pos`, ou None."""
    for start, end in tables:
        if start < pos < end:
            return start, end
    return None


def _snap_end(text: str, start: int, end: int, tables: list[tuple[int, int]]) -> int:
    """Recule `end` sur une frontière de paragraphe, sans couper un tableau."""
    if end >= len(text):
        return len(text)
    table = _inside_table(tables, end)
    if table is not None:
        _t_start, t_end = table
        if t_end - start <= (end - start) * 2:
            return min(t_end, len(text))
        return table[0] if table[0] > start else end

    window = text[start:end]
    snap = window.rfind("\n\n")
    if snap != -1 and snap > len(window) // 3:
        return start + snap
    snap_nl = window.rfind("\n")
    if snap_nl != -1 and snap_nl > len(window) // 2:
        return start + snap_nl
    return end


def chunk_markdown(
    text: str,
    *,
    document_id: str,
    max_chars: int = 2400,
    overlap_chars: int = 300,
    page_spans: list[PageSpan] | None = None,
) -> list[Chunk]:
    """Découpe le Markdown en fenêtres sur le texte original (offsets stables).

    Chaque chunk vérifie `text[start:end] == chunk.text`. Chevauchement et
    titres Markdown sont conservés ; les tableaux ne sont pas coupés au milieu
    si possible.

    Args:
        text: Markdown source (sortie LiteParse).
        document_id: Préfixe des `chunk_id` (`{document_id}:{index}`).
        max_chars: Taille cible d'une fenêtre.
        overlap_chars: Chevauchement entre deux fenêtres successives.
        page_spans: Offsets de pages pour renseigner `chunk.page`.

    Returns:
        Liste de chunks non vides, dans l'ordre du document.
    """
    if not text or not text.strip():
        return []

    n = len(text)
    tables = _table_block_ranges(text)
    chunks: list[Chunk] = []
    i = 0
    index = 0
    min_step = max(1, max_chars // 8)

    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        raw_end = min(i + max_chars, n)
        end = _snap_end(text, i, raw_end, tables)
        if end <= i:
            end = min(i + max_chars, n)
        slice_ = text[i:end]
        chunk_text = slice_.strip()
        if not chunk_text:
            i = end if end > i else i + 1
            continue
        leading = len(slice_) - len(slice_.lstrip())
        trailing = len(slice_) - len(slice_.rstrip())
        start = i + leading
        real_end = end - trailing
        chunk_id = f"{document_id}:{index}"
        page = page_for_span(page_spans or [], start, real_end) if page_spans else None
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=index,
                text=text[start:real_end],
                start=start,
                end=real_end,
                heading_path=_heading_path_at(text, start),
                page=page,
            )
        )
        index += 1
        if real_end >= n:
            break
        next_i = real_end - overlap_chars
        if next_i <= i:
            next_i = i + min_step
        i = max(next_i, i + min_step)

    return chunks
