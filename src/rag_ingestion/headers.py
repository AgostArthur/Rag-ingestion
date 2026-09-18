"""En-têtes PDF répétitifs : on garde la page 1, on les retire ensuite.

Les bandeaux (firme, n° de projet, « Page 12 ») polluent chunks et LangExtract.
La page 1 les conserve : titre, lot et identité y sont souvent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_MAX_HEADER_LINES = 8
_MAX_HEADER_LINE_LEN = 160
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_PAGE_NUM_LINE_RE = re.compile(
    r"(?i)^(?:(?:page|p\.?)\s*\d+(?:\s*(?:/|de)\s*\d+)?|"
    r"\d+\s*/\s*\d+|"
    r"[-–—]\s*\d+\s*[-–—])\s*$"
)
_PAGE_TOKEN_RE = re.compile(
    r"(?i)(?:page|p\.?)\s*\d+(?:\s*(?:/|de)\s*\d+)?|\d+\s*/\s*\d+"
)


@dataclass(frozen=True)
class HeaderStripStats:
    """Combien de chrome a été retiré des pages 2+."""

    header_keys: tuple[str, ...]
    pages_stripped: int
    lines_removed: int


def normalize_header_line(line: str) -> str:
    """Clé comparable : casse, markdown, numéros de page ignorés."""
    text = line.strip()
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = text.replace("*", "").replace("_", "")
    text = _PAGE_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def is_page_number_line(line: str) -> bool:
    """True si la ligne n'est qu'un folio (Page 12, 3/47, — 4 —)."""
    return bool(_PAGE_NUM_LINE_RE.match(line.strip()))


def _leading_candidate_lines(page: str, limit: int = _MAX_HEADER_LINES) -> list[str]:
    found: list[str] = []
    for raw in page.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _TABLE_LINE_RE.match(line):
            break
        if len(line) > _MAX_HEADER_LINE_LEN:
            break
        found.append(line)
        if len(found) >= limit:
            break
    return found


def _page_header_keys(page: str) -> set[str]:
    keys: set[str] = set()
    for line in _leading_candidate_lines(page):
        if is_page_number_line(line):
            keys.add("<page#>")
            continue
        key = normalize_header_line(line)
        if len(key) >= 2:
            keys.add(key)
    return keys


def detect_running_header_keys(pages: list[str]) -> set[str]:
    """Lignes d'en-tête qui se répètent en tête des pages après la première."""
    body = pages[1:]
    if not body:
        return set()
    per_page = [_page_header_keys(page) for page in body]
    n = len(per_page)
    page1_keys = _page_header_keys(pages[0]) if pages else set()
    if n == 1:
        return {k for k in per_page[0] if k == "<page#>" or k in page1_keys}
    threshold = (n // 2) + 1
    counts: Counter[str] = Counter()
    for keys in per_page:
        counts.update(keys)
    running: set[str] = set()
    for key, count in counts.items():
        if key == "<page#>":
            running.add(key)
            continue
        if count >= threshold:
            running.add(key)
    return running


def _strip_leading_headers(page: str, header_keys: set[str]) -> tuple[str, int]:
    if not header_keys:
        return page, 0
    lines = page.splitlines(keepends=True)
    index = 0
    removed = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue
        if removed >= _MAX_HEADER_LINES:
            break
        if _TABLE_LINE_RE.match(stripped):
            break
        is_folio = is_page_number_line(stripped)
        key = "<page#>" if is_folio else normalize_header_line(stripped)
        if key in header_keys or is_folio:
            index += 1
            removed += 1
            continue
        break
    rest = "".join(lines[index:]).strip()
    return rest, removed


def strip_running_headers(pages: list[str]) -> tuple[list[str], HeaderStripStats]:
    """Page 1 inchangée ; pages suivantes sans bandeau répétitif.

    Args:
        pages: Markdown d'une page chacune, déjà `strip`.

    Returns:
        Pages nettoyées et un compte-rendu (clés, pages touchées, lignes ôtées).
    """
    if len(pages) <= 1:
        return list(pages), HeaderStripStats((), 0, 0)
    header_keys = detect_running_header_keys(pages)
    if not header_keys:
        return list(pages), HeaderStripStats((), 0, 0)
    cleaned = [pages[0]]
    pages_stripped = 0
    lines_removed = 0
    for page in pages[1:]:
        rest, removed = _strip_leading_headers(page, header_keys)
        cleaned.append(rest)
        if removed:
            pages_stripped += 1
            lines_removed += removed
    keys = tuple(sorted(k for k in header_keys if k != "<page#>"))
    return cleaned, HeaderStripStats(keys, pages_stripped, lines_removed)
