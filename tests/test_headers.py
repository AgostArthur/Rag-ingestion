from rag_ingestion.headers import (
    detect_running_header_keys,
    is_page_number_line,
    normalize_header_line,
    strip_running_headers,
)


def _letterhead(page: int) -> str:
    return (
        "Enviro-Experts\n"
        f"ÉES phase II — Projet 4405 — Page {page}\n"
        "\n"
    )


def test_normalize_ignores_page_number():
    a = normalize_header_line("ÉES phase II — Projet 4405 — Page 2")
    b = normalize_header_line("# ÉES phase II — Projet 4405 — Page 12")
    assert a == b
    assert "4405" in a


def test_is_page_number_line():
    assert is_page_number_line("Page 12")
    assert is_page_number_line("3 / 47")
    assert is_page_number_line("— 4 —")
    assert not is_page_number_line("3. Méthodologie")
    assert not is_page_number_line("Projet 4405")


def test_page_one_keeps_header_later_pages_drop_it():
    pages = [
        _letterhead(1) + "Lettre d'accompagnement et n° de lot 2363352.",
        _letterhead(2) + "## 2. Méthodologie\n\nForages du 17 août 2019.",
        _letterhead(3) + "## 3. Résultats\n\nHydrocarbures C10-C50.",
    ]
    cleaned, stats = strip_running_headers(pages)
    assert "Enviro-Experts" in cleaned[0]
    assert "2363352" in cleaned[0]
    assert "Enviro-Experts" not in cleaned[1]
    assert "Enviro-Experts" not in cleaned[2]
    assert "Méthodologie" in cleaned[1]
    assert "Hydrocarbures" in cleaned[2]
    assert stats.pages_stripped == 2
    assert stats.lines_removed >= 4


def test_single_page_unchanged():
    pages = ["Enviro-Experts\n\nTout le rapport."]
    cleaned, stats = strip_running_headers(pages)
    assert cleaned == pages
    assert stats.lines_removed == 0


def test_two_page_only_strips_shared_top_lines():
    pages = [
        "Géosphère\n\nSommaire exécutif.",
        "Géosphère\n\n## Annexe A\n\nTableau des forages.",
    ]
    cleaned, stats = strip_running_headers(pages)
    assert cleaned[0].startswith("Géosphère")
    assert "Géosphère" not in cleaned[1]
    assert "Annexe A" in cleaned[1]
    assert stats.pages_stripped == 1


def test_unique_page_two_heading_is_kept():
    pages = [
        "Couverture seulement.",
        "## 1. Introduction\n\nTexte unique à la page 2.",
        "## 2. Terrain\n\nTexte unique à la page 3.",
    ]
    cleaned, stats = strip_running_headers(pages)
    assert "## 1. Introduction" in cleaned[1]
    assert "## 2. Terrain" in cleaned[2]
    assert stats.lines_removed == 0


def test_continued_table_not_treated_as_header():
    table = (
        "| Puits | Profondeur |\n"
        "| --- | --- |\n"
        "| F-1 | 3,0 m |\n"
    )
    pages = [
        "Rapport\n\nIntro.",
        table + "\nSuite du tableau.",
        table + "\nFin du tableau.",
    ]
    cleaned, _stats = strip_running_headers(pages)
    assert "| Puits | Profondeur |" in cleaned[1]
    assert "| Puits | Profondeur |" in cleaned[2]


def test_detect_requires_repetition():
    pages = [
        "Alpha\n\nP1",
        "Alpha\n\nP2",
        "Alpha\n\nP3",
        "Beta unique\n\nP4",
    ]
    keys = detect_running_header_keys(pages)
    assert "alpha" in keys
    assert "beta unique" not in keys
