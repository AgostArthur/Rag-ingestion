from pathlib import Path

import pytest

from rag_ingestion.extract import load_few_shots, load_prompt

ROOT = Path(__file__).resolve().parents[1]
LANGEXTRACT = ROOT / "config" / "langextract"
BASE_PROMPT = LANGEXTRACT / "prompt.base.txt"
PROFILES = {
    "default": LANGEXTRACT / "profiles" / "default" / "few_shots.json",
    "phase_1": LANGEXTRACT / "profiles" / "phase_1" / "few_shots.json",
    "phase_2": LANGEXTRACT / "profiles" / "phase_2" / "few_shots.json",
}


def test_load_base_prompt():
    text = load_prompt(BASE_PROMPT)
    assert "title" in text
    assert "doc_type" in text
    assert "ees_phase_1" in text
    assert "ees_phase_2" in text
    assert "fieldwork" in text
    assert "kind = contaminant" in text


@pytest.mark.parametrize("profile_id,path", list(PROFILES.items()))
def test_few_shots_extraction_text_is_verbatim_substring(profile_id: str, path: Path):
    examples = load_few_shots(path)
    assert examples
    for example in examples:
        for extraction in example.extractions:
            assert extraction.extraction_text in example.text, (
                f"{profile_id}: {extraction.extraction_text!r} missing from few-shot text"
            )


def test_phase2_few_shots_include_project_ids():
    examples = load_few_shots(PROFILES["phase_2"])
    project_ids = [
        e.extraction_text
        for ex in examples
        for e in ex.extractions
        if (e.attributes or {}).get("type") == "project_id"
    ]
    assert "5982" in project_ids


def test_phase1_few_shots_include_project_id():
    examples = load_few_shots(PROFILES["phase_1"])
    project_ids = [
        e.extraction_text
        for ex in examples
        for e in ex.extractions
        if (e.attributes or {}).get("type") == "project_id"
    ]
    assert "3188" in project_ids


def test_default_few_shots_are_generic():
    examples = load_few_shots(PROFILES["default"])
    assert "Hydro-Québec" in examples[0].text
    classes = {e.extraction_class for e in examples[0].extractions}
    assert {"title", "doc_type", "entity", "date", "topic", "location"} <= classes


def test_load_few_shots_rejects_object_instead_of_list(tmp_path: Path):
    path = tmp_path / "few.json"
    path.write_text(
        '{"text": "Titre", "extractions": [{"extraction_class": "title", "extraction_text": "Titre"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty JSON list"):
        load_few_shots(path)


def test_load_few_shots_rejects_bad_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_few_shots(bad)


def test_load_few_shots_rejects_missing_text(tmp_path: Path):
    path = tmp_path / "few.json"
    path.write_text(
        '[{"extractions": [{"extraction_class": "title", "extraction_text": "x"}]}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text"):
        load_few_shots(path)
