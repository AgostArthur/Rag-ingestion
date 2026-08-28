from pathlib import Path

import pytest

from rag_ingestion.extract import load_few_shots, load_prompt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "config" / "langextract" / "prompt.txt"
DEFAULT_FEW = ROOT / "config" / "langextract" / "few_shots.json"


def test_load_default_prompt():
    text = load_prompt(DEFAULT_PROMPT)
    assert "title" in text
    assert "doc_type" in text


def test_load_default_few_shots():
    examples = load_few_shots(DEFAULT_FEW)
    assert len(examples) >= 2
    assert "Hydro-Québec" in examples[0].text
    classes = {e.extraction_class for e in examples[0].extractions}
    assert {"title", "doc_type", "entity", "date", "topic", "location"} <= classes
    project_ids = [
        e.extraction_text
        for ex in examples
        for e in ex.extractions
        if (e.attributes or {}).get("type") == "project_id"
    ]
    assert "4405" in project_ids
    assert "2259" in project_ids


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
