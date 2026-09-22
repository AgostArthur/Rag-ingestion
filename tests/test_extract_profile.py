from pathlib import Path

import pytest

from rag_ingestion.config import Settings
from rag_ingestion.extract import load_few_shots, load_prompt
from rag_ingestion.extract_profile import resolve_extract_schema

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "langextract" / "profiles.json"


def _settings() -> Settings:
    return Settings(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks",
        embed_model="x",
        embed_dim=32,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=ROOT / "config" / "langextract" / "prompt.base.txt",
        langextract_few_shots_file=ROOT
        / "config"
        / "langextract"
        / "profiles"
        / "default"
        / "few_shots.json",
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=ROOT / "data",
        project_root=ROOT,
        langextract_profiles_file=PROFILES,
    )


def _resolve(name: str, markdown: str | None = None, override: str | None = None):
    return resolve_extract_schema(
        Path(name),
        markdown=markdown,
        override=override,
        settings=_settings(),
    )


def _assert_logged(caplog, *, profile_id: str, filename: str, source: str) -> None:
    assert f"type={profile_id}" in caplog.text
    assert f"fichier=« {filename} »" in caplog.text
    assert f"source={source}" in caplog.text


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ÉES Phase I.pdf", "ees_phase_1"),
        ("Rapport_phase_1.pdf", "ees_phase_1"),
        ("ESA-I_Laval.pdf", "ees_phase_1"),
        ("4405_Phase_II.pdf", "ees_phase_2"),
        ("ÉES phase 2.pdf", "ees_phase_2"),
        ("ESA-II.pdf", "ees_phase_2"),
        ("ees_phase_ii_619.pdf", "ees_phase_2"),
    ],
)
def test_filename_routes_phase_profiles(name: str, expected: str, caplog):
    with caplog.at_level("INFO"):
        schema = _resolve(name)
    assert schema.profile_id == expected
    assert schema.match_source == "filename"
    _assert_logged(caplog, profile_id=expected, filename=name, source="filename")


def test_phase_ii_does_not_match_phase_i():
    schema = _resolve("Évaluation_phase_II.pdf")
    assert schema.profile_id == "ees_phase_2"


def test_mute_filename_uses_default_and_warns(caplog):
    with caplog.at_level("WARNING"):
        schema = _resolve("scan_final.pdf")
    assert schema.profile_id == "default"
    assert schema.match_source == "default"
    _assert_logged(caplog, profile_id="default", filename="scan_final.pdf", source="default")
    assert "aucun motif de profil n'a matché" in caplog.text


def test_heading_fallback_when_filename_is_mute(caplog):
    with caplog.at_level("INFO"):
        schema = _resolve(
            "scan_final.pdf",
            markdown="# ÉES phase I\n\nHistorique d'usage du site.",
        )
    assert schema.profile_id == "ees_phase_1"
    assert schema.match_source == "heading"
    _assert_logged(
        caplog, profile_id="ees_phase_1", filename="scan_final.pdf", source="heading"
    )
    assert "aucun motif de profil n'a matché" not in caplog.text


def test_override_wins_over_filename(caplog):
    with caplog.at_level("INFO"):
        schema = _resolve("4405_Phase_II.pdf", override="ees_phase_1")
    assert schema.profile_id == "ees_phase_1"
    assert schema.match_source == "override"
    _assert_logged(
        caplog,
        profile_id="ees_phase_1",
        filename="4405_Phase_II.pdf",
        source="override",
    )


def test_override_default_skips_mute_warning(caplog):
    with caplog.at_level("INFO"):
        schema = _resolve("scan_final.pdf", override="default")
    assert schema.profile_id == "default"
    assert schema.match_source == "override"
    _assert_logged(
        caplog, profile_id="default", filename="scan_final.pdf", source="override"
    )
    assert "aucun motif de profil n'a matché" not in caplog.text


def test_unknown_override_raises():
    with pytest.raises(ValueError, match="Unknown LangExtract profile"):
        _resolve("a.pdf", override="phase3")


def test_composed_prompt_includes_base_and_addendum():
    phase2 = _resolve("phase_II.pdf")
    assert "Classes autorisées uniquement" in phase2.prompt
    assert "doc_type.normalized = ees_phase_2" in phase2.prompt
    phase1 = _resolve("phase_I.pdf")
    assert "doc_type.normalized = ees_phase_1" in phase1.prompt
    assert "campagne de forages" in phase1.prompt
    assert "lab_certificate" in phase2.prompt
    assert "information_request" in phase1.prompt


def test_profile_few_shots_are_phase_specific():
    phase2 = _resolve("phase_II.pdf")
    texts = [ex.text for ex in phase2.examples]
    assert any("5982" in text for text in texts)
    phase2_types = [
        (e.attributes or {}).get("normalized")
        for ex in phase2.examples
        for e in ex.extractions
        if e.extraction_class == "doc_type"
    ]
    assert set(phase2_types) == {"ees_phase_2"}
    phase2_roles = {
        (e.attributes or {}).get("role")
        for ex in phase2.examples
        for e in ex.extractions
        if e.extraction_class == "date"
    }
    assert {"fieldwork", "lab_certificate", "report"} <= phase2_roles
    assert "analysis" not in phase2_roles
    assert "other" not in phase2_roles
    assert any(
        (e.attributes or {}).get("kind") == "contaminant"
        for ex in phase2.examples
        for e in ex.extractions
        if e.extraction_class == "topic"
    )
    for example in phase2.examples:
        locations = [e for e in example.extractions if e.extraction_class == "location"]
        kinds = [(e.attributes or {}).get("type") for e in locations]
        assert kinds.count("address") == 1
        assert kinds.count("lot") == 1
        assert "city" not in kinds
        assert all("CSA" not in (e.extraction_text or "") for e in example.extractions)

    phase1 = _resolve("phase_I.pdf")
    texts = [ex.text for ex in phase1.examples]
    assert any("3188" in text for text in texts)
    phase1_types = [
        (e.attributes or {}).get("normalized")
        for ex in phase1.examples
        for e in ex.extractions
        if e.extraction_class == "doc_type"
    ]
    assert set(phase1_types) == {"ees_phase_1"}
    phase1_roles = {
        (e.attributes or {}).get("role")
        for ex in phase1.examples
        for e in ex.extractions
        if e.extraction_class == "date"
    }
    assert {"contract", "site_visit", "report"} <= phase1_roles
    assert "analysis" not in phase1_roles
    assert "other" not in phase1_roles
    assert any(
        (e.attributes or {}).get("kind") == "contaminant"
        for ex in phase1.examples
        for e in ex.extractions
        if e.extraction_class == "topic"
    )
    for example in phase1.examples:
        locations = [e for e in example.extractions if e.extraction_class == "location"]
        kinds = [(e.attributes or {}).get("type") for e in locations]
        assert kinds.count("address") == 1
        assert kinds.count("lot") == 1
        assert "city" not in kinds
        assert all("CSA" not in (e.extraction_text or "") for e in example.extractions)

    default = _resolve("notes.pdf")
    assert "Hydro-Québec" in default.examples[0].text
    assert all("4405" not in ex.text for ex in default.examples)
