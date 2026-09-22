"""Tests du module ingest_log (append JSONL, verrou, schéma, best-effort)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from rag_ingestion.ingest_log import (
    append_ingest_log,
    build_inbox_record,
    build_pipeline_record,
    now_utc_iso,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pipeline_record(**overrides) -> dict:
    base = dict(
        status="ingested",
        trigger="cli",
        source_path="/tmp/doc.pdf",
        dest_path=None,
        document_id="abc123",
        filename="doc.pdf",
        profile_id="ees_phase_2",
        profile_match_source="filename",
        skip_extract=False,
        langextract_model="gemma4:e4b",
        embed_model="paraphrase-multilingual",
        qdrant_collection="chunks",
        n_pages=10,
        n_needs_ocr=2,
        parse_quality="native",
        n_chunks=5,
        n_extractions=3,
        doc_type="ees_phase_2",
        site_id="lot:2363352",
        project_id="4405",
        step_seconds={"parse": 1.0, "embed": 0.5},
        total_seconds=2.1,
        warnings=[],
        error=None,
    )
    base.update(overrides)
    return build_pipeline_record(**base)


# ---------------------------------------------------------------------------
# now_utc_iso
# ---------------------------------------------------------------------------

def test_now_utc_iso_format():
    ts = now_utc_iso()
    # Format attendu : 2026-09-17T18:50:12Z
    assert len(ts) == 20
    assert ts.endswith("Z")
    assert ts[10] == "T"


# ---------------------------------------------------------------------------
# append_ingest_log
# ---------------------------------------------------------------------------

def test_append_creates_file_and_valid_json(tmp_path: Path):
    log = tmp_path / "log_ingest.jsonl"
    record = _minimal_pipeline_record()
    append_ingest_log(record, log)

    assert log.exists()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["status"] == "ingested"
    assert parsed["trigger"] == "cli"
    assert parsed["document_id"] == "abc123"


def test_append_multiple_lines(tmp_path: Path):
    log = tmp_path / "log_ingest.jsonl"
    for i in range(5):
        append_ingest_log(_minimal_pipeline_record(document_id=f"sha{i}"), log)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert json.loads(line)["document_id"] == f"sha{i}"


def test_append_creates_parent_dirs(tmp_path: Path):
    log = tmp_path / "nested" / "dir" / "log_ingest.jsonl"
    append_ingest_log(_minimal_pipeline_record(), log)
    assert log.exists()


def test_append_bestteffort_on_readonly_dir(tmp_path: Path):
    """Une erreur d'I/O ne lève pas d'exception (best-effort)."""
    bad_log = Path("/no/such/path/that/exists/log_ingest.jsonl")
    # Ne doit pas lever
    append_ingest_log(_minimal_pipeline_record(), bad_log)


def test_concurrent_appends_no_corruption(tmp_path: Path):
    """100 threads en parallèle → 100 lignes JSON valides."""
    log = tmp_path / "log_ingest.jsonl"
    errors: list[Exception] = []

    def _write(i: int) -> None:
        try:
            append_ingest_log(_minimal_pipeline_record(document_id=f"sha{i:04d}"), log)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    ids = {json.loads(l)["document_id"] for l in lines}
    assert len(ids) == 100, "Chaque enregistrement doit être distinct et complet"


# ---------------------------------------------------------------------------
# build_pipeline_record
# ---------------------------------------------------------------------------

def test_pipeline_record_all_keys(tmp_path: Path):
    rec = _minimal_pipeline_record()
    required_keys = {
        "ts", "status", "trigger", "filename", "source_path", "dest_path",
        "document_id", "profile_id", "profile_match_source", "skip_extract",
        "langextract_model", "embed_model", "qdrant_collection",
        "n_pages", "n_needs_ocr", "parse_quality",
        "n_chunks", "n_extractions", "doc_type",
        "site_id", "project_id",
        "step_seconds", "total_seconds", "warnings", "error",
    }
    assert required_keys.issubset(rec.keys())


def test_pipeline_record_step_seconds_rounded(tmp_path: Path):
    rec = build_pipeline_record(
        **{**dict(
            status="ingested", trigger="cli", source_path="/p", dest_path=None,
            document_id="x", filename="x.pdf", profile_id=None, profile_match_source=None,
            skip_extract=False, langextract_model="m", embed_model="e", qdrant_collection="c",
            n_pages=1, n_needs_ocr=0, parse_quality="ok", n_chunks=1, n_extractions=0,
            doc_type=None, site_id=None, project_id=None,
            step_seconds={"parse": 1.23456789},
            total_seconds=1.99999999,
            warnings=[], error=None,
        )}
    )
    assert rec["step_seconds"]["parse"] == round(1.23456789, 4)
    assert rec["total_seconds"] == round(1.99999999, 4)


# ---------------------------------------------------------------------------
# build_inbox_record
# ---------------------------------------------------------------------------

def test_inbox_record_structure():
    rec = build_inbox_record(
        status="skipped_duplicate",
        source_path="/tmp/in/report.pdf",
        dest_path="/data/archive/abc/report.pdf",
        document_id="abc123",
        filename="report.pdf",
        detail="already in catalog",
    )
    assert rec["status"] == "skipped_duplicate"
    assert rec["trigger"] == "inbox"
    assert rec["document_id"] == "abc123"
    assert rec["dest_path"] == "/data/archive/abc/report.pdf"
    assert rec["profile_id"] is None
    assert rec["n_chunks"] == 0
    assert rec["step_seconds"] == {}


def test_inbox_record_unstable():
    rec = build_inbox_record(
        status="unstable",
        source_path="/tmp/in/partial.pdf",
        dest_path=None,
        document_id="",
        filename="partial.pdf",
        detail="file still changing",
    )
    assert rec["status"] == "unstable"
    assert rec["dest_path"] is None
    assert rec["error"] == "file still changing"


# ---------------------------------------------------------------------------
# Intégration config : resolved_ingest_log_path
# ---------------------------------------------------------------------------

def test_resolved_ingest_log_path_default(tmp_path: Path):
    from rag_ingestion.config import Settings

    s = Settings(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks",
        embed_model="x",
        embed_dim=32,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=Path("config/langextract/prompt.base.txt"),
        langextract_few_shots_file=Path("config/langextract/profiles/default/few_shots.json"),
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=tmp_path / "data",
        project_root=tmp_path,
    )
    assert s.resolved_ingest_log_path() == tmp_path / "data" / "log_ingest.jsonl"


def test_resolved_ingest_log_path_custom(tmp_path: Path):
    from rag_ingestion.config import Settings

    custom = tmp_path / "logs" / "log_ingest.jsonl"
    s = Settings(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks",
        embed_model="x",
        embed_dim=32,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=Path("config/langextract/prompt.base.txt"),
        langextract_few_shots_file=Path("config/langextract/profiles/default/few_shots.json"),
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=tmp_path / "data",
        project_root=tmp_path,
        ingest_log_file=custom,
    )
    assert s.resolved_ingest_log_path() == custom


def test_ingest_log_file_env_relative_under_data_dir(tmp_path: Path, monkeypatch):
    """``INGEST_LOG_FILE=data/log_ingest.jsonl`` → ``{DATA_DIR}/log_ingest.jsonl`` (Docker-safe)."""
    from rag_ingestion import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("INGEST_LOG_FILE", "data/log_ingest.jsonl")
    # Évite de relire le .env du dépôt pour ce test.
    monkeypatch.setattr(cfg, "_load_dotenv_file", lambda: None)

    s = cfg.load_settings()
    assert s.resolved_ingest_log_path() == data_dir / "log_ingest.jsonl"


def test_langextract_max_char_buffer_default(tmp_path: Path, monkeypatch):
    from rag_ingestion import config as cfg

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("LANGEXTRACT_MAX_CHAR_BUFFER", raising=False)
    monkeypatch.setattr(cfg, "_load_dotenv_file", lambda: None)
    s = cfg.load_settings()
    assert s.langextract_max_char_buffer == 10000


def test_langextract_max_char_buffer_from_env(tmp_path: Path, monkeypatch):
    from rag_ingestion import config as cfg

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LANGEXTRACT_MAX_CHAR_BUFFER", "8000")
    monkeypatch.setattr(cfg, "_load_dotenv_file", lambda: None)
    s = cfg.load_settings()
    assert s.langextract_max_char_buffer == 8000
