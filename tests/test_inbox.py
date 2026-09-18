from pathlib import Path

from rag_ingestion.catalog import upsert_document_meta
from rag_ingestion.config import Settings
from rag_ingestion.inbox import list_inbox_pdfs, process_inbox, process_inbox_file
from rag_ingestion.models import DocumentMeta, IngestResult
from rag_ingestion.parse import document_id_from_bytes


def _settings(tmp_path: Path) -> Settings:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    return Settings(
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
        incoming_dir=incoming,
        archive_dir=tmp_path / "data" / "archive",
        failed_dir=tmp_path / "data" / "failed",
        ingest_skip_existing=True,
        ingest_skip_extract=True,
    )


def test_list_inbox_skips_hidden_and_non_pdf(tmp_path: Path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "ok.pdf").write_bytes(b"%PDF")
    (incoming / ".hidden.pdf").write_bytes(b"%PDF")
    (incoming / "notes.txt").write_text("no")
    (incoming / "sub").mkdir()
    (incoming / "sub" / "nested.pdf").write_bytes(b"%PDF")
    names = {p.name for p in list_inbox_pdfs(incoming)}
    assert names == {"ok.pdf", "nested.pdf"}


def test_duplicate_hash_is_archived_without_ingest(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    payload = b"%PDF-1.4 duplicate-body"
    pdf = settings.resolved_incoming_dir() / "report.pdf"
    pdf.write_bytes(payload)
    doc_id = document_id_from_bytes(payload)
    upsert_document_meta(
        DocumentMeta(
            document_id=doc_id,
            source_path=str(pdf),
            parse_quality="ok",
        ),
        settings=settings,
    )
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("ingest_path should not run for duplicates")

    monkeypatch.setattr("rag_ingestion.inbox.ingest_path", boom)
    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "skipped_duplicate"
    assert item.document_id == doc_id
    assert not pdf.exists()
    assert item.dest is not None and item.dest.is_file()
    assert called["n"] == 0


def test_same_document_id_does_not_duplicate_archive(tmp_path: Path, monkeypatch):
    """Ré-drop du même SHA → une seule copie sous archive/{sha16}/."""
    settings = _settings(tmp_path)
    payload = b"%PDF-1.4 same-sha-no-dup"
    doc_id = document_id_from_bytes(payload)
    archive_folder = settings.resolved_archive_dir() / doc_id[:16]

    def fake_ingest(path, *, settings=None, skip_extract=False, trigger="cli", dest_path=None):
        return IngestResult(
            document_id=doc_id,
            n_chunks=1,
            n_extractions=0,
            parse_quality="ok",
            warnings=[],
            skipped=False,
        )

    monkeypatch.setattr("rag_ingestion.inbox.ingest_path", fake_ingest)

    # Premier ingest → archive créée.
    first = settings.resolved_incoming_dir() / "report.pdf"
    first.write_bytes(payload)
    item1 = process_inbox_file(first, settings=settings, stable_wait=0)
    assert item1.action == "ingested"
    assert item1.dest is not None
    assert list(archive_folder.glob("*.pdf")) == [item1.dest]

    # Enregistrer au catalog pour simuler un skip doublon.
    upsert_document_meta(
        DocumentMeta(document_id=doc_id, source_path=str(item1.dest), parse_quality="ok"),
        settings=settings,
    )

    # Second drop (même contenu, autre nom) → pas de nouvelle copie.
    second = settings.resolved_incoming_dir() / "report_copy.pdf"
    second.write_bytes(payload)
    item2 = process_inbox_file(second, settings=settings, stable_wait=0)
    assert item2.action == "skipped_duplicate"
    assert not second.exists()
    assert item2.dest == item1.dest
    assert list(archive_folder.glob("*.pdf")) == [item1.dest]

    # Force re-ingest → toujours une seule copie, ingest depuis l'archive existante.
    third = settings.resolved_incoming_dir() / "again.pdf"
    third.write_bytes(payload)
    item3 = process_inbox_file(third, settings=settings, force=True, stable_wait=0)
    assert item3.action == "ingested"
    assert not third.exists()
    assert item3.dest == item1.dest
    assert list(archive_folder.glob("*.pdf")) == [item1.dest]


def test_successful_ingest_moves_to_archive(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    pdf = settings.resolved_incoming_dir() / "ok.pdf"
    pdf.write_bytes(b"%PDF-1.4 ok")
    doc_id = document_id_from_bytes(b"%PDF-1.4 ok")
    seen: dict[str, Path] = {}

    def fake_ingest(path, *, settings=None, skip_extract=False, trigger="cli", dest_path=None):
        seen["path"] = Path(path)
        seen["trigger"] = trigger
        return IngestResult(
            document_id=doc_id,
            n_chunks=3,
            n_extractions=0,
            parse_quality="ok",
            warnings=[],
            skipped=False,
        )

    monkeypatch.setattr("rag_ingestion.inbox.ingest_path", fake_ingest)
    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "ingested"
    assert not pdf.exists()
    assert item.dest is not None
    assert item.dest.is_file()
    assert doc_id[:16] in str(item.dest)
    assert seen["path"] == item.dest
    assert seen["trigger"] == "inbox"


def test_failed_ingest_moves_to_failed(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    pdf = settings.resolved_incoming_dir() / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.4 bad")

    def fake_ingest(path, *, settings=None, skip_extract=False, trigger="cli", dest_path=None):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr("rag_ingestion.inbox.ingest_path", fake_ingest)
    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "failed"
    assert not pdf.exists()
    assert item.dest is not None
    assert item.dest.is_file()
    assert "failed" in str(item.dest)


def test_process_inbox_empty_dir(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "rag_ingestion.inbox.ingest_path",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no files")),
    )
    assert process_inbox(settings=settings, stable_wait=0) == []


def test_duplicate_writes_jsonl_log(tmp_path: Path, monkeypatch):
    """Un doublon de SHA-256 doit écrire une ligne JSONL avec status=skipped_duplicate."""
    import json

    settings = _settings(tmp_path)
    payload = b"%PDF-1.4 dup-log"
    pdf = settings.resolved_incoming_dir() / "dup.pdf"
    pdf.write_bytes(payload)
    doc_id = document_id_from_bytes(payload)
    upsert_document_meta(
        DocumentMeta(document_id=doc_id, source_path=str(pdf), parse_quality="ok"),
        settings=settings,
    )
    monkeypatch.setattr(
        "rag_ingestion.inbox.ingest_path",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "skipped_duplicate"

    log_path = settings.resolved_ingest_log_path()
    assert log_path.exists(), "Le journal JSONL doit être créé"
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["status"] == "skipped_duplicate"
    assert rec["trigger"] == "inbox"
    assert rec["document_id"] == doc_id
    assert rec["filename"] == "dup.pdf"


def test_unstable_writes_jsonl_log(tmp_path: Path):
    """Un fichier instable doit écrire une ligne JSONL avec status=unstable."""
    import json

    settings = _settings(tmp_path)
    pdf = settings.resolved_incoming_dir() / "unstable.pdf"
    # Fichier vide → file_is_stable retourne False (st_size <= 0)
    pdf.write_bytes(b"")

    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "unstable"

    log_path = settings.resolved_ingest_log_path()
    assert log_path.exists()
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["status"] == "unstable"
    assert lines[0]["trigger"] == "inbox"


def test_inbox_ingest_writes_single_jsonl_line(tmp_path: Path, monkeypatch):
    """Un ingest inbox réussi → exactement une ligne JSONL (écrite par pipeline, pas inbox)."""
    import json

    settings = _settings(tmp_path)
    pdf = settings.resolved_incoming_dir() / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 single")
    doc_id = document_id_from_bytes(b"%PDF-1.4 single")
    log_calls: list[dict] = []

    def fake_ingest(path, *, settings=None, skip_extract=False, trigger="cli", dest_path=None):
        # Simule l'écriture du log par le pipeline.
        from rag_ingestion.ingest_log import append_ingest_log, build_pipeline_record

        rec = build_pipeline_record(
            status="ingested",
            trigger=trigger,
            source_path=str(path),
            dest_path=str(dest_path) if dest_path else None,
            document_id=doc_id,
            filename=path.name,
            profile_id="ees_phase_2",
            profile_match_source="filename",
            skip_extract=skip_extract,
            langextract_model="x",
            embed_model="x",
            qdrant_collection="chunks",
            n_pages=5,
            n_needs_ocr=0,
            parse_quality="native",
            n_chunks=3,
            n_extractions=2,
            doc_type="ees_phase_2",
            site_id="lot:123",
            project_id="4405",
            step_seconds={"parse": 0.1},
            total_seconds=0.5,
            warnings=[],
            error=None,
        )
        log_calls.append(rec)
        append_ingest_log(rec, settings.resolved_ingest_log_path())
        return IngestResult(
            document_id=doc_id, n_chunks=3, n_extractions=2, parse_quality="native", warnings=[]
        )

    monkeypatch.setattr("rag_ingestion.inbox.ingest_path", fake_ingest)
    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "ingested"

    log_path = settings.resolved_ingest_log_path()
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    # Une seule ligne : écrite par le pipeline simulé, pas une deuxième par l'inbox.
    assert len(lines) == 1
    rec = lines[0]
    assert rec["status"] == "ingested"
    assert rec["trigger"] == "inbox"
    assert rec["profile_id"] == "ees_phase_2"
    assert rec["n_pages"] == 5
    assert rec["doc_type"] == "ees_phase_2"


def test_watch_parser_flags():
    from rag_ingestion.cli import build_parser, cmd_watch

    args = build_parser().parse_args(
        ["watch", "--once", "--force", "--poll", "3", "--incoming", "/tmp/in"]
    )
    assert args.once is True
    assert args.force is True
    assert args.poll == 3
    assert args.incoming == "/tmp/in"
    assert args.func is cmd_watch
