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
    seed = settings.resolved_archive_dir() / doc_id[:16]
    seed.mkdir(parents=True)
    (seed / "kept.pdf").write_bytes(payload)
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


def test_empty_archive_reingests_even_if_catalog_has_hash(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    payload = b"%PDF-1.4 was-archived"
    pdf = settings.resolved_incoming_dir() / "again.pdf"
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

    def fake_ingest(path, *, settings=None, skip_extract=False):
        called["n"] += 1
        return IngestResult(
            document_id=doc_id,
            n_chunks=1,
            n_extractions=0,
            parse_quality="ok",
            warnings=[],
            skipped=False,
        )

    monkeypatch.setattr("rag_ingestion.inbox.ingest_path", fake_ingest)
    item = process_inbox_file(pdf, settings=settings, stable_wait=0)
    assert item.action == "ingested"
    assert called["n"] == 1
    assert not pdf.exists()


def test_successful_ingest_moves_to_archive(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    pdf = settings.resolved_incoming_dir() / "ok.pdf"
    pdf.write_bytes(b"%PDF-1.4 ok")
    doc_id = document_id_from_bytes(b"%PDF-1.4 ok")
    seen: dict[str, Path] = {}

    def fake_ingest(path, *, settings=None, skip_extract=False):
        seen["path"] = Path(path)
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


def test_failed_ingest_moves_to_failed(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    pdf = settings.resolved_incoming_dir() / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.4 bad")

    def fake_ingest(path, *, settings=None, skip_extract=False):
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
