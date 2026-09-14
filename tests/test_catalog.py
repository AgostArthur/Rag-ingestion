from pathlib import Path

from rag_ingestion.catalog import (
    document_exists,
    get_document,
    get_site,
    get_site_timeline,
    list_sites,
    parse_bbox,
    sites_geojson,
    upsert_document_meta,
)
from rag_ingestion.config import Settings
from rag_ingestion.models import DocumentMeta, TypedEvent


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks",
        embed_model="x",
        embed_dim=32,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=Path("config/langextract/prompt.txt"),
        langextract_few_shots_file=Path("config/langextract/few_shots.json"),
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=tmp_path,
        project_root=Path("."),
    )


def _meta(**overrides: object) -> DocumentMeta:
    base = dict(
        document_id="aaa",
        source_path="/tmp/a.pdf",
        parse_quality="ocr_heavy",
        project_ids=["4405"],
        title="ÉES phase II",
        doc_type="ees_phase_2",
        firm="Enviro-Experts",
        client=None,
        address="619, route 341, L'Épiphanie",
        lot_cadastral="2363352",
        city="L'Épiphanie",
        site_id="lot:2363352",
        report_date="2023-01-03",
        events=[
            TypedEvent(iso_date="2023-01-03", role="report", label="3 janvier 2023"),
            TypedEvent(iso_date="2019-08-17", role="fieldwork", label="17 août 2019"),
        ],
    )
    base.update(overrides)
    return DocumentMeta(**base)  # type: ignore[arg-type]


def test_two_projects_same_lot_share_site(tmp_path: Path):
    settings = _settings(tmp_path)
    upsert_document_meta(_meta(), settings=settings)
    upsert_document_meta(
        _meta(
            document_id="bbb",
            source_path="/tmp/b.pdf",
            project_ids=["2259"],
            firm="Géosphère",
            report_date="2024-01-03",
            events=[
                TypedEvent(iso_date="2024-01-03", role="report", label="03 janvier 2024"),
                TypedEvent(iso_date="2019-08-17", role="fieldwork", label="17 août 2019"),
            ],
        ),
        settings=settings,
    )
    a = get_document("aaa", settings=settings)
    b = get_document("bbb", settings=settings)
    assert a is not None and b is not None
    assert a["project_id"] == "4405"
    assert b["project_id"] == "2259"
    assert a["site_id"] == b["site_id"] == "lot:2363352"
    site = get_site("lot:2363352", settings=settings)
    assert site is not None
    assert set(site["document_ids"]) == {"aaa", "bbb"}
    timeline = get_site_timeline("lot:2363352", settings=settings)
    roles = {(e["iso_date"], e["role"], e["document_id"]) for e in timeline}
    assert ("2023-01-03", "report", "aaa") in roles
    assert ("2024-01-03", "report", "bbb") in roles
    assert ("2019-08-17", "fieldwork", "aaa") in roles
    assert ("2019-08-17", "fieldwork", "bbb") in roles


def test_address_only_document_joins_existing_lot_site(tmp_path: Path):
    settings = _settings(tmp_path)
    upsert_document_meta(_meta(), settings=settings)
    upsert_document_meta(
        _meta(
            document_id="ccc",
            project_ids=["9999"],
            lot_cadastral=None,
            site_id="addr:placeholder",
            firm="Autre",
        ),
        settings=settings,
    )
    row = get_document("ccc", settings=settings)
    assert row is not None
    assert row["site_id"] == "lot:2363352"


def test_reingest_replaces_events(tmp_path: Path):
    settings = _settings(tmp_path)
    upsert_document_meta(_meta(), settings=settings)
    upsert_document_meta(
        _meta(
            events=[TypedEvent(iso_date="2023-01-03", role="report", label="rapport")]
        ),
        settings=settings,
    )
    timeline = get_site_timeline("lot:2363352", settings=settings)
    assert len(timeline) == 1
    assert timeline[0]["role"] == "report"


def test_document_exists(tmp_path: Path):
    settings = _settings(tmp_path)
    assert document_exists("missing", settings=settings) is False
    upsert_document_meta(_meta(), settings=settings)
    assert document_exists("aaa", settings=settings) is True
    assert document_exists("bbb", settings=settings) is False


def test_timeline_includes_document_fields(tmp_path: Path):
    settings = _settings(tmp_path)
    upsert_document_meta(_meta(), settings=settings)
    timeline = get_site_timeline("lot:2363352", settings=settings)
    report = next(e for e in timeline if e["role"] == "report")
    assert report["title"] == "ÉES phase II"
    assert report["firm"] == "Enviro-Experts"
    assert report["project_id"] == "4405"


def test_list_sites_and_bbox(tmp_path: Path):
    settings = _settings(tmp_path)
    upsert_document_meta(_meta(), settings=settings)
    rows = list_sites(settings=settings)
    assert len(rows) == 1
    assert rows[0]["site_id"] == "lot:2363352"
    assert rows[0]["document_ids"] == ["aaa"]
    boxed = list_sites(bbox=(-80.0, 40.0, -70.0, 50.0), settings=settings)
    assert boxed == []
    assert parse_bbox(" -73.6,45.8,-73.4,45.9 ") == (-73.6, 45.8, -73.4, 45.9)


def test_sites_geojson_skips_null_coords(tmp_path: Path):
    settings = _settings(tmp_path)
    upsert_document_meta(_meta(), settings=settings)
    geo = sites_geojson(list_sites(settings=settings))
    assert geo["type"] == "FeatureCollection"
    assert geo["features"] == []


def test_geocode_on_upsert(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "rag_ingestion.geocode.geocode_address",
        lambda *a, **k: (45.849, -73.482),
    )
    settings = _settings(tmp_path)
    settings = Settings(
        **{**settings.__dict__, "geocode_enabled": True},
    )
    upsert_document_meta(_meta(), settings=settings)
    site = get_site("lot:2363352", settings=settings)
    assert site is not None
    assert site["lat"] == 45.849
    assert site["lon"] == -73.482
    assert site["geocode_status"] == "ok"
    geo = sites_geojson([site])
    assert geo["features"][0]["geometry"]["coordinates"] == [-73.482, 45.849]
