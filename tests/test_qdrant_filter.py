from qdrant_client.models import FieldCondition, MatchAny, MatchValue

from rag_ingestion.qdrant_store import build_filter


def test_build_filter_single_and_multi():
    f = build_filter({"doc_type": "rapport", "entities": "ACME,Hydro-Québec"})
    assert f is not None
    assert len(f.must) == 2
    first, second = f.must
    assert isinstance(first, FieldCondition)
    assert isinstance(first.match, MatchValue)
    assert first.match.value == "rapport"
    assert isinstance(second.match, MatchAny)
    assert second.match.any == ["ACME", "Hydro-Québec"]


def test_build_filter_empty():
    assert build_filter({}) is None
    assert build_filter({"doc_type": ""}) is None


def test_build_filter_contaminants():
    f = build_filter({"contaminants": "HAM,HAP", "doc_type": "ees_phase_2"})
    assert f is not None
    assert len(f.must) == 2
    first, second = f.must
    assert first.key == "contaminants"
    assert isinstance(first.match, MatchAny)
    assert first.match.any == ["HAM", "HAP"]
    assert second.key == "doc_type"
    assert second.match.value == "ees_phase_2"

