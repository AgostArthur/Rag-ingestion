from rag_ingestion.logging_setup import format_duration


def test_format_duration():
    assert format_duration(0.0004) == "<1 ms"
    assert format_duration(0.12) == "120 ms"
    assert format_duration(3.2).endswith(" s")
    assert "min" in format_duration(65)
