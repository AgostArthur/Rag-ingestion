from rag_ingestion.geocode import compose_query, geocode_address


def test_compose_query_adds_quebec():
    q = compose_query("619, route 341", "L'Épiphanie")
    assert q is not None
    assert "619" in q
    assert "Québec" in q or "Canada" in q


def test_compose_query_empty():
    assert compose_query(None) is None
    assert compose_query("  ") is None


def test_geocode_address_parses_nominatim(monkeypatch):
    payload = b'[{"lat": "45.849", "lon": "-73.482"}]'

    def opener(url, timeout):
        assert "nominatim" in url
        return payload

    coords = geocode_address("619 route 341", city="L'Épiphanie", opener=opener)
    assert coords == (45.849, -73.482)


def test_geocode_address_empty_payload(monkeypatch):
    def opener(url, timeout):
        return b"[]"

    assert geocode_address("x", opener=opener) is None
