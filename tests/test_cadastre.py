from rag_ingestion.cadastre import (
    format_cadastre_no_lot,
    geometry_centroid,
    lookup_lot_geometry,
)


def test_format_cadastre_no_lot_groups_from_right():
    assert format_cadastre_no_lot("2363352") == "2 363 352"
    assert format_cadastre_no_lot("lot 2 217 361") == "2 217 361"
    assert format_cadastre_no_lot("12") is None
    assert format_cadastre_no_lot(None) is None


def test_geometry_centroid_exterior_ring():
    geom = {
        "type": "Polygon",
        "coordinates": [
            [
                [-73.0, 45.0],
                [-73.0, 46.0],
                [-72.0, 46.0],
                [-72.0, 45.0],
                [-73.0, 45.0],
            ]
        ],
    }
    lat, lon = geometry_centroid(geom)
    assert lat == 45.5
    assert lon == -72.5


def test_lookup_lot_geometry_parses_geojson():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"NO_LOT": "2 363 352"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-73.51, 45.85],
                            [-73.51, 45.86],
                            [-73.50, 45.86],
                            [-73.50, 45.85],
                            [-73.51, 45.85],
                        ]
                    ],
                },
            }
        ],
    }

    def opener(url, timeout):
        assert "NO_LOT" in url
        assert "363" in url
        return json_bytes(payload)

    found = lookup_lot_geometry("2363352", opener=opener)
    assert found is not None
    assert found["no_lot"] == "2 363 352"
    assert found["geometry"]["type"] == "Polygon"
    assert 45.85 < found["lat"] < 45.86
    assert -73.51 < found["lon"] < -73.50


def json_bytes(obj):
    import json

    return json.dumps(obj).encode("utf-8")


def test_lookup_lot_geometry_empty():
    def opener(url, timeout):
        return b'{"type":"FeatureCollection","features":[]}'

    assert lookup_lot_geometry("2363352", opener=opener) is None
